import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


REQUIRED_WIDE = ["date", "ticker", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]
REQUIRED_LONG = ["date", "ticker", "field", "value"]


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_px: float
    shares: float
    stop: float
    target: float
    time_stop_date: pd.Timestamp
    atr_entry: float
    max_high: float
    min_low: float


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    universe = pd.read_csv(config["universe_path"])["ticker"].astype(str).tolist()
    px = _read_table(Path(config["prices_path"]))
    bench = _read_table(Path(config["benchmark_path"]))

    if set(REQUIRED_LONG).issubset(px.columns):
        px = px.pivot_table(index=["date", "ticker"], columns="field", values="value", aggfunc="last").reset_index()
    elif not set(REQUIRED_WIDE).issubset(px.columns):
        raise ValueError(f"Price file must contain either long columns {REQUIRED_LONG} or wide columns {REQUIRED_WIDE}")

    for c in REQUIRED_WIDE:
        if c not in px.columns:
            raise ValueError(f"Missing required column in price data: {c}")

    px["date"] = pd.to_datetime(px["date"])
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench[["date", "PX_LAST"]].rename(columns={"PX_LAST": "bench_px"})

    px = px[px["ticker"].astype(str).isin(universe)].copy()
    px["ticker"] = px["ticker"].astype(str)
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)

    logging.info("Loaded price rows=%s, universe=%s tickers", len(px), len(universe))
    return px, bench, universe


def _calc_features_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c = g["PX_LAST"]
    h = g["PX_HIGH"]
    l = g["PX_LOW"]
    v = g["PX_VOLUME"].clip(lower=0)

    g["ret_1d"] = c.pct_change(1)
    for p in [5, 21, 63, 126, 252]:
        g[f"ret_{p}d"] = c.pct_change(p)

    g["mom_12_1"] = (1 + g["ret_252d"]) / (1 + g["ret_21d"]) - 1
    g["mom_6_1"] = (1 + g["ret_126d"]) / (1 + g["ret_21d"]) - 1
    g["mom_3m"] = g["ret_63d"]
    g["rev_5d"] = -g["ret_5d"]

    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    g["atr_14"] = tr.rolling(14, min_periods=14).mean()
    g["atrp_14"] = g["atr_14"] / c.replace(0, np.nan)
    g["vol_20"] = g["ret_1d"].rolling(20, min_periods=20).std()

    dollar_vol = c * v
    g["dv_20"] = dollar_vol.rolling(20, min_periods=20).mean()
    g["illiq_amihud_20"] = (g["ret_1d"].abs() / dollar_vol.replace(0, np.nan)).rolling(20, min_periods=20).mean()

    sma50 = v.rolling(50, min_periods=20).mean()
    g["dryup"] = v / sma50.replace(0, np.nan)
    g["pivot_high_20"] = c.rolling(20, min_periods=20).max().shift(1)
    g["breakout"] = (c > g["pivot_high_20"]).astype(float)

    for fwd in [5, 10, 20]:
        g[f"fwd_ret_{fwd}d"] = c.shift(-fwd) / c - 1
    return g


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    feat = prices.groupby("ticker", group_keys=False).apply(_calc_features_ticker)
    feat = feat.reset_index(drop=True)
    logging.info("Features built rows=%s", len(feat))
    return feat


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def build_cross_sectional_scores(feat: pd.DataFrame, config: Dict) -> pd.DataFrame:
    k = config["signal"]["risk_k"]

    def per_day(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        liq_cut = g["dv_20"].quantile(0.2)
        g["eligible"] = (g["dv_20"] >= liq_cut)
        required = ["mom_12_1", "mom_6_1", "mom_3m", "rev_5d", "vol_20", "atr_14", "PX_OPEN"]
        g["eligible"] &= g[required].notna().all(axis=1)

        elig = g[g["eligible"]].copy()
        if len(elig) == 0:
            g["alpha_score"] = np.nan
            g["rank"] = np.nan
            g["decile"] = np.nan
            return g

        core = (
            _zscore(elig["mom_12_1"]) * 0.4
            + _zscore(elig["mom_6_1"]) * 0.3
            + _zscore(elig["mom_3m"]) * 0.2
            + _zscore(elig["rev_5d"]) * 0.1
        )
        core = core / (1 + k * elig["vol_20"].fillna(0))
        core = core + 0.3 * elig["breakout"].fillna(0)

        elig["alpha_score"] = core
        elig = elig.sort_values("alpha_score", ascending=False)
        elig["rank"] = np.arange(1, len(elig) + 1)
        elig["decile"] = pd.qcut(elig["alpha_score"].rank(method="first"), 10, labels=False, duplicates="drop")

        g["alpha_score"] = np.nan
        g["rank"] = np.nan
        g["decile"] = np.nan
        g.loc[elig.index, ["alpha_score", "rank", "decile"]] = elig[["alpha_score", "rank", "decile"]]
        return g

    scored = feat.groupby("date", group_keys=False).apply(per_day).reset_index(drop=True)
    return scored


def build_confidence_table(train: pd.DataFrame) -> pd.DataFrame:
    d = train.dropna(subset=["decile", "fwd_ret_20d"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["decile", "win_rate", "mean_ret_20d", "confidence"])

    stats = d.groupby("decile").agg(
        win_rate=("fwd_ret_20d", lambda x: (x > 0).mean()),
        mean_ret_20d=("fwd_ret_20d", "mean"),
        mean_ret_5d=("fwd_ret_5d", "mean"),
        mean_ret_10d=("fwd_ret_10d", "mean"),
    ).reset_index()
    wr_scaled = (stats["win_rate"] - stats["win_rate"].min()) / (stats["win_rate"].max() - stats["win_rate"].min() + 1e-9)
    ret_scaled = (stats["mean_ret_20d"] - stats["mean_ret_20d"].min()) / (
        stats["mean_ret_20d"].max() - stats["mean_ret_20d"].min() + 1e-9
    )
    stats["confidence"] = ((0.6 * wr_scaled + 0.4 * ret_scaled) * 100).round(1)
    return stats


def _calc_action(rank: float, held: bool, n: int, m: int) -> str:
    if pd.isna(rank):
        return "SELL" if held else "HOLD"
    if held and rank <= n + m:
        return "HOLD"
    if held and rank > n + m:
        return "SELL"
    if (not held) and rank <= n:
        return "BUY"
    return "HOLD"


def backtest(scored: pd.DataFrame, bench: pd.DataFrame, confidence_table: pd.DataFrame, config: Dict):
    cfg = config["backtest"]
    n_target = cfg["n_positions"]
    m_buffer = cfg["buffer"]
    cost_rate = cfg["one_way_cost_bps"] / 10000.0

    test_start = pd.to_datetime(cfg["test_start"])
    test_end = pd.to_datetime(cfg["test_end"])

    data = scored[(scored["date"] >= test_start) & (scored["date"] <= test_end)].copy()
    dates = sorted(data["date"].unique())

    conf_map = dict(zip(confidence_table["decile"], confidence_table["confidence"]))

    positions: Dict[str, Position] = {}
    pending_edge_exits: set[str] = set()
    prev_day_signal = None
    cash = 1.0
    portfolio_rows = []
    holdings_rows = []
    trades_rows = []
    signals_rows = []

    for i, d in enumerate(dates):
        day = data[data["date"] == d].copy()
        day = day.sort_values("ticker")
        open_map = dict(zip(day["ticker"], day["PX_OPEN"]))
        high_map = dict(zip(day["ticker"], day["PX_HIGH"]))
        low_map = dict(zip(day["ticker"], day["PX_LOW"]))
        close_map = dict(zip(day["ticker"], day["PX_LAST"]))
        atr_map = dict(zip(day["ticker"], day["atr_14"]))

        cost_paid = 0.0
        turnover = 0.0

        # 1) scheduled edge exits at open
        for t in list(pending_edge_exits):
            if t in positions and t in open_map:
                pos = positions.pop(t)
                px = open_map[t]
                gross = pos.shares * px
                cost = gross * cost_rate
                cash += gross - cost
                cost_paid += cost
                trades_rows.append({
                    "ticker": t,
                    "entry_date": pos.entry_date,
                    "entry_px": pos.entry_px,
                    "exit_date": d,
                    "exit_px": px,
                    "shares_or_weight": pos.shares,
                    "pnl_gross": (px - pos.entry_px) * pos.shares,
                    "pnl_net": (px - pos.entry_px) * pos.shares - cost - (pos.entry_px * pos.shares * cost_rate),
                    "reason": "edge",
                    "mae": (pos.min_low - pos.entry_px) / pos.entry_px,
                    "mfe": (pos.max_high - pos.entry_px) / pos.entry_px,
                    "holding_days": (d - pos.entry_date).days,
                    "cost": cost + (pos.entry_px * pos.shares * cost_rate),
                })
        pending_edge_exits = set()

        # 2) intraday stop/target/time exits
        for t in list(positions.keys()):
            if t not in day["ticker"].values:
                continue
            pos = positions[t]
            pos.max_high = max(pos.max_high, high_map[t])
            pos.min_low = min(pos.min_low, low_map[t])
            reason = None
            exit_px = None
            if low_map[t] <= pos.stop:
                reason = "stop"
                exit_px = pos.stop
            elif high_map[t] >= pos.target:
                reason = "target"
                exit_px = pos.target
            elif d >= pos.time_stop_date:
                reason = "time"
                exit_px = close_map[t]

            if reason:
                gross = pos.shares * exit_px
                cost = gross * cost_rate
                cash += gross - cost
                cost_paid += cost
                positions.pop(t)
                trades_rows.append({
                    "ticker": t,
                    "entry_date": pos.entry_date,
                    "entry_px": pos.entry_px,
                    "exit_date": d,
                    "exit_px": exit_px,
                    "shares_or_weight": pos.shares,
                    "pnl_gross": (exit_px - pos.entry_px) * pos.shares,
                    "pnl_net": (exit_px - pos.entry_px) * pos.shares - cost - (pos.entry_px * pos.shares * cost_rate),
                    "reason": reason,
                    "mae": (pos.min_low - pos.entry_px) / pos.entry_px,
                    "mfe": (pos.max_high - pos.entry_px) / pos.entry_px,
                    "holding_days": (d - pos.entry_date).days,
                    "cost": cost + (pos.entry_px * pos.shares * cost_rate),
                })

        # 3) execute buys and rebalance from previous day's close signals
        nav_open = cash + sum(pos.shares * open_map.get(t, 0.0) for t, pos in positions.items())
        if prev_day_signal is not None and nav_open > 0:
            candidates = prev_day_signal.sort_values("rank")
            top_n = set(candidates[candidates["rank"] <= n_target]["ticker"].tolist())

            for t in top_n:
                if t not in positions and t in open_map and np.isfinite(atr_map.get(t, np.nan)):
                    px = open_map[t]
                    target_dollar = nav_open / n_target
                    shares = target_dollar / px if px > 0 else 0.0
                    notional = shares * px
                    cost = notional * cost_rate
                    if shares > 0:
                        cash -= notional + cost
                        cost_paid += cost
                        positions[t] = Position(
                            ticker=t,
                            entry_date=d,
                            entry_px=px,
                            shares=shares,
                            stop=px - cfg["stop_atr_mult"] * atr_map[t],
                            target=px + cfg["target_atr_mult"] * atr_map[t],
                            time_stop_date=d + pd.tseries.offsets.BDay(cfg["time_stop_days"]),
                            atr_entry=atr_map[t],
                            max_high=high_map.get(t, px),
                            min_low=low_map.get(t, px),
                        )

            # rebalance to equal weights among active holdings (cap n_target)
            hold_list = list(positions.keys())[:n_target]
            if hold_list:
                nav_open = cash + sum(positions[t].shares * open_map.get(t, 0.0) for t in hold_list)
                tw = 1 / len(hold_list)
                for t in hold_list:
                    px = open_map.get(t)
                    if not px or px <= 0:
                        continue
                    pos = positions[t]
                    current_val = pos.shares * px
                    target_val = nav_open * tw
                    delta_val = target_val - current_val
                    if abs(delta_val) < 1e-10:
                        continue
                    delta_shares = delta_val / px
                    trade_notional = abs(delta_val)
                    cost = trade_notional * cost_rate
                    pos.shares += delta_shares
                    cash -= delta_val + cost
                    cost_paid += cost
                    turnover += trade_notional

        nav_close_gross = cash + sum(pos.shares * close_map.get(t, 0.0) for t, pos in positions.items()) + cost_paid
        nav_close_net = cash + sum(pos.shares * close_map.get(t, 0.0) for t, pos in positions.items())
        prev_nav = portfolio_rows[-1]["nav_net"] if portfolio_rows else 1.0
        ret_net = nav_close_net / prev_nav - 1
        ret_gross = nav_close_gross / (portfolio_rows[-1]["nav_gross"] if portfolio_rows else 1.0) - 1

        portfolio_rows.append(
            {
                "date": d,
                "nav_gross": nav_close_gross,
                "nav_net": nav_close_net,
                "ret_gross": ret_gross,
                "ret_net": ret_net,
                "turnover": turnover / max(prev_nav, 1e-12),
                "n_hold": len(positions),
                "cost_paid": cost_paid,
            }
        )

        held_set = set(positions.keys())
        rank_map = dict(zip(day["ticker"], day["rank"]))
        for t in held_set:
            holdings_rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "weight": (positions[t].shares * close_map.get(t, 0.0)) / max(nav_close_net, 1e-12),
                    "alpha_score": day.loc[day["ticker"] == t, "alpha_score"].iloc[0] if t in rank_map else np.nan,
                    "action": _calc_action(rank_map.get(t, np.nan), True, n_target, m_buffer),
                    "confidence": conf_map.get(day.loc[day["ticker"] == t, "decile"].iloc[0], 50) if t in rank_map else 50,
                }
            )

        # produce daily signal sheet
        for _, r in day.iterrows():
            held = r["ticker"] in held_set
            action = _calc_action(r["rank"], held, n_target, m_buffer)
            if held and action == "SELL":
                pending_edge_exits.add(r["ticker"])
            signals_rows.append(
                {
                    "date": d,
                    "ticker": r["ticker"],
                    "alpha_score": r.get("alpha_score", np.nan),
                    "rank": r.get("rank", np.nan),
                    "decile": r.get("decile", np.nan),
                    "action_reco": action,
                    "confidence": conf_map.get(r.get("decile", np.nan), 50),
                    "suggested_entry": r.get("PX_LAST", np.nan),
                    "suggested_stop": r.get("PX_LAST", np.nan) - cfg["stop_atr_mult"] * r.get("atr_14", np.nan),
                    "suggested_target": r.get("PX_LAST", np.nan) + cfg["target_atr_mult"] * r.get("atr_14", np.nan),
                    "suggested_time_stop": (d + pd.tseries.offsets.BDay(cfg["time_stop_days"])) if pd.notna(r.get("PX_LAST", np.nan)) else pd.NaT,
                    "key_features": json.dumps(
                        {
                            "mom_12_1": r.get("mom_12_1", np.nan),
                            "mom_6_1": r.get("mom_6_1", np.nan),
                            "mom_3m": r.get("mom_3m", np.nan),
                            "vol_20": r.get("vol_20", np.nan),
                            "dv_20": r.get("dv_20", np.nan),
                            "breakout": r.get("breakout", np.nan),
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        prev_day_signal = day[day["eligible"]].copy()

    portfolio = pd.DataFrame(portfolio_rows)
    holdings = pd.DataFrame(holdings_rows)
    trades = pd.DataFrame(trades_rows)
    signals = pd.DataFrame(signals_rows)

    portfolio = portfolio.merge(bench, on="date", how="left")
    return portfolio, holdings, trades, signals


def _calc_drawdown(nav: pd.Series) -> float:
    roll_max = nav.cummax()
    dd = nav / roll_max - 1
    return dd.min()


def build_diagnostics(portfolio: pd.DataFrame, scored_test: pd.DataFrame, confidence_table: pd.DataFrame) -> pd.DataFrame:
    ret = portfolio["ret_net"].fillna(0)
    ann = 252
    cagr = (portfolio["nav_net"].iloc[-1] / portfolio["nav_net"].iloc[0]) ** (ann / max(len(portfolio), 1)) - 1 if len(portfolio) > 1 else 0
    vol = ret.std() * np.sqrt(ann)
    sharpe = (ret.mean() * ann) / vol if vol > 0 else 0
    maxdd = _calc_drawdown(portfolio["nav_net"]) if not portfolio.empty else 0

    rows = [
        {"metric": "CAGR_net", "value": cagr},
        {"metric": "Vol_net", "value": vol},
        {"metric": "Sharpe_net", "value": sharpe},
        {"metric": "MaxDD_net", "value": maxdd},
    ]

    dec = confidence_table.copy()
    dec["metric"] = dec["decile"].apply(lambda x: f"decile_{x}")
    for _, r in dec.iterrows():
        rows.append({"metric": f"{r['metric']}_win_rate", "value": r["win_rate"]})
        rows.append({"metric": f"{r['metric']}_mean_ret_5d", "value": r["mean_ret_5d"]})
        rows.append({"metric": f"{r['metric']}_mean_ret_10d", "value": r["mean_ret_10d"]})
        rows.append({"metric": f"{r['metric']}_mean_ret_20d", "value": r["mean_ret_20d"]})

    return pd.DataFrame(rows)


def export_outputs(out_dir: Path, portfolio: pd.DataFrame, holdings: pd.DataFrame, trades: pd.DataFrame, signals: pd.DataFrame, diagnostics: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    portfolio.to_csv(out_dir / "portfolio_daily.csv", index=False)
    holdings.to_csv(out_dir / "holdings_daily.csv", index=False)
    trades.to_csv(out_dir / "trades.csv", index=False)
    signals.to_csv(out_dir / "signals_daily.csv", index=False)
    diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)


def run_pipeline(config: Dict):
    prices, bench, _ = load_data(config)
    feat = build_features(prices)
    scored = build_cross_sectional_scores(feat, config)

    train_start = pd.to_datetime(config["backtest"]["train_start"])
    train_end = pd.to_datetime(config["backtest"]["train_end"])
    train = scored[(scored["date"] >= train_start) & (scored["date"] <= train_end)]

    confidence_table = build_confidence_table(train)
    portfolio, holdings, trades, signals = backtest(scored, bench, confidence_table, config)
    diagnostics = build_diagnostics(portfolio, scored, confidence_table)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config["output_root"]) / f"run_{ts}"
    export_outputs(out_dir, portfolio, holdings, trades, signals, diagnostics)
    logging.info("Outputs written to %s", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
