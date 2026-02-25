# TOPIX500 Long-Only Daily Rebalance Backtester

Bloomberg API等で取得済みのローカル日次OHLCV（CSV/Parquet）を入力に、TOPIX500固定ユニバース向けのロングオンリー戦略をバックテストし、日次の `BUY/HOLD/SELL` 目線を出力するツールです。

## 実行

```bash
python run_backtest.py --config config/config.sample.yml
```

## 入力

- ユニバース: `config/universe_topix500.csv`（列: `ticker`）
- 価格データ（どちらか）
  - long形式: `date,ticker,field,value`（field: PX_OPEN/PX_HIGH/PX_LOW/PX_LAST/PX_VOLUME）
  - wide形式: `date,ticker,PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME`
- ベンチマーク: `data/benchmark_daily.(csv|parquet)`（`date,PX_LAST`）

## 実装要点

- シグナル確定: 当日引け
- 約定: 翌日始値（look-ahead回避）
- Long-only / 日次リバランス
- 保有数: 50銘柄、バッファ10
- コスト: 片道20bps（設定変更可）
- Exit優先順位: `stop/target/time` > `edge(順位劣化)`

## 主要関数

- `load_data()`
- `build_features()`
- `build_cross_sectional_scores()`
- `build_confidence_table(train)`
- `backtest(test)`
- `export_outputs()`

## 出力

`outputs/run_YYYYmmdd_HHMMSS/` に以下を保存:

1. `portfolio_daily.csv`
2. `holdings_daily.csv`
3. `trades.csv`
4. `signals_daily.csv`
5. `diagnostics.csv`

## Sanity check

- 必須列チェック（価格データ long/wide）
- 欠損で計算不可の銘柄は日次eligibilityから除外
- 処理継続（ログで件数表示）
