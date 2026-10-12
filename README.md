# AI-Powered Stock & Crypto Market Intelligence Analyzer

Streamlit dashboard for research on stocks (Alpha Vantage) and crypto (CoinGecko): candlesticks, technical
analysis, leak-free ML forecasts with uncertainty, BUY/SELL/HOLD research signals, news sentiment, anomaly
detection, cost-aware backtests, and downloadable reports. **Paper trading only. Not financial advice.**

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then edit .env with YOUR keys
streamlit run app.py
python -m pytest                                         # run tests
```
Keys: Alpha Vantage (https://www.alphavantage.co/support/#api-key, free) and CoinGecko demo key
(https://www.coingecko.com/en/api). Keys live only in `.env` (git-ignored). Do not share screenshots showing
them; if a key leaks, regenerate it in the provider dashboard.

### Keys not detected?
Open the sidebar **🔑 API keys** panel and paste them there (optionally tick "save to .env"). Common `.env` problems:
the file is named `.env.txt` (Windows hides extensions), it is in the wrong folder (it must sit next to `app.py`),
or Streamlit was not restarted after editing it. The panel shows exactly where the app looks for `.env`.

## Honesty rules built in
- **No fabricated or substituted data.** If a request fails you see an error, never demo data.
- Stocks = Alpha Vantage **end-of-day (delayed)**, labelled. Crypto = CoinGecko daily closes.
  CoinGecko's free history has no true OHLC, so candles are *derived* and flagged on-chart.
- Freshness banner on every dashboard: source, fetch time, latest bar, age.
- Free-tier friendly: results cached (`CACHE_TTL_SECONDS`), news/fundamentals load only on click,
  retry/backoff on 429/5xx, rate limiter, keys scrubbed from error messages.
- Alpha Vantage free keys may refuse `outputsize=full`; the app falls back to ~100 days and says so
  (ML/backtest then need more history and will say so).

## Methodology
- **Indicators:** RSI(Wilder), MACD, EMA/SMA, Bollinger, ATR, annualised volatility, swing support/resistance
  (pivots confirmed only after they form). All causal.
- **Forecast:** Random Forest + Gradient Boosting on past-only features predicting forward N-bar return.
  Chronological train / calibration / test (60/20/20) with an N-bar purge between segments. Test data is touched
  only for reporting (MAE, RMSE, directional accuracy, skill vs zero-return baseline, interval coverage).
  Intervals come from calibration residuals; direction confidence is the calibration hit rate shrunk toward 50%.
  If the ensemble does not beat the naive baseline, the signal ignores ML and says so.
- **Signal:** weighted technical (60%), ML (25%, only if skilful) and sentiment (15%, if loaded);
  confidence capped at 80%; risk from ATR%; stop/target = 2x/3x ATR scenarios.
- **Backtest:** long-only technical rule, signal at close t traded at t+1, fees + slippage, drawdown, Sharpe,
  win rate, vs buy & hold. Results are illustrative (in-sample idea, fixed rules).
- **Anomalies:** IsolationForest + robust z-score; descriptive, not predictive.

## Layout
```
app.py            Streamlit UI          src/data_alpha.py, data_coingecko.py   API clients
src/indicators.py src/features.py       src/models.py        forecasting
src/signals.py    src/backtest.py       src/sentiment.py, anomaly.py, trend.py
src/charts.py     src/report.py         src/paper.py         virtual portfolio
tests/            unit tests (synthetic data used ONLY inside tests)
```
Indian stocks: use Alpha Vantage BSE tickers (e.g. `RELIANCE.BSE`); coverage varies.
