"""Alpha Vantage client: daily OHLCV, fundamentals, news sentiment.
Free tier is ~25 requests/day, so the app only calls it on demand and caches results.
Daily data is END-OF-DAY (delayed) and is labelled that way."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config
from .common import (DataError, DataResult, PremiumEndpointError, RateLimiter,
                     RateLimitError, clean_ohlcv, get_json)

BASE_URL = "https://www.alphavantage.co/query"
_limiter = RateLimiter(1.2)
_RATE_WORDS = ("rate limit", "requests per day", "requests per minute", "call frequency", "api call volume")


def _call(params: dict) -> dict:
    key = config.alpha_key()
    if not key:
        raise DataError("ALPHA_VANTAGE_API_KEY is not set. Add it to your .env file (see README).")
    data = get_json(BASE_URL, params={**params, "apikey": key}, limiter=_limiter)
    if not isinstance(data, dict):
        raise DataError("Unexpected Alpha Vantage response.")
    if "Error Message" in data:
        raise DataError("Alpha Vantage rejected the request: unknown/unsupported symbol or bad parameters.")
    msg = data.get("Information") or data.get("Note")
    if msg:
        msg = str(msg).replace(key, "***")
        low = msg.lower()
        if "outputsize" in low:
            raise PremiumEndpointError("outputsize=full requires a premium Alpha Vantage plan.")
        if any(w in low for w in _RATE_WORDS):
            raise RateLimitError("Alpha Vantage rate/daily limit reached. " + msg[:200])
        raise DataError("Alpha Vantage: " + msg[:250])
    return data


def parse_daily(data: dict) -> pd.DataFrame:
    ts = data.get("Time Series (Daily)")
    if not ts:
        raise DataError("Alpha Vantage returned no daily time series for this symbol.")
    df = pd.DataFrame.from_dict(ts, orient="index")
    df = df.rename(columns={"1. open": "open", "2. high": "high", "3. low": "low",
                            "4. close": "close", "5. volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def fetch_daily(symbol: str, full: bool = True) -> DataResult:
    symbol = symbol.strip().upper()
    notes: list = []
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full" if full else "compact"}
    try:
        data = _call(params)
    except PremiumEndpointError:
        notes.append("Full history is premium-only; using the latest ~100 trading days. "
                     "ML forecasting/backtests may be unavailable with so little history.")
        data = _call({**params, "outputsize": "compact"})
    df, clean_notes = clean_ohlcv(parse_daily(data))
    if df.empty:
        raise DataError("No usable rows after cleaning.")
    return DataResult(df=df, source="Alpha Vantage", symbol=symbol, kind="End-of-day (delayed)",
                      fetched_at=datetime.now(timezone.utc), notes=notes + clean_notes)


_OVERVIEW_FIELDS = ["Name", "Sector", "Industry", "Currency", "MarketCapitalization", "PERatio", "PEGRatio",
                    "EPS", "DividendYield", "ProfitMargin", "Beta", "52WeekHigh", "52WeekLow"]


def fetch_overview(symbol: str) -> dict:
    data = _call({"function": "OVERVIEW", "symbol": symbol.strip().upper()})
    return {k: data[k] for k in _OVERVIEW_FIELDS if k in data and data[k] not in ("None", "-", "")}


def fetch_news(ticker: str, limit: int = 50) -> list:
    """Raw feed items (list of dicts). `ticker` e.g. 'AAPL' or 'CRYPTO:BTC'."""
    data = _call({"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": limit, "sort": "LATEST"})
    return data.get("feed", []) or []
