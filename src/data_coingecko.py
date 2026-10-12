"""CoinGecko client: current price, daily history (price/volume/market cap).
The free history endpoint returns daily CLOSE-style points, not true OHLC. Candles are therefore
derived (open = previous close) and flagged `ohlc_approximated=True`; the UI says so."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config
from .common import DataError, DataResult, RateLimiter, clean_ohlcv, get_json

_limiter = RateLimiter(2.2)  # demo plan ~30 calls/min
POPULAR = {"Bitcoin": "bitcoin", "Ethereum": "ethereum", "Solana": "solana", "XRP": "ripple",
           "Cardano": "cardano", "Dogecoin": "dogecoin", "BNB": "binancecoin", "Polkadot": "polkadot"}
SYMBOLS = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP",
           "cardano": "ADA", "dogecoin": "DOGE", "binancecoin": "BNB", "polkadot": "DOT"}


def _get(path: str, params: dict):
    if not config.coingecko_key():
        raise DataError("COINGECKO_API_KEY is not set. Add it to your .env file (see README).")
    return get_json(config.coingecko_base() + path, params=params,
                    headers=config.coingecko_header(), limiter=_limiter)


def parse_market_chart(data: dict) -> pd.DataFrame:
    prices = data.get("prices")
    if not prices:
        raise DataError("CoinGecko returned no price history for this coin id.")
    close = pd.Series({pd.Timestamp(t, unit="ms"): p for t, p in prices}, dtype=float)
    vol = pd.Series({pd.Timestamp(t, unit="ms"): v for t, v in data.get("total_volumes", [])}, dtype=float)
    mcap = pd.Series({pd.Timestamp(t, unit="ms"): v for t, v in data.get("market_caps", [])}, dtype=float)
    df = pd.DataFrame({"close": close, "volume": vol, "market_cap": mcap})
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    return df[["open", "high", "low", "close", "volume", "market_cap"]]


def fetch_history(coin_id: str, vs: str = "usd", days: int = 365) -> DataResult:
    days = min(days, 365) if config.coingecko_key_type() == "demo" else days
    data = _get(f"/coins/{coin_id.strip().lower()}/market_chart",
                {"vs_currency": vs, "days": days, "interval": "daily"})
    df, notes = clean_ohlcv(parse_market_chart(data))
    if df.empty:
        raise DataError("No usable rows after cleaning.")
    notes.append("Free CoinGecko history has no true OHLC: candle open = previous close, wicks are not "
                 "available. Volume/market cap are real. The last bar is the current (partial) UTC day.")
    return DataResult(df=df, source="CoinGecko", symbol=coin_id.lower(), kind="Daily close (near real-time)",
                      fetched_at=datetime.now(timezone.utc), notes=notes, ohlc_approximated=True)


def fetch_live_price(coin_ids: list, vs: str = "usd") -> dict:
    data = _get("/simple/price", {"ids": ",".join(coin_ids), "vs_currencies": vs, "include_24hr_change": "true",
                                  "include_last_updated_at": "true"})
    out = {}
    for cid in coin_ids:
        d = data.get(cid)
        if d and vs in d:
            out[cid] = {"price": d[vs], "change_24h_pct": d.get(f"{vs}_24h_change"),
                        "updated": datetime.fromtimestamp(d["last_updated_at"], tz=timezone.utc)
                        if d.get("last_updated_at") else None}
    return out


def ping() -> str:
    """Cheap connectivity + key check."""
    data = _get("/ping", {})
    return str(data.get("gecko_says", "OK"))
