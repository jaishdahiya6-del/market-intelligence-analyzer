"""Shared HTTP client (retry, backoff, rate limiting), error types, data cleaning."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests


class DataError(Exception):
    """Any data-retrieval problem. Messages never contain API keys."""


class RateLimitError(DataError):
    pass


class PremiumEndpointError(DataError):
    pass


class InsufficientDataError(DataError):
    pass


class RateLimiter:
    """Enforces a minimum interval between calls (thread-safe)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


def get_json(url, params=None, headers=None, limiter: RateLimiter | None = None,
             retries: int = 3, timeout: int = 20, backoff: float = 2.0, sleep=time.sleep):
    last: Exception | None = None
    for attempt in range(retries):
        if limiter:
            limiter.wait()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:  # do not echo exc: URL may contain a key
            last = DataError(f"Network error ({type(exc).__name__}).")
            sleep(backoff ** attempt)
            continue
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After", backoff ** (attempt + 1)))
            except ValueError:
                wait = backoff ** (attempt + 1)
            last = RateLimitError("Rate limit reached (HTTP 429). Wait a minute and retry.")
            if attempt < retries - 1:
                sleep(min(wait, 30))
                continue
            raise last
        if r.status_code in (401, 403):
            raise DataError(f"Authentication failed (HTTP {r.status_code}). Check the key in your .env file.")
        if r.status_code >= 500:
            last = DataError(f"Provider server error (HTTP {r.status_code}).")
            sleep(backoff ** attempt)
            continue
        if r.status_code >= 400:
            raise DataError(f"Request rejected (HTTP {r.status_code}). Check the symbol/coin id.")
        try:
            return r.json()
        except ValueError:
            raise DataError("Provider returned invalid JSON.")
    raise last or DataError("Request failed.")


@dataclass
class DataResult:
    df: pd.DataFrame
    source: str
    symbol: str
    kind: str                       # e.g. "End-of-day (delayed)"
    fetched_at: datetime
    notes: list = field(default_factory=list)
    ohlc_approximated: bool = False
    simulated: bool = False         # the app never sets this True: no demo data is ever substituted

    @property
    def last_bar(self) -> pd.Timestamp:
        return self.df.index[-1]

    def age_days(self) -> int:
        now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).normalize()
        return int((now - self.last_bar.normalize()).days)


def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Validate and repair an OHLCV frame. Returns (clean_df, notes). Nothing is invented:
    only short gaps in prices are forward-filled, everything else unusable is dropped."""
    notes: list = []
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()
    if df.index.duplicated().any():
        notes.append(f"Removed {int(df.index.duplicated().sum())} duplicate timestamps.")
        df = df[~df.index.duplicated(keep="last")]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    price_cols = ["open", "high", "low", "close"]
    bad = (df[price_cols] <= 0).any(axis=1)
    if bad.any():
        notes.append(f"Dropped {int(bad.sum())} rows with non-positive prices.")
        df = df[~bad]
    n_nan = int(df[price_cols].isna().any(axis=1).sum())
    if n_nan:
        df[["open", "high", "low"]] = df[["open", "high", "low"]].ffill(limit=2)
        before = len(df)
        df = df.dropna(subset=price_cols)
        notes.append(f"{n_nan} rows had missing prices; short gaps forward-filled, {before - len(df)} dropped.")
    if "volume" in df and df["volume"].isna().any():
        notes.append(f"{int(df['volume'].isna().sum())} missing volume values set to 0.")
        df["volume"] = df["volume"].fillna(0)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df, notes
