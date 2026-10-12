"""Technical indicators. Every value at row t uses data up to and including t only (causal)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
    return (100 - 100 / (1 + rs)).where(~((loss == 0) & (gain == 0)), 50.0)


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n=20, k=2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({"bb_mid": mid, "bb_upper": up, "bb_lower": lo,
                         "bb_pctb": (close - lo) / (up - lo).replace(0, np.nan),
                         "bb_bw": (up - lo) / mid})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def volatility(close: pd.Series, n: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualised rolling std of log returns."""
    return np.log(close / close.shift(1)).rolling(n, min_periods=n).std() * np.sqrt(periods_per_year)


def add_all(df: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    for n in (20, 50, 200):
        out[f"sma_{n}"] = sma(c, n)
    for n in (12, 26, 50):
        out[f"ema_{n}"] = ema(c, n)
    out["rsi_14"] = rsi(c)
    out = out.join(macd(c)).join(bollinger(c))
    out["atr_14"] = atr(out)
    out["atr_pct"] = out["atr_14"] / c
    out["ret_1"] = c.pct_change()
    out["vol_20"] = volatility(c, 20, periods_per_year)
    return out


def support_resistance(df: pd.DataFrame, window: int = 5, lookback: int = 150, n_levels: int = 3,
                       merge_pct: float = 0.015) -> dict:
    """Swing-pivot levels. A pivot is only confirmed `window` bars after it forms (no look-ahead)."""
    d = df.iloc[-lookback:]
    hi, lo = d["high"].to_numpy(), d["low"].to_numpy()
    highs, lows = [], []
    for i in range(window, len(d) - window):
        if hi[i] == hi[i - window:i + window + 1].max():
            highs.append(hi[i])
        if lo[i] == lo[i - window:i + window + 1].min():
            lows.append(lo[i])
    price = float(df["close"].iloc[-1])

    def cluster(levels):
        merged = []
        for lv in sorted(levels):
            if merged and abs(lv - merged[-1]) / merged[-1] < merge_pct:
                merged[-1] = (merged[-1] + lv) / 2
            else:
                merged.append(lv)
        return merged

    levels = cluster(highs + lows)
    sup = sorted([l for l in levels if l < price], reverse=True)[:n_levels]
    res = sorted([l for l in levels if l > price])[:n_levels]
    return {"support": sup, "resistance": res}
