"""Market trend detection via log-price regression over a recent window."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_trend(df: pd.DataFrame, window: int = 60, periods_per_year: int = 252) -> dict:
    c = np.log(df["close"].iloc[-window:].to_numpy())
    if len(c) < 20:
        return {"label": "Unknown", "slope_ann": np.nan, "r2": np.nan, "window": len(c)}
    x = np.arange(len(c))
    slope, intercept = np.polyfit(x, c, 1)
    fit = slope * x + intercept
    ss_res, ss_tot = ((c - fit) ** 2).sum(), ((c - c.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    total = slope * len(c)
    if r2 < 0.3 or abs(total) < 0.03:
        label = "Sideways / no clear trend"
    else:
        label = "Uptrend" if slope > 0 else "Downtrend"
    sma200 = df["close"].rolling(200).mean().iloc[-1]
    regime = None if np.isnan(sma200) else ("Above 200-SMA (long-term bullish)" if df["close"].iloc[-1] > sma200
                                            else "Below 200-SMA (long-term bearish)")
    return {"label": label, "slope_ann": float(slope * periods_per_year), "r2": r2,
            "window": len(c), "regime": regime}
