"""Feature engineering for forecasting. Features at row t use data <= t; the target is the
forward return over `horizon` bars (uses t+1..t+h, so it is NEVER a feature)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_all

FEATURES = ["ret_1", "ret_5", "ret_10", "ret_20", "rsi", "macd_hist_n", "pctb", "bb_bw", "ema_ratio",
            "sma50_ratio", "vol_20d", "atr_pct", "vol_z", "range_pct"]


def build_features(df: pd.DataFrame, horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    d = add_all(df)
    c = d["close"]
    X = pd.DataFrame(index=d.index)
    for n in (1, 5, 10, 20):
        X[f"ret_{n}"] = c.pct_change(n)
    X["rsi"] = d["rsi_14"]
    X["macd_hist_n"] = d["macd_hist"] / c
    X["pctb"] = d["bb_pctb"]
    X["bb_bw"] = d["bb_bw"]
    X["ema_ratio"] = d["ema_12"] / d["ema_26"] - 1
    X["sma50_ratio"] = c / d["sma_50"] - 1
    X["vol_20d"] = c.pct_change().rolling(20).std()
    X["atr_pct"] = d["atr_pct"]
    lv = np.log1p(d["volume"].astype(float))
    X["vol_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std().replace(0, np.nan)
    X["range_pct"] = (d["high"] - d["low"]) / c
    X = X.replace([np.inf, -np.inf], np.nan)
    y = c.shift(-horizon) / c - 1
    return X[FEATURES], y.rename("target")
