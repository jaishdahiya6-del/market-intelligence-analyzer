"""Descriptive anomaly detection (IsolationForest + robust z-score). Fitted on the whole window, so it
labels unusual days in hindsight; it is NOT a predictive signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02, seed: int = 42) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["ret"] = df["close"].pct_change()
    d["range_pct"] = (df["high"] - df["low"]) / df["close"]
    lv = np.log1p(df["volume"].astype(float))
    d["vol_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std().replace(0, np.nan)
    d = d.replace([np.inf, -np.inf], np.nan)
    med = d["ret"].median()
    mad = (d["ret"] - med).abs().median()
    d["robust_z"] = 0.6745 * (d["ret"] - med) / mad if mad else 0.0
    ok = d[["ret", "range_pct", "vol_z"]].dropna()
    d["anomaly"] = False
    if len(ok) >= 50:
        iso = IsolationForest(contamination=contamination, random_state=seed).fit(ok)
        flag = pd.Series(iso.predict(ok) == -1, index=ok.index)
        strong = (d.loc[ok.index, "robust_z"].abs() > 2.5) | (d.loc[ok.index, "vol_z"].abs() > 2.5)
        d.loc[ok.index, "anomaly"] = flag & strong
    return d
