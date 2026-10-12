"""Leak-free ML forecasting of the forward `horizon`-bar return.

Chronological 60/20/20 split into train / calibration / test, each separated by a `horizon`-bar
purge so overlapping targets never straddle a boundary:
  * train        -> fit models
  * calibration  -> prediction-interval residuals + direction-confidence calibration
  * test         -> untouched, used only for reporting honest out-of-sample metrics
The final live forecast refits on all labelled rows and predicts the latest (unlabelled) row."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from .common import InsufficientDataError
from .features import build_features

MIN_LABELED = 200


def _make_models(seed: int):
    return {
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                               max_features=0.5, n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                       subsample=0.7, min_samples_leaf=10, random_state=seed),
    }


def _metrics(y: np.ndarray, p: np.ndarray, baseline: bool = False) -> dict:
    err = y - p
    da = float(np.mean(y > 0)) if baseline else float(np.mean(np.sign(y) == np.sign(p)))
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "Directional accuracy": da, "MSE": float(np.mean(err ** 2))}


@dataclass
class ForecastResult:
    horizon: int
    last_close: float
    pred_return: float
    pred_price: float
    interval_return: tuple
    interval_price: tuple
    interval_level: float
    direction_confidence: float
    confidence_n: int
    metrics: pd.DataFrame
    skill_vs_naive: float
    interval_coverage_test: float
    feature_importance: pd.Series
    sizes: dict
    test_frame: pd.DataFrame
    warnings: list = field(default_factory=list)

    @property
    def has_skill(self) -> bool:
        return self.skill_vs_naive > 0


def direction_confidence(pred: float, calib_pred: np.ndarray, calib_y: np.ndarray, prior: int = 10):
    """P(direction correct) from calibration-set hit rate of same-sign predictions, shrunk to 0.5
    with a Beta(prior, prior) prior so small samples can't produce overconfident numbers."""
    mask = np.sign(calib_pred) == np.sign(pred)
    n = int(mask.sum())
    hits = int((np.sign(calib_y[mask]) == np.sign(pred)).sum()) if n else 0
    return (hits + prior) / (n + 2 * prior), n


def run_forecast(df: pd.DataFrame, horizon: int = 5, seed: int = 42, level: float = 0.90) -> ForecastResult:
    X, y = build_features(df, horizon)
    valid = y.notna() & X.notna().all(axis=1)
    Xl, yl = X[valid], y[valid]
    n = len(Xl)
    if n < MIN_LABELED:
        raise InsufficientDataError(
            f"Only {n} usable rows; need >= {MIN_LABELED} for a leak-free train/calibration/test split.")
    i1, i2 = int(n * 0.6), int(n * 0.8)
    tr = slice(0, i1 - horizon)
    ca = slice(i1, i2 - horizon)
    te = slice(i2, n)

    models = _make_models(seed)
    preds_cal, preds_te = {}, {}
    for name, m in models.items():
        m.fit(Xl.iloc[tr], yl.iloc[tr])
        preds_cal[name] = m.predict(Xl.iloc[ca])
        preds_te[name] = m.predict(Xl.iloc[te])
    cal_p = np.mean(list(preds_cal.values()), axis=0)
    te_p = np.mean(list(preds_te.values()), axis=0)
    y_cal, y_te = yl.iloc[ca].to_numpy(), yl.iloc[te].to_numpy()

    rows = {name: _metrics(y_te, p) for name, p in preds_te.items()}
    rows["Ensemble (avg)"] = _metrics(y_te, te_p)
    rows["Naive (zero return)"] = _metrics(y_te, np.zeros_like(y_te), baseline=True)
    metrics = pd.DataFrame(rows).T
    skill = 1 - rows["Ensemble (avg)"]["MSE"] / rows["Naive (zero return)"]["MSE"]

    a = (1 - level) / 2
    q_lo, q_hi = np.quantile(y_cal - cal_p, [a, 1 - a])
    coverage = float(np.mean((y_te >= te_p + q_lo) & (y_te <= te_p + q_hi)))

    # final refit on all labelled data, predict the newest row (no label yet)
    final_preds = []
    for name in models:
        m = _make_models(seed)[name]
        m.fit(Xl, yl)
        final_preds.append(m.predict(X.iloc[[-1]])[0])
        if name == "Random Forest":
            imp = pd.Series(m.feature_importances_, index=Xl.columns).sort_values(ascending=False)
    pred = float(np.mean(final_preds))
    last = float(df["close"].iloc[-1])
    conf, conf_n = direction_confidence(pred, cal_p, y_cal)

    warns = []
    if skill <= 0:
        warns.append("On unseen test data the ensemble did NOT beat the naive zero-return baseline: "
                     "treat this forecast as having no demonstrated edge.")
    if conf_n < 30:
        warns.append(f"Confidence is based on only {conf_n} calibration samples.")
    warns.append(f"Overlapping {horizon}-bar targets reduce the effective test sample size.")
    tf = pd.DataFrame({"actual": y_te, "predicted": te_p}, index=Xl.index[te])
    return ForecastResult(
        horizon=horizon, last_close=last, pred_return=pred, pred_price=last * (1 + pred),
        interval_return=(pred + q_lo, pred + q_hi),
        interval_price=(last * (1 + pred + q_lo), last * (1 + pred + q_hi)), interval_level=level,
        direction_confidence=conf, confidence_n=conf_n, metrics=metrics.drop(columns="MSE"),
        skill_vs_naive=float(skill), interval_coverage_test=coverage, feature_importance=imp,
        sizes={"train": len(Xl.iloc[tr]), "calibration": len(Xl.iloc[ca]), "test": len(Xl.iloc[te])},
        test_frame=tf, warnings=warns)
