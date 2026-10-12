"""Research signals (BUY / SELL / HOLD). These are decision-support scores, NOT financial advice."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DISCLAIMER = ("Research signal for educational purposes only. Not financial advice; no prediction is "
              "guaranteed. Past performance does not predict future results.")
BUY_T, SELL_T = 0.25, -0.25


def technical_score_series(d: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Causal technical score in [-1, 1] for every row, plus its components."""
    trend = 0.5 * np.sign(d["ema_12"] - d["ema_26"]) + 0.5 * np.sign(d["close"] - d["sma_50"])
    mom = np.sign(d["macd_hist"])
    r = d["rsi_14"]
    rsi_s = np.where(r < 30, 1.0, np.where(r > 70, -1.0, 0.0))
    rsi_s = pd.Series(rsi_s, index=d.index).where(r.notna())
    bb = np.where(d["bb_pctb"] < 0.05, 0.5, np.where(d["bb_pctb"] > 0.95, -0.5, 0.0))
    bb = pd.Series(bb, index=d.index).where(d["bb_pctb"].notna())
    comps = pd.DataFrame({"trend": trend, "momentum": mom, "rsi": rsi_s, "bollinger": bb}).fillna(0.0)
    score = 0.4 * comps["trend"] + 0.3 * comps["momentum"] + 0.2 * comps["rsi"] + 0.1 * comps["bollinger"]
    return score.clip(-1, 1), comps


@dataclass
class Signal:
    action: str
    score: float
    confidence: float
    risk_level: str
    reasons: list
    stop_loss: float
    target: float
    scenarios: pd.DataFrame
    components: dict
    disclaimer: str = DISCLAIMER


def risk_level(atr_pct: float, asset_type: str) -> str:
    lo, hi = (0.03, 0.07) if asset_type == "crypto" else (0.015, 0.035)
    return "Low" if atr_pct < lo else "Medium" if atr_pct < hi else "High"


def generate_signal(d: pd.DataFrame, asset_type: str = "stock", forecast=None,
                    sentiment_score: float | None = None, horizon: int = 5) -> Signal:
    tech_s, comps = technical_score_series(d)
    last, c = d.iloc[-1], float(d["close"].iloc[-1])
    parts = {"technical": (float(tech_s.iloc[-1]), 0.60)}
    reasons = []
    reasons.append(f"Trend: EMA12 {'>' if last['ema_12'] > last['ema_26'] else '<'} EMA26; price "
                   f"{'above' if c > last['sma_50'] else 'below'} 50-SMA." if pd.notna(last["sma_50"])
                   else "Not enough history for the 50-SMA trend filter.")
    reasons.append(f"MACD histogram is {'positive' if last['macd_hist'] > 0 else 'negative'} "
                   f"({'bullish' if last['macd_hist'] > 0 else 'bearish'} momentum).")
    rv = last["rsi_14"]
    reasons.append(f"RSI(14) = {rv:.1f} " + ("(oversold)" if rv < 30 else "(overbought)" if rv > 70 else "(neutral zone)") + ".")
    if pd.notna(last["bb_pctb"]):
        reasons.append(f"Price sits at {last['bb_pctb']*100:.0f}% of the Bollinger band range.")

    if forecast is not None:
        if forecast.has_skill:
            vol = max(float(d["close"].pct_change().rolling(20).std().iloc[-1]) * np.sqrt(horizon), 1e-4)
            parts["ml"] = (float(np.clip(forecast.pred_return / (2 * vol), -1, 1)), 0.25)
            lo, hi = forecast.interval_return
            reasons.append(f"ML ensemble expects {forecast.pred_return*100:+.2f}% over {horizon} bars "
                           f"({forecast.interval_level*100:.0f}% interval {lo*100:+.1f}% to {hi*100:+.1f}%)"
                           + (" - the interval spans zero, so the call is uncertain." if lo < 0 < hi else "."))
        else:
            reasons.append("ML forecast excluded: it did not beat the naive baseline on unseen data.")
    if sentiment_score is not None:
        parts["sentiment"] = (float(np.clip(sentiment_score / 0.35, -1, 1)), 0.15)
        reasons.append(f"News sentiment score {sentiment_score:+.2f}.")

    wsum = sum(w for _, w in parts.values())
    total = sum(s * w for s, w in parts.values()) / wsum
    action = "BUY" if total >= BUY_T else "SELL" if total <= SELL_T else "HOLD"
    sgn = np.sign(total)
    agree = np.mean([np.sign(s) == sgn for s, _ in parts.values() if s != 0] or [0.0])
    conf = 0.5 + 0.25 * min(abs(total) / 0.6, 1) * agree
    if action != "HOLD" and forecast is not None and forecast.has_skill and np.sign(forecast.pred_return) == sgn:
        conf = 0.5 * conf + 0.5 * forecast.direction_confidence
    conf = float(min(conf, 0.80)) if action != "HOLD" else 0.5

    atr_v = float(last["atr_14"]) if pd.notna(last["atr_14"]) else c * 0.02
    rl = risk_level(atr_v / c, asset_type)
    short = action == "SELL"
    stop = c + 2 * atr_v if short else c - 2 * atr_v
    target = c - 3 * atr_v if short else c + 3 * atr_v
    if forecast is not None:
        bear, bull = forecast.interval_price
        base = forecast.pred_price
    else:
        move = atr_v * np.sqrt(horizon)
        bear, base, bull = c - move, c, c + move
    scen = pd.DataFrame({"Scenario": ["Bear", "Base", "Bull"], "Price": [bear, base, bull],
                         "Return %": [(x / c - 1) * 100 for x in (bear, base, bull)]})
    reasons.append(f"Risk is {rl} (ATR = {atr_v/c*100:.1f}% of price). Stop/target use 2x / 3x ATR "
                   f"({'short' if short else 'long'} scenario).")
    return Signal(action, float(total), conf, rl, reasons, stop, target, scen,
                  {k: v[0] for k, v in parts.items()})
