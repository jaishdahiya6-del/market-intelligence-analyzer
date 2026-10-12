"""Vectorised long-only backtest of the technical score with costs, vs a buy-and-hold baseline.
Look-ahead protection: the signal computed at the close of bar t is traded on bar t+1 (shift(1))."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import BUY_T, technical_score_series


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    metrics: pd.DataFrame
    trades: pd.DataFrame
    position: pd.Series
    caveats: list


def _stats(eq: pd.Series, rets: pd.Series, ppy: int) -> dict:
    years = max(len(eq) / ppy, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    sd = rets.std()
    return {"Total return %": total * 100,
            "CAGR %": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100 if eq.iloc[-1] > 0 else -100.0,
            "Sharpe (rf=0)": float(rets.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
            "Max drawdown %": float((eq / eq.cummax() - 1).min() * 100)}


def run_backtest(d: pd.DataFrame, entry: float = BUY_T, exit_: float = 0.0, fee_bps: float = 10,
                 slippage_bps: float = 5, periods_per_year: int = 252, initial: float = 10_000) -> BacktestResult:
    score, _ = technical_score_series(d)
    state, pos = 0, np.zeros(len(d))
    for i, s in enumerate(score.to_numpy()):
        if state == 0 and s >= entry:
            state = 1
        elif state == 1 and s <= exit_:
            state = 0
        pos[i] = state
    position = pd.Series(pos, index=d.index).shift(1).fillna(0.0)   # trade next bar
    ret = d["close"].pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (fee_bps + slippage_bps) / 1e4
    strat = position * ret - cost
    bh = ret.copy()
    bh.iloc[0] = -(fee_bps + slippage_bps) / 1e4   # baseline pays the entry cost too
    eq = pd.DataFrame({"Strategy": initial * (1 + strat).cumprod(), "Buy & hold": initial * (1 + bh).cumprod()})

    tid = (position.diff() == 1).cumsum()
    in_trade = (position == 1) | ((position == 0) & (position.shift(1) == 1))
    tr = []
    for k, g in strat[in_trade & (tid > 0)].groupby(tid[in_trade & (tid > 0)]):
        tr.append({"entry": g.index[0], "exit": g.index[-1], "bars": len(g), "return %": ((1 + g).prod() - 1) * 100})
    trades = pd.DataFrame(tr, columns=["entry", "exit", "bars", "return %"])

    m_s, m_b = _stats(eq["Strategy"], strat, periods_per_year), _stats(eq["Buy & hold"], bh, periods_per_year)
    m_s["Trades"], m_b["Trades"] = len(trades), 1
    m_s["Win rate %"] = float((trades["return %"] > 0).mean() * 100) if len(trades) else float("nan")
    m_b["Win rate %"] = float("nan")
    m_s["Time in market %"], m_b["Time in market %"] = float(position.mean() * 100), 100.0
    m_s["Total costs %"] = float(cost.sum() * 100)
    m_b["Total costs %"] = (fee_bps + slippage_bps) / 100
    metrics = pd.DataFrame({"Strategy": m_s, "Buy & hold": m_b})
    caveats = ["Rules are fixed (not optimised) but the whole sample is in-sample for the strategy idea; "
               "treat results as illustrative, not as expected future performance.",
               "Long-only, fully invested when in a trade, no leverage, no taxes; slippage is a flat estimate."]
    return BacktestResult(eq, metrics, trades, position, caveats)
