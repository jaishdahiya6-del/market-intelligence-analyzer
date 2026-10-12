import numpy as np
import pandas as pd
from src import indicators as ind
from src.backtest import run_backtest
from src.signals import generate_signal


def test_no_lookahead_in_positions(synthetic_ohlcv):
    d = ind.add_all(synthetic_ohlcv)
    k = 450
    full = run_backtest(d).position
    part = run_backtest(d.iloc[:k]).position
    assert (full.iloc[:k].to_numpy() == part.to_numpy()).all()


def test_costs_reduce_returns(synthetic_ohlcv):
    d = ind.add_all(synthetic_ohlcv)
    free = run_backtest(d, fee_bps=0, slippage_bps=0).metrics.loc["Total return %", "Strategy"]
    paid = run_backtest(d, fee_bps=50, slippage_bps=50).metrics.loc["Total return %", "Strategy"]
    assert paid < free


def test_baseline_and_drawdown(synthetic_ohlcv):
    d = ind.add_all(synthetic_ohlcv)
    r = run_backtest(d, fee_bps=0, slippage_bps=0)
    expected = (d["close"].iloc[-1] / d["close"].iloc[0] - 1) * 100
    assert np.isclose(r.metrics.loc["Total return %", "Buy & hold"], expected, atol=1e-6)
    assert (r.metrics.loc["Max drawdown %"] <= 0).all()


def test_signal_structure_and_levels(synthetic_ohlcv):
    d = ind.add_all(synthetic_ohlcv)
    s = generate_signal(d, "stock")
    c = d["close"].iloc[-1]
    assert s.action in {"BUY", "SELL", "HOLD"} and 0.5 <= s.confidence <= 0.8
    assert s.risk_level in {"Low", "Medium", "High"} and s.reasons
    assert (s.stop_loss < c < s.target) if s.action != "SELL" else (s.target < c < s.stop_loss)


def test_uptrend_not_sell():
    idx = pd.bdate_range("2022-01-01", periods=300)
    c = pd.Series(np.linspace(100, 200, 300) * (1 + 0.003 * np.sin(np.arange(300))), index=idx)
    df = pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1000.0})
    assert generate_signal(ind.add_all(df), "stock").action != "SELL"
