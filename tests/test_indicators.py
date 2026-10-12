import numpy as np
import pandas as pd
from src import indicators as ind
from src.features import build_features


def test_sma_value():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ind.sma(s, 3).iloc[-1] == 4.0 and np.isnan(ind.sma(s, 3).iloc[1])


def test_rsi_bounds_and_extremes(synthetic_ohlcv):
    r = ind.rsi(synthetic_ohlcv["close"]).dropna()
    assert r.between(0, 100).all()
    assert ind.rsi(pd.Series(np.arange(1, 60, dtype=float))).iloc[-1] == 100.0


def test_macd_hist_identity(synthetic_ohlcv):
    m = ind.macd(synthetic_ohlcv["close"]).dropna()
    assert np.allclose(m["macd_hist"], m["macd"] - m["macd_signal"])


def test_bollinger_ordering(synthetic_ohlcv):
    b = ind.bollinger(synthetic_ohlcv["close"]).dropna()
    assert (b["bb_upper"] >= b["bb_mid"]).all() and (b["bb_mid"] >= b["bb_lower"]).all()


def test_support_resistance_sides(synthetic_ohlcv):
    lv = ind.support_resistance(synthetic_ohlcv)
    p = synthetic_ohlcv["close"].iloc[-1]
    assert all(x < p for x in lv["support"]) and all(x > p for x in lv["resistance"])


def test_features_are_causal(synthetic_ohlcv):
    """Features at row t must not change when future rows are removed (no look-ahead)."""
    k = 500
    full, _ = build_features(synthetic_ohlcv, 5)
    part, _ = build_features(synthetic_ohlcv.iloc[:k], 5)
    pd.testing.assert_frame_equal(full.iloc[:k], part, check_exact=False, atol=1e-10)


def test_target_is_forward_return(synthetic_ohlcv):
    _, y = build_features(synthetic_ohlcv, 5)
    c = synthetic_ohlcv["close"]
    assert np.isclose(y.iloc[10], c.iloc[15] / c.iloc[10] - 1)
    assert y.iloc[-5:].isna().all()
