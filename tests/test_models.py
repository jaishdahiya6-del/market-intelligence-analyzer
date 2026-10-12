import numpy as np
import pytest
from src.common import InsufficientDataError
from src.models import direction_confidence, run_forecast


def test_forecast_split_is_chronological_and_purged(synthetic_ohlcv):
    res = run_forecast(synthetic_ohlcv, horizon=5)
    tf = res.test_frame
    assert tf.index.is_monotonic_increasing
    assert res.sizes["train"] > 0 and res.sizes["test"] > 0
    assert {"MAE", "RMSE", "Directional accuracy"} <= set(res.metrics.columns)
    assert "Naive (zero return)" in res.metrics.index
    lo, hi = res.interval_return
    assert lo < res.pred_return < hi
    assert 0.0 <= res.interval_coverage_test <= 1.0
    assert 0.0 < res.direction_confidence < 1.0


def test_forecast_rejects_short_history(synthetic_ohlcv):
    with pytest.raises(InsufficientDataError):
        run_forecast(synthetic_ohlcv.iloc[:150])


def test_confidence_shrinks_small_samples():
    c, n = direction_confidence(0.01, np.array([0.02, 0.01]), np.array([0.03, 0.02]))
    assert n == 2 and c < 0.6   # 2/2 hits must NOT yield ~100%
