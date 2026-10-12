import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv():
    """SYNTHETIC random-walk data used ONLY inside unit tests (never by the app)."""
    rng = np.random.default_rng(0)
    n = 700
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    vol = rng.integers(1_000, 5_000, n).astype(float)
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
