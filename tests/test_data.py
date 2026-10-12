import pandas as pd
import pytest
from src import config, data_alpha, data_coingecko
from src.common import DataError, RateLimitError, clean_ohlcv
from src.paper import PaperPortfolio
from src.sentiment import analyze_feed

AV_OK = {"Time Series (Daily)": {
    "2024-01-03": {"1. open": "10", "2. high": "11", "3. low": "9", "4. close": "10.5", "5. volume": "100"},
    "2024-01-02": {"1. open": "9", "2. high": "10", "3. low": "8.5", "4. close": "9.8", "5. volume": "90"}}}


def test_alpha_parse_and_label(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "SECRETKEY")
    monkeypatch.setattr(data_alpha, "get_json", lambda *a, **k: AV_OK)
    r = data_alpha.fetch_daily("tcs.bse")
    assert r.symbol == "TCS.BSE" and "delayed" in r.kind and not r.simulated
    assert r.df.index.is_monotonic_increasing and r.df["close"].iloc[-1] == 10.5


def test_alpha_rate_limit_message_hides_key(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "SECRETKEY")
    msg = {"Information": "Key SECRETKEY: our standard API rate limit is 25 requests per day."}
    monkeypatch.setattr(data_alpha, "get_json", lambda *a, **k: msg)
    with pytest.raises(RateLimitError) as e:
        data_alpha.fetch_daily("AAPL")
    assert "SECRETKEY" not in str(e.value)


def test_missing_keys_raise(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(DataError):
        data_alpha.fetch_daily("AAPL")
    with pytest.raises(DataError):
        data_coingecko.fetch_history("bitcoin")


def test_coingecko_parse_flags_approximation(monkeypatch):
    monkeypatch.setenv("COINGECKO_API_KEY", "k")
    ms = lambda d: int(pd.Timestamp(d).timestamp() * 1000)
    payload = {"prices": [[ms("2024-01-01"), 100], [ms("2024-01-02"), 110]],
               "total_volumes": [[ms("2024-01-01"), 5], [ms("2024-01-02"), 6]],
               "market_caps": [[ms("2024-01-01"), 1e9], [ms("2024-01-02"), 1.1e9]]}
    monkeypatch.setattr(data_coingecko, "get_json", lambda *a, **k: payload)
    r = data_coingecko.fetch_history("bitcoin")
    assert r.ohlc_approximated and r.df["open"].iloc[1] == 100 and r.df["market_cap"].iloc[1] == 1.1e9


def test_clean_ohlcv_repairs():
    idx = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-03"])
    df = pd.DataFrame({"open": [1, 1, 1, None], "high": [2, 2, 2, None], "low": [0.5, 0.5, 0.5, None],
                       "close": [1.5, 1.5, 1.5, None], "volume": [1, 1, 1, 1]}, index=idx)
    out, notes = clean_ohlcv(df)
    assert out.index.is_monotonic_increasing and len(out) == 2 and notes


def test_sentiment_parse():
    feed = [{"title": "Great profits and strong growth", "summary": "Excellent", "source": "x",
             "time_published": "20240101T000000", "url": "u",
             "ticker_sentiment": [{"ticker": "AAPL", "ticker_sentiment_score": "0.4", "relevance_score": "0.9"}]}]
    r = analyze_feed(feed, "AAPL")
    assert r.label == "Positive" and r.n_articles == 1
    assert analyze_feed([], "AAPL").score is None


def test_paper_portfolio_rules():
    p = PaperPortfolio(cash=1000, fee_bps=0)
    p.buy("X", 5, 100)
    with pytest.raises(ValueError):
        p.buy("X", 100, 100)
    with pytest.raises(ValueError):
        p.sell("X", 6, 100)
    p.sell("X", 5, 120)
    assert p.cash == 1100 and not p.positions
