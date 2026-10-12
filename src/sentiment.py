"""News sentiment: Alpha Vantage's per-ticker scores (primary) cross-checked with VADER on headlines."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentResult:
    score: float | None          # -1..1 (AV score if available, else VADER)
    label: str
    av_score: float | None
    vader_score: float | None
    n_articles: int
    items: pd.DataFrame
    note: str = ""


def label_for(score) -> str:
    if score is None:
        return "No data"
    return "Positive" if score >= 0.15 else "Negative" if score <= -0.15 else "Neutral"


def analyze_feed(feed: list, ticker: str) -> SentimentResult:
    if not feed:
        return SentimentResult(None, "No data", None, None, 0, pd.DataFrame(),
                               "No news returned for this ticker (coverage for some markets is limited).")
    an = SentimentIntensityAnalyzer()
    rows, w_sum, w_tot = [], 0.0, 0.0
    for it in feed:
        text = f"{it.get('title', '')}. {it.get('summary', '')}"
        v = an.polarity_scores(text)["compound"]
        av = None
        for ts in it.get("ticker_sentiment", []) or []:
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    av, rel = float(ts["ticker_sentiment_score"]), float(ts.get("relevance_score", 1))
                    w_sum += av * rel
                    w_tot += rel
                except (KeyError, ValueError):
                    av = None
        rows.append({"time": it.get("time_published"), "source": it.get("source"), "title": it.get("title"),
                     "av_score": av, "vader": v, "url": it.get("url")})
    items = pd.DataFrame(rows)
    av_score = w_sum / w_tot if w_tot > 0 else None
    vader = float(items["vader"].mean())
    score = av_score if av_score is not None else vader
    return SentimentResult(score, label_for(score), av_score, vader, len(items), items)
