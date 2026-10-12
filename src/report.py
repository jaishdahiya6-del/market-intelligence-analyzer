"""Downloadable report builders (Markdown + CSV)."""
from __future__ import annotations

from datetime import datetime, timezone


def build_markdown(symbol, asset_type, data, signal, forecast, backtest, trend, sentiment, anomalies_n, levels) -> str:
    L = [f"# Market Intelligence Report: {symbol}",
         f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_", "",
         "## Data provenance",
         f"- Source: **{data.source}** ({data.kind})",
         f"- Fetched: {data.fetched_at:%Y-%m-%d %H:%M UTC}; latest bar: {data.last_bar:%Y-%m-%d}",
         f"- Simulated data: **{'YES' if data.simulated else 'No'}**"]
    L += [f"- Note: {n}" for n in data.notes]
    L += ["", "## Signal (research only)",
          f"- **{signal.action}**, score {signal.score:+.2f}, calibrated confidence {signal.confidence*100:.0f}%, "
          f"risk {signal.risk_level}",
          f"- Stop-loss scenario: {signal.stop_loss:,.2f} | Target scenario: {signal.target:,.2f}", "", "Reasoning:"]
    L += [f"- {r}" for r in signal.reasons]
    L += ["", signal.scenarios.round(2).to_markdown(index=False) if _has_tabulate() else signal.scenarios.round(2).to_string(index=False)]
    L += ["", "## Trend", f"- {trend['label']} (R² {trend['r2']:.2f}, window {trend['window']} bars)"]
    if trend.get("regime"):
        L.append(f"- {trend['regime']}")
    if levels:
        L += [f"- Support: {', '.join(f'{x:,.2f}' for x in levels['support']) or 'n/a'}",
              f"- Resistance: {', '.join(f'{x:,.2f}' for x in levels['resistance']) or 'n/a'}"]
    L += ["", "## Forecast"]
    if forecast is None:
        L.append("Not available (insufficient history).")
    else:
        L += [f"- {forecast.horizon}-bar return: {forecast.pred_return*100:+.2f}% "
              f"({forecast.interval_level*100:.0f}% interval {forecast.interval_return[0]*100:+.2f}% to "
              f"{forecast.interval_return[1]*100:+.2f}%)",
              f"- Skill vs naive baseline on unseen data: {forecast.skill_vs_naive*100:+.1f}%",
              f"- Test interval coverage: {forecast.interval_coverage_test*100:.0f}%"]
        L += [f"- WARNING: {w}" for w in forecast.warnings]
    L += ["", "## Sentiment",
          f"- {sentiment.label} ({sentiment.n_articles} articles)" if sentiment else "- Not loaded",
          "", "## Anomalies", f"- {anomalies_n} unusual bars flagged (descriptive, hindsight)", "", "## Backtest"]
    if backtest is None:
        L.append("Not run.")
    else:
        L.append(backtest.metrics.round(2).to_string())
        L += [f"- {c}" for c in backtest.caveats]
    L += ["", "---", signal.disclaimer]
    return "\n".join(L)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
