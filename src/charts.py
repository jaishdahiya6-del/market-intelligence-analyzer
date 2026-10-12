"""Plotly figures (dark theme)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OVERLAYS = {"SMA 20": "sma_20", "SMA 50": "sma_50", "SMA 200": "sma_200", "EMA 12": "ema_12",
            "EMA 26": "ema_26", "EMA 50": "ema_50"}
UP, DOWN = "#26a69a", "#ef5350"


def price_figure(d: pd.DataFrame, overlays=(), bollinger=False, levels=None, anomalies=None,
                 show_rsi=True, show_macd=True, approximated=False) -> go.Figure:
    rows = 2 + int(show_rsi) + int(show_macd)
    heights = [0.5, 0.15] + [0.175] * (rows - 2)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=heights)
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    for name in overlays:
        fig.add_trace(go.Scatter(x=d.index, y=d[OVERLAYS[name]], name=name, line=dict(width=1.2)), 1, 1)
    if bollinger:
        for col, nm in (("bb_upper", "BB upper"), ("bb_lower", "BB lower")):
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=nm, line=dict(width=1, dash="dot", color="#9e9e9e")), 1, 1)
    for kind, color in (("support", "#26a69a"), ("resistance", "#ef5350")):
        for lv in (levels or {}).get(kind, []):
            fig.add_hline(y=lv, line=dict(color=color, dash="dash", width=1), opacity=0.6, row=1, col=1,
                          annotation_text=f"{kind[:3].upper()} {lv:,.2f}", annotation_position="right")
    if anomalies is not None and anomalies.any():
        pts = d[anomalies.reindex(d.index, fill_value=False)]
        fig.add_trace(go.Scatter(x=pts.index, y=pts["close"], mode="markers", name="Anomaly",
                                 marker=dict(color="#ffca28", size=9, symbol="diamond")), 1, 1)
    colors = [UP if c >= o else DOWN for c, o in zip(d["close"], d["open"])]
    fig.add_trace(go.Bar(x=d.index, y=d["volume"], marker_color=colors, name="Volume", opacity=0.7), 2, 1)
    r = 3
    if show_rsi:
        fig.add_trace(go.Scatter(x=d.index, y=d["rsi_14"], name="RSI 14", line=dict(color="#ab47bc")), r, 1)
        for lv in (30, 70):
            fig.add_hline(y=lv, line=dict(color="#757575", dash="dot", width=1), row=r, col=1)
        r += 1
    if show_macd:
        fig.add_trace(go.Bar(x=d.index, y=d["macd_hist"], name="MACD hist",
                             marker_color=[UP if v >= 0 else DOWN for v in d["macd_hist"].fillna(0)]), r, 1)
        fig.add_trace(go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color="#42a5f5")), r, 1)
        fig.add_trace(go.Scatter(x=d.index, y=d["macd_signal"], name="Signal", line=dict(color="#ffa726")), r, 1)
    fig.update_layout(template="plotly_dark", height=250 * rows // 1 + 120 if rows < 4 else 880,
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04),
                      margin=dict(l=10, r=60, t=30, b=10))
    if approximated:
        fig.add_annotation(text="Candles derived from daily closes (open = prev. close)", xref="paper", yref="paper",
                           x=0, y=1.09, showarrow=False, font=dict(size=11, color="#ffca28"))
    return fig


def forecast_figure(d: pd.DataFrame, f, n_hist: int = 90) -> go.Figure:
    h = d["close"].iloc[-n_hist:]
    fx = h.index[-1] + pd.Timedelta(days=f.horizon)
    lo, hi = f.interval_price
    fig = go.Figure(go.Scatter(x=h.index, y=h.values, name="Close", line=dict(color="#42a5f5")))
    fig.add_trace(go.Scatter(x=[h.index[-1], fx], y=[f.last_close, f.pred_price], name="Forecast",
                             mode="lines+markers", line=dict(color="#ffa726", dash="dash"),
                             error_y=dict(type="data", symmetric=False, array=[0, hi - f.pred_price],
                                          arrayminus=[0, f.pred_price - lo], color="#ffa726")))
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"{f.horizon}-bar forecast with {f.interval_level*100:.0f}% interval")
    return fig


def equity_figure(eq: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in eq.columns:
        fig.add_trace(go.Scatter(x=eq.index, y=eq[c], name=c))
    dd = eq["Strategy"] / eq["Strategy"].cummax() - 1
    fig.add_trace(go.Scatter(x=eq.index, y=dd * 100, name="Strategy drawdown %", yaxis="y2",
                             fill="tozeroy", line=dict(width=0.5, color="#ef5350"), opacity=0.35))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="DD %"),
                      legend=dict(orientation="h", y=1.08))
    return fig
