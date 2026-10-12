"""AI-Powered Stock & Crypto Market Intelligence Analyzer (Streamlit). Research tool; paper trading only."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import (anomaly, backtest as bt, charts, config, data_alpha, data_coingecko, indicators as ind,
                 models, paper, report, sentiment as sent, signals, trend)
from src.common import DataError, InsufficientDataError

st.set_page_config(page_title="Market Intelligence Analyzer", page_icon="📈", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem}</style>", unsafe_allow_html=True)

STOCKS = {"Global": ["AAPL", "MSFT", "NVDA", "TSLA", "IBM"],
          "India (BSE)": ["RELIANCE.BSE", "TCS.BSE", "INFY.BSE", "HDFCBANK.BSE"]}
ss = st.session_state
ss.setdefault("watch_stock", ["AAPL", "RELIANCE.BSE"])
ss.setdefault("watch_crypto", ["bitcoin", "ethereum"])
ss.setdefault("portfolio", paper.PaperPortfolio())
ss.setdefault("last_prices", {})   # symbol -> dict(price, kind, fetched_at)
ss.setdefault("news", {})


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner="Fetching stock data from Alpha Vantage...")
def load_stock(symbol): return data_alpha.fetch_daily(symbol)


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner="Fetching crypto data from CoinGecko...")
def load_crypto(coin, vs, days): return data_coingecko.fetch_history(coin, vs, days)


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner="Training models on past data only...")
def cached_forecast(df, horizon): return models.run_forecast(df, horizon)


@st.cache_data(ttl=config.CACHE_TTL_SECONDS)
def cached_overview(symbol): return data_alpha.fetch_overview(symbol)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 Market Intelligence")
page = st.sidebar.radio("Dashboard", ["Stocks", "Crypto", "Paper trading"])

with st.sidebar.expander("🔑 API keys", expanded=not (config.alpha_key() and config.coingecko_key())):
    st.caption("Paste your keys here (kept in memory for this session) or put them in the .env file.")
    in_av = st.text_input("Alpha Vantage key", type="password", key="in_av")
    in_cg = st.text_input("CoinGecko key", type="password", key="in_cg")
    in_type = st.selectbox("CoinGecko plan", ["demo", "pro"],
                           index=0 if config.coingecko_key_type() == "demo" else 1)
    in_save = st.checkbox("Also save to local .env file", value=False)
    if st.button("Apply keys"):
        config.set_keys(in_av or None, in_cg or None, in_type, persist=in_save)
        st.cache_data.clear()
        st.rerun()
    if st.button("Test CoinGecko key"):
        try:
            st.success("CoinGecko OK: " + data_coingecko.ping())
        except DataError as e:
            st.error(str(e))
    st.caption(f".env location: {config.ENV_PATH} ({'found' if config.ENV_PATH.exists() else 'not found'})")
st.sidebar.caption(f"Alpha Vantage key: {'✅ set' if config.alpha_key() else '❌ missing'} · "
                   f"CoinGecko key: {'✅ set' if config.coingecko_key() else '❌ missing'}")
if st.sidebar.button("🔄 Force refresh data"):
    st.cache_data.clear()
    ss.news = {}
    st.rerun()
st.sidebar.markdown("**Analysis settings**")
horizon = st.sidebar.slider("Forecast horizon (bars)", 1, 20, 5)
fee_bps = st.sidebar.number_input("Fee (bps per side)", 0.0, 100.0, 10.0)
slip_bps = st.sidebar.number_input("Slippage (bps per side)", 0.0, 100.0, 5.0)
overlays = st.sidebar.multiselect("Overlays", list(charts.OVERLAYS), ["SMA 50", "EMA 12"])
show_bb = st.sidebar.checkbox("Bollinger Bands", True)
show_levels = st.sidebar.checkbox("Support / resistance", True)
show_anom = st.sidebar.checkbox("Mark anomalies", True)
st.sidebar.info("Research tool only. No trades are ever executed. Not financial advice.")


def freshness_banner(r):
    age = r.age_days()
    color = "green" if age <= 1 else "orange" if age <= 5 else "red"
    st.markdown(f":{color}[●] **{r.source}** · {r.kind} · fetched {r.fetched_at:%Y-%m-%d %H:%M UTC} · "
                f"latest bar **{r.last_bar:%Y-%m-%d}** ({age} day(s) old) · simulated data: **no**")
    for n in r.notes:
        st.caption(f"ℹ️ {n}")


def analysis_view(r, asset_type, news_ticker, currency):
    ppy = 365 if asset_type == "crypto" else 252
    d = ind.add_all(r.df, ppy)
    last, prev = d["close"].iloc[-1], d["close"].iloc[-2]
    ss.last_prices[f"{r.symbol}"] = {"price": float(last), "kind": r.kind, "fetched_at": r.fetched_at}
    freshness_banner(r)
    c = st.columns(5)
    c[0].metric("Last close", f"{last:,.2f} {currency}", f"{(last/prev-1)*100:+.2f}%")
    c[1].metric("RSI(14)", f"{d['rsi_14'].iloc[-1]:.1f}")
    c[2].metric("Volatility (ann.)", f"{d['vol_20'].iloc[-1]*100:.1f}%")
    c[3].metric("Bars loaded", f"{len(d):,}")
    tr = trend.detect_trend(d, periods_per_year=ppy)
    c[4].metric("Trend", tr["label"].split(" /")[0])

    forecast = None
    try:
        forecast = cached_forecast(r.df, horizon)
    except InsufficientDataError as e:
        st.warning(f"Forecast unavailable: {e}")
    sres = ss.news.get(r.symbol)
    sig = signals.generate_signal(d, asset_type, forecast, sres.score if sres else None, horizon)
    levels = ind.support_resistance(d) if show_levels else None
    an = anomaly.detect_anomalies(d)
    t_chart, t_tech, t_fc, t_sig, t_news, t_bt, t_rep = st.tabs(
        ["Chart", "Technicals", "Forecast", "Signal", "News & anomalies", "Backtest", "Report"])

    with t_chart:
        n = st.select_slider("Bars shown", [60, 120, 250, 500, len(d)] if len(d) > 500 else [60, 120, len(d)], 120)
        view = d.iloc[-n:]
        st.plotly_chart(charts.price_figure(view, overlays, show_bb, levels,
                                            an["anomaly"] if show_anom else None,
                                            approximated=r.ohlc_approximated), width="stretch")
        perf = pd.DataFrame({p: [(d["close"].iloc[-1] / d["close"].iloc[-k - 1] - 1) * 100 if len(d) > k else None]
                             for p, k in {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}.items()},
                            index=["Return %"])
        st.dataframe(perf.round(2), width="stretch")
        if "market_cap" in d:
            st.caption(f"Market cap (latest): {d['market_cap'].iloc[-1]:,.0f} {currency} · "
                       f"24h volume: {d['volume'].iloc[-1]:,.0f} {currency}")
        if asset_type == "stock":
            if st.button("Load fundamentals (1 API call)"):
                try:
                    st.json(cached_overview(r.symbol))
                except DataError as e:
                    st.error(str(e))

    with t_tech:
        L = d.iloc[-1]
        st.dataframe(pd.DataFrame({"Value": {"SMA20": L["sma_20"], "SMA50": L["sma_50"], "SMA200": L["sma_200"],
                                             "EMA12": L["ema_12"], "EMA26": L["ema_26"], "RSI14": L["rsi_14"],
                                             "MACD": L["macd"], "MACD signal": L["macd_signal"],
                                             "Bollinger upper": L["bb_upper"], "Bollinger lower": L["bb_lower"],
                                             "ATR14": L["atr_14"], "Volatility (ann.)": L["vol_20"]}}).round(4),
                     width="stretch")
        st.write(f"**Trend:** {tr['label']} (R²={tr['r2']:.2f}, {tr['window']} bars)" +
                 (f" · {tr['regime']}" if tr.get("regime") else ""))
        if levels:
            st.write("**Support:**", ", ".join(f"{x:,.2f}" for x in levels["support"]) or "n/a",
                     " · **Resistance:**", ", ".join(f"{x:,.2f}" for x in levels["resistance"]) or "n/a")
        if r.ohlc_approximated:
            st.caption("ATR / support-resistance use derived candles (no true highs/lows) and understate true ranges.")

    with t_fc:
        if forecast is None:
            st.info("Need more history for a leak-free train / calibration / test split.")
        else:
            st.plotly_chart(charts.forecast_figure(d, forecast), width="stretch")
            a, b, c2 = st.columns(3)
            a.metric(f"Expected {horizon}-bar return", f"{forecast.pred_return*100:+.2f}%")
            b.metric("Direction confidence (calibrated)", f"{forecast.direction_confidence*100:.0f}%")
            c2.metric("Skill vs naive (test)", f"{forecast.skill_vs_naive*100:+.1f}%")
            for w in forecast.warnings:
                (st.warning if "NOT" in w else st.caption)(w)
            st.write(f"Walk-forward split sizes: {forecast.sizes} · {forecast.interval_level*100:.0f}% interval "
                     f"coverage on unseen test data: **{forecast.interval_coverage_test*100:.0f}%**")
            st.markdown("**Out-of-sample test metrics** (data the models never saw)")
            st.dataframe(forecast.metrics.style.format("{:.4f}"), width="stretch")
            st.bar_chart(forecast.feature_importance.head(8))

    with t_sig:
        col = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[sig.action]
        st.markdown(f"## :{col}[{sig.action}] &nbsp; score {sig.score:+.2f}")
        a, b, c2, d2 = st.columns(4)
        a.metric("Confidence", "n/a (no edge)" if sig.action == "HOLD" else f"{sig.confidence*100:.0f}%")
        b.metric("Risk", sig.risk_level)
        c2.metric("Stop-loss scenario", f"{sig.stop_loss:,.2f}")
        d2.metric("Target scenario", f"{sig.target:,.2f}")
        st.markdown("**Why**")
        for x in sig.reasons:
            st.markdown(f"- {x}")
        st.dataframe(sig.scenarios.round(2), width="stretch", hide_index=True)
        st.caption(sig.disclaimer)

    with t_news:
        if st.button("Load news sentiment (1 API call)", key="news_btn"):
            try:
                ss.news[r.symbol] = sent.analyze_feed(data_alpha.fetch_news(news_ticker), news_ticker)
                st.rerun()
            except DataError as e:
                st.error(str(e))
        if sres:
            st.metric("News sentiment", f"{sres.label} ({sres.score:+.2f})" if sres.score is not None else "No data",
                      f"{sres.n_articles} articles")
            if sres.note:
                st.info(sres.note)
            if not sres.items.empty:
                st.dataframe(sres.items[["time", "source", "title", "av_score", "vader"]], width="stretch")
        else:
            st.caption("News is loaded on demand to protect the Alpha Vantage free-tier quota.")
        st.markdown("**Anomalies** (descriptive, flagged in hindsight)")
        flagged = an[an["anomaly"]].join(d[["close"]]).tail(15)
        st.dataframe(flagged.round(4), width="stretch") if len(flagged) else st.caption("None flagged.")

    with t_bt:
        res = bt.run_backtest(d, fee_bps=fee_bps, slippage_bps=slip_bps, periods_per_year=ppy)
        st.plotly_chart(charts.equity_figure(res.equity), width="stretch")
        st.dataframe(res.metrics.round(2), width="stretch")
        if len(res.trades):
            st.dataframe(res.trades.round(2).tail(20), width="stretch")
        for cv in res.caveats:
            st.caption("⚠️ " + cv)

    with t_rep:
        md = report.build_markdown(r.symbol, asset_type, r, sig, forecast, res, tr, sres, int(an["anomaly"].sum()), levels)
        st.download_button("⬇️ Report (Markdown)", md, f"{r.symbol}_report.md", "text/markdown")
        st.download_button("⬇️ Data + indicators (CSV)", d.to_csv().encode(), f"{r.symbol}_indicators.csv", "text/csv")
        st.markdown(md)


def watchlist_ui(key, current, fmt=str):
    wl = ss[key]
    c1, c2 = st.sidebar.columns(2)
    if c1.button("➕ Watch") and current not in wl:
        wl.append(current)
    if c2.button("➖ Unwatch") and current in wl:
        wl.remove(current)
    st.sidebar.write("Watchlist: " + (", ".join(fmt(x) for x in wl) or "empty"))


# ---------------------------------------------------------------- pages
if page == "Stocks":
    st.title("Stocks Dashboard")
    grp = st.sidebar.selectbox("Universe", list(STOCKS) + ["Watchlist"])
    options = ss.watch_stock if grp == "Watchlist" else STOCKS[grp]
    pick = st.sidebar.selectbox("Symbol", options) if options else ""
    custom = st.sidebar.text_input("…or custom symbol (e.g. TCS.BSE)")
    symbol = (custom or pick).strip().upper()
    if symbol:
        watchlist_ui("watch_stock", symbol)
    st.caption("Indian equities use Alpha Vantage's BSE tickers (e.g. RELIANCE.BSE); coverage varies. "
               "If a symbol has no data you will see an error: nothing is substituted.")
    if not symbol:
        st.stop()
    try:
        analysis_view(load_stock(symbol), "stock", symbol.split(".")[0], "INR" if symbol.endswith((".BSE", ".NSE")) else "USD")
    except DataError as e:
        st.error(f"⚠️ {e}")
        st.stop()

elif page == "Crypto":
    st.title("Cryptocurrency Dashboard")
    names = {**data_coingecko.POPULAR}
    pick = st.sidebar.selectbox("Coin", list(names))
    custom = st.sidebar.text_input("…or CoinGecko id (e.g. chainlink)")
    coin = (custom or names[pick]).strip().lower()
    vs = st.sidebar.selectbox("Quote currency", ["usd", "inr", "eur"])
    days = st.sidebar.select_slider("History (days)", [90, 180, 365], 365)
    watchlist_ui("watch_crypto", coin)
    if ss.watch_crypto and st.sidebar.button("Refresh watchlist prices"):
        try:
            live = data_coingecko.fetch_live_price(ss.watch_crypto, vs)
            st.dataframe(pd.DataFrame(live).T.rename(columns={"price": f"price ({vs})"}), width="stretch")
        except DataError as e:
            st.error(str(e))
    try:
        analysis_view(load_crypto(coin, vs, days), "crypto", "CRYPTO:" + data_coingecko.SYMBOLS.get(coin, coin.upper()), vs.upper())
    except DataError as e:
        st.error(f"⚠️ {e}")
        st.stop()

else:
    st.title("Paper Trading (virtual money)")
    st.warning("Simulation only. No broker is connected and no real order can be placed.")
    pf: paper.PaperPortfolio = ss.portfolio
    pf.fee_bps = fee_bps
    if not ss.last_prices:
        st.info("Open a Stocks or Crypto dashboard first so a price is available (prices are never invented).")
    else:
        sym = st.selectbox("Symbol (from analysed assets)", list(ss.last_prices))
        info = ss.last_prices[sym]
        st.caption(f"Reference price {info['price']:,.4f} · {info['kind']} · fetched {info['fetched_at']:%Y-%m-%d %H:%M UTC}")
        qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
        b, s = st.columns(2)
        for btn, side in ((b, "buy"), (s, "sell")):
            if btn.button(side.upper()):
                try:
                    getattr(pf, side)(sym, qty, info["price"], info["kind"])
                    st.success(f"Paper {side} recorded.")
                except ValueError as e:
                    st.error(str(e))
    prices = {k: v["price"] for k, v in ss.last_prices.items()}
    c = st.columns(2)
    c[0].metric("Cash", f"{pf.cash:,.2f}")
    c[1].metric("Equity (at last reference prices)", f"{pf.equity(prices):,.2f}")
    st.dataframe(pd.DataFrame(pf.positions.items(), columns=["symbol", "qty"]), width="stretch")
    st.dataframe(pd.DataFrame(pf.log), width="stretch")
    if st.button("Reset portfolio"):
        ss.portfolio = paper.PaperPortfolio()
        st.rerun()
