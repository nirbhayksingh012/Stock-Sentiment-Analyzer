from datetime import date

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from stock_analyzer import charts, forecast, styles
from stock_analyzer.constants import (
    DEFAULT_START_DATE,
    DEFAULT_TICKER_SELECTION,
    PAGE_TITLE,
    TICKER_DICT,
)
from stock_analyzer.data import fetch_stock_data


def _inject_styles() -> None:
    st.markdown(styles.app_shell_css(), unsafe_allow_html=True)
    st.markdown(styles.title_block_html(), unsafe_allow_html=True)
    st.markdown(styles.hide_streamlit_chrome_css(), unsafe_allow_html=True)


def _sidebar_inputs():
    ticker_options = [f"{ticker} – {name}" for ticker, name in TICKER_DICT.items()]
    st.sidebar.header("📊 Input Parameters")
    selected_display = st.sidebar.multiselect(
        "💼 Choose Tickers",
        options=ticker_options,
        default=DEFAULT_TICKER_SELECTION,
    )
    tickers = [item.split(" – ")[0] for item in selected_display]
    start_date = st.sidebar.date_input("📅 Start Date", DEFAULT_START_DATE)
    end_date = st.sidebar.date_input("📅 End Date", date.today())

    if start_date > end_date:
        st.sidebar.error("⚠ Start Date cannot be after End Date.")
        st.stop()

    enable_refresh = st.sidebar.checkbox("🔁 Enable Auto-Refresh", value=False)
    refresh_interval = (
        st.sidebar.slider("⏲ Refresh Interval (seconds)", 10, 120, 30)
        if enable_refresh
        else 30
    )
    show_combined = st.sidebar.checkbox("📈 Show Combined Chart", value=True)

    return tickers, start_date, end_date, enable_refresh, refresh_interval, show_combined


def _render_forecast_section(ticker: str, data) -> None:
    if data.empty or "Date" not in data or "Close" not in data:
        st.warning(f"Not enough data to predict {ticker}")
        return

    st.subheader(f"📈 5-Year Forecast for {ticker}")

    lr = forecast.make_lr_forecast(ticker, data)
    if lr is None:
        st.warning("Not enough data for reliable forecast.")
        return

    st.plotly_chart(lr.figure, use_container_width=True)
    st.success(f"📈 LR Forecast Gain in 5 Years: {lr.gain_pct:.2f}%")

    prophet_result = forecast.make_prophet_forecast(ticker, data)
    if prophet_result is None:
        st.warning(f"❌ Not enough valid data for Prophet prediction for {ticker}")
        return

    st.plotly_chart(prophet_result.figure, use_container_width=True)
    st.success(f"📈 Prophet Forecast Gain in 5 Years: {prophet_result.gain_pct:.2f}%")


def _plot_stock(ticker: str, start_date, end_date) -> None:
    st.markdown(f"<h2>📊 {ticker} Stock Analysis</h2>", unsafe_allow_html=True)
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        st.warning(f"No data found for '{ticker}'")
        return

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    st.subheader("📉 Candlestick Chart")
    st.plotly_chart(charts.make_candlestick_figure(data), use_container_width=True)

    st.subheader("📈 Closing Price Line Chart")
    st.plotly_chart(charts.make_close_line_figure(data), use_container_width=True)

    st.subheader("📊 Volume Chart")
    st.plotly_chart(charts.make_volume_figure(data), use_container_width=True)

    st.subheader("📄 Raw Data")
    st.dataframe(
        data[["Date", "Open", "High", "Low", "Close", "Volume", "MA20", "MA50"]],
        use_container_width=True,
    )

    _render_forecast_section(ticker, data)


def _plot_combined_chart(tickers: list[str], start_date, end_date) -> None:
    fig = charts.make_combined_normalized_figure(
        tickers, start_date, end_date, fetch_stock_data
    )
    if fig is None:
        st.warning("No valid data for combined chart.")
        return
    st.markdown(
        "<h2>📊 Combined Performance Chart (Normalized)</h2>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def _run_app(
    tickers: list[str],
    start_date,
    end_date,
    show_combined: bool,
) -> None:
    if not tickers:
        st.warning("Please select at least one ticker.")
        return
    for ticker in tickers:
        with st.expander(f"🔍 {ticker} Stock Details", expanded=True):
            _plot_stock(ticker, start_date, end_date)
    if show_combined and len(tickers) > 1:
        st.markdown("---")
        _plot_combined_chart(tickers, start_date, end_date)


def run() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    _inject_styles()

    tickers, start_date, end_date, enable_refresh, refresh_interval, show_combined = (
        _sidebar_inputs()
    )

    if enable_refresh:
        st_autorefresh(interval=refresh_interval * 1000, limit=None, key="refresh")

    _run_app(tickers, start_date, end_date, show_combined)
