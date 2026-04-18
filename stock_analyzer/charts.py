from collections.abc import Callable

import pandas as pd
import plotly.graph_objs as go

from stock_analyzer.constants import CHART_LAYOUT


def _apply_dark_layout(fig: go.Figure, **extra) -> go.Figure:
    layout = {**CHART_LAYOUT, **extra}
    fig.update_layout(**layout)
    return fig


def make_candlestick_figure(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["MA20"],
            mode="lines",
            name="MA20",
            line=dict(color="#4e9af1"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["MA50"],
            mode="lines",
            name="MA50",
            line=dict(color="#f5a623"),
        )
    )
    return _apply_dark_layout(fig)


def make_close_line_figure(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#56ccf2"),
        )
    )
    return _apply_dark_layout(fig)


def make_volume_figure(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data["Date"],
            y=data["Volume"],
            name="Volume",
            marker_color="#bb86fc",
        )
    )
    return _apply_dark_layout(fig)


def make_combined_normalized_figure(
    tickers: list[str],
    start_date,
    end_date,
    fetch_stock_data: Callable[..., pd.DataFrame],
) -> go.Figure | None:
    combined_df = pd.DataFrame()
    for ticker in tickers:
        raw = fetch_stock_data(ticker, start_date, end_date)
        if raw.empty:
            continue
        series = raw["Close"].dropna()
        series.name = ticker
        combined_df = pd.concat([combined_df, series], axis=1)
    if combined_df.empty:
        return None
    normalized_df = combined_df / combined_df.iloc[0] * 100
    fig = go.Figure()
    for col in normalized_df.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized_df.index,
                y=normalized_df[col],
                mode="lines",
                name=col,
            )
        )
    return _apply_dark_layout(fig)
