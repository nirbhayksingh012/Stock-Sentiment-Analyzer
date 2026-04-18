from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression

from stock_analyzer.constants import CHART_LAYOUT, FORECAST_DAYS, MIN_ROWS_FOR_LR_FORECAST


@dataclass(frozen=True)
class LrForecastResult:
    figure: go.Figure
    gain_pct: float


@dataclass(frozen=True)
class ProphetForecastResult:
    figure: go.Figure
    gain_pct: float


def _dark_forecast_layout(title: str) -> dict:
    return {
        "title": title,
        "xaxis_title": "Date",
        "yaxis_title": "Price (USD)",
        **CHART_LAYOUT,
    }


def make_lr_forecast(ticker: str, data: pd.DataFrame) -> LrForecastResult | None:
    if data.empty or "Date" not in data or "Close" not in data:
        return None

    df = data[["Date", "Close"]].copy()
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days
    X = df[["Days"]]
    y = df["Close"]

    if len(X) < MIN_ROWS_FOR_LR_FORECAST:
        return None

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_days = np.arange(X["Days"].max() + 1, X["Days"].max() + FORECAST_DAYS)
    future_dates = [
        df["Date"].max() + pd.Timedelta(days=int(i)) for i in range(1, len(future_days) + 1)
    ]
    future_X = pd.DataFrame({"Days": future_days})
    y_pred_lr = model_lr.predict(future_X)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Close"], name="Historical", line=dict(color="deepskyblue"))
    )
    fig.add_trace(
        go.Scatter(x=future_dates, y=y_pred_lr, name="LR Forecast", line=dict(color="orange"))
    )
    fig.update_layout(
        _dark_forecast_layout(f"{ticker} – Linear Regression 5-Year Forecast"),
    )

    gain_lr = (y_pred_lr[-1] - y.iloc[-1]) / y.iloc[-1] * 100
    return LrForecastResult(figure=fig, gain_pct=float(gain_lr))


def make_prophet_forecast(ticker: str, data: pd.DataFrame) -> ProphetForecastResult | None:
    if data.empty or "Date" not in data or "Close" not in data:
        return None

    df = data[["Date", "Close"]].copy()
    df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})
    df_prophet.dropna(inplace=True)
    if len(df_prophet) < 2:
        return None

    m = Prophet(daily_seasonality=False, yearly_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = m.predict(future)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_prophet["ds"],
            y=df_prophet["y"],
            name="Historical",
            line=dict(color="#56ccf2"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Prophet Forecast",
            line=dict(color="#feca57"),
        )
    )
    fig.update_layout(_dark_forecast_layout(f"{ticker} – Prophet 5-Year Forecast"))

    gain_prophet = (
        (forecast["yhat"].iloc[-1] - df_prophet["y"].iloc[-1]) / df_prophet["y"].iloc[-1] * 100
    )
    return ProphetForecastResult(figure=fig, gain_pct=float(gain_prophet))
