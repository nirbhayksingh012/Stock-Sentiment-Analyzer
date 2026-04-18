import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            threads=False,
            auto_adjust=True,
        )
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data.reset_index(inplace=True)
        return data
    except Exception:
        return pd.DataFrame()
