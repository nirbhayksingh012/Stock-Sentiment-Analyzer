import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import date
from sklearn.linear_model import LinearRegression
import numpy as np
from prophet import Prophet
from streamlit_autorefresh import st_autorefresh


# === App config ===
st.set_page_config(page_title="📊 Stock Sentiment Analyzer", layout="wide")

# === Styling ===
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: #FAFAFA;
        font-family: 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(135deg, rgba(32, 50, 73, 0.6), rgba(57, 108, 139, 0.5));
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 25px 20px;
        margin: 10px;
    }}
    .floating-btn {{
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #4e9af1;
        color: white;
        border: none;
        padding: 15px 18px;
        border-radius: 50px;
        font-size: 20px;
        box-shadow: 0 5px 15px rgba(78, 154, 241, 0.4);
        cursor: pointer;
        z-index: 9999;
    }}
    </style>

""", unsafe_allow_html=True)

# === Animated Headline ===
st.markdown("""
    <style>
    .animated-title {
        font-size: 48px;
        text-align: center;
        margin-top: 30px;
        color: #FAFAFA;
        animation: fadeInSlide 2s ease-in-out forwards;
        opacity: 0;
    }

    @keyframes fadeInSlide {
        0% {
            opacity: 0;
            transform: translateY(-40px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>

    <h1 class="animated-title">📊 Stock Sentiment Analyzer</h1>
    
""", unsafe_allow_html=True)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# === Sidebar Inputs ===
ticker_dict = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "META": "Meta Platforms Inc.", "NFLX": "Netflix Inc.", "NVDA": "NVIDIA Corporation", "AMD": "Advanced Micro Devices", "INTC": "Intel Corporation", "BABA": "Alibaba Group", "ADBE": "Adobe Inc.", "ORCL": "Oracle Corporation", "CRM": "Salesforce.com Inc.", "PYPL": "PayPal Holdings", "UBER": "Uber Technologies", "JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America", "WMT": "Walmart Inc.", "T": "AT&T Inc.", "VZ": "Verizon Communications", "DIS": "The Walt Disney Company", "PEP": "PepsiCo Inc.", "KO": "Coca-Cola Company", "NKE": "Nike Inc.", "COST": "Costco Wholesale", "MCD": "McDonald’s Corporation", "IBM": "IBM", "GE": "General Electric", "SBUX": "Starbucks", "PFE": "Pfizer", "MRNA": "Moderna", "XOM": "Exxon Mobil", "CVX": "Chevron"}
ticker_options = [f"{ticker} – {name}" for ticker, name in ticker_dict.items()]
st.sidebar.header("📊 Input Parameters")
selected_display = st.sidebar.multiselect("💼 Choose Tickers", options=ticker_options, default=["AAPL – Apple Inc.", "MSFT – Microsoft Corporation"])
tickers = [item.split(" – ")[0] for item in selected_display]
start_date = st.sidebar.date_input("📅 Start Date", date(2023, 1, 1))
end_date = st.sidebar.date_input("📅 End Date", date.today())

if start_date > end_date:
    st.sidebar.error("⚠ Start Date cannot be after End Date.")
    st.stop()

enable_refresh = st.sidebar.checkbox("🔁 Enable Auto-Refresh", value=False)
refresh_interval = st.sidebar.slider("⏲ Refresh Interval (seconds)", 10, 120, 30) if enable_refresh else 30
show_combined = st.sidebar.checkbox("📈 Show Combined Chart", value=True)

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data.reset_index(inplace=True)
        return data
    except Exception:
        return pd.DataFrame()

def plot_prediction_chart(ticker, data):
    if data.empty or 'Date' not in data or 'Close' not in data:
        st.warning(f"Not enough data to predict {ticker}")
        return

    st.subheader(f"📈 5-Year Forecast for {ticker}")

    df = data[['Date', 'Close']].copy()
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Close']

    if len(X) < 100:
        st.warning("Not enough data for reliable forecast.")
        return

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_days = np.arange(X['Days'].max() + 1, X['Days'].max() + 5*365)
    future_dates = [df['Date'].max() + pd.Timedelta(days=int(i)) for i in range(1, len(future_days)+1)]
    y_pred_lr = model_lr.predict(future_days.reshape(-1, 1))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical", line=dict(color='deepskyblue')))
    fig1.add_trace(go.Scatter(x=future_dates, y=y_pred_lr, name="LR Forecast", line=dict(color='orange')))
    fig1.update_layout(title=f"{ticker} – Linear Regression 5-Year Forecast", xaxis_title="Date", yaxis_title="Price (USD)", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#FAFAFA"))
    st.plotly_chart(fig1, use_container_width=True)

    gain_lr = (y_pred_lr[-1] - y.iloc[-1]) / y.iloc[-1] * 100
    st.success(f"📈 LR Forecast Gain in 5 Years: {gain_lr:.2f}%")

    df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})
    df_prophet.dropna(inplace=True)
    if len(df_prophet) < 2:
        st.warning(f"❌ Not enough valid data for Prophet prediction for {ticker}")
        return

    m = Prophet(daily_seasonality=False, yearly_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=5 * 365)
    forecast = m.predict(future)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name="Historical", line=dict(color="#56ccf2")))
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Prophet Forecast", line=dict(color="#feca57")))
    fig2.update_layout(title=f"{ticker} – Prophet 5-Year Forecast", xaxis_title="Date", yaxis_title="Price (USD)", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#FAFAFA"))
    st.plotly_chart(fig2, use_container_width=True)

    gain_prophet = (forecast['yhat'].iloc[-1] - df_prophet['y'].iloc[-1]) / df_prophet['y'].iloc[-1] * 100
    st.success(f"📈 Prophet Forecast Gain in 5 Years: {gain_prophet:.2f}%")

def plot_stock(ticker):
    st.markdown(f"<h2>📊 {ticker} Stock Analysis</h2>", unsafe_allow_html=True)
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        st.warning(f"No data found for '{ticker}'")
        return
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    st.subheader("📉 Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], increasing_line_color='green', decreasing_line_color='red')])
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], mode='lines', name='MA20', line=dict(color="#4e9af1")))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], mode='lines', name='MA50', line=dict(color="#f5a623")))
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#FAFAFA"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Closing Price Line Chart")
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color="#56ccf2")))
    line_fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#FAFAFA"))
    st.plotly_chart(line_fig, use_container_width=True)

    st.subheader("📊 Volume Chart")
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color="#bb86fc"))
    vol_fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#FAFAFA"))
    st.plotly_chart(vol_fig, use_container_width=True)

    st.subheader("📄 Raw Data")
    st.dataframe(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']], use_container_width=True)

    plot_prediction_chart(ticker, data)

def plot_combined_chart(tickers):
    combined_df = pd.DataFrame()
    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            continue
        data = data['Close'].dropna()
        data.name = ticker
        combined_df = pd.concat([combined_df, data], axis=1)
    if combined_df.empty:
        st.warning("No valid data for combined chart.")
        return
    normalized_df = combined_df / combined_df.iloc[0] * 100
    st.markdown("<h2>📊 Combined Performance Chart (Normalized)</h2>", unsafe_allow_html=True)
    fig = go.Figure()
    for ticker in normalized_df.columns:
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df[ticker], mode='lines', name=ticker))
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#FAFAFA"))
    st.plotly_chart(fig, use_container_width=True)

def run_app():
    if not tickers:
        st.warning("Please select at least one ticker.")
        return
    for ticker in tickers:
        with st.expander(f"🔍 {ticker} Stock Details", expanded=True):
            plot_stock(ticker)
    if show_combined and len(tickers) > 1:
        st.markdown("---")
        plot_combined_chart(tickers)

if enable_refresh:
    st_autorefresh(interval=refresh_interval * 1000, limit=None, key="refresh")
run_app()
