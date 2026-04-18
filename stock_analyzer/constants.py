from datetime import date

PAGE_TITLE = "📊 Stock Sentiment Analyzer"

DEFAULT_START_DATE = date(2023, 1, 1)

DEFAULT_TICKER_SELECTION = ["AAPL – Apple Inc.", "MSFT – Microsoft Corporation"]

TICKER_DICT: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NFLX": "Netflix Inc.",
    "NVDA": "NVIDIA Corporation",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel Corporation",
    "BABA": "Alibaba Group",
    "ADBE": "Adobe Inc.",
    "ORCL": "Oracle Corporation",
    "CRM": "Salesforce.com Inc.",
    "PYPL": "PayPal Holdings",
    "UBER": "Uber Technologies",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America",
    "WMT": "Walmart Inc.",
    "T": "AT&T Inc.",
    "VZ": "Verizon Communications",
    "DIS": "The Walt Disney Company",
    "PEP": "PepsiCo Inc.",
    "KO": "Coca-Cola Company",
    "NKE": "Nike Inc.",
    "COST": "Costco Wholesale",
    "MCD": "McDonald’s Corporation",
    "IBM": "IBM",
    "GE": "General Electric",
    "SBUX": "Starbucks",
    "PFE": "Pfizer",
    "MRNA": "Moderna",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
}

CHART_LAYOUT = {
    "paper_bgcolor": "#0e1117",
    "plot_bgcolor": "#0e1117",
    "font": {"color": "#FAFAFA"},
}

MIN_ROWS_FOR_LR_FORECAST = 100

FORECAST_DAYS = 5 * 365
