"""
data.py — central data fetcher with in-memory cache
All routes import from here so yfinance is called once per symbol per session.
"""

import time, threading
import numpy as np
import pandas as pd
import yfinance as yf

_cache: dict = {}
_lock  = threading.Lock()
CACHE_TTL = 300   # seconds

# ── symbol maps ────────────────────────────────────────────────
SYMBOL_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS":      "TCS.NS",
    "INFY":     "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "WIPRO":    "WIPRO.NS",
    "SBIN":     "SBIN.NS",
    "SUNPHARMA":"SUNPHARMA.NS",
    "AAPL":     "AAPL",
    "TSLA":     "TSLA",
    "NVDA":     "NVDA",
    "MSFT":     "MSFT",
    "BTC":      "BTC-USD",
    "ETH":      "ETH-USD",
    "NIFTY50":  "^NSEI",
    "SP500":    "^GSPC",
    "GOLD":     "GC=F",
    "CRUDE":    "CL=F",
}

CATEGORIES = {
    "RELIANCE":"indian","TCS":"indian","INFY":"indian","HDFCBANK":"indian",
    "WIPRO":"indian","SBIN":"indian","SUNPHARMA":"indian",
    "AAPL":"us","TSLA":"us","NVDA":"us","MSFT":"us",
    "BTC":"crypto","ETH":"crypto",
    "NIFTY50":"indices","SP500":"indices",
    "GOLD":"commodities","CRUDE":"commodities",
}

RISK_BASE = {
    "RELIANCE":28,"TCS":22,"INFY":32,"HDFCBANK":24,"WIPRO":38,
    "SBIN":35,"SUNPHARMA":30,"AAPL":30,"TSLA":76,"NVDA":42,
    "MSFT":26,"BTC":88,"ETH":72,"NIFTY50":18,"SP500":20,
    "GOLD":14,"CRUDE":22,
}


def _key(sym, period):
    return f"{sym}|{period}"


def get_history(sym: str, period: str = "1y") -> pd.DataFrame:
    """Return OHLCV DataFrame for a symbol, using cache."""
    k = _key(sym, period)
    with _lock:
        entry = _cache.get(k)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            return entry["df"].copy()

    ticker_sym = SYMBOL_MAP.get(sym, sym)
    try:
        df = yf.download(ticker_sym, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df = df.dropna()
        with _lock:
            _cache[k] = {"df": df, "ts": time.time()}
        return df.copy()
    except Exception as e:
        print(f"[data] yfinance error for {sym}: {e}")
        return pd.DataFrame()


def get_quote(sym: str) -> dict:
    """Return latest quote dict for a symbol."""
    df = get_history(sym, "5d")
    if df.empty:
        return {}
    row  = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row
    close  = float(row["Close"])
    pclose = float(prev["Close"])
    chg    = round((close - pclose) / pclose * 100, 2)
    hi52   = round(float(df["High"].max()), 2)
    lo52   = round(float(df["Low"].min()), 2)
    vol    = int(row["Volume"]) if "Volume" in row else 0

    return {
        "sym":   sym,
        "name":  _name(sym),
        "price": round(close, 2),
        "change": chg,
        "volume": _fmt_vol(vol),
        "hi52":  hi52,
        "lo52":  lo52,
        "cat":   CATEGORIES.get(sym, "other"),
        "risk":  _calc_risk(df, sym),
    }


def _calc_risk(df: pd.DataFrame, sym: str) -> int:
    """Compute 0-100 risk score from 30-day rolling volatility."""
    if len(df) < 5:
        return RISK_BASE.get(sym, 50)
    ret = df["Close"].pct_change().dropna()
    vol30 = float(ret.tail(30).std()) * np.sqrt(252)  # annualised
    # map 0-200% annualised vol → 0-100 score, then blend with base
    dynamic = min(100, int(vol30 * 50))
    base    = RISK_BASE.get(sym, 50)
    return int(dynamic * 0.6 + base * 0.4)


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix used by all ML models."""
    d = df[["Close","Volume"]].copy()
    d.columns = ["Close","Volume"]
    d["Return"]      = d["Close"].pct_change()
    d["Volatility"]  = d["Return"].rolling(20).std()
    d["SMA_20"]      = d["Close"].rolling(20).mean()
    d["SMA_50"]      = d["Close"].rolling(50).mean()
    d["EMA_10"]      = d["Close"].ewm(span=10).mean()
    d["Vol_ratio"]   = d["Volume"] / d["Volume"].rolling(20).mean()

    # RSI
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    d["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = d["Close"].ewm(span=12).mean()
    ema26 = d["Close"].ewm(span=26).mean()
    d["MACD"]        = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9).mean()

    # Bollinger position
    mid  = d["Close"].rolling(20).mean()
    std  = d["Close"].rolling(20).std()
    d["BB_pos"] = (d["Close"] - mid) / (2 * std + 1e-9)

    d = d.dropna()
    return d


FEATURE_COLS = [
    "Close","Return","Volatility","SMA_20","EMA_10",
    "Vol_ratio","RSI","MACD","BB_pos",
]


# ── helpers ────────────────────────────────────────────────────
_NAMES = {
    "RELIANCE":"Reliance Industries","TCS":"Tata Consultancy",
    "INFY":"Infosys Ltd","HDFCBANK":"HDFC Bank","WIPRO":"Wipro Ltd",
    "SBIN":"State Bank of India","SUNPHARMA":"Sun Pharma",
    "AAPL":"Apple Inc","TSLA":"Tesla Inc","NVDA":"NVIDIA Corp",
    "MSFT":"Microsoft Corp","BTC":"Bitcoin","ETH":"Ethereum",
    "NIFTY50":"NIFTY 50 Index","SP500":"S&P 500 Index",
    "GOLD":"Gold Futures","CRUDE":"Crude Oil WTI",
}

def _name(sym):
    return _NAMES.get(sym, sym)

def _fmt_vol(v: int) -> str:
    if v >= 1_000_000_000: return f"${v/1e9:.1f}B"
    if v >= 1_000_000:     return f"{v/1e6:.1f}M"
    if v >= 1_000:         return f"{v/1e3:.1f}K"
    return str(v)
