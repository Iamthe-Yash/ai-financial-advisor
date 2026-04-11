"""routes/market.py — live market data endpoints"""
from flask import Blueprint, jsonify, request
import numpy as np
from data import get_quote, get_history, get_features, SYMBOL_MAP

market_bp = Blueprint("market", __name__, url_prefix="/api/market")

ALL_SYMBOLS = list(SYMBOL_MAP.keys())


@market_bp.route("/quotes")
def quotes():
    """Return live quotes for all symbols (or ?symbols=A,B,C)."""
    syms_param = request.args.get("symbols", "")
    syms = [s.strip().upper() for s in syms_param.split(",") if s.strip()] \
           if syms_param else ALL_SYMBOLS
    result = []
    for s in syms:
        q = get_quote(s)
        if q:
            result.append(q)
    return jsonify(result)


@market_bp.route("/quote/<sym>")
def quote_single(sym):
    sym = sym.upper()
    q = get_quote(sym)
    if not q:
        return jsonify({"error": f"No data for {sym}"}), 404
    return jsonify(q)


@market_bp.route("/candles/<sym>")
def candles(sym):
    """OHLCV candle data for candlestick chart."""
    sym    = sym.upper()
    period = request.args.get("period", "3mo")
    df     = get_history(sym, period)
    if df.empty:
        return jsonify({"error": "no data"}), 404
    rows = []
    for ts, row in df.iterrows():
        rows.append({
            "t":  ts.strftime("%Y-%m-%d"),
            "o":  round(float(row["Open"]),  2),
            "h":  round(float(row["High"]),  2),
            "l":  round(float(row["Low"]),   2),
            "c":  round(float(row["Close"]), 2),
            "v":  int(row["Volume"]) if "Volume" in row else 0,
        })
    return jsonify(rows)


@market_bp.route("/indicators/<sym>")
def indicators(sym):
    """RSI + MACD for charting panels."""
    sym = sym.upper()
    df  = get_history(sym, "6mo")
    if df.empty:
        return jsonify({"error": "no data"}), 404
    feat = get_features(df)
    tail = feat.tail(60)
    rsi_data  = [{"t": str(ts.date()), "v": round(float(v), 2)}
                 for ts, v in tail["RSI"].items()]
    macd_data = [{"t": str(ts.date()), "macd": round(float(m), 4),
                  "signal": round(float(s), 4)}
                 for (ts, m), (_, s) in
                 zip(tail["MACD"].items(), tail["MACD_signal"].items())]
    return jsonify({"rsi": rsi_data, "macd": macd_data})


@market_bp.route("/indices")
def indices():
    """Dashboard index cards: NIFTY50, SP500, BTC, Fear&Greed proxy."""
    result = []
    for sym in ["NIFTY50", "SP500", "BTC"]:
        q = get_quote(sym)
        if q:
            result.append({"n": q["name"], "p": _fmt(q["price"]), "c": q["change"]})
    # Fear & Greed proxy: 14-day RSI of SP500 mapped 0-100
    df = get_history("SP500", "3mo")
    fg = 50
    if not df.empty:
        feat = get_features(df)
        if not feat.empty:
            fg = int(feat["RSI"].iloc[-1])
    result.append({"n": "FEAR/GREED", "p": str(fg), "c": round((fg - 50) / 10, 1)})
    return jsonify(result)


@market_bp.route("/movers")
def movers():
    """Top 5 movers by absolute % change."""
    all_q = [get_quote(s) for s in ALL_SYMBOLS]
    all_q = [q for q in all_q if q]
    all_q.sort(key=lambda q: abs(q["change"]), reverse=True)
    return jsonify(all_q[:5])


def _fmt(p):
    if p >= 10000: return f"{p:,.0f}"
    if p >= 1000:  return f"{p:,.2f}"
    return f"{p:.2f}"
