"""routes/screener.py — AI stock screener with technical signals"""
from flask import Blueprint, jsonify, request
import numpy as np
from data import get_history, get_features, SYMBOL_MAP, _calc_risk, _name

screener_bp = Blueprint("screener", __name__, url_prefix="/api/screener")

ALL_SYMBOLS = [s for s in SYMBOL_MAP if s not in ("NIFTY50","SP500","GOLD","CRUDE")]


@screener_bp.route("/scan")
def scan():
    """Scan all symbols and return screener cards with signals."""
    results = []
    for sym in ALL_SYMBOLS:
        df = get_history(sym, "3mo")
        if df.empty or len(df) < 30: continue
        feat = get_features(df)
        if feat.empty: continue

        close  = float(df["Close"].iloc[-1])
        prev   = float(df["Close"].iloc[-2]) if len(df) > 1 else close
        chg    = round((close - prev) / prev * 100, 2)
        rsi    = round(float(feat["RSI"].iloc[-1]), 1)
        macd   = float(feat["MACD"].iloc[-1])
        sig    = float(feat["MACD_signal"].iloc[-1])
        vol30  = float(feat["Volatility"].iloc[-1]) * np.sqrt(252)
        risk   = _calc_risk(df, sym)

        # Signal
        score = 0
        if rsi < 35:   score += 2
        elif rsi > 65: score -= 2
        if macd > sig: score += 1
        else:          score -= 1
        ret_10 = (close / float(df["Close"].iloc[-10]) - 1) * 100 if len(df) > 10 else 0
        if ret_10 > 0: score += 1
        else:          score -= 1

        signal = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"

        results.append({
            "sym":    sym,
            "name":   _name(sym),
            "price":  round(close, 2),
            "change": chg,
            "rsi":    rsi,
            "macd":   round(macd, 4),
            "vol":    round(vol30 * 100, 2),
            "risk":   risk,
            "signal": signal,
            "score":  score,
        })

    results.sort(key=lambda x: -abs(x["change"]))
    buy_count  = sum(1 for r in results if r["signal"] == "BUY")
    sell_count = sum(1 for r in results if r["signal"] == "SELL")
    high_risk  = sum(1 for r in results if r["risk"] > 60)

    return jsonify({
        "stocks":     results,
        "total":      len(results),
        "buy_signals":buy_count,
        "sell_signals":sell_count,
        "high_risk":  high_risk,
    })
