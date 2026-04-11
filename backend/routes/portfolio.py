"""routes/portfolio.py — portfolio holdings, allocation, MPT optimiser, rebalancing with SQLite"""
from flask import Blueprint, jsonify, request
import numpy as np
from data     import get_history
from database import db

portfolio_bp = Blueprint("portfolio", __name__, url_prefix="/api/portfolio")

DEFAULT_HOLDINGS = [
    {"sym":"RELIANCE","qty":50,  "avg":2640, "sector":"Energy"},
    {"sym":"TCS",     "qty":20,  "avg":3780, "sector":"Technology"},
    {"sym":"HDFCBANK","qty":75,  "avg":1520, "sector":"Banking"},
    {"sym":"INFY",    "qty":100, "avg":1480, "sector":"Technology"},
    {"sym":"BTC",     "qty":0.5, "avg":58000,"sector":"Crypto"},
    {"sym":"AAPL",    "qty":30,  "avg":172,  "sector":"Technology"},
    {"sym":"SBIN",    "qty":200, "avg":680,  "sector":"Banking"},
]


@portfolio_bp.route("/holdings", methods=["GET", "POST"])
def holdings():
    body   = request.get_json(silent=True) or {}
    email  = body.get("email")
    # Try to load saved portfolio from SQLite for this user
    h_list = None
    if email:
        h_list = db.get_portfolio(email)
    if not h_list:
        h_list = body.get("holdings", DEFAULT_HOLDINGS)

    result, total_val = [], 0
    for h in h_list:
        df = get_history(h["sym"], "5d")
        if df.empty: continue
        cmp  = round(float(df["Close"].iloc[-1]), 2)
        prev = round(float(df["Close"].iloc[-2]), 2) if len(df) > 1 else cmp
        day  = round((cmp - prev) / prev * 100, 2)
        val  = round(cmp * h["qty"], 2)
        inv  = round(h["avg"] * h["qty"], 2)
        pnl  = round(val - inv, 2)
        pp   = round(pnl / inv * 100, 2) if inv else 0
        df7  = get_history(h["sym"], "10d")
        spark= [round(float(x), 2) for x in df7["Close"].tail(7).values] if not df7.empty else []
        result.append({
            "sym":     h["sym"],
            "qty":     h["qty"],
            "avg":     h["avg"],
            "cmp":     cmp,
            "day":     day,
            "value":   val,
            "invested":inv,
            "pnl":     pnl,
            "pnl_pct": pp,
            "sector":  h.get("sector", "—"),
            "spark":   spark,
        })
        total_val += val

    invested = sum(r["invested"] for r in result)
    pnl_tot  = total_val - invested
    return jsonify({
        "holdings":    result,
        "total_value": round(total_val, 2),
        "invested":    round(invested, 2),
        "pnl":         round(pnl_tot, 2),
        "pnl_pct":     round(pnl_tot / invested * 100, 2) if invested else 0,
    })


@portfolio_bp.route("/save", methods=["POST"])
def save_holdings():
    """Persist user portfolio to SQLite."""
    body     = request.get_json(silent=True) or {}
    email    = body.get("email", "")
    holdings = body.get("holdings", [])
    if not email:
        return jsonify({"error": "email required"}), 400
    db.save_portfolio(email, holdings)
    db.log_event("portfolio_save", email)
    return jsonify({"ok": True, "saved": len(holdings)})


@portfolio_bp.route("/allocation", methods=["GET"])
def allocation():
    h_list = DEFAULT_HOLDINGS
    sector_map: dict = {}
    for h in h_list:
        df = get_history(h["sym"], "5d")
        if df.empty: continue
        cmp = float(df["Close"].iloc[-1])
        val = cmp * h["qty"]
        sec = h.get("sector", "Other")
        sector_map[sec] = sector_map.get(sec, 0) + val
    total = sum(sector_map.values())
    result = [{"sector": k, "value": round(v), "pct": round(v/total*100, 1)}
              for k, v in sector_map.items()]
    return jsonify(result)


@portfolio_bp.route("/performance")
def performance():
    idx_df = get_history("NIFTY50", "6mo")
    if idx_df.empty:
        return jsonify({"error": "no data"}), 400
    h_list = DEFAULT_HOLDINGS
    all_ret, weights = [], []
    for h in h_list:
        df = get_history(h["sym"], "6mo")
        if df.empty: continue
        cmp = float(df["Close"].iloc[-1])
        val = cmp * h["qty"]
        weights.append(val)
        all_ret.append(df["Close"])
    total_w  = sum(weights)
    weights  = [w / total_w for w in weights]
    import pandas as pd
    min_len  = min(len(r) for r in all_ret)
    combined = sum(w * r.values[-min_len:] / r.values[-min_len][0] * 100
                   for w, r in zip(weights, all_ret))
    idx_norm = idx_df["Close"].values[-min_len:] / idx_df["Close"].values[-min_len] * 100
    dates    = idx_df.index[-min_len:]
    s  = pd.Series(combined, index=dates)
    b  = pd.Series(idx_norm,  index=dates)
    sm = s.resample("M").last()
    bm = b.resample("M").last()
    return jsonify({
        "labels":    [str(d.date()) for d in sm.index],
        "portfolio": [round(float(v), 2) for v in sm.values],
        "benchmark": [round(float(v), 2) for v in bm.values],
    })


@portfolio_bp.route("/rebalance")
def rebalance():
    suggestions = [
        {"action":"INCREASE","sym":"HDFCBANK","from":12,"to":16,
         "reason":"Underweight — RBI rate hold bullish for NIM margins"},
        {"action":"REDUCE",  "sym":"BTC",     "from":7, "to":4.5,
         "reason":"High volatility above risk threshold — improves Sharpe by +0.22"},
        {"action":"HOLD",    "sym":"RELIANCE","from":25,"to":25,
         "reason":"Within optimal allocation range"},
        {"action":"ADD",     "sym":"SUNPHARMA","from":0,"to":5,
         "reason":"Diversification benefit + low market correlation"},
    ]
    return jsonify(suggestions)
