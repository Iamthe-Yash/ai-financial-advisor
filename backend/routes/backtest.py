"""routes/backtest.py — real historical strategy backtesting on yfinance data"""
from flask import Blueprint, jsonify, request
import numpy as np
import pandas as pd
from data import get_history

backtest_bp = Blueprint("backtest", __name__, url_prefix="/api/backtest")


@backtest_bp.route("/run", methods=["POST"])
def run_backtest():
    body     = request.get_json(silent=True) or {}
    sym      = body.get("symbol",   "RELIANCE").upper()
    strategy = body.get("strategy", "sma_crossover")
    years    = int(body.get("years", 5))

    period_map = {1:"1y", 2:"2y", 3:"5y", 5:"5y", 10:"10y"}
    period     = period_map.get(years, "5y")

    df = get_history(sym, period)
    if df.empty or len(df) < 60:
        return jsonify({"error": f"Insufficient data for {sym}"}), 400

    result = _run_strategy(df, strategy, sym)
    return jsonify(result)


@backtest_bp.route("/compare", methods=["POST", "GET"])
def compare_strategies():
    """Run all 7 strategies on a symbol and return comparison table."""
    body  = request.get_json(silent=True) or {}
    sym   = body.get("symbol", "RELIANCE").upper()
    df    = get_history(sym, "5y")
    if df.empty:
        return jsonify({"error": "no data"}), 400

    strategies = [
        "momentum_10d","sma_crossover","mean_reversion",
        "ema_crossover","rsi_strategy","macd_signal","buy_and_hold",
    ]
    labels = {
        "momentum_10d":  "Momentum (10d)",
        "sma_crossover": "SMA 20/50",
        "mean_reversion":"Mean Reversion",
        "ema_crossover": "EMA 10/20",
        "rsi_strategy":  "RSI Strategy",
        "macd_signal":   "MACD Signal",
        "buy_and_hold":  "Buy & Hold",
    }
    rows = []
    for s in strategies:
        r = _run_strategy(df, s, sym)
        rows.append({
            "name":   labels[s],
            "return": r["total_return"],
            "sharpe": r["sharpe"],
            "win_rate": r["win_rate"],
            "alpha":  r["alpha"],
            "trades": r["n_trades"],
        })
    rows.sort(key=lambda x: -x["return"])
    return jsonify(rows)


# ── strategy engine ────────────────────────────────────────────────────────────

def _run_strategy(df: pd.DataFrame, strategy: str, sym: str) -> dict:
    close = df["Close"].values.astype(float)
    n     = len(close)

    if strategy == "buy_and_hold":
        signals = np.ones(n)
    elif strategy == "sma_crossover":
        sma20 = _sma(close, 20)
        sma50 = _sma(close, 50)
        signals = np.where(sma20 > sma50, 1, -1)
    elif strategy == "ema_crossover":
        ema10 = _ema(close, 10)
        ema20 = _ema(close, 20)
        signals = np.where(ema10 > ema20, 1, -1)
    elif strategy == "momentum_10d":
        mom = np.full(n, 0.0)
        for i in range(10, n):
            mom[i] = close[i] / close[i-10] - 1
        signals = np.where(mom > 0, 1, -1)
    elif strategy == "mean_reversion":
        sma20  = _sma(close, 20)
        std20  = _rolling_std(close, 20)
        upper  = sma20 + 1.5 * std20
        lower  = sma20 - 1.5 * std20
        signals = np.where(close < lower, 1, np.where(close > upper, -1, 0))
    elif strategy == "rsi_strategy":
        rsi = _rsi(close, 14)
        signals = np.where(rsi < 35, 1, np.where(rsi > 65, -1, 0))
    elif strategy == "macd_signal":
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd  = ema12 - ema26
        sig   = _ema(macd, 9)
        signals = np.where(macd > sig, 1, -1)
    else:
        signals = np.ones(n)

    # ── simulate returns ───────────────────────────────────────────────────
    daily_ret = np.diff(close) / close[:-1]
    pos       = signals[:-1]
    strat_ret = pos * daily_ret

    # Buy & Hold
    bh_ret    = daily_ret

    # Cumulative value (start ₹100,000)
    init      = 100_000
    strat_val = init * np.cumprod(1 + strat_ret)
    bh_val    = init * np.cumprod(1 + bh_ret)

    total_return = round(float((strat_val[-1] / init - 1) * 100), 2)
    bh_return    = round(float((bh_val[-1]    / init - 1) * 100), 2)
    alpha        = round(total_return - bh_return, 2)

    # Sharpe
    rf_daily = 0.065 / 252
    excess   = strat_ret - rf_daily
    sharpe   = round(float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252)), 2)

    # Max drawdown
    cum  = np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-9)
    max_dd = round(float(dd.min()) * 100, 2)

    # Win rate
    trades  = np.diff(pos)
    n_trades= int(np.sum(trades != 0))
    wins    = int(np.sum((strat_ret > 0) & (pos > 0))) + int(np.sum((strat_ret < 0) & (pos < 0)))
    total_t = max(1, n_trades)
    win_rate= round(wins / len(strat_ret) * 100, 1)

    # Monthly labels & values
    dates = df.index[1:]
    monthly_idx = pd.DatetimeIndex(dates)
    df_strat = pd.Series(strat_val, index=monthly_idx)
    df_bh    = pd.Series(bh_val,    index=monthly_idx)
    monthly_s = df_strat.resample("M").last()
    monthly_b = df_bh.resample("M").last()

    labels     = [str(d.date()) for d in monthly_s.index]
    strat_vals = [round(float(v)) for v in monthly_s.values]
    bh_vals    = [round(float(v)) for v in monthly_b.values]

    # Drawdown series (monthly)
    df_dd = pd.Series(dd, index=monthly_idx).resample("M").min()
    dd_vals = [round(float(v) * 100, 2) for v in df_dd.values]

    return {
        "symbol":       sym,
        "strategy":     strategy,
        "total_return": total_return,
        "bh_return":    bh_return,
        "alpha":        alpha,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "win_rate":     win_rate,
        "n_trades":     total_t,
        "labels":       labels,
        "strat_vals":   strat_vals,
        "bh_vals":      bh_vals,
        "dd_vals":      dd_vals,
        "thesis": _bt_thesis(strategy, total_return, bh_return, alpha, sharpe, win_rate, total_t),
    }


def _bt_thesis(strategy, ret, bh, alpha, sharpe, wr, trades):
    q = "excellent" if sharpe > 1.5 else "solid" if sharpe > 1.0 else "moderate"
    a = "outperforms" if alpha > 0 else "underperforms"
    return (
        f"{strategy.replace('_',' ').title()} delivered +{ret}% over the period, "
        f"generating {'+' if alpha>=0 else ''}{alpha}% alpha vs buy-and-hold. "
        f"Sharpe ratio {sharpe} indicates {q} risk-adjusted performance. "
        f"Win rate {wr}% across {trades} trades. "
        f"Strategy {a} — {'recommend continuing with 2-4% position sizing.' if alpha>0 else 'consider switching to Momentum which historically shows better alpha.'}"
    )


# ── indicator helpers ──────────────────────────────────────────────────────────

def _sma(arr, n):
    out = np.full_like(arr, np.nan)
    for i in range(n-1, len(arr)):
        out[i] = arr[i-n+1:i+1].mean()
    return out

def _ema(arr, span):
    s = pd.Series(arr)
    return s.ewm(span=span, adjust=False).mean().values

def _rsi(arr, n=14):
    delta = np.diff(arr)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(alpha=1/n, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(alpha=1/n, adjust=False).mean().values
    rs    = avg_g / (avg_l + 1e-9)
    rsi   = 100 - 100 / (1 + rs)
    return np.concatenate([[50], rsi])

def _rolling_std(arr, n):
    out = np.full_like(arr, 0.0)
    for i in range(n-1, len(arr)):
        out[i] = arr[i-n+1:i+1].std()
    return out
