"""routes/risk.py — portfolio risk metrics: VaR, CVaR, Sharpe, Monte Carlo, stress tests"""
from flask import Blueprint, jsonify, request
import numpy as np
from data import get_history

risk_bp = Blueprint("risk", __name__, url_prefix="/api/risk")

# Default holdings mirroring the UI's HOLDINGS array
DEFAULT_HOLDINGS = [
    {"sym":"RELIANCE","qty":50, "avg":2640},
    {"sym":"TCS",     "qty":20, "avg":3780},
    {"sym":"HDFCBANK","qty":75, "avg":1520},
    {"sym":"INFY",    "qty":100,"avg":1480},
    {"sym":"BTC",     "qty":0.5,"avg":58000},
    {"sym":"AAPL",    "qty":30, "avg":172},
    {"sym":"SBIN",    "qty":200,"avg":680},
]


@risk_bp.route("/portfolio", methods=["POST", "GET"])
def portfolio_risk():
    holdings = (request.get_json(silent=True) or {}).get("holdings", DEFAULT_HOLDINGS)
    metrics  = _compute_risk(holdings)
    return jsonify(metrics)


@risk_bp.route("/montecarlo", methods=["POST", "GET"])
def monte_carlo():
    holdings   = (request.get_json(silent=True) or {}).get("holdings", DEFAULT_HOLDINGS)
    n_paths    = int(request.args.get("paths", 500))
    n_days     = int(request.args.get("days", 252))

    total_val, mu, sigma = _portfolio_params(holdings)
    if total_val == 0:
        return jsonify({"error": "no data"}), 400

    dt = 1 / 252
    paths = []
    rng   = np.random.default_rng(42)
    for p in [5, 25, 50, 75, 95]:
        v = total_val
        series = [round(v)]
        for _ in range(n_days):
            z  = rng.standard_normal()
            v *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            series.append(round(v))
        paths.append({"percentile": p, "data": series})

    labels = [f"D{(i+1)*10}" for i in range(n_days // 10)] + ["End"]
    sampled_paths = []
    for path in paths:
        sampled = [path["data"][i*10] for i in range(n_days // 10)] + [path["data"][-1]]
        sampled_paths.append({"percentile": path["percentile"], "data": sampled})

    return jsonify({"labels": labels, "paths": sampled_paths, "portfolio_value": round(total_val)})


@risk_bp.route("/stress")
def stress_tests():
    holdings = DEFAULT_HOLDINGS
    total_val, _, _ = _portfolio_params(holdings)
    scenarios = [
        {"name":"COVID-19 (2020)",  "drop":-0.38,"recovery":"5-9 mo", "severity":"EXTREME"},
        {"name":"2008 Crisis",      "drop":-0.52,"recovery":"18-24 mo","severity":"EXTREME"},
        {"name":"Mild Correction",  "drop":-0.10,"recovery":"2-3 mo", "severity":"MODERATE"},
        {"name":"Flash Crash",      "drop":-0.09,"recovery":"2-4 wks","severity":"LOW"},
    ]
    for s in scenarios:
        s["loss_inr"] = round(total_val * abs(s["drop"]))
        s["drop_pct"] = f"{s['drop']*100:.0f}%"
    return jsonify(scenarios)


@risk_bp.route("/breakdown")
def risk_breakdown():
    """Per-asset risk score + contribution to portfolio vol."""
    holdings = DEFAULT_HOLDINGS
    total_val, _, _ = _portfolio_params(holdings)
    result = []
    for h in holdings:
        df   = get_history(h["sym"], "6mo")
        if df.empty: continue
        ret  = df["Close"].pct_change().dropna()
        vol  = float(ret.std()) * np.sqrt(252)
        val  = h["qty"] * float(df["Close"].iloc[-1])
        wt   = val / total_val if total_val else 0
        score= min(100, int(vol * 50))
        result.append({
            "sym":   h["sym"],
            "risk":  score,
            "vol":   round(vol * 100, 2),
            "weight":round(wt * 100, 2),
        })
    result.sort(key=lambda x: -x["risk"])
    return jsonify(result)


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_risk(holdings):
    returns_list, weights, total_val = [], [], 0

    for h in holdings:
        df = get_history(h["sym"], "1y")
        if df.empty: continue
        cmp = float(df["Close"].iloc[-1])
        val = h["qty"] * cmp
        total_val += val
        ret = df["Close"].pct_change().dropna().values
        returns_list.append(ret)
        weights.append(val)

    if not returns_list or total_val == 0:
        return {"error": "insufficient data"}

    weights  = np.array(weights) / total_val
    min_len  = min(len(r) for r in returns_list)
    R        = np.column_stack([r[-min_len:] for r in returns_list])
    port_ret = R @ weights

    mu       = float(port_ret.mean())
    sigma    = float(port_ret.std())
    ann_ret  = mu * 252
    ann_vol  = sigma * np.sqrt(252)

    # VaR / CVaR
    var95 = float(np.percentile(port_ret, 5))
    var99 = float(np.percentile(port_ret, 1))
    cvar  = float(port_ret[port_ret <= var95].mean())

    var95_inr = round(var95 * total_val)
    var99_inr = round(var99 * total_val)
    cvar_inr  = round(cvar  * total_val)

    # Sharpe (risk-free 6.5% for India)
    rf      = 0.065 / 252
    sharpe  = round((mu - rf) / (sigma + 1e-9) * np.sqrt(252), 2)
    sortino = _sortino(port_ret, rf, 252)

    # Max drawdown
    cum   = np.cumprod(1 + port_ret)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / (peak + 1e-9)
    max_dd= round(float(dd.min()) * 100, 2)

    # Beta vs NIFTY50
    idx_df  = get_history("NIFTY50", "1y")
    beta    = 1.0
    alpha_a = 0.0
    if not idx_df.empty:
        idx_ret = idx_df["Close"].pct_change().dropna().values
        common  = min(len(port_ret), len(idx_ret))
        pr, ir  = port_ret[-common:], idx_ret[-common:]
        cov     = np.cov(pr, ir)
        beta    = round(cov[0,1] / (cov[1,1] + 1e-9), 2)
        alpha_a = round((ann_ret - rf * 252) - beta * (float(ir.mean()) * 252 - rf * 252), 4)

    return {
        "total_value":  round(total_val),
        "var95_pct":    round(var95 * 100, 2),
        "var99_pct":    round(var99 * 100, 2),
        "cvar_pct":     round(cvar  * 100, 2),
        "var95_inr":    var95_inr,
        "var99_inr":    var99_inr,
        "cvar_inr":     cvar_inr,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "beta":         beta,
        "alpha_pct":    round(alpha_a * 100, 2),
        "max_drawdown": max_dd,
        "ann_return":   round(ann_ret * 100, 2),
        "ann_vol":      round(ann_vol * 100, 2),
    }


def _portfolio_params(holdings):
    total_val, mu_sum, sig_sum, n = 0, 0, 0, 0
    for h in holdings:
        df = get_history(h["sym"], "1y")
        if df.empty: continue
        cmp = float(df["Close"].iloc[-1])
        val = h["qty"] * cmp
        total_val += val
        ret   = df["Close"].pct_change().dropna()
        mu_sum += float(ret.mean()) * 252
        sig_sum+= float(ret.std())  * np.sqrt(252)
        n += 1
    mu    = mu_sum / n  if n else 0.08
    sigma = sig_sum/ n  if n else 0.18
    return total_val, mu, sigma


def _sortino(returns, rf_daily, ann_factor):
    excess   = returns - rf_daily
    downside = returns[returns < 0]
    down_dev = float(downside.std()) * np.sqrt(ann_factor) if len(downside) > 1 else 1e-9
    return round(float(excess.mean()) * ann_factor / down_dev, 2)
