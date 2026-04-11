"""routes/predict.py — AI price prediction endpoint with SQLite persistence"""
from flask import Blueprint, jsonify, request
from ml_models import predict_symbol
from database  import db

predict_bp = Blueprint("predict", __name__, url_prefix="/api/predict")


@predict_bp.route("/run", methods=["POST"])
def run_prediction():
    body = request.get_json(silent=True) or {}
    sym  = body.get("symbol", "RELIANCE").upper()
    days = int(body.get("days", 7))
    days = max(1, min(days, 30))

    result = predict_symbol(sym, days)
    if "error" in result:
        return jsonify(result), 400

    # ── persist to SQLite ──────────────────────────────────────────────────
    row_id = db.save_prediction(sym, days, result)
    result["db_id"] = row_id

    # ── record current price snapshot ──────────────────────────────────────
    db.record_price(sym, result.get("current_price", 0), result.get("change_pct"))

    return jsonify(result)


@predict_bp.route("/history/<sym>")
def prediction_history(sym):
    """Last 20 predictions for a symbol from SQLite."""
    sym  = sym.upper()
    rows = db.get_predictions(sym, limit=20)
    # Return lightweight summary (not full result blob)
    out = []
    for r in rows:
        res = r.get("result", {})
        out.append({
            "id":         r["id"],
            "symbol":     r["symbol"],
            "days":       r["days"],
            "signal":     r["signal"],
            "rmse":       r["rmse"],
            "mae":        r["mae"],
            "change_pct": res.get("change_pct"),
            "created_at": r["created_at"],
        })
    return jsonify(out)


@predict_bp.route("/stats/<sym>")
def prediction_stats(sym):
    """Aggregate prediction accuracy stats from SQLite history."""
    sym   = sym.upper()
    stats = db.get_prediction_stats(sym)
    return jsonify(stats)


@predict_bp.route("/status/<sym>")
def model_status(sym):
    """Check if a model is already cached (trained) for a symbol."""
    from ml_models import _model_cache
    sym     = sym.upper()
    trained = sym in _model_cache
    db_stats= db.get_prediction_stats(sym)
    return jsonify({"symbol": sym, "trained": trained, "db_history": db_stats})
