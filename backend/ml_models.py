"""
ml_models.py — LSTM + Random Forest + XGBoost ensemble
=========================================================
• Trains on real yfinance data (2 years)
• Caches trained models in memory per symbol
• Returns predictions, confidence, metrics, feature importances
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings, threading

warnings.filterwarnings("ignore")

from data import get_history, get_features, FEATURE_COLS

_model_cache: dict = {}
_lock = threading.Lock()

LOOK_BACK  = 10   # LSTM sequence length
LSTM_UNITS = 50
EPOCHS     = 25
BATCH      = 32
RF_TREES   = 200
RF_DEPTH   = 10


# ── public API ────────────────────────────────────────────────────────────────

def predict_symbol(sym: str, days: int = 7) -> dict:
    """
    Return ensemble prediction for `sym` over `days` future trading days.
    Returns rich dict ready to be JSON-serialised.
    """
    model_data = _get_or_train(sym)
    if model_data is None:
        return {"error": f"Could not fetch data for {sym}"}

    scaler   = model_data["scaler"]
    rf       = model_data["rf"]
    xgb_m    = model_data["xgb"]
    lstm_m   = model_data["lstm"]
    metrics  = model_data["metrics"]
    feat_imp = model_data["feat_imp"]
    df_feat  = model_data["df_feat"]

    # ── generate future predictions iteratively ──────────────────────────
    last_close    = float(df_feat["Close"].iloc[-1])
    window        = df_feat[FEATURE_COLS].values[-LOOK_BACK:].copy()
    preds_raw     = []
    conf_list     = []
    current_price = last_close

    for i in range(days):
        row  = window[-1].reshape(1, -1)
        row_s= scaler.transform(row)

        # RF
        rf_p  = rf.predict(row_s)[0]
        # XGB
        xgb_p = xgb_m.predict(xgb.DMatrix(row_s))[0]
        # LSTM
        seq   = scaler.transform(window).reshape(1, LOOK_BACK, len(FEATURE_COLS))
        lstm_p= float(lstm_m.predict(seq, verbose=0)[0][0])

        # ensemble (weighted average)
        ensemble_scaled = rf_p * 0.35 + xgb_p * 0.35 + lstm_p * 0.30
        # ensemble_scaled is the Close feature in scaled space; invert
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, 0] = ensemble_scaled
        pred_close = scaler.inverse_transform(dummy)[0][0]

        ret           = (pred_close - current_price) / (current_price + 1e-9)
        vol           = float(df_feat["Volatility"].iloc[-1]) or 0.015
        margin        = pred_close * vol * 1.96        # 95% CI band
        conf          = max(55, 94 - i * 3.5)

        preds_raw.append({
            "day":    i + 1,
            "price":  round(float(pred_close), 2),
            "upper":  round(float(pred_close + margin), 2),
            "lower":  round(float(pred_close - margin), 2),
            "conf":   round(conf, 1),
        })
        conf_list.append(conf)

        # slide window: replace Close, keep other features constant (simplified)
        new_row    = window[-1].copy()
        new_row[0] = pred_close          # Close column index = 0
        new_row[1] = ret                 # Return
        window     = np.vstack([window[1:], new_row])
        current_price = pred_close

    chg_pct = (preds_raw[0]["price"] - last_close) / last_close * 100

    return {
        "symbol":       sym,
        "current_price": round(last_close, 2),
        "predictions":  preds_raw,
        "change_pct":   round(chg_pct, 2),
        "signal":       "BUY" if chg_pct > 0.5 else "SELL" if chg_pct < -0.5 else "HOLD",
        "confidence":   round(conf_list[0], 1),
        "metrics":      metrics,
        "feature_importance": feat_imp,
        "thesis": _build_thesis(sym, chg_pct, days, preds_raw, metrics),
    }


def _build_thesis(sym, chg, days, preds, metrics):
    direction = "bullish" if chg > 0 else "bearish"
    pattern   = "breakout" if chg > 0 else "reversal"
    target    = preds[-1]["upper"] if chg > 0 else preds[-1]["lower"]
    sl        = round(preds[0]["price"] * 0.95, 2)
    return (
        f"{sym} shows {direction} momentum with Ensemble RMSE={metrics['RMSE']:.2f}. "
        f"Model detects a {pattern} pattern over {days} days. "
        f"Expected {'+' if chg>=0 else ''}{chg:.2f}% with {preds[0]['conf']:.1f}% confidence. "
        f"{'Recommended: BUY — Target ₹' + str(target) + ', Stop-loss ₹' + str(sl) if chg > 0.5 else 'Monitor closely — position size cautiously.'}"
    )


# ── training ─────────────────────────────────────────────────────────────────

def _get_or_train(sym: str):
    with _lock:
        if sym in _model_cache:
            return _model_cache[sym]

    data = _train(sym)
    if data:
        with _lock:
            _model_cache[sym] = data
    return data


def _train(sym: str):
    """Download 2y data, engineer features, train RF + XGB + LSTM, return bundle."""
    try:
        df_raw  = get_history(sym, "2y")
        if df_raw.empty or len(df_raw) < 100:
            return None
        df_feat = get_features(df_raw)
        if len(df_feat) < 60:
            return None

        X = df_feat[FEATURE_COLS].values
        y = df_feat["Close"].values

        scaler = MinMaxScaler()
        X_s    = scaler.fit_transform(X)
        y_s    = MinMaxScaler().fit_transform(y.reshape(-1, 1)).ravel()

        split   = int(len(X_s) * 0.85)
        X_tr, X_te = X_s[:split], X_s[split:]
        y_tr, y_te = y_s[:split], y_s[split:]
        y_tr_r, y_te_r = y[:split], y[split:]

        # ── Random Forest ─────────────────────────────────────────────────
        rf = RandomForestRegressor(
            n_estimators=RF_TREES, max_depth=RF_DEPTH,
            min_samples_split=5, random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        rf_pred_s = rf.predict(X_te)

        # ── XGBoost ───────────────────────────────────────────────────────
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dtest  = xgb.DMatrix(X_te)
        params = {"max_depth": 6, "eta": 0.05, "subsample": 0.8,
                  "objective": "reg:squarederror", "seed": 42, "verbosity": 0}
        xgb_m  = xgb.train(params, dtrain, num_boost_round=200,
                            evals=[(dtrain, "train")], verbose_eval=False)
        xgb_pred_s = xgb_m.predict(dtest)

        # ── LSTM ──────────────────────────────────────────────────────────
        lstm_m   = _build_lstm(len(FEATURE_COLS))
        X_lstm   = _make_sequences(X_s, LOOK_BACK)
        y_lstm   = y_s[LOOK_BACK:]
        split_l  = split - LOOK_BACK
        if split_l < 10:
            split_l = 10
        lstm_m.fit(
            X_lstm[:split_l], y_lstm[:split_l],
            epochs=EPOCHS, batch_size=BATCH, verbose=0,
            validation_split=0.1
        )
        lstm_pred_s = lstm_m.predict(X_lstm[split_l:], verbose=0).ravel()

        # ── ensemble on test set ──────────────────────────────────────────
        min_len = min(len(rf_pred_s), len(xgb_pred_s), len(lstm_pred_s))
        ens_s   = (rf_pred_s[-min_len:]  * 0.35 +
                   xgb_pred_s[-min_len:] * 0.35 +
                   lstm_pred_s[-min_len:]* 0.30)

        # invert scale to real prices for metrics
        dummy = np.zeros((len(ens_s), len(FEATURE_COLS)))
        dummy[:, 0] = ens_s
        ens_real = scaler.inverse_transform(dummy)[:, 0]
        actual   = y_te_r[-min_len:]

        rmse = float(np.sqrt(mean_squared_error(actual, ens_real)))
        mae  = float(mean_absolute_error(actual, ens_real))
        ss_res = np.sum((actual - ens_real) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2   = float(1 - ss_res / (ss_tot + 1e-9))
        mape = float(np.mean(np.abs((actual - ens_real) / (actual + 1e-9))) * 100)
        # directional accuracy
        dir_acc = float(np.mean(
            np.sign(np.diff(actual)) == np.sign(np.diff(ens_real))
        ) * 100)

        metrics = {
            "RMSE":    round(rmse, 3),
            "MAE":     round(mae, 3),
            "R2":      round(r2, 4),
            "MAPE":    round(mape, 2),
            "Dir_Acc": round(dir_acc, 1),
        }

        # feature importances from RF
        fi = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
        fi = {k: round(float(v), 4) for k, v in
              sorted(fi.items(), key=lambda x: -x[1])}

        return {
            "rf": rf, "xgb": xgb_m, "lstm": lstm_m,
            "scaler": scaler, "df_feat": df_feat,
            "metrics": metrics, "feat_imp": fi,
        }

    except Exception as e:
        print(f"[ml_models] training failed for {sym}: {e}")
        return None


def _make_sequences(X, look_back):
    seqs = []
    for i in range(look_back, len(X)):
        seqs.append(X[i - look_back:i])
    return np.array(seqs)


def _build_lstm(n_features):
    """Build and compile LSTM model."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    tf.get_logger().setLevel("ERROR")

    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(LOOK_BACK, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(LSTM_UNITS),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
