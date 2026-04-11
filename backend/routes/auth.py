"""routes/auth.py — auth with full SQLite persistence via database.py"""
from flask import Blueprint, request, jsonify
import hashlib, time
from database import db

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/login", methods=["POST"])
def login():
    body  = request.get_json(silent=True) or {}
    email = body.get("email", "").strip().lower()
    pwd   = body.get("password", "")
    if not email or len(pwd) < 6:
        return jsonify({"error": "Invalid credentials"}), 400

    user = db.get_user(email)
    if user and user["pwd_hash"] != _hash(pwd):
        return jsonify({"error": "Incorrect password"}), 401

    # Auto-register on first login (demo convenience)
    if not user:
        name = email.split("@")[0].capitalize()
        db.create_user(email, name, _hash(pwd))
        db.log_event("auto_register", email)

    token = _make_token(email)
    name  = (user["name"] if user else email.split("@")[0].capitalize())
    db.log_event("login", email)
    return jsonify({"token": token, "name": name, "email": email})


@auth_bp.route("/signup", methods=["POST"])
def signup():
    body  = request.get_json(silent=True) or {}
    email = body.get("email", "").strip().lower()
    pwd   = body.get("password", "")
    name  = body.get("name", email.split("@")[0]).strip()
    if not email or len(pwd) < 6:
        return jsonify({"error": "Email and password (min 6 chars) required"}), 400

    ok = db.create_user(email, name, _hash(pwd))
    if not ok:
        return jsonify({"error": "Email already registered"}), 409

    token = _make_token(email)
    db.log_event("signup", email)
    return jsonify({"token": token, "name": name, "email": email})


@auth_bp.route("/users/count")
def user_count():
    """Admin: how many users are stored in SQLite."""
    s = db.stats()
    return jsonify({"users": s["users"], "predictions": s["predictions"]})


def _hash(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def _make_token(email):
    raw = f"{email}:{time.time()}:aifin"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
