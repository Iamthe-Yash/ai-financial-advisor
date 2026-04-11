"""
app.py — Render.com entry point
Serves BOTH the Flask backend API AND the frontend index.html
from a single process on a single port.
"""

import os
import sys
import time
import threading
import logging

# Make sure the backend package is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

from routes.market    import market_bp
from routes.predict   import predict_bp
from routes.portfolio import portfolio_bp
from routes.risk      import risk_bp
from routes.sentiment import sentiment_bp
from routes.backtest  import backtest_bp
from routes.screener  import screener_bp
from routes.chat      import chat_bp
from routes.auth      import auth_bp

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


def create_app():
    app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
    app.config["SECRET_KEY"] = os.getenv(
        "SECRET_KEY",
        "03726189c8afc95d8e35e9d5929e312d0fcce1dc93ec915b6b832882ad6967f0"
    )
    CORS(app, origins="*")

    for bp in [auth_bp, market_bp, predict_bp, portfolio_bp,
               risk_bp, sentiment_bp, backtest_bp, screener_bp, chat_bp]:
        app.register_blueprint(bp)

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "version": "1.0.0", "host": "render"})

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        full_path = os.path.join(FRONTEND_DIR, path)
        if os.path.isfile(full_path):
            return send_from_directory(FRONTEND_DIR, path)
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "endpoint not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "internal server error", "detail": str(e)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🚀  AI Financial Advisor starting on port {port}")
    print(f"    Frontend:  http://0.0.0.0:{port}/")
    print(f"    API:       http://0.0.0.0:{port}/api/health\n")
    app.run(host="0.0.0.0", port=port, debug=False)
