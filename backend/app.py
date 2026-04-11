"""
AI Financial Advisor — Flask Backend (Render-ready)
"""

from flask import Flask, jsonify
from flask_cors import CORS
import logging
import os

from routes.market    import market_bp
from routes.predict   import predict_bp
from routes.portfolio import portfolio_bp
from routes.risk      import risk_bp
from routes.sentiment import sentiment_bp
from routes.backtest  import backtest_bp
from routes.screener  import screener_bp
from routes.chat      import chat_bp
from routes.auth      import auth_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "aifin-secret-change-in-prod")
    CORS(app, origins="*")

    for bp in [auth_bp, market_bp, predict_bp, portfolio_bp,
               risk_bp, sentiment_bp, backtest_bp, screener_bp, chat_bp]:
        app.register_blueprint(bp)

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "version": "1.0.0"})

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "endpoint not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "internal server error", "detail": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
