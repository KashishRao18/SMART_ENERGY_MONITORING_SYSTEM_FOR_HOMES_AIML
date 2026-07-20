import io
import logging
import os
from flask import Flask, render_template, send_file, request, session
from flask_session import Session
from dotenv import load_dotenv
from app.config import Config


def create_app():
    # Load environment variables from .env if present (non-destructive to real env)
    load_dotenv(".env", override=False)
    logging.basicConfig(
        level=logging.DEBUG if Config.DEBUG else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    app = Flask(__name__)
    app.config.from_object(Config)
    app.jinja_env.auto_reload = bool(app.config.get("TEMPLATES_AUTO_RELOAD", False))

    if app.config.get("SESSION_TYPE") == "filesystem":
        os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

    # Initialize Flask-Session
    Session(app)

    # Initialize database
    from app.utils.database import init_db

    init_db()

    from app.routes.api import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    from app.routes.auth import auth_bp, login_required

    app.register_blueprint(auth_bp, url_prefix="/auth")

    @app.route("/", endpoint="index")
    @login_required
    def index_view():
        return render_template("index.html")

    @app.route("/dashboard", endpoint="dashboard")
    @login_required
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/analytics", endpoint="analytics")
    @login_required
    def analytics():
        return render_template("analytics.html")

    @app.route("/prediction", endpoint="prediction")
    @login_required
    def prediction():
        return render_template("prediction.html")

    @app.route("/anomaly", endpoint="anomaly")
    @login_required
    def anomaly():
        return render_template("anomaly.html")

    @app.route("/reports", endpoint="reports")
    @login_required
    def reports():
        return render_template("reports.html")

    @app.route("/my-data", endpoint="my_data")
    @login_required
    def my_data():
        return render_template("my_data.html")

    # Expose route functions to satisfy static analysis tools
    # These functions are accessed via Flask's url_for() at runtime
    _ = [index_view, dashboard, analytics, prediction, anomaly, reports]

    @app.route("/favicon.ico")
    def favicon():
        """Serve favicon - returns a simple 1x1 transparent PNG"""
        # Minimal 16x16 transparent PNG
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
            0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0xF3, 0xFF, 0x61, 0x00, 0x00, 0x00,
            0x01, 0x73, 0x52, 0x47, 0x42, 0x00, 0xAE, 0xCE, 0x1C, 0xE9, 0x00, 0x00,
            0x00, 0x04, 0x67, 0x41, 0x4D, 0x41, 0x00, 0x00, 0xB1, 0x8F, 0x0B, 0xFC,
            0x61, 0x05, 0x00, 0x00, 0x00, 0x09, 0x70, 0x48, 0x59, 0x73, 0x00, 0x00,
            0x0E, 0xC3, 0x00, 0x00, 0x0E, 0xC3, 0x01, 0xC7, 0x6F, 0xA8, 0x64, 0x00,
            0x00, 0x00, 0x19, 0x74, 0x45, 0x58, 0x74, 0x53, 0x6F, 0x66, 0x74, 0x77,
            0x61, 0x72, 0x65, 0x00, 0x70, 0x61, 0x69, 0x6E, 0x74, 0x2E, 0x6E, 0x65,
            0x74, 0x20, 0x34, 0x2E, 0x30, 0x2E, 0x32, 0x31, 0x78, 0xDA, 0x63, 0x64,
            0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0xF8, 0x0F, 0x00, 0x01, 0x04, 0x01,
            0x00, 0x63, 0x28, 0xBF, 0x02, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x03, 0x00,
            0x30, 0x00, 0x02, 0x00, 0x01, 0x8A, 0x77, 0x0D, 0x45, 0x00, 0x00, 0x00,
            0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        return send_file(io.BytesIO(png_data), mimetype="image/png")

    @app.after_request
    def apply_security_headers(response):
        """
        Prevent browser caching for authenticated pages/API responses so
        back-navigation after logout cannot reveal protected content.
        """
        is_static = (request.endpoint or "").startswith("static")
        if not is_static and session.get("user_id") is not None:
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
