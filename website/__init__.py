"""Flask application factory and blueprint registration."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask
from dotenv import load_dotenv


def _env_int(name: str, default: int) -> int:
    """Safely read an integer environment variable."""

    value = os.getenv(name)
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def create_app() -> Flask:
    """Application factory used by tests and the development server."""

    load_dotenv()

    app = Flask(__name__)

    # Secret key and logger configuration
    app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
    app.logger.setLevel("INFO")

    # PayPal and SMTP configuration
    app.config.update(
        PAYPAL_ENV=os.getenv("PAYPAL_ENV", "sandbox"),
        PAYPAL_CLIENT_ID=os.getenv("PAYPAL_CLIENT_ID", ""),
        PAYPAL_SECRET=os.getenv("PAYPAL_SECRET", ""),
        PAYPAL_PLAN_ID_STARTER=os.getenv("PAYPAL_PLAN_ID_STARTER", ""),
        PAYPAL_PLAN_ID_PRO=os.getenv("PAYPAL_PLAN_ID_PRO", ""),
        ADMIN_EMAIL=os.getenv("ADMIN_EMAIL", ""),
        SMTP_HOST=os.getenv("SMTP_HOST", ""),
        SMTP_PORT=_env_int("SMTP_PORT", 587),
        SMTP_USER=os.getenv("SMTP_USER", ""),
        SMTP_PASS=os.getenv("SMTP_PASS", ""),
    )

    app.config["PAYPAL_BASE_URL"] = (
        "https://api-m.paypal.com"
        if app.config["PAYPAL_ENV"] == "live"
        else "https://api-m.sandbox.paypal.com"
    )
    app.config["PLANS"] = {"starter": 30.0, "pro": 50.0}

    # Data directory used by several modules
    data_dir = Path(__file__).resolve().parents[1] / "data"
    app.config["DATA_DIR"] = data_dir

    app.logger.info(
        "PAYPAL_PLAN_ID_STARTER: %r", app.config["PAYPAL_PLAN_ID_STARTER"]
    )
    app.logger.info("PAYPAL_PLAN_ID_PRO: %r", app.config["PAYPAL_PLAN_ID_PRO"])

    # Register blueprints
    from .auth import auth_bp
    from .payments import payments_bp
    from .routes.generator import generator_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(payments_bp)
    app.register_blueprint(generator_bp)

    return app


__all__ = ["create_app"]

