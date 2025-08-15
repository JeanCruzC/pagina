import logging
import os
from flask import Flask
from dotenv import load_dotenv

from .extensions import csrf, scheduler
from .blueprints.core import bp as core_bp
from .blueprints.apps import bp as apps_bp


def create_app(config=None):
    """Application factory."""
    load_dotenv()

    app = Flask(__name__)

    if config is not None:
        app.config.from_object(config)

    # Secret keys
    secret = os.getenv("SECRET_KEY", "dev-secret")
    app.secret_key = secret
    app.config["WTF_CSRF_SECRET_KEY"] = os.getenv("CSRF_SECRET", secret)

    # Logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(level=numeric_level)
    app.logger.setLevel(numeric_level)

    # Extensions
    csrf.init_app(app)
    scheduler.init_app(app)

    # Blueprints
    app.register_blueprint(core_bp)
    app.register_blueprint(apps_bp)

    return app
