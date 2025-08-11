import logging
import os

from flask import Flask
from flask_wtf import CSRFProtect

csrf = CSRFProtect()


def create_app() -> Flask:
    """Application factory for the project."""
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
    )
    app.secret_key = os.getenv('SECRET_KEY', 'dev-secret')
    app.config['WTF_CSRF_ENABLED'] = False

    # Configure logging
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(level=numeric_level)
    app.logger.setLevel(numeric_level)

    # Initialise extensions
    csrf.init_app(app)

    # Register blueprints
    from .auth.routes import auth_bp
    from .routes.generator import generator_bp
    from .payments.routes import payments_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(generator_bp)
    app.register_blueprint(payments_bp)

    return app


# Convenience: create a default application instance
app = create_app()

# Re-export commonly used helpers
from .auth.utils import add_to_allowlist
