from flask import Flask


def create_app() -> Flask:
    from .app import app
    from .blueprints.payments.routes import payments_bp

    app.register_blueprint(payments_bp, url_prefix="/api/paypal")
    return app
