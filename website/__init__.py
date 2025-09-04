from flask import Flask
from .generator_routes import bp as generator_bp
try:
    from flask_wtf.csrf import CSRFProtect
    csrf = CSRFProtect()
except Exception:  # pragma: no cover - optional dependency
    csrf = None


def create_app():
    app = Flask(__name__)
    # Registrar solo el blueprint síncrono
    app.register_blueprint(generator_bp)

    # Habilitar CSRF si está disponible y eximir el blueprint
    if csrf:
        csrf.init_app(app)
        csrf.exempt(generator_bp)

    return app
