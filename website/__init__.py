from flask import Flask
from .generator_routes import bp as generator_bp

def create_app():
    app = Flask(__name__)
    app.config["TIME_SOLVER"] = 240
    app.register_blueprint(generator_bp)
    return app
