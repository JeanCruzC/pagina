from flask import Blueprint, render_template
from flask_wtf.csrf import CSRFError

from .auth.routes import login_required

bp = Blueprint("core", __name__)


@bp.route("/generador")
@login_required
def generador():
    return render_template("generador.html")


@bp.route("/resultados")
@login_required
def resultados():
    return render_template("resultados.html")


@bp.route("/configuracion")
@login_required
def configuracion():
    return render_template("configuracion.html")


@bp.app_errorhandler(CSRFError)
def handle_csrf_error(error):
    return render_template("400.html", description=error.description), 400
