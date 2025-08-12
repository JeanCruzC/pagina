import os
import json
from functools import wraps
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
)
from flask_wtf.csrf import CSRFError

from ..utils.allowlist import verify_user

bp = Blueprint("core", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("core.login"))
        return view(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/")
def landing():
    return render_template("landing.html")
    # o: return redirect(url_for("core.login"))


@bp.route("/index")
def index():
    return render_template("index.html")

@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if verify_user(email, password):
            session["user"] = email
            return redirect(url_for("core.generador"))
        flash("Credenciales inválidas", "warning")
    return render_template("login.html")


@bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("core.login"))


@bp.route("/register")
def register():
    return redirect(url_for("core.login"))


@bp.route("/generador", methods=["GET", "POST"])
@login_required
def generador():
    if request.method == "POST":
        excel_file = request.files.get("excel")
        if not excel_file:
            flash("Se requiere un archivo Excel", "warning")
            return render_template("generador.html"), 400

        config = {}
        for key, value in request.form.items():
            if key == "csrf_token":
                continue
            low = value.lower()
            if low in {"on", "true", "1"}:
                config[key] = True
            elif low in {"off", "false", "0"}:
                config[key] = False
            else:
                try:
                    config[key] = int(value) if value.isdigit() else float(value)
                except ValueError:
                    config[key] = value

        jean_file = request.files.get("jean_file")
        if jean_file and jean_file.filename:
            try:
                config.update(json.load(jean_file))
            except Exception:
                pass

        from ..scheduler import run_complete_optimization

        result = run_complete_optimization(excel_file, config=config)

        if request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]:
            return jsonify(result)
        return render_template("resultados.html", resultado=result)

    return render_template("generador.html")


@bp.route("/resultados")
@login_required
def resultados():
    return render_template("resultados.html")


@bp.route("/configuracion")
@login_required
def configuracion():
    return render_template("configuracion.html")


@bp.route("/contacto")
def contacto():
    return render_template("contacto.html")


@bp.route("/subscribe")
def subscribe():
    plan = request.args.get("plan", "starter")
    plans = {
        "starter": {
            "paypal_plan_id": os.getenv("PAYPAL_PLAN_ID_STARTER"),
            "amount": float(os.getenv("STARTER_AMOUNT", "0")),
        },
        "pro": {
            "paypal_plan_id": os.getenv("PAYPAL_PLAN_ID_PRO"),
            "amount": float(os.getenv("PRO_AMOUNT", "0")),
        },
    }
    data = plans.get(plan, plans["starter"])
    return render_template(
        "subscribe.html",
        paypal_plan_id=data["paypal_plan_id"],
        paypal_client_id=os.getenv("PAYPAL_CLIENT_ID"),
        paypal_env=os.getenv("PAYPAL_ENV", "sandbox"),
        amount=data["amount"],
        tier=plan if plan in plans else "starter",
    )


@bp.route("/subscribe/success")
def subscribe_success():
    return render_template("subscribe_success.html")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@bp.app_errorhandler(CSRFError)
def handle_csrf_error(error):
    return render_template("400.html", description=error.description), 400
