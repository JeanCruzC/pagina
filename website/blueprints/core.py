import os
import json
import uuid
from io import BytesIO
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
    send_file,
    abort,
    after_this_request,
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
            if key in {"csrf_token", "generate_charts"}:
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
        job_id = uuid.uuid4().hex
        json_path = os.path.join("/tmp", f"{job_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
        session["job_id"] = job_id
        if request.accept_mimetypes["application/json"] > request.accept_mimetypes["text/html"]:
            return jsonify({"job_id": job_id})
        return redirect(url_for("core.resultados"))

    return render_template("generador.html")


@bp.route("/resultados")
@login_required
def resultados():
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("core.generador"))

    json_path = os.path.join("/tmp", f"{job_id}.json")
    if not os.path.exists(json_path):
        return redirect(url_for("core.generador"))

    with open(json_path) as f:
        resultado = json.load(f)

    return render_template("resultados.html", resultado=resultado, job_id=job_id)


@bp.route("/download/<job_id>")
@login_required
def download_excel(job_id):
    path = os.path.join("/tmp", f"{job_id}.xlsx")
    if not os.path.exists(path):
        abort(404)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(path)
        except OSError:
            pass
        return response

    return send_file(path, as_attachment=True)


@bp.route("/download/csv/<job_id>")
@login_required
def download_csv(job_id):
    path = os.path.join("/tmp", f"{job_id}.csv")
    if not os.path.exists(path):
        abort(404)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(path)
        except OSError:
            pass
        return response

    return send_file(path, as_attachment=True)


@bp.route("/heatmap/<job_id>/<path:filename>")
@login_required
def heatmap(job_id, filename):
    path = os.path.join("/tmp", job_id, filename)
    if not os.path.exists(path):
        abort(404)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(path)
            os.rmdir(os.path.join("/tmp", job_id))
        except OSError:
            pass
        return response

    return send_file(path, mimetype="image/png")


@bp.route("/generate_excel/<job_id>")
@login_required
def generate_excel(job_id):
    json_path = os.path.join("/tmp", f"{job_id}.json")
    if not os.path.exists(json_path):
        abort(404)
    with open(json_path) as f:
        data = json.load(f)
    from ..scheduler import generate_excel as gen_excel
    assignments = data.get("assignments", {})
    patterns = data.get("patterns", {})
    excel_bytes, _ = gen_excel(assignments, patterns)
    if not excel_bytes:
        abort(404)
    return send_file(
        BytesIO(excel_bytes),
        as_attachment=True,
        download_name="resultado.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@bp.route("/generate_charts/<job_id>")
@login_required
def generate_charts(job_id):
    json_path = os.path.join("/tmp", f"{job_id}.json")
    if not os.path.exists(json_path):
        abort(404)
    with open(json_path) as f:
        data = json.load(f)
    from ..scheduler import generate_heatmaps
    demand_matrix = data.get("demand_matrix")
    metrics = data.get("metrics", {})
    zip_bytes = generate_heatmaps(demand_matrix, metrics)
    if not zip_bytes:
        abort(404)
    return send_file(
        BytesIO(zip_bytes),
        as_attachment=True,
        download_name="heatmaps.zip",
        mimetype="application/zip",
    )



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
