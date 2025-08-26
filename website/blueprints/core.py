import os
import json
import uuid
import tempfile
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

# Default temporary directory
temp_dir = tempfile.gettempdir()


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

        generate_charts = (
            request.form.get("generate_charts", "false").lower() in {"on", "true", "1"}
        )

        from ..scheduler import run_complete_optimization

        result, excel_bytes, csv_bytes = run_complete_optimization(
            excel_file, config=config, generate_charts=generate_charts
        )

        job_id = uuid.uuid4().hex

        heatmaps = result.get("heatmaps", {})
        if heatmaps:
            heatmap_dir = os.path.join(temp_dir, job_id)
            os.makedirs(heatmap_dir, exist_ok=True)
            for key, path in list(heatmaps.items()):
                try:
                    new_name = f"{key}.png"
                    dest = os.path.join(heatmap_dir, new_name)
                    os.replace(path, dest)
                    heatmaps[key] = new_name
                except OSError:
                    heatmaps[key] = None
            result["heatmaps"] = heatmaps

        if excel_bytes:
            xlsx_path = os.path.join(temp_dir, f"{job_id}.xlsx")
            with open(xlsx_path, "wb") as f:
                f.write(excel_bytes)
            result["download_url"] = url_for("core.download_excel", job_id=job_id)

        if csv_bytes:
            csv_path = os.path.join(temp_dir, f"{job_id}.csv")
            with open(csv_path, "wb") as f:
                f.write(csv_bytes)
            result["csv_url"] = url_for("core.download_csv", job_id=job_id)

        json_path = os.path.join(temp_dir, f"{job_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        session["job_id"] = job_id

        if request.accept_mimetypes["application/json"] > request.accept_mimetypes["text/html"]:
            return jsonify({"job_id": job_id, **result})

        return redirect(url_for("core.resultados"))

    return render_template("generador.html")


@bp.route("/resultados")
@login_required
def resultados():
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("core.generador"))

    json_path = os.path.join(temp_dir, f"{job_id}.json")
    if not os.path.exists(json_path):
        return redirect(url_for("core.generador"))

    with open(json_path) as f:
        resultado = json.load(f)

    heatmaps = resultado.get("heatmaps", {})
    for key, fname in list(heatmaps.items()):
        if fname:
            heatmaps[key] = url_for("core.heatmap", job_id=job_id, filename=fname)
        else:
            heatmaps[key] = None

    try:
        os.remove(json_path)
    except OSError:
        pass

    session.pop("job_id", None)

    return render_template("resultados.html", resultado=resultado)


@bp.route("/download/<job_id>")
@login_required
def download_excel(job_id):
    path = os.path.join(temp_dir, f"{job_id}.xlsx")
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
    path = os.path.join(temp_dir, f"{job_id}.csv")
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
    path = os.path.join(temp_dir, job_id, filename)
    if not os.path.exists(path):
        abort(404)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(path)
            os.rmdir(os.path.join(temp_dir, job_id))
        except OSError:
            pass
        return response

    return send_file(path, mimetype="image/png")



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
