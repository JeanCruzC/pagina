import os
import json
import uuid
import pickle
from functools import wraps
from io import BytesIO
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

        from ..scheduler import run_optimization

        result, context = run_optimization(excel_file, config=config)

        job_id = uuid.uuid4().hex

        json_path = os.path.join("/tmp", f"{job_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        context_path = os.path.join("/tmp", f"{job_id}_ctx.pkl")
        with open(context_path, "wb") as f:
            pickle.dump(context, f)

        result["download_url"] = url_for("core.download_excel", job_id=job_id)
        result["csv_url"] = url_for("core.download_csv", job_id=job_id)

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

    json_path = os.path.join("/tmp", f"{job_id}.json")
    if not os.path.exists(json_path):
        return redirect(url_for("core.generador"))

    with open(json_path) as f:
        resultado = json.load(f)

    ctx_path = os.path.join("/tmp", f"{job_id}_ctx.pkl")
    if os.path.exists(ctx_path):
        resultado["heatmaps"] = {
            "demand": url_for("core.heatmap", job_id=job_id, map_type="demand"),
            "coverage": url_for("core.heatmap", job_id=job_id, map_type="coverage"),
            "difference": url_for("core.heatmap", job_id=job_id, map_type="difference"),
        }

    try:
        os.remove(json_path)
    except OSError:
        pass

    session.pop("job_id", None)

    return render_template("resultados.html", resultado=resultado)


@bp.route("/download/<job_id>")
@login_required
def download_excel(job_id):
    ctx_path = os.path.join("/tmp", f"{job_id}_ctx.pkl")
    if not os.path.exists(ctx_path):
        abort(404)
    with open(ctx_path, "rb") as f:
        context = pickle.load(f)
    from ..scheduler import export_excel

    excel_bytes, _, timing = export_excel(context)
    print(f"[TIMING] export_excel: {timing['export_excel']:.2f}s")
    bio = BytesIO(excel_bytes)
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name=f"{job_id}.xlsx")


@bp.route("/download/csv/<job_id>")
@login_required
def download_csv(job_id):
    ctx_path = os.path.join("/tmp", f"{job_id}_ctx.pkl")
    if not os.path.exists(ctx_path):
        abort(404)
    with open(ctx_path, "rb") as f:
        context = pickle.load(f)
    from ..scheduler import export_excel

    _, csv_bytes, timing = export_excel(context)
    print(f"[TIMING] export_excel: {timing['export_excel']:.2f}s")
    bio = BytesIO(csv_bytes)
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name=f"{job_id}.csv", mimetype="text/csv")


@bp.route("/heatmap/<job_id>/<map_type>")
@login_required
def heatmap(job_id, map_type):
    ctx_path = os.path.join("/tmp", f"{job_id}_ctx.pkl")
    if not os.path.exists(ctx_path):
        abort(404)
    with open(ctx_path, "rb") as f:
        context = pickle.load(f)
    from ..scheduler import generate_charts

    maps, timing = generate_charts(context)
    print(f"[TIMING] charts: {timing['charts']:.2f}s")
    fig = maps.get(map_type)
    if fig is None:
        abort(404)
    bio = BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight")
    bio.seek(0)
    return send_file(bio, mimetype="image/png")



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
