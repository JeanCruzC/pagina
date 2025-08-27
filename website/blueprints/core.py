import os
import json
import uuid
import tempfile
import time
import shutil
from functools import wraps
from threading import Thread
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
    after_this_request,
    Response,
    stream_with_context,
    current_app,
)
from flask_wtf.csrf import CSRFError

from ..utils.allowlist import verify_user
from ..extensions import csrf
import importlib

# Ensure we import the ``website.scheduler`` module, not the extension object
scheduler = importlib.import_module("website.scheduler")

core = Blueprint("core", __name__)
bp = core  # backward compatibility

# Default temporary directory
temp_dir = tempfile.gettempdir()

_RESULTS = {}  # {job_id: {"resultado": ..., "excel": ..., "csv": ..., "timestamp": ...}}

# In-memory job store for background optimization tasks (used by /cancel)
JOBS = {}


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _cleanup_job_dir(job_id):
    """Remove temporary files associated with ``job_id`` if they exist."""
    try:  # pragma: no cover - best effort
        shutil.rmtree(os.path.join(temp_dir, job_id))
    except OSError:
        pass


def cleanup_results(max_age=None, max_entries=None):
    """Remove old entries from ``_RESULTS``.

    Entries older than ``max_age`` seconds or exceeding ``max_entries`` are
    discarded. Default values are taken from application configuration keys
    ``RESULT_TTL`` and ``RESULT_MAX_ENTRIES``.
    """

    app = current_app._get_current_object()
    now = time.time()
    max_age = max_age if max_age is not None else app.config.get("RESULT_TTL", 3600)
    max_entries = (
        max_entries
        if max_entries is not None
        else app.config.get("RESULT_MAX_ENTRIES", 100)
    )

    # Remove entries older than ``max_age``
    for job_id, data in list(_RESULTS.items()):
        ts = data.get("timestamp", 0)
        if now - ts > max_age:
            _RESULTS.pop(job_id, None)
            _cleanup_job_dir(job_id)

    # Trim if exceeding ``max_entries`` keeping the newest results
    if len(_RESULTS) > max_entries:
        sorted_items = sorted(
            _RESULTS.items(), key=lambda item: item[1].get("timestamp", 0)
        )
        for job_id, _ in sorted_items[:-max_entries]:
            _RESULTS.pop(job_id, None)
            _cleanup_job_dir(job_id)


_cleanup_thread_started = False


@bp.before_app_request
def _start_cleanup_thread():  # pragma: no cover - background job
    """Start a background thread to periodically clean ``_RESULTS``."""

    global _cleanup_thread_started
    if _cleanup_thread_started:
        return
    app = current_app._get_current_object()
    interval = app.config.get("RESULT_CLEAN_INTERVAL", 600)

    def _worker():
        while True:
            time.sleep(interval)
            with app.app_context():
                cleanup_results()

    Thread(target=_worker, daemon=True).start()
    _cleanup_thread_started = True


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


@core.get("/generador")
@login_required
def generador():
    return render_template("generador.html")


@core.post("/generador")
@login_required
def generar():
    file = request.files.get("excel")
    if not file:
        return jsonify({"error": "Se requiere un archivo Excel"}), 400

    cfg = {}
    for key, value in request.form.items():
        if key in {"csrf_token", "generate_charts", "job_id"}:
            continue
        if value == "":
            continue
        low = value.lower()
        if low in {"on", "true", "1"}:
            cfg[key] = True
        elif low in {"off", "false", "0"}:
            cfg[key] = False
        else:
            try:
                cfg[key] = int(value) if value.isdigit() else float(value)
            except ValueError:
                cfg[key] = value

    jean_file = request.files.get("jean_file")
    if jean_file and jean_file.filename:
        try:
            cfg.update(json.load(jean_file))
        except Exception:
            pass

    generate_charts = (
        request.form.get("generate_charts", "false").lower() in {"on", "true", "1"}
    )
    job_id = request.form.get("job_id") or uuid.uuid4().hex
    excel_bytes = file.read()
    app = current_app._get_current_object()

    def _worker():
        with app.app_context():
            result, excel_out, csv_out = scheduler.run_complete_optimization(
                BytesIO(excel_bytes),
                config=cfg,
                generate_charts=generate_charts,
                job_id=job_id,
            )
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
            _RESULTS[job_id] = {
                "resultado": result,
                "excel": excel_out,
                "csv": csv_out,
                "timestamp": time.time(),
            }
            cleanup_results()

    Thread(target=_worker, daemon=True).start()
    session["last_job_id"] = job_id
    return jsonify({"job_id": job_id}), 202


@core.get("/generador/status/<job_id>")
@login_required
def generador_status(job_id):
    if job_id in _RESULTS:
        return jsonify({"status": "finished"})
    active = getattr(scheduler, "active_jobs", {})
    if active.get(job_id):
        return jsonify({"status": "running"})
    return jsonify({"status": "error"}), 404


@bp.route("/cancel", methods=["POST"])
@login_required
@csrf.exempt
def cancel_job():
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    if job_id:
        active = getattr(scheduler, "active_jobs", {})
        thread = active.get(job_id)
        stopper = getattr(scheduler, "_stop_thread", None)
        if thread and stopper:
            stopper(thread)
        active.pop(job_id, None)
        JOBS[job_id] = {"status": "cancelled"}
        if session.get("last_job_id") == job_id:
            session.pop("last_job_id", None)
    return "", 204


@core.get("/resultados")
@login_required
def resultados():
    job_id = session.get("last_job_id")
    data = _RESULTS.get(job_id)
    if not job_id or not data:
        return redirect(url_for("core.generador"))

    resultado = data.get("resultado", {}) or {}
    resultado["download_url"] = url_for("core.descargar_excel", job_id=job_id)
    resultado["csv_url"] = url_for("core.descargar_csv", job_id=job_id)
    heatmaps = resultado.get("heatmaps", {})
    for key, fname in list(heatmaps.items()):
        if fname:
            heatmaps[key] = url_for("core.heatmap", job_id=job_id, filename=fname)
        else:
            heatmaps[key] = None
    resultado["heatmaps"] = heatmaps
    return render_template("resultados.html", resultado=resultado)


@core.get("/descargar/excel/<job_id>")
@login_required
def descargar_excel(job_id):
    data = _RESULTS.get(job_id)
    if not data or not data.get("excel"):
        abort(404)
    return send_file(
        BytesIO(data["excel"]),
        as_attachment=True,
        download_name=f"{job_id}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@core.get("/descargar/csv/<job_id>")
@login_required
def descargar_csv(job_id):
    data = _RESULTS.get(job_id)
    if not data or not data.get("csv"):
        abort(404)
    return send_file(
        BytesIO(data["csv"]),
        as_attachment=True,
        download_name=f"{job_id}.csv",
        mimetype="text/csv",
    )


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
