import os
import tempfile
import time
import shutil
from functools import wraps
from threading import Thread

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
    """Remove old entries from the scheduler results store.

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

    results = scheduler._store(app).setdefault("results", {})

    # Remove entries older than ``max_age``
    for job_id, data in list(results.items()):
        ts = data.get("timestamp", 0)
        if now - ts > max_age:
            results.pop(job_id, None)
            _cleanup_job_dir(job_id)

    # Trim if exceeding ``max_entries`` keeping the newest results
    if len(results) > max_entries:
        sorted_items = sorted(
            results.items(), key=lambda item: item[1].get("timestamp", 0)
        )
        for job_id, _ in sorted_items[:-max_entries]:
            results.pop(job_id, None)
            _cleanup_job_dir(job_id)


_cleanup_thread_started = False


@bp.before_app_request
def _start_cleanup_thread():  # pragma: no cover - background job
    """Start a background thread to periodically clean results."""

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


@bp.post("/")
@csrf.exempt
def landing_post():
    """Handle stray POST requests to the landing page.

    Some automated clients may POST to ``/`` which previously resulted in a
    405 error being logged. Redirect them back to the landing page instead.
    """
    return redirect(url_for("core.landing"))


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
            return redirect(url_for("generator.generador"))
        flash("Credenciales inválidas", "warning")
    return render_template("login.html")


@bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("core.login"))


@bp.route("/register")
def register():
    return redirect(url_for("core.login"))

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
