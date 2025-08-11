"""Authentication blueprint and related helpers."""

from __future__ import annotations

import json
from pathlib import Path
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
from werkzeug.security import generate_password_hash, check_password_hash
import click


auth_bp = Blueprint("auth", __name__)


# Paths for storing allowlisted users
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ALLOWLIST_FILE = DATA_DIR / "allowlist.json"


def load_allowlist() -> dict:
    if ALLOWLIST_FILE.exists():
        try:
            with ALLOWLIST_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
    return {}


def add_to_allowlist(email: str, raw_password: str) -> dict:
    allowlist = load_allowlist()
    email = email.strip().lower()
    allowlist[email] = generate_password_hash(raw_password)
    ALLOWLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ALLOWLIST_FILE.open("w", encoding="utf-8") as f:
        json.dump(allowlist, f, indent=2)
    return allowlist


def verify_user(email: str, password: str) -> bool:
    allowlist = load_allowlist()
    email = email.strip().lower()
    hashed = allowlist.get(email)
    if not hashed:
        return False
    return check_password_hash(hashed, password)


@auth_bp.cli.command("allowlist-add")
@click.argument("email")
@click.argument("password")
def allowlist_add(email: str, password: str) -> None:
    """CLI helper to add a user to the allowlist."""

    add_to_allowlist(email, password)
    click.echo(f"Usuario {email} añadido al allowlist")


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        user = session.get("user")

        if not user:
            if "application/json" in request.headers.get("Accept", ""):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("auth.login"))

        from .payments import has_active_subscription

        if not has_active_subscription(user):
            if "application/json" in request.headers.get("Accept", ""):
                return jsonify({"error": "Payment required"}), 402
            flash("Suscripción activa requerida")
            return redirect(url_for("payments.subscribe"))

        return f(*args, **kwargs)

    return wrapped


@auth_bp.route("/register", endpoint="register")
def register():
    return redirect(url_for("auth.login"))


@auth_bp.route("/login", methods=["GET", "POST"], endpoint="login")
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form.get("password", "")
        if verify_user(email, password):
            session["user"] = email
            return redirect(url_for("routes.generador"))
        flash("Credenciales inválidas")
    return render_template("login.html")


@auth_bp.route("/logout", endpoint="logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("routes.landing"))


__all__ = [
    "auth_bp",
    "add_to_allowlist",
    "load_allowlist",
    "verify_user",
    "login_required",
    "ALLOWLIST_FILE",
]

