from flask import Blueprint, redirect, url_for

bp = Blueprint("apps", __name__, url_prefix="/apps")


@bp.route("/")
def index():
    """Redirect to the default app or show a menu."""
    return redirect(url_for("apps.erlang"))


@bp.route("/erlang")
def erlang():
    """Placeholder for the Erlang app."""
    return "Erlang app coming soon"
