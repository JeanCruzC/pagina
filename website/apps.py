from flask import Blueprint, render_template, request
import json
import importlib

bp = Blueprint("apps", __name__)


@bp.route("/erlang", methods=["GET", "POST"])
def erlang():
    metrics = {}
    figures = {}
    messages = []

    if request.method == "POST":
        data = request.get_json(silent=True)
        if not data:
            data = request.form.to_dict()
        submodule = data.pop("submodule", None)

        # Convert parameter values using JSON if possible
        params = {}
        for k, v in data.items():
            try:
                params[k] = json.loads(v)
            except Exception:
                params[k] = v

        try:
            erlang_core = importlib.import_module("erlang_core")
            func = getattr(erlang_core, submodule, None)
            if callable(func):
                result = func(**params)
                figs = {}
                if isinstance(result, tuple):
                    metrics = result[0] if len(result) > 0 else {}
                    figs = result[1] if len(result) > 1 else {}
                elif isinstance(result, dict):
                    metrics = result.get("metrics", {})
                    figs = result.get("figures", {})
                else:
                    metrics = result
                for name, fig in figs.items():
                    try:
                        figures[name] = fig.to_json()
                    except Exception:
                        pass
            else:
                messages.append("Submódulo inválido")
        except Exception as exc:
            messages.append(str(exc))

    return render_template(
        "erlang.html", metrics=metrics, figures=figures, messages=messages
    )
