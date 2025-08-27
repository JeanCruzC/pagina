import os
import uuid
import tempfile
from threading import Thread
from io import BytesIO
import importlib
import os

from flask import (
    Blueprint,
    render_template,
    request,
    session,
    jsonify,
    send_file,
    abort,
    current_app,
    url_for,
)

from .extensions import csrf
from .blueprints.core import login_required

# Ensure scheduler module is imported (not extension)
scheduler = importlib.import_module("website.scheduler")

bp = Blueprint("generator", __name__)

# In-memory job store
# {job_id: {"status": "running"|"finished"|"error"|"cancelled", "result": {...}}}
JOBS = {}



def _worker(app, job_id, excel_bytes, cfg, generate_charts):
    """Background worker that runs the optimization and stores results."""
    with app.app_context():
        try:
            result, excel_out, csv_out = scheduler.run_complete_optimization(
                BytesIO(excel_bytes),
                config=cfg,
                generate_charts=generate_charts,
                job_id=job_id,
            )
            # Save Excel output
            if excel_out:
                token = uuid.uuid4().hex + ".xlsx"
                path = os.path.join(tempfile.gettempdir(), token)
                with open(path, "wb") as tmp:
                    tmp.write(excel_out)
                result["download_url"] = url_for("generator.download", token=token)
            # Save CSV output
            if csv_out:
                token = uuid.uuid4().hex + ".csv"
                path = os.path.join(tempfile.gettempdir(), token)
                with open(path, "wb") as tmp:
                    tmp.write(csv_out)
                result["csv_url"] = url_for("generator.download", token=token, csv=1)
            JOBS[job_id] = {"status": "finished", "result": result}
        except Exception as e:
            JOBS[job_id] = {"status": "error", "error": str(e)}


@bp.get("/generador")
@login_required
def generador():
    return render_template("generador.html")


@bp.post("/generador")
@login_required
def generador_form():
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
            import json

            cfg.update(json.load(jean_file))
        except Exception:
            pass

    generate_charts = request.form.get("generate_charts", "false").lower() in {
        "on",
        "true",
        "1",
    }
    job_id = request.form.get("job_id") or uuid.uuid4().hex
    excel_bytes = file.read()

    JOBS[job_id] = {"status": "running"}

    app_obj = current_app._get_current_object()
    Thread(
        target=_worker,
        args=(app_obj, job_id, excel_bytes, cfg, generate_charts),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id}), 202


@bp.get("/generador/status/<job_id>")
@login_required
def generador_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "unknown"}), 200

    st = job.get("status")
    if st == "finished":
        session["resultado"] = job["result"]
        print(f"[STATUS] job={job_id} -> finished")
        return jsonify({"status": "finished"})
    if st == "error":
        print(f"[STATUS] job={job_id} -> error: {job.get('error')}")
        return jsonify({"status": "error", "error": job.get("error")})
    return jsonify({"status": "running"})


@bp.get("/resultados")
@login_required
def resultados():
    resultado = session.get("resultado")
    return render_template("resultados.html", resultado=resultado)


@bp.get("/download/<token>")
@login_required
def download(token):
    path = os.path.join(tempfile.gettempdir(), token)
    if not os.path.exists(path):
        abort(404)
    if request.args.get("csv") == "1":
        filename = "horarios.csv"
        mimetype = "text/csv"
    else:
        filename = "horarios.xlsx"
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return send_file(
        path,
        as_attachment=True,
        download_name=filename,
        mimetype=mimetype,
    )


@bp.route("/cancel", methods=["POST"])
@login_required
@csrf.exempt
def cancel_job():
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    if job_id:
        active = getattr(scheduler, "active_jobs", {}) or {}
        thread = active.get(job_id) if isinstance(active, dict) else None
        stopper = getattr(scheduler, "_stop_thread", None)
        if thread and stopper:
            stopper(thread)
        if isinstance(active, dict):
            active.pop(job_id, None)
        job_info = JOBS.pop(job_id, None)
        if job_info:
            for key in ("excel_path", "csv_path"):
                path = job_info.get(key) or job_info.get("result", {}).get(key)
                if path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
    return "", 204
