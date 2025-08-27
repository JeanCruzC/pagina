import os
import uuid
import tempfile
import threading
import json
from io import BytesIO
import numpy as np
import pandas as pd

from flask import (
    Blueprint,
    render_template,
    request,
    session,
    jsonify,
    send_file,
    abort,
    current_app,
)

from .extensions import csrf
from .blueprints.core import login_required

bp = Blueprint("generator", __name__)

# In-memory job store
# {job_id: {"status": "running"|"finished"|"error"|"cancelled", "result": {...}}}
JOBS = {}


def _to_jsonable(obj):
    """Recursively convert pandas and numpy objects to JSON-serializable types."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _worker(app, job_id, file_bytes, config, generate_charts):
    """Background worker that runs the optimization and stores results."""
    from website import scheduler

    with app.app_context():
        try:
            result, excel_bytes, csv_bytes = scheduler.run_complete_optimization(
                BytesIO(file_bytes),
                config=config,
                generate_charts=generate_charts,
                job_id=job_id,
            )
            excel_path = csv_path = None
            result = _to_jsonable(result)
            if excel_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                    tmp.write(excel_bytes)
                    excel_path = tmp.name
                token = os.path.basename(excel_path)
                result["download_url"] = f"/download/{token}"
            if csv_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(csv_bytes)
                    csv_path = tmp.name
                token = os.path.basename(csv_path)
                result["csv_url"] = f"/download/{token}?csv=1"
            JOBS[job_id] = {
                "status": "finished",
                "result": result,
                "excel_path": excel_path,
                "csv_path": csv_path,
            }
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

    cfg.setdefault("solver_time", 20)
    cfg.setdefault("iterations", 5)

    JOBS[job_id] = {"status": "running"}

    app_obj = current_app._get_current_object()
    threading.Thread(
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

    status = job.get("status")
    if status == "finished":
        session["resultado"] = job.get("result")
        print(f"\u2705 [GENERATOR] Job {job_id} finished")
        return jsonify({"status": "finished"})
    if status == "error":
        print(f"\u274C [GENERATOR] Job {job_id} error: {job.get('error')}")
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
    from website import scheduler

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
        job_info = JOBS.get(job_id)
        if job_info:
            job_info["status"] = "cancelled"
            for key in ("excel_path", "csv_path"):
                path = job_info.get(key) or job_info.get("result", {}).get(key)
                if path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    job_info[key] = None
    return "", 204
