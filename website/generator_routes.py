import os
import uuid
import tempfile
import threading
import json
from io import BytesIO
import numpy as np
import pandas as pd
import importlib

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    send_file,
    abort,
    current_app,
)

from .extensions import csrf
from .blueprints.core import login_required

from website.extensions import scheduler as store
# Motor de optimización (módulo)
scheduler = importlib.import_module("website.scheduler")
# Utility to stop running threads
try:
    from website.scheduler import _stop_thread
except ImportError:
    def _stop_thread(thread):
        return None

bp = Blueprint("generator", __name__)


def _cfg_from_request(form):
    # Usa tus claves actuales; dejo valores por defecto razonables
    return {
        "profile": form.get("profile", "JEAN"),
        "use_ft": form.get("use_ft", "true") == "true",
        "use_pt": form.get("use_pt", "true") == "true",
        "allow_8h": True,
        "allow_10h8": True,
        "allow_pt_4h": True,
        "allow_pt_5h": True,
        "allow_pt_6h": True,
        "break_from_start": float(form.get("break_from_start", 2)),
        "break_from_end": float(form.get("break_from_end", 2)),
        "solver_time": int(form.get("solver_time", current_app.config.get("TIME_SOLVER", 120))),
        "solver_msg": True,
        "coverage": float(form.get("coverage", 98)),
        "agent_limit_factor": int(form.get("agent_limit_factor", 30)),
        "excess_penalty": float(form.get("excess_penalty", 5)),
        "peak_bonus": float(form.get("peak_bonus", 2)),
        "critical_bonus": float(form.get("critical_bonus", 2.5)),
        "iterations": int(form.get("iterations", 6)),
        "use_pulp": True,
        "use_greedy": True,
        "export_files": False,
        "ft_first_pt_last": True,
        "optimization_profile": form.get("optimization_profile", "JEAN"),
    }


@bp.route("/generador_sync", methods=["POST"])
def generar_sync():
    xls = request.files.get("file") or request.files.get("excel")  # clave según tu form
    if not xls:
        return jsonify({"error": "Falta el archivo Excel"}), 400
    cfg = _cfg_from_request(request.form)
    # Ejecuta TODO en esta request y devuelve payload listo para la vista
    payload = scheduler.run_complete_optimization(
        xls, config=cfg, generate_charts=True, job_id=None, return_payload=True
    )
    return render_template("resultados.html", payload=payload, mode="sync")


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
    """Background worker que ejecuta optimización paralela."""
    print(f"[WORKER] Starting job {job_id}")
    with app.app_context():
        try:
            print(f"[WORKER] Running optimization for job {job_id}")

            result = None
            excel_bytes = None
            csv_bytes = None
            if hasattr(scheduler, "run_complete_optimization"):
                result, excel_bytes, csv_bytes = scheduler.run_complete_optimization(
                    BytesIO(file_bytes), config=config, generate_charts=generate_charts, job_id=job_id
                )
            else:
                from .parallel_optimizer import run_parallel_optimization
                result, excel_bytes, csv_bytes = run_parallel_optimization(
                    BytesIO(file_bytes), config=config, generate_charts=generate_charts, job_id=job_id
                )
            
            print(f"[WORKER] Optimization completed for job {job_id}")
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
            print(f"[WORKER] job={job_id} FINISHED -> calling mark_finished")
            scheduler.mark_finished(job_id, result, excel_path, csv_path, app=app)
            try:
                scheduler.active_jobs.pop(job_id, None)
            except Exception:
                pass
            print(f"[WORKER] mark_finished called for job {job_id}")
            
        except KeyboardInterrupt:
            print(f"[WORKER] Job {job_id} interrupted")
            scheduler.mark_error(job_id, "Interrupted by user", app=app)
            try:
                scheduler.active_jobs.pop(job_id, None)
            except Exception:
                pass
            return
        except Exception as e:
            print(f"[WORKER] ERROR in job {job_id}: {str(e)}")
            current_app.logger.exception(e)
            scheduler.mark_error(job_id, str(e), app=app)
            try:
                scheduler.active_jobs.pop(job_id, None)
            except Exception:
                pass


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
    
    # Manejar el perfil de optimización
    if "profile" in cfg:
        cfg["optimization_profile"] = cfg["profile"]

    jean_file = request.files.get("jean_file")
    if jean_file and jean_file.filename:
        try:
            jean_config = json.load(jean_file)
            cfg["custom_shifts_json"] = jean_config
            cfg["use_custom_shifts"] = True
            print(f"[GENERATOR] Cargado archivo JSON personalizado: {jean_file.filename}")
        except Exception as e:
            print(f"[GENERATOR] Error cargando JSON: {e}")
            pass

    generate_charts = request.form.get("generate_charts", "false").lower() in {
        "on",
        "true",
        "1",
    }
    job_id = request.form.get("job_id") or uuid.uuid4().hex
    excel_bytes = file.read()

    cfg.setdefault("solver_time", 10)
    cfg.setdefault("iterations", 5)
    cfg.setdefault("use_pulp", True)
    cfg.setdefault("use_greedy", True)
    cfg.setdefault("optimization_profile", "Equilibrado (Recomendado)")

    scheduler.mark_running(job_id)

    app_obj = current_app._get_current_object()
    thread = threading.Thread(
        target=_worker,
        args=(app_obj, job_id, excel_bytes, cfg, generate_charts),
        daemon=True,
    )
    try:
        scheduler.active_jobs[job_id] = thread
    except Exception:
        pass
    thread.start()

    return jsonify({"job_id": job_id}), 202


@bp.get("/generador/status/<job_id>")
@login_required
def generador_status(job_id):
    st = scheduler.get_status(job_id)
    status = st.get("status")
    current_app.logger.info(f"[STATUS] job={job_id} -> {status}")
    print(f"[STATUS] job={job_id} -> {status}")
    
    # Obtener información de progreso
    progress = st.get("progress", {})
    
    if status == "finished":
        print(f"[GENERATOR] Job {job_id} finished")
        return jsonify({"status": "finished", "redirect": f"/resultados/{job_id}"})
    if status == "error":
        print(f"[GENERATOR] Job {job_id} error: {st.get('error')}")
        return jsonify({"status": "error", "error": st.get("error")})
    
    response = {"status": status or "unknown"}
    if progress:
        response["progress"] = progress
    
    return jsonify(response)


@bp.get("/resultados/<job_id>")
@login_required
def resultados(job_id):
    """Render results page and load from disk fallback if needed."""
    # ALWAYS try disk first - this ensures we get the latest results
    try:
        import json, os, tempfile
        path = os.path.join(tempfile.gettempdir(), f"scheduler_result_{job_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
            
            # Debug: verificar que tenemos resultados de PuLP
            has_pulp = bool(result.get("pulp_results", {}).get("assignments"))
            has_greedy = bool(result.get("greedy_results", {}).get("assignments"))
            print(f"[RESULTADOS] Cargando desde disco - PuLP: {has_pulp}, Greedy: {has_greedy}")
            
            # Update scheduler store for future requests
            try:
                scheduler.mark_finished(job_id, result, None, None)
            except Exception:
                pass
            return render_template("resultados.html", resultado=result)
    except Exception as e:
        print(f"[RESULTADOS] Error loading from disk for {job_id}: {e}")
        import traceback
        print(f"[RESULTADOS] Traceback: {traceback.format_exc()}")
    
    # Fallback to scheduler store
    payload = scheduler.get_payload(job_id)
    if payload:
        result = payload["result"]
        has_pulp = bool(result.get("pulp_results", {}).get("assignments"))
        has_greedy = bool(result.get("greedy_results", {}).get("assignments"))
        print(f"[RESULTADOS] Cargando desde store - PuLP: {has_pulp}, Greedy: {has_greedy}")
        return render_template("resultados.html", resultado=result)

    # If job exists and is running, show placeholder (200).
    # Only return 500 if status is unknown/error without payload.
    try:
        st = scheduler.get_status(job_id)
        status = st.get("status")
        if status in {"running", "pending", "queued"}:
            print(f"[RESULTADOS] Job {job_id} running - placeholder")
            return render_template(
                "resultados.html",
                job_id=job_id,
                running=True,
                has_pulp=False,
                has_greedy=False,
                pulp_assignments=0,
                greedy_assignments=0,
                resultado=None,
            ), 200
        if status in {"error", "cancelled", "unknown"}:
            print(f"[RESULTADOS] No hay datos para {job_id} - 500")
            return render_template("500.html", message="Resultado no disponible"), 500
    except Exception:
        pass

    # Unknown without state/payload -> 500
    print(f"[RESULTADOS] No hay datos para {job_id} - 500")
    return render_template("500.html", message="Resultado no disponible"), 500


@bp.get("/download/<token>")
@login_required
def download(token):
    if os.path.basename(token) != token:
        abort(404)
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


@bp.get("/heatmap/<filename>")
@login_required
def serve_heatmap(filename):
    """Serve heatmap images from temporary files."""
    if not filename.endswith('.png'):
        abort(404)
    
    # Security check - only allow files that look like temp files
    if not (filename.startswith('tmp') or 'tmp' in filename):
        abort(404)
    
    path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(path):
        abort(404)
    
    return send_file(path, mimetype='image/png')


@bp.get("/resultados/<job_id>/refresh")
@login_required
def refresh_results(job_id):
    """Force refresh of results; return flags to drive auto-reload."""
    # Primero verificar archivo en disco
    import os, tempfile, json
    path = os.path.join(tempfile.gettempdir(), f"scheduler_result_{job_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                disk_result = json.load(f)

            pulp_data = disk_result.get("pulp_results", {})
            greedy_data = disk_result.get("greedy_results", {})
            has_pulp = bool(pulp_data.get("assignments"))
            has_greedy = bool(greedy_data.get("assignments"))

            print(f"[REFRESH] Disco - PuLP: {has_pulp}, Greedy: {has_greedy}")
            print(f"[REFRESH] PuLP assignments: {len(pulp_data.get('assignments', {}))}")
            print(f"[REFRESH] Greedy assignments: {len(greedy_data.get('assignments', {}))}")

            resp = {
                "has_pulp_results": has_pulp,
                "has_greedy_results": has_greedy,
                "has_greedy_charts": False,  # No se generan gráficos por defecto
                "pulp_status": pulp_data.get("status", "PENDING"),
                "greedy_status": greedy_data.get("status", "PENDING"),
                # Seguir refrescando mientras no haya resultados completos
                "should_refresh": not has_pulp,
            }
            if pulp_data.get("iteration") is not None:
                resp["jean_iter"] = pulp_data.get("iteration")
            if pulp_data.get("factor") is not None:
                resp["jean_factor"] = pulp_data.get("factor")
            # Incluir progreso en memoria si está disponible
            try:
                st = scheduler.get_status(job_id)
                progress = st.get("progress", {}) if isinstance(st, dict) else {}
            except Exception:
                progress = {}
            if isinstance(progress, dict):
                if progress.get("jean_status") is not None:
                    resp["jean_status"] = progress.get("jean_status")
                if progress.get("jean_time") is not None:
                    resp["jean_time"] = progress.get("jean_time")
            return jsonify(resp)
        except Exception as e:
            print(f"[REFRESH] Error leyendo disco: {e}")
            import traceback
            print(f"[REFRESH] Traceback: {traceback.format_exc()}")
    
    # Fallback al scheduler store
    payload = scheduler.get_payload(job_id)
    if not payload:
        print(f"[REFRESH] No payload para {job_id} - verificando disco nuevamente")
        # Verificar disco una vez más antes de dar up
        if os.path.exists(path):
            print(f"[REFRESH] Archivo existe pero no se pudo leer - forzando reload")
            return jsonify({
                "has_pulp_results": True,  # Forzar reload
                "has_greedy_results": True,
                "has_greedy_charts": False,
                "pulp_status": "READY",
                "greedy_status": "READY",
                "should_refresh": False,  # Forzar parada de refresh
            })

        # Consultar estado antes de indicar que debe seguir refrescando
        st = scheduler.get_status(job_id) or {}
        state = st.get("status")
        if state in {"error", "cancelled"} or state == "finished":
            return jsonify({
                "has_pulp_results": False,
                "has_greedy_results": False,
                "has_greedy_charts": False,
                "pulp_status": "PENDING",
                "greedy_status": "PENDING",
                "should_refresh": False,
                "error": state or "unknown",
            })

        return jsonify({
            "has_pulp_results": False,
            "has_greedy_results": False,
            "has_greedy_charts": False,
            "pulp_status": "PENDING",
            "greedy_status": "PENDING",
            "should_refresh": True,
        })
    
    result = payload["result"]
    has_pulp = bool(result.get("pulp_results", {}).get("assignments"))
    has_greedy = bool(result.get("greedy_results", {}).get("assignments"))

    print(f"[REFRESH] Store - PuLP: {has_pulp}, Greedy: {has_greedy}")
    # Incluir progreso si el store lo tiene
    progress = {}
    try:
        st = scheduler.get_status(job_id)
        progress = st.get("progress", {}) if isinstance(st, dict) else {}
    except Exception:
        progress = {}

    resp = {
        "has_pulp_results": has_pulp,
        "has_greedy_results": has_greedy,
        "has_greedy_charts": False,
        "pulp_status": result.get("pulp_results", {}).get("status", "PENDING"),
        "greedy_status": result.get("greedy_results", {}).get("status", "PENDING"),
        "should_refresh": False,  # Parar refresh cuando hay resultados
    }
    # Mapear campos de progreso esperados por el front
    if isinstance(progress, dict):
        if progress.get("pulp_progress"):
            resp["pulp_progress"] = progress.get("pulp_progress")
        if progress.get("pulp_time"):
            resp["pulp_time"] = progress.get("pulp_time")
        if progress.get("greedy_progress"):
            resp["greedy_progress"] = progress.get("greedy_progress")
        if progress.get("greedy_time"):
            resp["greedy_time"] = progress.get("greedy_time")
        if progress.get("jean_iter") is not None:
            resp["jean_iter"] = progress.get("jean_iter")
        if progress.get("jean_factor") is not None:
            resp["jean_factor"] = progress.get("jean_factor")
        if progress.get("jean_status") is not None:
            resp["jean_status"] = progress.get("jean_status")
        if progress.get("jean_time") is not None:
            resp["jean_time"] = progress.get("jean_time")
        # Campos adicionales para barra de progreso detallada
        if progress.get("stage"):
            resp["stage"] = progress.get("stage")
        if progress.get("patterns_count") is not None:
            resp["patterns_count"] = progress.get("patterns_count")
        if progress.get("total_agents") is not None:
            resp["total_agents"] = progress.get("total_agents")
        if progress.get("pulp_agents") is not None:
            resp["pulp_agents"] = progress.get("pulp_agents")
        if progress.get("greedy_agents") is not None:
            resp["greedy_agents"] = progress.get("greedy_agents")
        if progress.get("algorithm"):
            resp["algorithm"] = progress.get("algorithm")
        if progress.get("final_coverage") is not None:
            resp["final_coverage"] = progress.get("final_coverage")
    return jsonify(resp)


@bp.route("/cancel", methods=["POST"])
@login_required
@csrf.exempt
def cancel_job():
    data = request.get_json(silent=True) or {}
    if not data:
        try:
            data = json.loads(request.get_data(as_text=True))
        except Exception:
            data = {}
    job_id = data.get("job_id")
    if job_id:
        thread = None
        try:
            thread = scheduler.active_jobs.get(job_id)
        except Exception:
            pass
        if thread:
            _stop_thread(thread)
            try:
                scheduler.active_jobs.pop(job_id, None)
            except Exception:
                pass
        scheduler.mark_cancelled(job_id)
        # Obtener payload con compatibilidad de nombres
        payload = None
        if hasattr(scheduler, "get_payload"):
            payload = scheduler.get_payload(job_id)
        elif hasattr(scheduler, "get_result"):
            payload = scheduler.get_result(job_id)
        if payload:
            for key in ("excel_path", "csv_path"):
                path = payload.get(key)
                if path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    payload[key] = None
    return "", 204


# Debug route to inspect scheduler store
@bp.get("/__debug/scheduler")
def _dbg_scheduler():
    from flask import jsonify
    # Preferir el almacn del m3dulo scheduler si expone _store()
    try:
        if hasattr(scheduler, "_store"):
            s = scheduler._store()
            return jsonify({
                "jobs": s.get("jobs", {}),
                "results_keys": list(s.get("results", {}).keys()),
            })
    except Exception:
        pass
    return jsonify({"jobs": {}, "results_keys": []})
