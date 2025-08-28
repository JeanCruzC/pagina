from flask import current_app
from flask_wtf import CSRFProtect

csrf = CSRFProtect()

class SchedulerStore:
    def init_app(self, app):
        if not hasattr(app, "extensions"):
            app.extensions = {}
        if "scheduler" not in app.extensions:
            app.extensions["scheduler"] = {
                "jobs": {},      # job_id -> {"status": "running|finished|error|cancelled", ...}
                "results": {},   # job_id -> {"result": {...}, "excel_path": "...", "csv_path": "..."}
                "active_jobs": {}     # job_id -> thread (opcional)
            }

    def _s(self, app=None):
        app = app or current_app
        if not hasattr(app, "extensions"):
            app.extensions = {}
        return app.extensions.setdefault(
            "scheduler", {"jobs": {}, "results": {}, "active_jobs": {}}
        )

    # --- API usada por tus rutas/worker ---
    def mark_running(self, job_id, app=None):
        self._s(app)["jobs"][job_id] = {"status": "running", "progress": {}}
    
    def update_progress(self, job_id, info: dict, app=None):
        """
        Mezcla 'info' dentro de jobs[job_id]['progress'] sin romper el status actual.
        Estructura esperada: info = {"fase": "...", "pct": 0..100, ...}
        """
        s = self._s(app)
        jobs = s.setdefault("jobs", {})
        job = jobs.setdefault(job_id, {"status": "running"})
        progress = job.get("progress", {})
        if not isinstance(progress, dict):
            progress = {}
        if not isinstance(info, dict):
            info = {"msg": str(info)}
        progress.update(info)
        job["progress"] = progress
        jobs[job_id] = job

    def mark_finished(self, job_id, result_dict, excel_path, csv_path, app=None):
        s = self._s(app)
        # ⚠️ CLAVE: actualizar SIEMPRE el estado
        s["jobs"][job_id] = {"status": "finished"}
        s["results"][job_id] = {
            "result": result_dict,
            "excel_path": excel_path,
            "csv_path": csv_path,
        }

    def mark_error(self, job_id, msg, app=None):
        self._s(app)["jobs"][job_id] = {"status": "error", "error": msg}

    def mark_cancelled(self, job_id, app=None):
        self._s(app)["jobs"][job_id] = {"status": "cancelled"}

    def get_status(self, job_id, app=None):
        return self._s(app)["jobs"].get(job_id, {"status": "unknown"})

    def get_payload(self, job_id, app=None):
        return self._s(app)["results"].get(job_id)

    # Alias si en algún punto llamas get_result(...)
    def get_result(self, job_id, app=None):
        return self.get_payload(job_id, app=app)

    @property
    def active_jobs(self):
        return self._s().setdefault("active_jobs", {})

scheduler = SchedulerStore()