from flask_wtf import CSRFProtect
from flask import current_app

csrf = CSRFProtect()

class SchedulerStore:
    def init_app(self, app):
        if not hasattr(app, "extensions"):
            app.extensions = {}
        if "scheduler" not in app.extensions:
            app.extensions["scheduler"] = {
                "jobs": {},
                "results": {},
                "active_jobs": {},
            }

    def _s(self, app=None):
        app = app or current_app
        if not hasattr(app, "extensions"):
            app.extensions = {}
        return app.extensions.setdefault(
            "scheduler", {"jobs": {}, "results": {}, "active_jobs": {}}
        )

    def mark_running(self, job_id, app=None):
        s = self._s(app)
        s["jobs"][job_id] = {"status": "running", "progress": {}}

    def update_progress(self, job_id, info, app=None):
        """Merge progress ``info`` into ``jobs[job_id]['progress']`` without
        altering the job status."""
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
        # CRITICAL: Update status to finished
        s["jobs"][job_id] = {"status": "finished"}
        s["results"][job_id] = {
            "result": result_dict,
            "excel_path": excel_path,
            "csv_path": csv_path,
        }

    def mark_error(self, job_id, msg, app=None):
        s = self._s(app)
        s["jobs"][job_id] = {"status": "error", "error": msg}

    def mark_cancelled(self, job_id, app=None):
        s = self._s(app)
        s["jobs"][job_id] = {"status": "cancelled"}

    def get_status(self, job_id, app=None):
        s = self._s(app)
        return s["jobs"].get(job_id, {"status": "unknown"})

    def get_payload(self, job_id, app=None):
        s = self._s(app)
        return s["results"].get(job_id)

    @property
    def active_jobs(self):
        return self._s().setdefault("active_jobs", {})

scheduler = SchedulerStore()
