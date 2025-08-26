from flask_wtf import CSRFProtect


csrf = CSRFProtect()


class Scheduler:
    """Simple placeholder scheduler extension."""

    def init_app(self, app):  # pragma: no cover - placeholder
        pass


scheduler = Scheduler()
