import os
from flask import (
    request,
    session,
    flash,
    redirect,
    url_for,
    send_file,
    after_this_request,
)


def form_on(name: str) -> bool:
    v = request.form.get(name)
    return v is not None and str(v).lower() in {"on", "1", "true", "yes"}


def serve_excel(file_path: str):
    if not file_path or not os.path.exists(file_path):
        flash("No hay archivo para descargar.")
        session.pop("last_excel_file", None)
        return redirect(url_for("generator.generador"))

    @after_this_request
    def cleanup(response):
        try:
            os.remove(file_path)
        except Exception:
            pass
        session.pop("last_excel_file", None)
        return response

    return send_file(file_path, download_name="horario.xlsx", as_attachment=True)
