# -*- coding: utf-8 -*-
"""
Rutas del generador
"""

from flask import Blueprint, request, render_template, jsonify, redirect, url_for
from .scheduler import run_complete_optimization
import io

bp = Blueprint('generator', __name__)

@bp.route('/', methods=['GET'])
def root_index():
    """Enrutador raíz: redirige al generador.

    Algunos navegadores/extensiones hacen POSTs de sondeo a '/'.
    Definimos GET explícito para evitar 404 y redirigir al flujo principal.
    """
    return redirect(url_for('generator.generador'))

@bp.route('/', methods=['POST', 'OPTIONS', 'HEAD'])
def root_sink():
    """Sumidero de POST/OPTIONS/HEAD a '/'.

    Evita el spam de 404 cuando el cliente envía peticiones a la raíz.
    Devuelve 204 No Content.
    """
    return ('', 204)

@bp.route('/generador', methods=['GET', 'POST'])
def generador():
    if request.method == 'GET':
        return render_template('generador.html')
    
    try:
        # Obtener archivo
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Construir configuración
        cfg = {
            "optimization_profile": request.form.get('optimization_profile', 'Equilibrado (Recomendado)'),
            "hpo_trials": int(request.form.get('hpo_trials', 12)),
            "solver_time": int(request.form.get('solver_time', 180)),
            "TARGET_COVERAGE": float(request.form.get('coverage', 100.0)),
            "use_ft": 'use_ft' in request.form,
            "use_pt": 'use_pt' in request.form,
            "allow_8h": True,
            "allow_pt_4h": True,
            "allow_pt_6h": True,
        }
        
        # Ejecutar optimización
        file_stream = io.BytesIO(file.read())
        result = run_complete_optimization(
            file_stream, 
            config=cfg, 
            return_payload=True
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def register_routes(app):
    """Registra las rutas en la app"""
    app.register_blueprint(bp)
