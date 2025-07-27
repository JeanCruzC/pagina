from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify, make_response
from functools import wraps
from . import scheduler
import io
import json
import base64
import os
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
secret = os.getenv("SECRET_KEY")
if not secret:
    warnings.warn("SECRET_KEY environment variable not set, using insecure default")
    secret = "change-me"
app.secret_key = secret

users = {}


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        user = session.get('user')
        logger.debug(f"\U0001F50D [AUTH] User: {user}, Method: {request.method}")

        if not user:
            if 'application/json' in request.headers.get('Accept', ''):
                logger.info("\u274C [AUTH] No user, returning JSON error")
                return jsonify({'error': 'Unauthorized'}), 401
            logger.info("\u274C [AUTH] No user, redirecting to login")
            return redirect(url_for('login'))

        logger.debug("\u2705 [AUTH] User authorized, proceeding")
        return f(*args, **kwargs)
    return wrapped


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users[request.form['username']] = request.form['password']
        flash('Usuario creado. Ingrese ahora.')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']
        if users.get(user) == pw:
            session['user'] = user
            return redirect(url_for('generador'))
        flash('Credenciales invalidas')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/generador', methods=['GET', 'POST'])
@login_required
def generador():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    logger.debug(f"\U0001F50D [DEBUG] Request method: {request.method}")
    logger.debug(f"\U0001F50D [DEBUG] Content-Type: {request.content_type}")
    logger.debug(f"\U0001F50D [DEBUG] Files: {list(request.files.keys())}")
    logger.debug(f"\U0001F50D [DEBUG] Form: {list(request.form.keys())}")
    logger.debug(f"\U0001F50D [DEBUG] User session: {session.get('user', 'NO_USER')}")

    if request.method == 'POST':
        try:
            logger.debug("\u2705 [DEBUG] Entrando a lógica POST")
            logger.debug("\U0001F680 [DEBUG] Iniciando procesamiento POST")

            excel = request.files.get('excel')
            if not excel:
                logger.error("\u274C [ERROR] No se recibió archivo")
                return {'error': 'No file provided'}, 400

            logger.debug(f"\u2705 [DEBUG] Archivo recibido: {excel.filename}")

            logger.debug("🔄 [DEBUG] Construyendo configuración...")

            cfg = {
                'TIME_SOLVER': request.form.get('solver_time', type=int),
                'TARGET_COVERAGE': request.form.get('coverage', type=float),
                'use_ft': bool(request.form.get('use_ft')),
                'use_pt': bool(request.form.get('use_pt')),
                'allow_8h': bool(request.form.get('allow_8h')),
                'allow_10h8': bool(request.form.get('allow_10h8')),
                'allow_pt_4h': bool(request.form.get('allow_pt_4h')),
                'allow_pt_6h': bool(request.form.get('allow_pt_6h')),
                'allow_pt_5h': bool(request.form.get('allow_pt_5h')),
                'break_from_start': request.form.get('break_from_start', type=float),
                'break_from_end': request.form.get('break_from_end', type=float),
                'optimization_profile': request.form.get('profile'),
                'agent_limit_factor': request.form.get('agent_limit_factor', type=int),
                'excess_penalty': request.form.get('excess_penalty', type=float),
                'peak_bonus': request.form.get('peak_bonus', type=float),
                'critical_bonus': request.form.get('critical_bonus', type=float),
                'iterations': request.form.get('iterations', type=int),
            }

            logger.debug(f"✅ [DEBUG] Configuración creada: {cfg}")
            logger.debug("🚀 [DEBUG] Llamando scheduler.run_complete_optimization...")

            jean_template = request.files.get('jean_file')
            if jean_template and jean_template.filename:
                try:
                    cfg.update(json.load(jean_template))
                except Exception:
                    flash('Plantilla JEAN inválida')

            try:
                result = scheduler.run_complete_optimization(excel, config=cfg)
                logger.debug("✅ [DEBUG] Scheduler completado exitosamente")
                logger.debug(f"✅ [DEBUG] Tipo de resultado: {type(result)}")
                logger.debug(f"✅ [DEBUG] Keys en resultado: {list(result.keys()) if isinstance(result, dict) else 'No es dict'}")
            except Exception as e:
                logger.exception(f"❌ [ERROR] EXCEPCIÓN EN SCHEDULER: {str(e)}")
                return {"error": f"Error en optimización: {str(e)}"}, 500

            logger.debug("🎯 [DEBUG] Agregando download_url...")
            result["download_url"] = url_for("download_excel") if session.get("last_excel_result") else None

            logger.debug("📤 [DEBUG] Enviando respuesta al frontend...")
            logger.debug(f"📤 [DEBUG] Tamaño de respuesta: {len(str(result))} caracteres")

            return result

        except Exception as e:
            logger.exception(f"\u274C [ERROR] Exception en POST: {str(e)}")
            code = 400 if isinstance(e, ValueError) else 500
            return {"error": f'Server error: {str(e)}'}, code

    return render_template('generador.html')


@app.route('/download_excel')
@login_required
def download_excel():
    data_b64 = session.get('last_excel_result')
    if not data_b64:
        flash('No hay archivo para descargar.')
        return redirect(url_for('generador'))
    data = base64.b64decode(data_b64)
    return send_file(io.BytesIO(data), download_name='horario.xlsx', as_attachment=True)
