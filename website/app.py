from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file,
    jsonify,
    make_response,
    after_this_request,
)
from functools import wraps
from . import scheduler
import json, io
import os
import warnings
import tempfile
from datetime import datetime

app = Flask(__name__)
secret = os.getenv("SECRET_KEY")
if not secret:
    warnings.warn("SECRET_KEY environment variable not set, using insecure default")
    secret = "change-me"
app.secret_key = secret

users = {}

# Allowed email list obtained from environment variable "ALLOWED_EMAILS".
# If the variable is empty or undefined, all users are considered allowed.
allowed_emails = {
    e.strip().lower()
    for e in os.getenv("ALLOWED_EMAILS", "").split(",")
    if e.strip()
}


def is_allowed(email: str) -> bool:
    """Check if the given email is in the allowed list."""
    return not allowed_emails or email.lower() in allowed_emails


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        user = session.get('user')
        print(f"\U0001F50D [AUTH] User: {user}, Method: {request.method}")

        if not user:
            if 'application/json' in request.headers.get('Accept', ''):
                print("\u274C [AUTH] No user, returning JSON error")
                return jsonify({'error': 'Unauthorized'}), 401
            print("\u274C [AUTH] No user, redirecting to login")
            return redirect(url_for('login'))

        print("\u2705 [AUTH] User authorized, proceeding")
        return f(*args, **kwargs)
    return wrapped


def _on(name: str) -> bool:
    v = request.form.get(name)
    return v is not None and str(v).lower() in {'on', '1', 'true', 'yes'}


@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html', title='Schedules', year=datetime.now().year)


@app.route('/app', methods=['GET'])
def app_entry():
    if session.get('user'):
        return redirect(url_for('generador'))
    return redirect(url_for('login'))


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
        user_email = request.form['username']
        pw = request.form['password']
        if users.get(user_email) == pw:
            if not is_allowed(user_email):
                flash('Correo no autorizado')
                return render_template('login.html')
            session['user'] = user_email
            return redirect(url_for('generador'))
        flash('Credenciales invalidas')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))


@app.route('/generador', methods=['GET', 'POST'])
@login_required
def generador():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    print(f"\U0001F50D [DEBUG] Request method: {request.method}")
    print(f"\U0001F50D [DEBUG] Content-Type: {request.content_type}")
    print(f"\U0001F50D [DEBUG] Files: {list(request.files.keys())}")
    print(f"\U0001F50D [DEBUG] Form: {list(request.form.keys())}")
    print(f"\U0001F50D [DEBUG] User session: {session.get('user', 'NO_USER')}")

    if request.method == 'POST':
        try:
            print("\u2705 [DEBUG] Entrando a lógica POST")
            print("\U0001F680 [DEBUG] Iniciando procesamiento POST")

            excel = request.files.get('excel')
            if not excel:
                print("\u274C [ERROR] No se recibió archivo")
                return {'error': 'No file provided'}, 400

            print(f"\u2705 [DEBUG] Archivo recibido: {excel.filename}")

            print("🔄 [DEBUG] Construyendo configuración...")

            cfg = {
                'TIME_SOLVER': request.form.get('solver_time', type=int),
                'TARGET_COVERAGE': request.form.get('coverage', type=float),
                'use_ft': _on('use_ft'),
                'use_pt': _on('use_pt'),
                'allow_8h': _on('allow_8h'),
                'allow_10h8': _on('allow_10h8'),
                'allow_pt_4h': _on('allow_pt_4h'),
                'allow_pt_6h': _on('allow_pt_6h'),
                'allow_pt_5h': _on('allow_pt_5h'),
                'break_from_start': request.form.get('break_from_start', type=float),
                'break_from_end': request.form.get('break_from_end', type=float),
                'optimization_profile': request.form.get('profile'),
                'agent_limit_factor': request.form.get('agent_limit_factor', type=int),
                'excess_penalty': request.form.get('excess_penalty', type=float),
                'peak_bonus': request.form.get('peak_bonus', type=float),
                'critical_bonus': request.form.get('critical_bonus', type=float),
                'iterations': request.form.get('iterations', type=int),
            }

            print(f"✅ [DEBUG] Configuración creada: {cfg}")
            print("🚀 [DEBUG] Llamando scheduler.run_complete_optimization...")

            jean_template = request.files.get('jean_file')
            if jean_template and jean_template.filename:
                try:
                    with io.TextIOWrapper(jean_template, encoding='utf-8') as fh:
                        cfg.update(json.load(fh))
                except Exception:
                    flash('Plantilla JEAN inválida')

            try:
                result = scheduler.run_complete_optimization(excel, config=cfg)
                print(f"✅ [DEBUG] Scheduler completado exitosamente")
                print(f"✅ [DEBUG] Tipo de resultado: {type(result)}")
                print(f"✅ [DEBUG] Keys en resultado: {list(result.keys()) if isinstance(result, dict) else 'No es dict'}")
                print("📊 [DEBUG] Verificando librerías...")
                try:
                    import pulp
                    print(f"✅ PuLP disponible: {pulp.__version__}")
                except Exception:
                    print("❌ PuLP no disponible")
                try:
                    import numpy as np
                    print(f"✅ NumPy: {np.__version__}")
                except Exception:
                    print("❌ NumPy no disponible")
            except Exception as e:
                print(f"❌ [ERROR] EXCEPCIÓN EN SCHEDULER: {str(e)}")
                import traceback
                print("❌ [ERROR] STACK TRACE COMPLETO:")
                traceback.print_exc()
                return {"error": f"Error en optimización: {str(e)}"}, 500

            print("🎯 [DEBUG] Agregando download_url...")
            result["download_url"] = url_for("download_excel") if session.get("last_excel_file") else None

            # Persist result to a temporary file and store its path in session
            if session.get('last_result_file'):
                try:
                    os.remove(session['last_result_file'])
                except Exception:
                    pass
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8')
            json.dump(result, tmp)
            tmp.flush()
            tmp.close()
            session['last_result_file'] = tmp.name
            session['effective_config'] = cfg

            return redirect(url_for('resultados'))

        except Exception as e:
            print(f"\u274C [ERROR] Exception en POST: {str(e)}")
            import traceback
            traceback.print_exc()
            code = 400 if isinstance(e, ValueError) else 500
            return {"error": f'Server error: {str(e)}'}, code

    return render_template('generador.html')


@app.route('/download_excel')
@login_required
def download_excel():
    file_path = session.get('last_excel_file')
    if not file_path or not os.path.exists(file_path):
        flash('No hay archivo para descargar.')
        session.pop('last_excel_file', None)
        return redirect(url_for('generador'))

    @after_this_request
    def cleanup(response):
        try:
            os.remove(file_path)
        except Exception:
            pass
        session.pop('last_excel_file', None)
        return response

    return send_file(file_path, download_name='horario.xlsx', as_attachment=True)


@app.route('/resultados')
@login_required
def resultados():
    result_file = session.get('last_result_file')
    cfg = session.get('effective_config')
    excel_file = session.get('last_excel_file')
    if not result_file or not os.path.exists(result_file) or cfg is None:
        flash('No hay resultados disponibles. Genera un nuevo horario.')
        return redirect(url_for('generador'))
    with open(result_file) as f:
        result = json.load(f)
    result['download_url'] = url_for('download_excel') if excel_file else None
    result['effective_config'] = cfg
    return render_template('resultados.html', resultado=result)


@app.route('/configuracion')
@login_required
def configuracion():
    return render_template('configuracion.html')


@app.route('/perfil')
@login_required
def perfil():
    return redirect(url_for('configuracion'))
