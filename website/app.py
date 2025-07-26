from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from functools import wraps
from . import scheduler
import io
import json
import base64

app = Flask(__name__)
app.secret_key = "change-me"

users = {}


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('user'):
            return redirect(url_for('login'))
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
    if request.method == 'POST':
        excel = request.files.get('excel')
        if not excel:
            return {'error': 'No file provided'}, 400

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

        jean_template = request.files.get('jean_file')
        if jean_template and jean_template.filename:
            try:
                cfg.update(json.load(jean_template))
            except Exception:
                flash('Plantilla JEAN inválida')

        result = scheduler.run_complete_optimization(excel, config=cfg)
        result["download_url"] = url_for("download_excel") if session.get("last_excel_result") else None
        return result

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
