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
        file = request.files.get('excel')
        if not file:
            return {"error": "missing file"}, 400

        cfg = {
            "max_iter": int(request.form.get("iterations", 30)),
            "solver_time": int(request.form.get("solver_time", 240)),
            "target_coverage": float(request.form.get("coverage", 98)),
            "use_ft": bool(request.form.get("use_ft")),
            "use_pt": bool(request.form.get("use_pt")),
            "allow_8h": bool(request.form.get("allow_8h")),
            "allow_10h8": bool(request.form.get("allow_10h8")),
            "allow_pt_4h": bool(request.form.get("allow_pt_4h")),
            "allow_pt_6h": bool(request.form.get("allow_pt_6h")),
            "allow_pt_5h": bool(request.form.get("allow_pt_5h")),
            "break_from_start": float(request.form.get("break_from_start", 2.5)),
            "break_from_end": float(request.form.get("break_from_end", 2.5)),
            "profile": request.form.get("profile"),
        }
        if "agent_limit_factor" in request.form:
            cfg["agent_limit_factor"] = float(request.form.get("agent_limit_factor", 0))
        if "excess_penalty" in request.form:
            cfg["excess_penalty"] = float(request.form.get("excess_penalty", 0.5))
        if "peak_bonus" in request.form:
            cfg["peak_bonus"] = float(request.form.get("peak_bonus", 1.5))
        if "critical_bonus" in request.form:
            cfg["critical_bonus"] = float(request.form.get("critical_bonus", 2.0))

        jean_file = request.files.get("jean_file")
        if jean_file and jean_file.filename:
            try:
                cfg["jean_template"] = json.load(jean_file)
            except Exception:
                cfg["jean_template"] = None

        result = scheduler.run_complete_optimization(file, cfg)
        session['last_excel_result'] = result.get('excel_bytes')

        cov_b64 = base64.b64encode(result.get('coverage_image', b"")).decode('utf-8')
        dem_b64 = base64.b64encode(result.get('demand_image', b"")).decode('utf-8')

        return {
            "metrics": result.get("metrics"),
            "coverage_image": cov_b64,
            "demand_image": dem_b64,
            "excel_url": url_for('download_excel'),
        }

    return render_template('generador.html')


@app.route('/download_excel')
@login_required
def download_excel():
    data = session.get('last_excel_result')
    if not data:
        return redirect(url_for('generador'))
    return send_file(io.BytesIO(data), download_name='horario.xlsx', as_attachment=True)
