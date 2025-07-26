from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from functools import wraps
from . import scheduler
import io

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
            return {'error': 'No file provided'}, 400

        cfg = {
            'TIME_SOLVER': int(request.form.get('solver_time', 240)),
            'TARGET_COVERAGE': float(request.form.get('coverage', 98)),
            'use_ft': bool(request.form.get('use_ft')),
            'use_pt': bool(request.form.get('use_pt')),
            'allow_8h': bool(request.form.get('allow_8h')),
            'allow_10h8': bool(request.form.get('allow_10h8')),
            'allow_pt_4h': bool(request.form.get('allow_pt_4h')),
            'allow_pt_6h': bool(request.form.get('allow_pt_6h')),
            'allow_pt_5h': bool(request.form.get('allow_pt_5h')),
            'break_from_start': float(request.form.get('break_from_start', 2.5)),
            'break_from_end': float(request.form.get('break_from_end', 2.5)),
            'optimization_profile': request.form.get('profile', 'Equilibrado (Recomendado)'),
            'agent_limit_factor': request.form.get('agent_limit_factor', type=int),
            'excess_penalty': request.form.get('excess_penalty', type=float),
            'peak_bonus': request.form.get('peak_bonus', type=float),
            'critical_bonus': request.form.get('critical_bonus', type=float),
        }

        dm = scheduler.load_demand_excel(file)
        patterns = next(scheduler.generate_shifts_coverage_corrected(cfg=cfg))
        assigns = scheduler.solve_in_chunks_optimized(patterns, dm, cfg=cfg)
        metrics = scheduler.analyze_results(assigns, patterns, dm)
        schedule = metrics['total_coverage'] if metrics else dm
        img_io = scheduler.heatmap(schedule, 'Cobertura')
        d_io = scheduler.heatmap(dm, 'Demanda')
        path = 'static/result.png'
        with open('website/' + path, 'wb') as f:
            f.write(img_io.read())
        path2 = 'static/demand.png'
        with open('website/' + path2, 'wb') as f:
            f.write(d_io.read())
        image_url = url_for('static', filename='result.png')
        demand_url = url_for('static', filename='demand.png')
        excel = scheduler.export_detailed_schedule(assigns, patterns)
        download_url = None
        if excel:
            app.config['LAST_EXCEL'] = excel
            download_url = url_for('download')
        return {
            'image_url': image_url,
            'demand_url': demand_url,
            'metrics': metrics,
            'download_url': download_url,
            'diff_matrix': metrics.get('diff_matrix').tolist() if metrics else None,
        }

    return render_template('generador.html')


@app.route('/download')
@login_required
def download():
    data = app.config.get('LAST_EXCEL')
    if not data:
        return redirect(url_for('generador'))
    return send_file(io.BytesIO(data), download_name='horario.xlsx', as_attachment=True)
