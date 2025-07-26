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
    image_url = None
    demand_url = None
    metrics = None
    download = False
    if request.method == 'POST':
        file = request.files.get('excel')
        if file:
            dm = scheduler.load_demand_excel(file)
            patterns = next(scheduler.generate_shifts_coverage_corrected())
            assigns = scheduler.solve_in_chunks_optimized(patterns, dm)
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
            if excel:
                app.config['LAST_EXCEL'] = excel
                download = True
    return render_template('generador.html', image_url=image_url, demand_url=demand_url, metrics=metrics, download=download)


@app.route('/download')
@login_required
def download():
    data = app.config.get('LAST_EXCEL')
    if not data:
        return redirect(url_for('generador'))
    return send_file(io.BytesIO(data), download_name='horario.xlsx', as_attachment=True)
