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
    if request.method == 'POST':
        file = request.files['excel']
        if file:
            dm = scheduler.load_demand_excel(file)
            schedule = scheduler.generate_schedule(dm)
            img_io = scheduler.heatmap(schedule, 'Cobertura')
            path = 'static/result.png'
            with open('website/' + path, 'wb') as f:
                f.write(img_io.read())
            image_url = url_for('static', filename='result.png')
    return render_template('generador.html', image_url=image_url)
