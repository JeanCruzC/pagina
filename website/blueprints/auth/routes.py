from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, current_app
from functools import wraps

from ...app import verify_user, has_active_subscription

auth_bp = Blueprint('auth', __name__)


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        user = session.get('user')
        current_app.logger.debug("[AUTH] User: %s, Method: %s", user, request.method)

        if not user:
            if 'application/json' in request.headers.get('Accept', ''):
                current_app.logger.warning("[AUTH] No user, returning JSON error")
                return jsonify({'error': 'Unauthorized'}), 401
            current_app.logger.warning("[AUTH] No user, redirecting to login")
            return redirect(url_for('auth.login'))
        if not has_active_subscription(user):
            if 'application/json' in request.headers.get('Accept', ''):
                return jsonify({'error': 'Payment required'}), 402
            flash('Suscripción activa requerida')
            return redirect(url_for('subscribe'))

        current_app.logger.debug("[AUTH] User authorized, proceeding")
        return f(*args, **kwargs)

    return wrapped


@auth_bp.route('/register')
def register():
    return redirect(url_for('auth.login'))


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form.get('password', '')
        if verify_user(email, password):
            session['user'] = email
            return redirect(url_for('generador'))
        flash('Credenciales inválidas')
    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))
