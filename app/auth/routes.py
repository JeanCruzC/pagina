from functools import wraps
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)

from .utils import verify_user

auth_bp = Blueprint('auth', __name__)


def login_required(view):
    """Simple session-based authentication decorator."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('auth.login'))
        return view(*args, **kwargs)
    return wrapped


@auth_bp.route('/register')
def register():
    """Public route that redirects to the login page."""
    return redirect(url_for('auth.login'))


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Authenticate a user against the allowlist."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form.get('password', '')
        if verify_user(email, password):
            session['user'] = email
            return redirect(url_for('generator.generador'))
        flash('Credenciales inválidas')
    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('auth.login'))
