from flask import Blueprint, render_template

from ..auth.routes import login_required

generator_bp = Blueprint('generator', __name__)


@generator_bp.route('/generador', methods=['GET'])
@login_required
def generador():
    return render_template('generador.html')


@generator_bp.route('/resultados')
@login_required
def resultados():
    return render_template('resultados.html')


@generator_bp.route('/configuracion')
@login_required
def configuracion():
    return render_template('configuracion.html')
