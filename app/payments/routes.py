from flask import Blueprint, jsonify, request, session

from ..auth.utils import add_to_allowlist
from ..utils.email import send_admin_email
from ..utils.paypal import paypal_create_order, paypal_capture_order

payments_bp = Blueprint('payments', __name__, url_prefix='/api/paypal')

PLANS = {'starter': 30.0, 'pro': 50.0}


@payments_bp.route('/create-order', methods=['POST'])
def paypal_create_order_endpoint():
    data = request.get_json(silent=True) or {}
    plan = data.get('plan')
    amount = PLANS.get(plan)
    if amount is None:
        return jsonify({'error': 'invalid plan'}), 400
    order = paypal_create_order(amount)
    order_id = order.get('id')
    if not order_id:
        return jsonify({'error': 'invalid response'}), 400
    return jsonify({'orderID': order_id})


@payments_bp.route('/capture-order', methods=['POST'])
def paypal_capture_order_endpoint():
    data = request.get_json(silent=True) or {}
    order_id = data.get('orderID')
    email = (data.get('email') or session.get('pending_email', '')).strip().lower()
    if not order_id or not email:
        return jsonify({'error': 'missing data'}), 400
    capture = paypal_capture_order(order_id)
    status = capture.get('status')
    if status != 'COMPLETED':
        return jsonify({'error': 'payment not completed'}), 400
    add_to_allowlist(email)
    send_admin_email('Nuevo pago', f'{email} completó el pago {order_id}')
    return jsonify({'status': 'ok'})
