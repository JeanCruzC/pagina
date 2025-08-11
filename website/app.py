import os
import logging
from dotenv import load_dotenv
load_dotenv()  # CARGAR .env SIEMPRE

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
import io
import json
import smtplib
import ssl
import tempfile
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
import click
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import CSRFProtect
from flask_wtf.csrf import CSRFError

from . import scheduler

# PayPal
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox")  # 'sandbox' | 'live'
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET", "")
PAYPAL_PLAN_ID_STARTER = os.getenv("PAYPAL_PLAN_ID_STARTER", "")
PAYPAL_PLAN_ID_PRO = os.getenv("PAYPAL_PLAN_ID_PRO", "")

# SMTP opcional (no debe romper si está vacío)
def _env_int(name, default):
    v = os.getenv(name)
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
SMTP_HOST   = os.getenv("SMTP_HOST", "")
SMTP_PORT   = _env_int("SMTP_PORT", 587)
SMTP_USER   = os.getenv("SMTP_USER", "")
SMTP_PASS   = os.getenv("SMTP_PASS", "")

SECRET_KEY  = os.getenv("SECRET_KEY", "dev-secret")

PAYPAL_BASE_URL = (
    "https://api-m.paypal.com"
    if PAYPAL_ENV == "live"
    else "https://api-m.sandbox.paypal.com"
)

PLANS = {"starter": 30.0, "pro": 50.0}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, LOG_LEVEL, logging.INFO)

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['WTF_CSRF_SECRET_KEY'] = os.environ.get('CSRF_SECRET', SECRET_KEY)
csrf = CSRFProtect(app)
logging.basicConfig(level=numeric_level)
app.logger.setLevel(numeric_level)
app.logger.info("PAYPAL_PLAN_ID_STARTER: %r", PAYPAL_PLAN_ID_STARTER)
app.logger.info("PAYPAL_PLAN_ID_PRO: %r", PAYPAL_PLAN_ID_PRO)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ALLOWLIST_FILE = DATA_DIR / "allowlist.json"

def load_allowlist() -> dict:
    if ALLOWLIST_FILE.exists():
        try:
            with ALLOWLIST_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
    return {}


def add_to_allowlist(email: str, raw_password: str) -> dict:
    allowlist = load_allowlist()
    email = email.strip().lower()
    allowlist[email] = generate_password_hash(raw_password)
    ALLOWLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ALLOWLIST_FILE.open("w", encoding="utf-8") as f:
        json.dump(allowlist, f, indent=2)
    return allowlist


def verify_user(email: str, password: str) -> bool:
    allowlist = load_allowlist()
    email = email.strip().lower()
    hashed = allowlist.get(email)
    if not hashed:
        return False
    return check_password_hash(hashed, password)


@app.cli.command('allowlist-add')
@click.argument('email')
@click.argument('password')
def allowlist_add(email: str, password: str) -> None:
    """CLI para añadir un usuario al allowlist."""
    add_to_allowlist(email, password)
    click.echo(f"Usuario {email} añadido al allowlist")


SUBSCRIPTIONS_FILE = DATA_DIR / "subscriptions.json"


def load_subscriptions() -> dict:
    if SUBSCRIPTIONS_FILE.exists():
        try:
            with SUBSCRIPTIONS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_subscriptions(subs: dict) -> None:
    SUBSCRIPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SUBSCRIPTIONS_FILE.open("w", encoding="utf-8") as f:
        json.dump(subs, f, indent=2)


def add_subscription_record(email: str, subscription_id: str, status: str) -> None:
    subs = load_subscriptions()
    subs[email] = {
        "subscription_id": subscription_id,
        "status": status,
    }
    save_subscriptions(subs)


def has_active_subscription(email: str) -> bool:
    email = email.strip().lower()
    subs = load_subscriptions()
    info = subs.get(email)
    if info and info.get("status") == "ACTIVE":
        return True
    if info and info.get("subscription_id"):
        try:
            sub = paypal_get_subscription(info["subscription_id"])
            info["status"] = sub.get("status")
            subs[email] = info
            save_subscriptions(subs)
            return info.get("status") == "ACTIVE"
        except Exception:
            return False
    return email in load_allowlist()


def send_admin_email(subject: str, html_body: str) -> None:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and ADMIN_EMAIL):
        app.logger.warning("SMTP not configured; skipping admin email: %s", subject)
        return
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = ADMIN_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, ADMIN_EMAIL, msg.as_string())


def paypal_token() -> str:
    auth = (PAYPAL_CLIENT_ID, PAYPAL_SECRET)
    resp = requests.post(
        f"{PAYPAL_BASE_URL}/v1/oauth2/token",
        data={"grant_type": "client_credentials"},
        auth=auth,
    )
    resp.raise_for_status()
    return resp.json().get("access_token", "")


def paypal_get_subscription(sub_id: str) -> dict:
    token = paypal_token()
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{PAYPAL_BASE_URL}/v1/billing/subscriptions/{sub_id}", headers=headers
    )
    resp.raise_for_status()
    return resp.json()


def paypal_create_order(value: float, currency: str = "USD") -> dict:
    token = paypal_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [
            {"amount": {"currency_code": currency, "value": str(value)}}
        ],
    }
    resp = requests.post(
        f"{PAYPAL_BASE_URL}/v2/checkout/orders", headers=headers, json=payload
    )
    resp.raise_for_status()
    return resp.json()


def paypal_capture_order(order_id: str) -> dict:
    token = paypal_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{PAYPAL_BASE_URL}/v2/checkout/orders/{order_id}/capture",
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        user = session.get('user')
        app.logger.debug("[AUTH] User: %s, Method: %s", user, request.method)

        if not user:
            if 'application/json' in request.headers.get('Accept', ''):
                app.logger.warning("[AUTH] No user, returning JSON error")
                return jsonify({'error': 'Unauthorized'}), 401
            app.logger.warning("[AUTH] No user, redirecting to login")
            return redirect(url_for('login'))
        if not has_active_subscription(user):
            if 'application/json' in request.headers.get('Accept', ''):
                return jsonify({'error': 'Payment required'}), 402
            flash('Suscripción activa requerida')
            return redirect(url_for('subscribe'))

        app.logger.debug("[AUTH] User authorized, proceeding")
        return f(*args, **kwargs)
    return wrapped


def _on(name: str) -> bool:
    v = request.form.get(name)
    return v is not None and str(v).lower() in {'on', '1', 'true', 'yes'}


@app.route('/')
def landing():
    return render_template(
        'landing.html',
        title='Schedules',
        year=datetime.now().year,
    )


@app.route('/app', methods=['GET'])
def app_entry():
    user = session.get('user')
    if user:
        if has_active_subscription(user):
            return redirect(url_for('generador'))
        flash('Suscripción activa requerida')
        return redirect(url_for('subscribe'))
    return redirect(url_for('login'))


@app.route('/register')
def register():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form.get('password', '')
        if verify_user(email, password):
            session['user'] = email
            return redirect(url_for('generador'))
        flash('Credenciales inválidas')
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
    app.logger.debug("Request method: %s", request.method)
    app.logger.debug("Content-Type: %s", request.content_type)
    app.logger.debug("Files: %s", list(request.files.keys()))
    app.logger.debug("Form: %s", list(request.form.keys()))
    app.logger.debug("User session: %s", session.get('user', 'NO_USER'))

    if request.method == 'POST':
        try:
            app.logger.debug("Entering POST logic")
            app.logger.debug("Starting POST processing")

            excel = request.files.get('excel')
            if not excel:
                app.logger.error("No file received")
                return {'error': 'No file provided'}, 400

            app.logger.debug("Received file: %s", excel.filename)

            app.logger.debug("Building configuration...")

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

            app.logger.debug("Configuration created: %s", cfg)
            app.logger.debug("Calling scheduler.run_complete_optimization...")

            jean_template = request.files.get('jean_file')
            if jean_template and jean_template.filename:
                try:
                    with io.TextIOWrapper(jean_template, encoding='utf-8') as fh:
                        cfg.update(json.load(fh))
                except Exception:
                    flash('Plantilla JEAN inválida')

            try:
                result = scheduler.run_complete_optimization(excel, config=cfg)
                app.logger.info("Scheduler completed successfully")
                app.logger.debug("Result type: %s", type(result))
                app.logger.debug(
                    "Result keys: %s",
                    list(result.keys()) if isinstance(result, dict) else 'No es dict',
                )
                app.logger.debug("Verifying libraries...")
                try:
                    import pulp
                    app.logger.debug("PuLP available: %s", pulp.__version__)
                except Exception:
                    app.logger.warning("PuLP not available")
                try:
                    import numpy as np
                    app.logger.debug("NumPy: %s", np.__version__)
                except Exception:
                    app.logger.warning("NumPy not available")
            except Exception as e:
                app.logger.exception("Scheduler exception: %s", e)
                return {"error": f"Error en optimización: {str(e)}"}, 500

            app.logger.debug("Adding download_url...")
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
            app.logger.exception("Exception in POST: %s", e)
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


@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        plan = request.form.get('plan')
        if email:
            session['pending_email'] = email
        if plan == 'subscribe':
            return redirect(url_for('subscribe'))
        if plan in PLANS:
            return redirect(url_for('checkout', plan=plan))
    return render_template('contacto.html')


@app.get("/checkout/<plan>")
def checkout(plan):
    amount = PLANS.get(plan, 50.0)
    return render_template(
        "checkout.html",
        amount=amount,
        paypal_client_id=PAYPAL_CLIENT_ID,
        paypal_env=PAYPAL_ENV,
        title="Pago"
    )


@app.get("/subscribe")
def subscribe():
    tier = request.args.get("plan", "pro")
    if tier == "starter":
        plan_id = PAYPAL_PLAN_ID_STARTER
        amount = 30
        tier_name = "Starter"
    else:
        plan_id = PAYPAL_PLAN_ID_PRO
        amount = 50
        tier_name = "Pro"

    app.logger.info(f"[PAYPAL] ENV={PAYPAL_ENV} TIER={tier_name} PLAN_ID={repr(plan_id)}")
    return render_template(
        "subscribe.html",
        paypal_client_id=PAYPAL_CLIENT_ID,
        paypal_env=PAYPAL_ENV,
        paypal_plan_id=plan_id,
        amount=amount,
        tier=tier_name,
        title="Suscripción",
    )


@app.get("/subscribe/success")
def subscribe_success():
    """Handle PayPal subscription success callback."""
    sub_id = request.args.get("sub", "")
    status = None
    email = session.get("user") or session.get("pending_email")
    try:
        if sub_id:
            sub = paypal_get_subscription(sub_id)
            status = sub.get("status")
            if email:
                add_subscription_record(email, sub_id, status or "")
                add_to_allowlist(email)
    except Exception:
        app.logger.exception("Error verifying PayPal subscription")
    return render_template(
        "subscribe_success.html",
        subscription_id=sub_id,
        status=status,
        title="Suscripción exitosa",
    )


@app.route("/api/paypal/subscription-activate", methods=["POST"])
def paypal_subscription_activate():
    data = request.get_json(silent=True) or {}
    sub_id = data.get("subscription_id")
    if not sub_id:
        return jsonify({"error": "subscription_id requerido"}), 400

    try:
        sub = paypal_get_subscription(sub_id)
        status = sub.get("status", "")
        email = session.get("user") or "anon"
        add_subscription_record(email, sub_id, status)
        ok_status = {"ACTIVE", "APPROVAL_PENDING", "APPROVED"}
        if status in ok_status:
            return jsonify({"ok": True, "status": status})
        return jsonify({"ok": False, "status": status}), 422
    except requests.HTTPError as e:
        try:
            return jsonify({"ok": False, "error": "HTTPError", "details": e.response.json()}), 502
        except Exception:
            return jsonify({"ok": False, "error": "HTTPError"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/paypal/plan/create", methods=["POST"])
def paypal_create_plan():
    """Create a PayPal product and subscription plan."""
    try:
        token = paypal_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        prod_payload = {"name": "Pro plan", "type": "SERVICE"}
        r_prod = requests.post(
            f"{PAYPAL_BASE_URL}/v1/catalogs/products",
            headers=headers,
            json=prod_payload,
        )
        r_prod.raise_for_status()
        product_id = r_prod.json().get("id")

        plan_payload = {
            "product_id": product_id,
            "name": "Pro plan",
            "billing_cycles": [
                {
                    "frequency": {"interval_unit": "MONTH", "interval_count": 1},
                    "tenure_type": "REGULAR",
                    "sequence": 1,
                    "total_cycles": 0,
                    "pricing_scheme": {
                        "fixed_price": {"value": "50", "currency_code": "USD"}
                    },
                }
            ],
            "payment_preferences": {"auto_bill_outstanding": True},
        }
        r_plan = requests.post(
            f"{PAYPAL_BASE_URL}/v1/billing/plans",
            headers=headers,
            json=plan_payload,
        )
        r_plan.raise_for_status()
        plan_id = r_plan.json().get("id")
        app.logger.info("Created PayPal plan %s", plan_id)
        return jsonify(
            {
                "plan_id": plan_id,
                "message": "Update PAYPAL_PLAN_ID_PRO with this value",
            }
        )
    except requests.HTTPError as e:
        try:
            return (
                jsonify(
                    {
                        "error": "HTTPError",
                        "status_code": e.response.status_code,
                        "details": e.response.json(),
                    }
                ),
                502,
            )
        except Exception:
            return jsonify({"error": "HTTPError"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(CSRFError)
def handle_csrf_error(error):
    app.logger.warning("CSRF error: %s", error.description)
    return render_template("400.html", description=error.description), 400


@app.errorhandler(404)
def not_found(error):
    app.logger.error("404 error: %s", error)
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.exception("500 error: %s", error)
    return render_template("500.html"), 500


@app.route('/api/paypal/create-order', methods=['POST'])
def paypal_create_order_endpoint():
    data = request.get_json(silent=True) or {}
    plan = data.get('plan') or session.get('pending_plan')
    if plan not in PLANS:
        return jsonify({'error': 'invalid plan'}), 400
    try:
        order = paypal_create_order(PLANS[plan])
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    order_id = order.get('id')
    if not order_id:
        return jsonify({'error': 'invalid response from PayPal'}), 400
    return jsonify({'orderID': order_id})


@app.route('/api/paypal/capture-order', methods=['POST'])
def paypal_capture_order_endpoint():
    data = request.get_json(silent=True) or {}
    order_id = data.get('orderID')
    email = (data.get('email') or session.get('pending_email', '')).strip().lower()
    if not order_id or not email:
        return jsonify({'error': 'missing data'}), 400
    try:
        capture = paypal_capture_order(order_id)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    status = capture.get('status')
    if status != 'COMPLETED':
        try:
            status = capture['purchase_units'][0]['payments']['captures'][0]['status']
        except Exception:
            status = None
        if status != 'COMPLETED':
            return jsonify({'error': 'payment not completed'}), 400
    payer_email = capture.get('payer', {}).get('email_address', email)
    add_to_allowlist(payer_email)
    send_admin_email('Nuevo pago', f'{payer_email} completó el pago {order_id}')
    return jsonify({'status': 'ok'})

@app.route("/api/paypal/diagnose", methods=["GET"])
def paypal_diagnose():
    try:
        plan_id = request.args.get("plan", PAYPAL_PLAN_ID_PRO)
        token = paypal_token()
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"{PAYPAL_BASE_URL}/v1/billing/plans/{plan_id}", headers=headers)
        body = r.json() if r.content else {}
        return jsonify({
            "ok": r.status_code == 200,
            "plan_id": plan_id,
            "status": body.get("status"),
            "product_id": body.get("product_id"),
            "raw": body
        }), (200 if r.status_code == 200 else 500)
    except requests.HTTPError as e:
        try:
            return jsonify({
                "ok": False,
                "error": "HTTPError",
                "status_code": e.response.status_code,
                "body": e.response.json()
            }), 500
        except Exception:
            return jsonify({"ok": False, "error": "HTTPError"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/paypal/plan/check", methods=["GET"])
def paypal_plan_check():
    try:
        plan_id = request.args.get("plan", PAYPAL_PLAN_ID_PRO)
        token = paypal_token()
        headers = {"Authorization": f"Bearer {token}"}
        user_info = {}
        try:
            uresp = requests.get(
                f"{PAYPAL_BASE_URL}/v1/identity/oauth2/userinfo?schema=paypalv1.1",
                headers=headers,
            )
            if uresp.content:
                user_info = uresp.json()
        except Exception as e:
            user_info = {"error": str(e)}
        app.logger.info("paypal plan check %r %s", plan_id, user_info)
        r = requests.get(
            f"{PAYPAL_BASE_URL}/v1/billing/plans/{plan_id}",
            headers=headers,
        )
        r.raise_for_status()
        resp = make_response(r.text, r.status_code)
        resp.headers["Content-Type"] = r.headers.get("Content-Type", "application/json")
        return resp
    except requests.HTTPError as e:
        r = e.response
        body = r.text if r is not None else ""
        status = r.status_code if r is not None else 500
        resp = make_response(body, status)
        if r is not None:
            resp.headers["Content-Type"] = r.headers.get("Content-Type", "application/json")
        else:
            resp.headers["Content-Type"] = "application/json"
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500
