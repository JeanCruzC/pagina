"""Payment and subscription related routes and helpers."""

from __future__ import annotations

import json
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)


payments_bp = Blueprint("payments", __name__)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
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
    from .auth import load_allowlist

    return email in load_allowlist()


def send_admin_email(subject: str, html_body: str) -> None:
    cfg = current_app.config
    if not (
        cfg.get("SMTP_HOST")
        and cfg.get("SMTP_USER")
        and cfg.get("SMTP_PASS")
        and cfg.get("ADMIN_EMAIL")
    ):
        print("[EMAIL] SMTP no configurado; se omite envío:", subject)
        return
    msg = MIMEMultipart()
    msg["From"] = cfg["SMTP_USER"]
    msg["To"] = cfg["ADMIN_EMAIL"]
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(cfg["SMTP_HOST"], cfg["SMTP_PORT"], context=context) as server:
        server.login(cfg["SMTP_USER"], cfg["SMTP_PASS"])
        server.sendmail(cfg["SMTP_USER"], cfg["ADMIN_EMAIL"], msg.as_string())


def paypal_token() -> str:
    cfg = current_app.config
    auth = (cfg["PAYPAL_CLIENT_ID"], cfg["PAYPAL_SECRET"])
    resp = requests.post(
        f"{cfg['PAYPAL_BASE_URL']}/v1/oauth2/token",
        data={"grant_type": "client_credentials"},
        auth=auth,
    )
    resp.raise_for_status()
    return resp.json().get("access_token", "")


def paypal_get_subscription(sub_id: str) -> dict:
    token = paypal_token()
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{current_app.config['PAYPAL_BASE_URL']}/v1/billing/subscriptions/{sub_id}",
        headers=headers,
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
        f"{current_app.config['PAYPAL_BASE_URL']}/v2/checkout/orders",
        headers=headers,
        json=payload,
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
        f"{current_app.config['PAYPAL_BASE_URL']}/v2/checkout/orders/{order_id}/capture",
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


@payments_bp.route("/contacto", methods=["GET", "POST"], endpoint="contacto")
def contacto():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        plan = request.form.get("plan")
        if email:
            session["pending_email"] = email
        if plan == "subscribe":
            return redirect(url_for("payments.subscribe"))
        if plan in current_app.config["PLANS"]:
            return redirect(url_for("payments.checkout", plan=plan))
    return render_template("contacto.html")


@payments_bp.get("/checkout/<plan>", endpoint="checkout")
def checkout(plan):
    amount = current_app.config["PLANS"].get(plan, 50.0)
    return render_template(
        "checkout.html",
        amount=amount,
        paypal_client_id=current_app.config["PAYPAL_CLIENT_ID"],
        paypal_env=current_app.config["PAYPAL_ENV"],
        title="Pago",
    )


@payments_bp.get("/subscribe", endpoint="subscribe")
def subscribe():
    tier = request.args.get("plan", "pro")
    if tier == "starter":
        plan_id = current_app.config["PAYPAL_PLAN_ID_STARTER"]
        amount = 30
        tier_name = "Starter"
    else:
        plan_id = current_app.config["PAYPAL_PLAN_ID_PRO"]
        amount = 50
        tier_name = "Pro"

    current_app.logger.info(
        "[PAYPAL] ENV=%s TIER=%s PLAN_ID=%r",
        current_app.config["PAYPAL_ENV"],
        tier_name,
        plan_id,
    )
    return render_template(
        "subscribe.html",
        paypal_client_id=current_app.config["PAYPAL_CLIENT_ID"],
        paypal_env=current_app.config["PAYPAL_ENV"],
        paypal_plan_id=plan_id,
        amount=amount,
        tier=tier_name,
        title="Suscripción",
    )


@payments_bp.get("/subscribe/success", endpoint="subscribe_success")
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
                from .auth import add_to_allowlist

                add_to_allowlist(email)
    except Exception:
        current_app.logger.exception("Error verifying PayPal subscription")
    return render_template(
        "subscribe_success.html",
        subscription_id=sub_id,
        status=status,
        title="Suscripción exitosa",
    )


@payments_bp.route("/api/paypal/subscription-activate", methods=["POST"])
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
            return (
                jsonify(
                    {"ok": False, "error": "HTTPError", "details": e.response.json()}
                ),
                502,
            )
        except Exception:
            return jsonify({"ok": False, "error": "HTTPError"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@payments_bp.route("/api/paypal/plan/create", methods=["POST"])
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
            f"{current_app.config['PAYPAL_BASE_URL']}/v1/catalogs/products",
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
            f"{current_app.config['PAYPAL_BASE_URL']}/v1/billing/plans",
            headers=headers,
            json=plan_payload,
        )
        r_plan.raise_for_status()
        plan_id = r_plan.json().get("id")
        current_app.logger.info("Created PayPal plan %s", plan_id)
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


@payments_bp.route("/api/paypal/create-order", methods=["POST"])
def paypal_create_order_endpoint():
    data = request.get_json(silent=True) or {}
    plan = data.get("plan") or session.get("pending_plan")
    if plan not in current_app.config["PLANS"]:
        return jsonify({"error": "invalid plan"}), 400
    try:
        order = paypal_create_order(current_app.config["PLANS"][plan])
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    order_id = order.get("id")
    if not order_id:
        return jsonify({"error": "invalid response from PayPal"}), 400
    return jsonify({"orderID": order_id})


@payments_bp.route("/api/paypal/capture-order", methods=["POST"])
def paypal_capture_order_endpoint():
    data = request.get_json(silent=True) or {}
    order_id = data.get("orderID")
    email = (data.get("email") or session.get("pending_email", "")).strip().lower()
    if not order_id or not email:
        return jsonify({"error": "missing data"}), 400
    try:
        capture = paypal_capture_order(order_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    status = capture.get("status")
    if status != "COMPLETED":
        try:
            status = capture["purchase_units"][0]["payments"]["captures"][0]["status"]
        except Exception:
            status = None
        if status != "COMPLETED":
            return jsonify({"error": "payment not completed"}), 400
    payer_email = capture.get("payer", {}).get("email_address", email)
    from .auth import add_to_allowlist

    add_to_allowlist(payer_email)
    send_admin_email("Nuevo pago", f"{payer_email} completó el pago {order_id}")
    return jsonify({"status": "ok"})


@payments_bp.route("/api/paypal/diagnose", methods=["GET"])
def paypal_diagnose():
    try:
        plan_id = request.args.get("plan", current_app.config["PAYPAL_PLAN_ID_PRO"])
        token = paypal_token()
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(
            f"{current_app.config['PAYPAL_BASE_URL']}/v1/billing/plans/{plan_id}",
            headers=headers,
        )
        body = r.json() if r.content else {}
        return jsonify(
            {
                "ok": r.status_code == 200,
                "plan_id": plan_id,
                "status": body.get("status"),
                "product_id": body.get("product_id"),
                "raw": body,
            }
        ), (200 if r.status_code == 200 else 500)
    except requests.HTTPError as e:
        try:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "HTTPError",
                        "status_code": e.response.status_code,
                        "body": e.response.json(),
                    }
                ),
                500,
            )
        except Exception:
            return jsonify({"ok": False, "error": "HTTPError"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@payments_bp.route("/api/paypal/plan/check", methods=["GET"])
def paypal_plan_check():
    try:
        plan_id = request.args.get("plan", current_app.config["PAYPAL_PLAN_ID_PRO"])
        token = paypal_token()
        headers = {"Authorization": f"Bearer {token}"}
        user_info = {}
        try:
            uresp = requests.get(
                f"{current_app.config['PAYPAL_BASE_URL']}/v1/identity/oauth2/userinfo?schema=paypalv1.1",
                headers=headers,
            )
            if uresp.content:
                user_info = uresp.json()
        except Exception as e:
            user_info = {"error": str(e)}
        current_app.logger.info("paypal plan check %r %s", plan_id, user_info)
        r = requests.get(
            f"{current_app.config['PAYPAL_BASE_URL']}/v1/billing/plans/{plan_id}",
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
            resp.headers["Content-Type"] = r.headers.get(
                "Content-Type", "application/json"
            )
        else:
            resp.headers["Content-Type"] = "application/json"
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


__all__ = [
    "payments_bp",
    "has_active_subscription",
    "add_subscription_record",
]

