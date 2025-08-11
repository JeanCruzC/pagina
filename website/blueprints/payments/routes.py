import requests
from flask import Blueprint, jsonify, request, session, make_response, current_app

from ...config import (PAYPAL_BASE_URL, PAYPAL_PLAN_ID_PRO, PLANS)
from ...utils.email import send_admin_email
from ...app import (
    paypal_token,
    paypal_get_subscription,
    paypal_create_order,
    paypal_capture_order,
    add_subscription_record,
    add_to_allowlist,
)

payments_bp = Blueprint("payments", __name__)


@payments_bp.route("/subscription-activate", methods=["POST"])
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
                jsonify({"ok": False, "error": "HTTPError", "details": e.response.json()}),
                502,
            )
        except Exception:
            return jsonify({"ok": False, "error": "HTTPError"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@payments_bp.route("/plan/create", methods=["POST"])
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


@payments_bp.route("/create-order", methods=["POST"])
def paypal_create_order_endpoint():
    data = request.get_json(silent=True) or {}
    plan = data.get("plan") or session.get("pending_plan")
    if plan not in PLANS:
        return jsonify({"error": "invalid plan"}), 400
    try:
        order = paypal_create_order(PLANS[plan])
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    order_id = order.get("id")
    if not order_id:
        return jsonify({"error": "invalid response from PayPal"}), 400
    return jsonify({"orderID": order_id})


@payments_bp.route("/capture-order", methods=["POST"])
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
    add_to_allowlist(payer_email)
    send_admin_email("Nuevo pago", f"{payer_email} complet\u00f3 el pago {order_id}")
    return jsonify({"status": "ok"})


@payments_bp.route("/diagnose", methods=["GET"])
def paypal_diagnose():
    try:
        plan_id = request.args.get("plan", PAYPAL_PLAN_ID_PRO)
        token = paypal_token()
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"{PAYPAL_BASE_URL}/v1/billing/plans/{plan_id}", headers=headers)
        body = r.json() if r.content else {}
        return (
            jsonify(
                {
                    "ok": r.status_code == 200,
                    "plan_id": plan_id,
                    "status": body.get("status"),
                    "product_id": body.get("product_id"),
                    "raw": body,
                }
            ),
            (200 if r.status_code == 200 else 500),
        )
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


@payments_bp.route("/plan/check", methods=["GET"])
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
        current_app.logger.info("paypal plan check %r %s", plan_id, user_info)
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
            resp.headers["Content-Type"] = r.headers.get(
                "Content-Type", "application/json"
            )
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500
