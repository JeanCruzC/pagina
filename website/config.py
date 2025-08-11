import os

# PayPal configuration
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox")  # 'sandbox' | 'live'
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET", "")
PAYPAL_PLAN_ID_STARTER = os.getenv("PAYPAL_PLAN_ID_STARTER", "")
PAYPAL_PLAN_ID_PRO = os.getenv("PAYPAL_PLAN_ID_PRO", "")

PAYPAL_BASE_URL = (
    "https://api-m.paypal.com"
    if PAYPAL_ENV == "live"
    else "https://api-m.sandbox.paypal.com"
)

PLANS = {"starter": 30.0, "pro": 50.0}
