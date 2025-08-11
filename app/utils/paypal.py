import os
import requests

PAYPAL_ENV = os.getenv('PAYPAL_ENV', 'sandbox')
PAYPAL_CLIENT_ID = os.getenv('PAYPAL_CLIENT_ID', '')
PAYPAL_SECRET = os.getenv('PAYPAL_SECRET', '')

PAYPAL_BASE_URL = (
    'https://api-m.paypal.com'
    if PAYPAL_ENV == 'live'
    else 'https://api-m.sandbox.paypal.com'
)


def paypal_token() -> str:
    auth = (PAYPAL_CLIENT_ID, PAYPAL_SECRET)
    resp = requests.post(
        f'{PAYPAL_BASE_URL}/v1/oauth2/token',
        auth=auth,
        data={'grant_type': 'client_credentials'},
    )
    resp.raise_for_status()
    return resp.json()['access_token']


def paypal_create_order(value: float, currency: str = 'USD') -> dict:
    token = paypal_token()
    headers = {'Authorization': f'Bearer {token}'}
    body = {
        'intent': 'CAPTURE',
        'purchase_units': [
            {'amount': {'currency_code': currency, 'value': f'{value:.2f}'}}
        ],
    }
    resp = requests.post(
        f'{PAYPAL_BASE_URL}/v2/checkout/orders',
        json=body,
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


def paypal_capture_order(order_id: str) -> dict:
    token = paypal_token()
    headers = {'Authorization': f'Bearer {token}'}
    resp = requests.post(
        f'{PAYPAL_BASE_URL}/v2/checkout/orders/{order_id}/capture',
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()
