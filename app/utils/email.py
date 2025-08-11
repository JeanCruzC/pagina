import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_admin_email(subject: str, html_body: str) -> None:
    """Send an email to the configured admin address.

    If any of the required SMTP settings are missing the function simply
    returns without raising an exception so that calling code can proceed
    normally in development environments.
    """
    admin = os.getenv('ADMIN_EMAIL')
    host = os.getenv('SMTP_HOST')
    user = os.getenv('SMTP_USER')
    password = os.getenv('SMTP_PASS')
    port = int(os.getenv('SMTP_PORT', '587'))

    if not all([admin, host, user, password]):
        return

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = admin
    msg.attach(MIMEText(html_body, 'html'))

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(user, password)
        server.send_message(msg)
