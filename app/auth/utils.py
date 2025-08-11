import json
from pathlib import Path
from typing import Dict

from werkzeug.security import generate_password_hash, check_password_hash

# Path to the allowlist file; can be overridden in tests
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
ALLOWLIST_FILE = DATA_DIR / 'allowlist.json'


def load_allowlist() -> Dict[str, str]:
    """Load the allowlist mapping emails to password hashes."""
    if ALLOWLIST_FILE.exists():
        try:
            with ALLOWLIST_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
    return {}


def add_to_allowlist(email: str, raw_password: str = '') -> Dict[str, str]:
    """Add a user to the allowlist, storing a hashed password."""
    allowlist = load_allowlist()
    email = email.strip().lower()
    allowlist[email] = generate_password_hash(raw_password)
    ALLOWLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ALLOWLIST_FILE.open('w', encoding='utf-8') as f:
        json.dump(allowlist, f, indent=2)
    return allowlist


def verify_user(email: str, password: str) -> bool:
    """Return ``True`` if the provided credentials are valid."""
    allowlist = load_allowlist()
    email = email.strip().lower()
    hashed = allowlist.get(email)
    if not hashed:
        return False
    return check_password_hash(hashed, password)
