#!/usr/bin/env python3
"""Hash a password for manual insertion into data/allowlist.json.

This script accepts a single positional argument (the plaintext password),
uses ``werkzeug.security.generate_password_hash`` to hash it and prints the
result to stdout.
"""

from __future__ import annotations

import argparse
from werkzeug.security import generate_password_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash a password using Werkzeug")
    parser.add_argument("password", help="Plaintext password to hash")
    args = parser.parse_args()
    hashed = generate_password_hash(args.password)
    print(hashed)


if __name__ == "__main__":
    main()
