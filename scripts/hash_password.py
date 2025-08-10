#!/usr/bin/env python3
"""Generate a Werkzeug password hash for a given plaintext password."""

from argparse import ArgumentParser
from werkzeug.security import generate_password_hash


def main() -> None:
    parser = ArgumentParser(description="Hash a plaintext password")
    parser.add_argument("password", help="Plaintext password to hash")
    args = parser.parse_args()

    hashed = generate_password_hash(args.password)
    print(hashed)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
