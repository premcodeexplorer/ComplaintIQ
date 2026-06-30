"""
ComplaintIQ — Create Bank Admin Account
-----------------------------------------
One-time CLI tool to provision a new bank admin account.
Requires SUPABASE_URL, SUPABASE_SERVICE_KEY, and DATABASE_URL in .env.

Usage:
    python scripts/create_admin.py --email admin@bank.com --name "Admin Name" --password "SecurePass123"

Flags:
    --email     Admin's email address  (required)
    --name      Admin's display name   (required)
    --password  Initial password       (required, min 8 chars)
"""

import sys
import argparse
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auth.supabase_auth import create_admin_user  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Create a bank admin account for ComplaintIQ."
    )
    parser.add_argument("--email",    required=True, help="Admin email address")
    parser.add_argument("--name",     required=True, help="Admin display name")
    parser.add_argument("--password", required=True, help="Initial password (min 8 chars)")
    args = parser.parse_args()

    if len(args.password) < 8:
        print("❌  Password must be at least 8 characters.")
        sys.exit(1)

    print(f"\nCreating admin account...")
    print(f"  Email : {args.email}")
    print(f"  Name  : {args.name}")
    print()

    try:
        profile = create_admin_user(
            email=args.email,
            password=args.password,
            full_name=args.name,
        )
        print("✅  Admin account created successfully!")
        print(f"    ID    : {profile['id']}")
        print(f"    Email : {profile['email']}")
        print(f"    Name  : {profile['full_name']}")
        print(f"    Role  : {profile['role']}")
        print()
        print("The admin can now log in at the ComplaintIQ dashboard.")
        print()
    except RuntimeError as e:
        print(f"❌  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
