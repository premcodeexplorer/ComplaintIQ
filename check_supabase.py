"""
Supabase Connection Test for ComplaintIQ
----------------------------------------
Tests a direct PostgreSQL connection via psycopg2 using the
Supabase shared pooler URL (IPv4-compatible).

Usage:
    python check_supabase.py

Requirements:
    pip install psycopg2-binary python-dotenv
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding so UTF-8 emojis print correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def check_env_vars() -> bool:
    """Check that DATABASE_URL is present and not a placeholder."""
    url = os.getenv("DATABASE_URL", "")

    if not url or url.startswith("your_"):
        print("❌  DATABASE_URL is missing or still set to placeholder.")
        print("    Open .env and paste your Supabase shared pooler URL:")
        print("    Supabase Dashboard → Project Settings → Database → Connection string → Transaction pooler")
        return False

    if not url.startswith("postgresql://") and not url.startswith("postgres://"):
        print(f"❌  DATABASE_URL doesn't look like a PostgreSQL URL: {url[:40]}...")
        return False

    # Mask password for display
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        masked = url.replace(p.password or "", "****") if p.password else url
    except Exception:
        masked = url[:40] + "..."

    print(f"✅  DATABASE_URL = {masked}")
    return True


def check_psycopg2() -> bool:
    """Ensure psycopg2 is installed."""
    try:
        import psycopg2  # noqa: F401
        print("✅  psycopg2 package is installed")
        return True
    except ImportError:
        print("❌  psycopg2 is not installed.")
        print("    Run:  pip install psycopg2-binary")
        return False


def check_connection() -> bool:
    """Attempt a live PostgreSQL connection and run a simple query."""
    import psycopg2

    url = os.getenv("DATABASE_URL", "")
    print(f"\n🔌  Connecting to Supabase (shared pooler) ...")

    try:
        conn = psycopg2.connect(url, connect_timeout=10)
        cursor = conn.cursor()

        # Fetch server version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅  Connected! Server: {version.split(',')[0]}")

        cursor.close()
        conn.close()
        return True

    except psycopg2.OperationalError as e:
        err = str(e).strip()
        if "password authentication" in err:
            print("❌  Auth failed — wrong password in DATABASE_URL.")
        elif "could not connect" in err or "timeout" in err.lower():
            print("❌  Network error — check your internet connection or pooler URL.")
        else:
            print(f"❌  Connection error: {err}")
        return False
    except Exception as e:
        print(f"❌  Unexpected error: {e}")
        return False


def check_table_exists() -> bool:
    """Check if the 'complaints' table already exists in Supabase."""
    import psycopg2

    url = os.getenv("DATABASE_URL", "")
    try:
        conn = psycopg2.connect(url, connect_timeout=10)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'complaints'
            );
        """)
        exists = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        if exists:
            print("✅  'complaints' table already exists in Supabase.")
        else:
            print("⚠️   'complaints' table does NOT exist yet.")
            print("    → Schema migration needed (next step).")
        return True
    except Exception as e:
        print(f"⚠️   Table check error: {e}")
        return False


def main():
    print("=" * 55)
    print("  ComplaintIQ — Supabase Connection Check")
    print("=" * 55)
    print()

    steps = [
        ("1. Environment variable", check_env_vars),
        ("2. psycopg2 package",     check_psycopg2),
    ]

    for label, fn in steps:
        print(f"-- {label} --")
        ok = fn()
        print()
        if not ok:
            print("🛑  Fix the issues above before continuing.\n")
            sys.exit(1)

    print("-- 3. Live connection --")
    connected = check_connection()
    print()

    if connected:
        print("-- 4. Table existence --")
        check_table_exists()
        print()
        print("🎉  All checks passed! Supabase is reachable.\n")
    else:
        print("🛑  Could not connect to Supabase. Check your DATABASE_URL.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
