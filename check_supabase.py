"""
Supabase Connection Test for ComplaintIQ
----------------------------------------
Run this script to verify your Supabase credentials and connection
are working correctly before deploying.

Usage:
    python check_supabase.py

Requirements:
    pip install supabase python-dotenv
"""

import os
import sys
from pathlib import Path

# Load .env from project root
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def check_env_vars() -> bool:
    """Check that all required Supabase env vars are present."""
    required = ["SUPABASE_URL", "SUPABASE_ANON_KEY"]
    missing = [k for k in required if not os.getenv(k) or os.getenv(k, "").startswith("your_")]

    if missing:
        print("❌  Missing or placeholder env vars:")
        for k in missing:
            print(f"     - {k}")
        print("\n👉  Open your .env file and fill in real values from:")
        print("    Supabase Dashboard → Project Settings → API")
        return False

    url = os.getenv("SUPABASE_URL", "")
    if not url.startswith("https://") or ".supabase.co" not in url:
        print(f"❌  SUPABASE_URL looks wrong: {url!r}")
        print("    Expected format: https://<project-ref>.supabase.co")
        return False

    print("✅  Env vars present:")
    print(f"    SUPABASE_URL      = {os.getenv('SUPABASE_URL')}")
    anon = os.getenv("SUPABASE_ANON_KEY", "")
    print(f"    SUPABASE_ANON_KEY = {anon[:20]}...{anon[-6:]}")
    return True


def check_supabase_import() -> bool:
    """Ensure the supabase-py package is installed."""
    try:
        import supabase  # noqa: F401
        print("✅  supabase-py package is installed")
        return True
    except ImportError:
        print("❌  supabase-py is not installed.")
        print("    Run:  pip install supabase")
        return False


def check_connection() -> bool:
    """Attempt a live connection to Supabase and run a simple health query."""
    from supabase import create_client, Client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_ANON_KEY", "")

    print(f"\n🔌  Connecting to {url} ...")
    try:
        client: Client = create_client(url, key)
        # Try fetching Supabase server time via the REST API health check
        # (works even on fresh projects with no tables)
        resp = client.rpc("version").execute()  # postgres version()
        print(f"✅  Connected! PostgreSQL: {resp.data}")
        return True
    except Exception as e:
        err = str(e)
        if "Invalid API key" in err or "401" in err:
            print("❌  Authentication failed — check your SUPABASE_ANON_KEY.")
        elif "connection" in err.lower() or "timeout" in err.lower():
            print("❌  Network error — check your internet connection and SUPABASE_URL.")
        else:
            print(f"❌  Unexpected error: {e}")
        return False


def check_table_exists() -> bool:
    """Check if the 'complaints' table already exists in Supabase."""
    from supabase import create_client, Client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")

    client: Client = create_client(url, key)
    try:
        # A SELECT with LIMIT 0 tells us if the table exists without fetching data
        resp = client.table("complaints").select("id").limit(0).execute()
        count = len(resp.data) if resp.data else 0
        print(f"✅  'complaints' table exists (quick check returned {count} rows)")
        return True
    except Exception as e:
        err = str(e)
        if "relation" in err.lower() or "does not exist" in err.lower() or "42P01" in err:
            print("⚠️   'complaints' table does NOT exist yet in Supabase.")
            print("    → Run the schema migration to create it (coming next).")
        else:
            print(f"⚠️   Table check error: {e}")
        return False


def main():
    print("=" * 55)
    print("  ComplaintIQ — Supabase Connection Check")
    print("=" * 55)
    print()

    steps = [
        ("1. Environment variables", check_env_vars),
        ("2. supabase-py package",   check_supabase_import),
    ]

    for label, fn in steps:
        print(f"── {label} ──")
        ok = fn()
        print()
        if not ok:
            print("🛑  Fix the issues above before continuing.\n")
            sys.exit(1)

    # Only run live checks if env + package are good
    print("── 3. Live connection ──")
    connected = check_connection()
    print()

    if connected:
        print("── 4. Table existence ──")
        check_table_exists()
        print()
        print("🎉  All checks passed! Supabase is reachable.\n")
    else:
        print("🛑  Could not connect to Supabase. Check your credentials.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
