"""
ComplaintIQ — Supabase Auth utilities
--------------------------------------
Handles bank-admin authentication via Supabase Auth (email + password).

Usage in Streamlit:
    from auth.supabase_auth import sign_in, sign_out, get_user_profile

Admin account creation (CLI only, not exposed to browser):
    python scripts/create_admin.py --email ... --name ... --password ...
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── Client factory ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_auth_client():
    """Return a cached supabase-py client (anon key — for auth operations)."""
    from supabase import create_client
    url  = os.getenv("SUPABASE_URL", "")
    key  = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env "
            "for authentication to work."
        )
    return create_client(url, key)


@lru_cache(maxsize=1)
def get_admin_client():
    """Return a cached supabase-py client (service role key — for admin ops)."""
    from supabase import create_client
    url  = os.getenv("SUPABASE_URL", "")
    key  = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env "
            "for admin user creation to work."
        )
    return create_client(url, key)


# ── Sign-in / Sign-out ───────────────────────────────────────────────────────

def sign_in(email: str, password: str) -> dict[str, Any]:
    """
    Authenticate a bank admin with email + password.

    Returns a session dict:
        {
            "access_token":  str,
            "refresh_token": str,
            "user": {
                "id":    str (UUID),
                "email": str,
            }
        }

    Raises RuntimeError on bad credentials or if the user is not an admin.
    """
    url  = os.getenv("SUPABASE_URL", "")
    key  = os.getenv("SUPABASE_ANON_KEY", "")
    
    if not url or not key:
        # Mock fallback for offline/demo run
        if email == "admin@bank.com" and password == "admin123":
            return {
                "access_token": "mock_access_token",
                "refresh_token": "mock_refresh_token",
                "user": {
                    "id": "00000000-0000-0000-0000-000000000000",
                    "email": email,
                },
                "profile": {
                    "id": "00000000-0000-0000-0000-000000000000",
                    "email": email,
                    "full_name": "Demo Admin",
                    "role": "admin",
                    "created_at": None,
                    "last_login": None,
                }
            }
        else:
            raise RuntimeError(
                "Supabase not configured. To login offline, use:\n"
                "Email: admin@bank.com\n"
                "Password: admin123"
            )

    client = get_auth_client()
    try:
        resp = client.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        raise RuntimeError(f"Login failed: {e}") from e

    if not resp.session:
        raise RuntimeError("Login failed: no session returned.")

    session = {
        "access_token":  str(resp.session.access_token),
        "refresh_token": str(resp.session.refresh_token),
        "user": {
            "id":    str(resp.user.id),
            "email": str(resp.user.email),
        },
    }

    # Verify the user has an admin profile in our user_profiles table
    profile = get_user_profile(resp.user.id)
    if profile is None:
        # Auth succeeded but no profile row — shouldn't happen if admin was
        # created via create_admin.py, but guard against it.
        sign_out(session)
        raise RuntimeError(
            "Your account is not registered as a bank admin. "
            "Contact your system administrator."
        )

    session["profile"] = profile
    return session


def sign_out(session: dict[str, Any]) -> None:
    """Invalidate the current session on Supabase's side."""
    try:
        client = get_auth_client()
        client.auth.sign_out()
    except Exception:
        pass  # Best-effort — local session state will be cleared by caller


# ── Profile helpers ──────────────────────────────────────────────────────────

def get_user_profile(user_id: str) -> dict[str, Any] | None:
    """
    Fetch the user_profiles row for `user_id`.
    Returns None if no row exists (user is not a registered admin).
    Uses psycopg2 (the same connection as the rest of the app).
    """
    import psycopg2

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set.")

    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
        cur  = conn.cursor()
        cur.execute(
            "SELECT id, email, full_name, role, created_at, last_login "
            "FROM user_profiles WHERE id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Could not fetch user profile: {e}") from e

    if row is None:
        return None

    return {
        "id":         str(row[0]),
        "email":      row[1],
        "full_name":  row[2],
        "role":       row[3],
        "created_at": row[4].isoformat() if row[4] else None,
        "last_login": row[5].isoformat() if row[5] else None,
    }


def update_last_login(user_id: str) -> None:
    """Stamp last_login = NOW() for the given user after successful sign-in."""
    if user_id == "00000000-0000-0000-0000-000000000000":
        return
    import psycopg2
    from datetime import datetime, timezone

    db_url = os.getenv("DATABASE_URL", "")
    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
        cur  = conn.cursor()
        cur.execute(
            "UPDATE user_profiles SET last_login = %s WHERE id = %s",
            (datetime.now(timezone.utc).isoformat(), user_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass  # Non-critical — don't break login if this fails


# ── Admin creation (called by scripts/create_admin.py) ───────────────────────

def create_admin_user(email: str, password: str, full_name: str) -> dict[str, Any]:
    """
    Create a new bank admin account.
    1. Creates the Supabase Auth user (service role — no email confirmation needed).
    2. Inserts a user_profiles row with role = 'admin'.

    Returns the new profile dict.
    Only call this from create_admin.py — never expose to the browser.
    """
    import psycopg2

    # Step 1: create Supabase Auth user (service role bypasses email confirmation)
    admin_client = get_admin_client()
    try:
        resp = admin_client.auth.admin.create_user({
            "email":            email,
            "password":         password,
            "email_confirm":    True,   # mark as confirmed immediately
        })
    except Exception as e:
        raise RuntimeError(f"Failed to create Supabase Auth user: {e}") from e

    user_id = resp.user.id

    # Step 2: insert into user_profiles
    db_url = os.getenv("DATABASE_URL", "")
    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO user_profiles (id, email, full_name, role) "
            "VALUES (%s, %s, %s, 'admin') "
            "ON CONFLICT (id) DO UPDATE SET full_name = EXCLUDED.full_name",
            (user_id, email, full_name),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        # Rollback: delete the auth user we just created to avoid orphans
        try:
            admin_client.auth.admin.delete_user(user_id)
        except Exception:
            pass
        raise RuntimeError(f"Failed to create user profile: {e}") from e

    return {
        "id":        user_id,
        "email":     email,
        "full_name": full_name,
        "role":      "admin",
    }
