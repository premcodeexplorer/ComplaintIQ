"""
ComplaintIQ — Supabase Schema Migration Runner
------------------------------------------------
Applies all pending SQL migrations in migrations/ to Supabase
in order, using psycopg2 + the DATABASE_URL from .env.

Usage:
    python migrations/apply_migrations.py

Safe to re-run: all statements use IF NOT EXISTS.
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MIGRATIONS_DIR = Path(__file__).resolve().parent


def get_connection():
    import psycopg2
    url = os.getenv("DATABASE_URL", "")
    if not url:
        print("❌  DATABASE_URL not set in .env")
        sys.exit(1)
    return psycopg2.connect(url, connect_timeout=10)


def apply_migrations():
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        print("No migration files found in migrations/")
        return

    print("=" * 55)
    print("  ComplaintIQ -- Supabase Schema Migration")
    print("=" * 55)

    conn = get_connection()
    cursor = conn.cursor()

    print(f"\n✅  Connected to Supabase\n")

    for sql_file in sql_files:
        print(f"-- Applying: {sql_file.name} --")
        sql = sql_file.read_text(encoding="utf-8")

        try:
            cursor.execute(sql)
            conn.commit()
            print(f"✅  {sql_file.name} applied successfully\n")
        except Exception as e:
            conn.rollback()
            print(f"❌  Error in {sql_file.name}: {e}\n")
            cursor.close()
            conn.close()
            sys.exit(1)

    # Verify tables were created
    print("-- Verifying tables --")
    expected_tables = ["complaints", "root_cause_alerts", "feedback"]
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = ANY(%s)
        ORDER BY table_name;
    """, (expected_tables,))
    found = [row[0] for row in cursor.fetchall()]

    for table in expected_tables:
        if table in found:
            # Get column count
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
            """, (table,))
            col_count = cursor.fetchone()[0]
            print(f"  ✅  {table:<25} ({col_count} columns)")
        else:
            print(f"  ❌  {table:<25} MISSING!")

    cursor.close()
    conn.close()

    print(f"\n🎉  Migration complete! All tables are ready in Supabase.\n")


if __name__ == "__main__":
    apply_migrations()
