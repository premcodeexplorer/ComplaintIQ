import os
import sys
import sqlite3
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PG_URL = os.getenv("DATABASE_URL")
SQLITE_PATH = ROOT / "data" / "complaintiq.sqlite"

def main():
    print(f"Connecting to SQLite: {SQLITE_PATH}")
    sl_conn = sqlite3.connect(SQLITE_PATH)
    sl_conn.row_factory = sqlite3.Row
    
    print(f"Connecting to Postgres")
    pg_conn = psycopg2.connect(PG_URL)
    
    # Initialize Postgres schema using existing db module
    sys.path.insert(0, str(ROOT))
    from database import db
    db.init_db()

    # 1. Migrate complaints in two passes to avoid foreign key violations
    complaints = sl_conn.execute("SELECT * FROM complaints").fetchall()
    if complaints:
        cols = list(complaints[0].keys())
        dup_idx = cols.index("duplicate_of")
        
        pass1_rows = []
        updates = []
        for row in complaints:
            r = list(row)
            if r[dup_idx] is not None:
                updates.append((r[dup_idx], row["id"])) # (dup_val, id) for update
                r[dup_idx] = None
            pass1_rows.append(tuple(r))
            
        query = f"INSERT INTO complaints ({','.join(cols)}) VALUES %s ON CONFLICT (id) DO NOTHING"
        with pg_conn.cursor() as cur:
            execute_values(cur, query, pass1_rows)
            
            if updates:
                execute_values(cur, "UPDATE complaints SET duplicate_of = data.dup FROM (VALUES %s) AS data(dup, id) WHERE complaints.id = data.id", updates)
                
        pg_conn.commit()
        print(f"Migrated {len(complaints)} complaints.")

    # 2. Migrate root_cause_alerts
    alerts = sl_conn.execute("SELECT * FROM root_cause_alerts").fetchall()
    if alerts:
        cols = list(alerts[0].keys())
        query = f"INSERT INTO root_cause_alerts ({','.join(cols)}) VALUES %s ON CONFLICT (id) DO NOTHING"
        with pg_conn.cursor() as cur:
            execute_values(cur, query, [tuple(row) for row in alerts])
            # Update sequence
            cur.execute("SELECT setval('root_cause_alerts_id_seq', (SELECT MAX(id) FROM root_cause_alerts));")
        pg_conn.commit()
        print(f"Migrated {len(alerts)} alerts.")

    # 3. Migrate feedback
    feedbacks = sl_conn.execute("SELECT * FROM feedback").fetchall()
    if feedbacks:
        cols = list(feedbacks[0].keys())
        idx = cols.index("is_correct")
        fb_rows = []
        for row in feedbacks:
            r = list(row)
            r[idx] = bool(r[idx])
            fb_rows.append(tuple(r))
            
        query = f"INSERT INTO feedback ({','.join(cols)}) VALUES %s ON CONFLICT (id) DO NOTHING"
        with pg_conn.cursor() as cur:
            execute_values(cur, query, fb_rows)
            # Update sequence
            cur.execute("SELECT setval('feedback_id_seq', (SELECT MAX(id) FROM feedback));")
        pg_conn.commit()
        print(f"Migrated {len(feedbacks)} feedback records.")

    sl_conn.close()
    pg_conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    main()
