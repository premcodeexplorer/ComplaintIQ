import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.db import connect, ensure_schema

def seed_users():
    # Make sure schema is up to date
    ensure_schema()
    
    users = [
        ("ACC-1001", "John Doe", "john.doe@example.com"),
        ("ACC-1002", "Jane Smith", "jane.smith@example.com"),
        ("ACC-1003", "Alice Johnson", "alice.j@example.com"),
    ]
    
    with connect() as c:
        for acc, name, email in users:
            try:
                # Use insert or ignore for SQLite, ON CONFLICT DO NOTHING for PG
                try:
                    c.execute("INSERT OR IGNORE INTO users (account_no, name, email) VALUES (?, ?, ?)", (acc, name, email))
                except Exception:
                    # If PG
                    c.cursor().execute("INSERT INTO users (account_no, name, email) VALUES (%s, %s, %s) ON CONFLICT (account_no) DO NOTHING", (acc, name, email))
            except Exception as e:
                print(f"Failed to insert {acc}: {e}")
                
    print(f"Successfully seeded {len(users)} users.")

if __name__ == "__main__":
    seed_users()
