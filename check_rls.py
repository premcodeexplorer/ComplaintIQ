import psycopg2
import os

url = "postgresql://postgres:Prajwanil%402006@db.ujadranhlvrbiksujvum.supabase.co:5432/postgres"
conn = psycopg2.connect(url)
cur = conn.cursor()

# Check if RLS is enabled on complaints
cur.execute("SELECT relname, relrowsecurity FROM pg_class WHERE relname = 'complaints';")
result = cur.fetchone()
if result:
    print(f"Table {result[0]} RLS enabled: {result[1]}")
else:
    print("Table complaints not found!")

from datetime import datetime

try:
    cur.execute("""
        INSERT INTO complaints (
            id, customer_name, channel, complaint_text, language, date, 
            location, account_type, amount_involved, status
        ) VALUES (
            'PORTAL-12345678', 'Test User', 'portal', 'Test complaint', 'English', 
            %s, 'Nagpur', 'Savings Account', 0.0, 'open'
        )
    """, (datetime.utcnow().isoformat(),))
    conn.commit()
    print("Insert succeeded!")
except Exception as e:
    print("Insert failed with error:")
    print(e)
