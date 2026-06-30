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

try:
    cur.execute("""
        SELECT id, channel, customer_name, complaint_text, date, status 
        FROM complaints 
        ORDER BY date DESC 
        LIMIT 5;
    """)
    rows = cur.fetchall()
    print("Latest 5 complaints in Supabase:")
    for r in rows:
        print(f"ID: {r[0]} | Channel: {r[1]} | Name: {r[2]} | Date: {r[4]}")
        print(f"Text snippet: {r[3][:60]}...")
        print("-" * 40)
except Exception as e:
    print("Select failed with error:")
    print(e)

