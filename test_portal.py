import urllib.request
import json

url = "https://magnificent-salamander-02e31c.netlify.app/api/submit-complaint"
data = {
    "customer_name": "Test User",
    "complaint_text": "Test complaint from AI",
    "language": "English",
    "location": "Pune",
    "account_type": "Savings Account",
    "amount_involved": "500"
}

req = urllib.request.Request(url, json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req) as res:
        print("Status:", res.status)
        print("Body:", res.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print("HTTP Error Status:", e.code)
    print("Error Body:", e.read().decode("utf-8"))
except Exception as e:
    print("Other Error:", str(e))
