import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

def send_reply_email(to_email: str, subject: str, body: str) -> bool:
    """
    Sends an email using the configured SMTP server over TLS.
    Returns True if successful, False otherwise.
    """
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        print("Warning: SMTP credentials not configured. Mocking email delivery.")
        print(f"--- MOCK EMAIL ---")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Body: {body}")
        print(f"------------------")
        return True

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email

        # Connect to the server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        # Login Credentials for sending the mail
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send the email
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")
        return False
