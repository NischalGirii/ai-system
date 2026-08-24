import os
from twilio.rest import Client

# Option A: Replace these strings directly with your actual credentials from the Twilio Console
# Option B: Set them via PowerShell ($env:TWILIO_ACCOUNT_SID="...")
account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "YOUR_CREDENTIALS")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "YOUR_CREDENTIALS")

client = Client(account_sid, auth_token)

call = client.calls.create(
    # Point this directly to your live ngrok FastAPI endpoint!
    url="YOUR_URL",
    to="YOUR_PHONE_NUMBER",  # Ensure this number is verified in your Twilio console
    from_="TRAIL_NUMBER",  # Your Twilio trial phone number
)

print(f"Call initiated successfully! SID: {call.sid}")