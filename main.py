import os
import uuid
import asyncio
import time
import uvicorn
import edge_tts
import traceback
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv

load_dotenv()

from src.rag.rag_engine import answer_user_query, init_pipeline

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

BASE_URL = os.getenv("BASE_URL")
if not BASE_URL:
    raise ValueError("BASE_URL not set in .env")

ACTION_URL = f"{BASE_URL}/process_speech"
VOICE_URL = f"{BASE_URL}/voice"
LISTEN_URL = f"{BASE_URL}/listen"

RETRY_MP3 = f"{BASE_URL}/static/retry.mp3"
GOODBYE_MP3 = f"{BASE_URL}/static/goodbye.mp3"
GREETING_MP3 = f"{BASE_URL}/static/greeting.mp3"

os.makedirs("static", exist_ok=True)

# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------
async def generate_greeting_mp3():
    """Generate the greeting MP3 using edge-tts if it does not exist."""
    filepath = os.path.join("static", "greeting.mp3")
    if os.path.exists(filepath):
        print("Greeting MP3 already exists, skipping generation.")
        return
    print("Generating greeting MP3 with edge-tts...")
    try:
        communicate = edge_tts.Communicate(
            "यो विपद् व्यवस्थापन जानकारी पोर्टल हो। म कसरी तपाईंलाई मद्दत गर्न सक्छु?",
            "ne-NP-HemkalaNeural"
        )
        await communicate.save(filepath)
        print(f"Greeting MP3 saved: {filepath}")
    except Exception as e:
        print(f"Failed to generate greeting MP3: {e}")

async def clean_old_mp3s():
    """Periodically delete old answer MP3s (older than 1 hour)."""
    while True:
        await asyncio.sleep(3600 * 6)  # every 6 hours
        now = time.time()
        for f in os.listdir("static"):
            if f.startswith("answer_") and f.endswith(".mp3"):
                path = os.path.join("static", f)
                try:
                    if os.path.getmtime(path) < now - 3600:
                        os.remove(path)
                        print(f"Cleaned old MP3: {f}")
                except Exception as e:
                    print(f"Error cleaning {f}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up RAG engine...")
    await asyncio.to_thread(init_pipeline)
    print("RAG engine ready.")

    # Generate greeting MP3 in background
    asyncio.create_task(generate_greeting_mp3())

    # Start periodic cleanup
    cleanup_task = asyncio.create_task(clean_old_mp3s())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

# ------------------------------------------------------------------
# Middleware & Static
# ------------------------------------------------------------------
class AddNgrokSkipHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response

app.add_middleware(AddNgrokSkipHeaderMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------------------------------------------------
# TTS helper
# ------------------------------------------------------------------
async def synthesize_nepali_speech(text: str) -> str:
    print(f"Synthesizing: {text[:50]}...")
    try:
        filename = f"answer_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("static", filename)
        communicate = edge_tts.Communicate(text, "ne-NP-HemkalaNeural")
        await communicate.save(filepath)
        print(f"MP3 saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"TTS error: {e}")
        traceback.print_exc()
        raise

async def delete_mp3_after_delay(filepath: str, delay_seconds: int = 30):
    """Delete the MP3 file after a delay to ensure Twilio has fetched it."""
    await asyncio.sleep(delay_seconds)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted MP3: {filepath}")
    except Exception as e:
        print(f"Failed to delete MP3: {e}")

async def process_nepali_query(user_text: str, call_sid: Optional[str] = None) -> str:
    try:
        return await asyncio.to_thread(answer_user_query, user_text, call_sid)
    except Exception as e:
        print(f"RAG error: {e}")
        traceback.print_exc()
        return "क्षमा गर्नुहोस्, जानकारी खोज्न समस्या भयो।"

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.api_route("/voice", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    print("/voice called")
    response = VoiceResponse()
    try:
        response.play(GREETING_MP3)
    except Exception:
        response.say("यो विपद् व्यवस्थापन जानकारी पोर्टल हो। म कसरी तपाईंलाई मद्दत गर्न सक्छु?")
    response.redirect(f"{LISTEN_URL}?first=1", method="POST")
    twiml = str(response)
    print(f"[VOICE TWIML] {twiml}")
    return Response(content=twiml, media_type="application/xml")

@app.api_route("/listen", methods=["GET", "POST"])
async def listen_for_speech(request: Request):
    print("/listen called")
    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action=ACTION_URL,
        language="ne-NP",
        timeout=10,
        speechTimeout="auto",
    )
    response.append(gather)
    response.redirect(LISTEN_URL, method="POST")
    twiml = str(response)
    print(f"[LISTEN TWIML] {twiml}")
    return Response(content=twiml, media_type="application/xml")

@app.api_route("/process_speech", methods=["GET", "POST"])
async def process_speech(request: Request, SpeechResult: str = Form(None), Digits: str = Form(None)):
    print("/process_speech called")
    form = await request.form()
    call_sid = form.get("CallSid", None)
    user_input = SpeechResult or Digits
    print(f"User input: {user_input}")

    response = VoiceResponse()
    try:
        if user_input:
            answer_text = await process_nepali_query(user_input, call_sid)
            print(f"RAG answer: {answer_text}")

            if answer_text.strip() == "धन्यवाद। फेरि भेटौँला।":
                response.play(GOODBYE_MP3)
                response.hangup()
                twiml = str(response)
                print(f"[TWIML RESPONSE] {twiml}")
                return Response(content=twiml, media_type="application/xml")

            try:
                filepath = await synthesize_nepali_speech(answer_text)
                filename = os.path.basename(filepath)
                mp3_url = f"{BASE_URL}/static/{filename}"
                response.play(mp3_url)
                print(f"[PLAY] {mp3_url}")
                asyncio.create_task(delete_mp3_after_delay(filepath, 30))
            except Exception:
                print("TTS failed, using <Say>")
                response.say(answer_text)
        else:
            response.play(RETRY_MP3)
            response.redirect(LISTEN_URL, method="POST")
            twiml = str(response)
            print(f"[TWIML RESPONSE] {twiml}")
            return Response(content=twiml, media_type="application/xml")

        response.redirect(LISTEN_URL, method="POST")
        twiml = str(response)
        print(f"[TWIML RESPONSE] {twiml}")
        return Response(content=twiml, media_type="application/xml")

    except Exception as e:
        print("Unhandled exception in /process_speech:")
        traceback.print_exc()
        response = VoiceResponse()
        response.say("क्षमा गर्नुहोस्, प्रणालीमा समस्या भयो।")
        response.hangup()
        twiml = str(response)
        return Response(content=twiml, media_type="application/xml")

# ------------------------------------------------------------------
# Outbound endpoint
# ------------------------------------------------------------------
@app.post("/outbound")
async def make_outbound_call(to: str = Form(...), message: str = Form(None)):
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        return {"error": "Twilio credentials or phone number not configured."}
    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        call = client.calls.create(url=VOICE_URL, to=to, from_=TWILIO_PHONE_NUMBER)
        return {"status": "success", "call_sid": call.sid, "message": "Call initiated."}
    except Exception as e:
        print(f"Outbound call error: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/outbound", response_class=HTMLResponse)
async def outbound_form():
    return """
    <html>
        <head><title>Outbound Call Tester</title></head>
        <body>
            <h2>Initiate Outbound Call</h2>
            <form method="post" action="/outbound">
                <label>Phone Number (e.g. +97798XXXXXXXX):</label>
                <input type="text" name="to" placeholder="+97798XXXXXXXX" required>
                <br><br>
                <button type="submit">Call Now</button>
            </form>
        </body>
    </html>
    """

@app.get("/")
async def root():
    return {"status": "online", "message": "Voice-to-Voice RAG with Outbound Support"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)