import os
import uuid
import asyncio
import uvicorn
import edge_tts
import traceback
from fastapi import FastAPI, Form, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from twilio.twiml.voice_response import VoiceResponse, Gather, Redirect
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

# Static MP3 URLs
GREETING_MP3 = f"{BASE_URL}/static/greeting.mp3"
RETRY_MP3 = f"{BASE_URL}/static/retry.mp3"
GOODBYE_MP3 = f"{BASE_URL}/static/goodbye.mp3"

os.makedirs("static", exist_ok=True)
app = FastAPI()

class AddNgrokSkipHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response

app.add_middleware(AddNgrokSkipHeaderMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    print("🔥 Warming up RAG engine...")
    await asyncio.to_thread(init_pipeline)
    print("🚀 RAG engine ready!")

# ------------------------------------------------------------------
# TTS helper (edge‑tts)
# ------------------------------------------------------------------
async def synthesize_nepali_speech(text: str) -> str:
    print(f"🔊 Synthesizing: {text[:50]}...")
    try:
        filename = f"answer_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("static", filename)
        communicate = edge_tts.Communicate(text, "ne-NP-HemkalaNeural")
        await communicate.save(filepath)
        print(f"✅ MP3 saved: {filepath}")
        return filename
    except Exception as e:
        print(f"❌ TTS error: {e}")
        traceback.print_exc()
        raise

async def process_nepali_query(user_text: str) -> str:
    try:
        return await asyncio.to_thread(answer_user_query, user_text)
    except Exception as e:
        print(f"❌ RAG error: {e}")
        traceback.print_exc()
        return "क्षमा गर्नुहोस्, जानकारी खोज्न समस्या भयो।"

# ------------------------------------------------------------------
# /voice – plays greeting and redirects to /listen?first=1
# ------------------------------------------------------------------
@app.api_route("/voice", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    print("📞 /voice called")
    response = VoiceResponse()
    
    try:
        response.play(GREETING_MP3)
    except:
        response.say("नमस्ते! स्वचालित सूचना सेवामा स्वागत छ। तपाईं के जानकारी चाहनुहुन्छ?")
    
    # Redirect to /listen with first=1 (we'll ignore this flag now, but it's harmless)
    response.redirect(f"{LISTEN_URL}?first=1", method="POST")
    
    twiml = str(response)
    print(f"[VOICE TWIML] {twiml}")
    return Response(content=twiml, media_type="application/xml")

# ------------------------------------------------------------------
# /listen – silent <Gather> (no prompt)
# ------------------------------------------------------------------
@app.api_route("/listen", methods=["GET", "POST"])
async def listen_for_speech(request: Request):
    print("🎙️ /listen called")
    response = VoiceResponse()
    
    # No Say – just listen silently
    gather = Gather(
        input="speech",
        action=ACTION_URL,
        language="ne-NP",
        timeout=10,
        speechTimeout="auto",
    )
    response.append(gather)
    
    # If no speech, redirect back to /listen (silent loop)
    response.redirect(LISTEN_URL, method="POST")
    
    twiml = str(response)
    print(f"[LISTEN TWIML] {twiml}")
    return Response(content=twiml, media_type="application/xml")

# ------------------------------------------------------------------
# /process_speech – uses edge‑tts for answer, then redirects silently
# ------------------------------------------------------------------
@app.api_route("/process_speech", methods=["GET", "POST"])
async def process_speech(request: Request, SpeechResult: str = Form(None), Digits: str = Form(None)):
    print("🔔 /process_speech called")
    user_input = SpeechResult or Digits
    print(f"User input: {user_input}")

    response = VoiceResponse()
    try:
        if user_input:
            answer_text = await process_nepali_query(user_input)
            print(f"RAG answer: {answer_text}")

            if answer_text.strip() == "धन्यवाद। फेरि भेटौँला।":
                response.play(GOODBYE_MP3)
                response.hangup()
                twiml = str(response)
                print(f"[TWIML RESPONSE] {twiml}")
                return Response(content=twiml, media_type="application/xml")

            # Generate MP3 via edge‑tts and play it
            try:
                filename = await synthesize_nepali_speech(answer_text)
                mp3_url = f"{BASE_URL}/static/{filename}"
                response.play(mp3_url)
                print(f"[PLAY] {mp3_url}")
            except Exception:
                print("⚠️ TTS failed, using <Say>")
                response.say(answer_text)
        else:
            # No speech – play retry and redirect to /listen
            response.play(RETRY_MP3)
            response.redirect(LISTEN_URL, method="POST")
            twiml = str(response)
            print(f"[TWIML RESPONSE] {twiml}")
            return Response(content=twiml, media_type="application/xml")

        # After answer, redirect to /listen silently (no prompt)
        response.redirect(LISTEN_URL, method="POST")
        twiml = str(response)
        print(f"[TWIML RESPONSE] {twiml}")
        return Response(content=twiml, media_type="application/xml")

    except Exception as e:
        print("❌ Unhandled exception in /process_speech:")
        traceback.print_exc()
        response = VoiceResponse()
        response.say("क्षमा गर्नुहोस्, प्रणालीमा समस्या भयो।")
        response.hangup()
        twiml = str(response)
        return Response(content=twiml, media_type="application/xml")

# ------------------------------------------------------------------
# Outbound endpoint (unchanged)
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
        print(f"❌ Outbound call error: {e}")
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
    return {"status": "online", "message": "Voice‑to‑Voice RAG with Outbound Support"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)