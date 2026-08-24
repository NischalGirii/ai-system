import os
import uuid
import asyncio
import uvicorn
import edge_tts
import traceback
from fastapi import FastAPI, Form, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from twilio.twiml.voice_response import VoiceResponse, Gather

from src.rag.rag_engine import answer_user_query, init_pipeline

# ------------------------------------------------------------------
# Create static directory and mount it
# ------------------------------------------------------------------
os.makedirs("static", exist_ok=True)
app = FastAPI()

# ------------------------------------------------------------------
# Middleware to skip ngrok browser warning (free tier)
# ------------------------------------------------------------------
class AddNgrokSkipHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response

app.add_middleware(AddNgrokSkipHeaderMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------------------------------------------------
# Configuration – update BASE_URL if your ngrok URL changes
# ------------------------------------------------------------------
BASE_URL = "https://mocha-slip-pretense.ngrok-free.dev"   # <-- CHANGE THIS
ACTION_URL = f"{BASE_URL}/process_speech"

# ------------------------------------------------------------------
# Startup: initialise RAG pipeline
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# RAG query helper (async wrapper)
# ------------------------------------------------------------------
async def process_nepali_query(user_text: str) -> str:
    try:
        return await asyncio.to_thread(answer_user_query, user_text)
    except Exception as e:
        print(f"❌ RAG error: {e}")
        traceback.print_exc()
        return "क्षमा गर्नुहोस्, जानकारी खोज्न समस्या भयो।"

# ------------------------------------------------------------------
# Endpoint: /voice – initial greeting and gather
# ------------------------------------------------------------------
@app.api_route("/voice", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    print("📞 /voice called")
    response = VoiceResponse()

    # Use edge‑tts for greeting
    try:
        filename = await synthesize_nepali_speech("नमस्ते! सोध्नुहोस्, म कसरी मद्दत गर्न सक्छु?")
        gather = Gather(
            input="speech dtmf",
            action=ACTION_URL,
            language="ne-NP",
            timeout=5,
            speechTimeout="auto",
            numDigits=1,
        )
        gather.play(f"{BASE_URL}/static/{filename}")
    except Exception:
        # Fallback to <Say> if TTS fails
        gather = Gather(
            input="speech dtmf",
            action=ACTION_URL,
            language="ne-NP",
            timeout=5,
            speechTimeout="auto",
            numDigits=1,
        )
        gather.say("नमस्ते! कृपया प्रश्न सोध्नुहोस्।")

    response.append(gather)
    return Response(content=str(response), media_type="application/xml")

# ------------------------------------------------------------------
# Endpoint: /process_speech – process user input and reply
# ------------------------------------------------------------------
@app.api_route("/process_speech", methods=["GET", "POST"])
async def process_speech(request: Request, SpeechResult: str = Form(None), Digits: str = Form(None)):
    print("🔔 /process_speech called")
    user_input = SpeechResult or Digits
    print(f"User input: {user_input}")

    response = VoiceResponse()
    try:
        if user_input:
            # 1. Get answer from RAG
            answer_text = await process_nepali_query(user_input)
            print(f"RAG answer: {answer_text}")

            # 2. Play answer via TTS (or fallback to <Say>)
            try:
                filename = await synthesize_nepali_speech(answer_text)
                response.play(f"{BASE_URL}/static/{filename}")
            except Exception:
                print("⚠️ TTS failed, using <Say>")
                response.say(answer_text)

            # 3. If goodbye, hang up
            if answer_text.strip() == "धन्यवाद। फेरि भेटौँला।":
                response.hangup()
                return Response(content=str(response), media_type="application/xml")
        else:
            # No input – fallback
            try:
                filename = await synthesize_nepali_speech("मैले तपाईंको कुरा बुझिन। कृपया पुनः भन्नुहोस्।")
                response.play(f"{BASE_URL}/static/{filename}")
            except Exception:
                response.say("मैले तपाईंको कुरा बुझिन। कृपया पुनः भन्नुहोस्।")

        # 4. Continue conversation – prompt for next question
        try:
            prompt_filename = await synthesize_nepali_speech("अर्को प्रश्न सोध्नुहोस्।")
            next_gather = Gather(
                input="speech dtmf",
                action=ACTION_URL,
                language="ne-NP",
                timeout=5,
                speechTimeout="auto",
                numDigits=1,
            )
            next_gather.play(f"{BASE_URL}/static/{prompt_filename}")
        except Exception:
            next_gather = Gather(
                input="speech dtmf",
                action=ACTION_URL,
                language="ne-NP",
                timeout=5,
                speechTimeout="auto",
                numDigits=1,
            )
            next_gather.say("अर्को प्रश्न सोध्नुहोस्।")
        response.append(next_gather)

    except Exception as e:
        print("❌ Unhandled exception in /process_speech:")
        traceback.print_exc()
        response = VoiceResponse()
        response.say("क्षमा गर्नुहोस्, प्रणालीमा समस्या भयो।")
        response.hangup()

    return Response(content=str(response), media_type="application/xml")

# ------------------------------------------------------------------
# Root endpoint (health check)
# ------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "online", "message": "Voice‑to‑Voice RAG with Nepali TTS"}

# ------------------------------------------------------------------
# Run with uvicorn (if executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)