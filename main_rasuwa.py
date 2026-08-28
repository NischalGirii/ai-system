# main_rasuwa.py
import os
import uuid
import asyncio
import time
import uvicorn
import edge_tts
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from twilio.twiml.voice_response import VoiceResponse, Gather
from dotenv import load_dotenv

load_dotenv()

from src.rag.rag_rasuwa import answer_user_query, init_pipeline

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL")
if not BASE_URL:
    raise ValueError("BASE_URL not set in .env")

LISTEN_URL = f"{BASE_URL}/listen"
ACTION_URL = f"{BASE_URL}/process_speech"
GREETING_MP3 = f"{BASE_URL}/static/greeting.mp3"
GOODBYE_MP3 = f"{BASE_URL}/static/goodbye.mp3"

os.makedirs("static", exist_ok=True)

# ------------------------------------------------------------------
# Parse emergency TXT directly to extract contact numbers
# ------------------------------------------------------------------
EMERGENCY_DOC_PATH = os.getenv(
    "EMERGENCY_DOC_PATH",
    "data/rasuwa_nuwakot_dhading_chitwan_flood_emergency_nepali.txt"
)

def parse_emergency_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    headers = re.finditer(r"(?m)^={4,}\n(\d+\. .+?)\n={4,}$", text)
    matches = list(headers)
    sections = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx+1].start() if idx+1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({
            "title": title,
            "content": content,
            "section_number": idx+1,
        })
    return sections

def extract_contact_numbers():
    try:
        sections = parse_emergency_sections(EMERGENCY_DOC_PATH)
    except FileNotFoundError:
        print(f"[WARNING] Emergency file not found: {EMERGENCY_DOC_PATH}")
        return set()

    if not sections:
        return set()

    numbers = set()
    for sec in sections:
        content = sec["content"]
        hyphen_numbers = re.findall(r'[०-९0-9]+[-–—][०-९0-9]+', content)
        numbers.update(hyphen_numbers)
        standalone = re.findall(r'(?<![०-९0-9])[०-९0-9]{3,}(?![०-९0-9])', content)
        numbers.update(standalone)
        keyword_pattern = r'(?:नम्बर|फोन|टोल-फ्री|हटलाइन|सम्पर्क)\s*[:–\-]?\s*([०-९0-9]+)'
        matches = re.findall(keyword_pattern, content)
        for num in matches:
            if len(num) >= 3:
                numbers.add(num)

    helplines = {"100", "102", "1149", "1234", "16600141516",
                 "१००", "१०२", "११४९", "१२३४", "१६६००१४१५१६"}
    numbers.update(helplines)

    print(f"[CONTACT NUMBERS] Loaded {len(numbers)} unique contact numbers.")
    return numbers

_CONTACT_NUMBERS = extract_contact_numbers()

# ------------------------------------------------------------------
# Text preparation for TTS (digit‑by‑digit only for contact numbers)
# ------------------------------------------------------------------
ENGLISH_TO_NEPALI = {
    "DEOC": "डीईओसी",
    "NEOC": "एनईओसी",
    "DAO": "डीएओ",
    "NDRRMA": "एनडीआरआरएमए",
    "GIS": "जीआईएस",
    "GPS": "जीपीएस",
    "VHF": "भीएचएफ",
    "UHF": "यूएचएफ",
    "SMS": "एसएमएस",
    "AI": "एआई",
}

DIGIT_MAP = {
    "०": "शून्य", "१": "एक", "२": "दुई", "३": "तीन", "४": "चार",
    "५": "पाँच", "६": "छ", "७": "सात", "८": "आठ", "९": "नौ",
    "0": "शून्य", "1": "एक", "2": "दुई", "3": "तीन", "4": "चार",
    "5": "पाँच", "6": "छ", "7": "सात", "8": "आठ", "9": "नौ",
}

def expand_digits(seq: str) -> str:
    parts = []
    for ch in seq:
        if ch in DIGIT_MAP:
            parts.append(DIGIT_MAP[ch])
        else:
            parts.append(ch)
    return " ".join(parts)

def prepare_tts_text(text: str) -> str:
    if not text:
        return text

    for eng, nep in sorted(ENGLISH_TO_NEPALI.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'(?<![A-Za-z])' + re.escape(eng) + r'(?![A-Za-z])'
        text = re.sub(pattern, nep, text, flags=re.IGNORECASE)

    def replace_number(match):
        num_str = match.group(0)
        if '-' in num_str or '–' in num_str or '—' in num_str:
            return expand_digits(num_str)
        if num_str in _CONTACT_NUMBERS:
            return expand_digits(num_str)
        return num_str

    digit_pattern = r'(?<![०-९0-9])[०-९0-9]+(?![०-९0-9])'
    text = re.sub(digit_pattern, replace_number, text)

    hyphen_pattern = r'[०-९0-9]+[-–—][०-९0-9]+'
    text = re.sub(hyphen_pattern, lambda m: expand_digits(m.group(0)), text)

    text = re.sub(r'\s+', ' ', text)
    return text

# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------
async def generate_greeting_mp3():
    filepath = os.path.join("static", "greeting.mp3")
    if os.path.exists(filepath):
        return
    print("Generating greeting MP3...")
    try:
        communicate = edge_tts.Communicate(
            "रसुवा, नुवाकोट, धादिङ र चितवनका बाढी प्रभावितहरूको लागि आपतकालीन सहायता सेवामा स्वागत छ। कृपया आफ्नो समस्या वा प्रश्न भन्नुहोस्।",
            "ne-NP-HemkalaNeural"
        )
        await communicate.save(filepath)
        print("Greeting MP3 saved.")
    except Exception as e:
        print(f"Greeting generation failed: {e}")

async def generate_goodbye_mp3():
    filepath = os.path.join("static", "goodbye.mp3")
    if os.path.exists(filepath):
        return
    try:
        communicate = edge_tts.Communicate(
            "धन्यवाद। फेरि भेटौँला।",
            "ne-NP-HemkalaNeural"
        )
        await communicate.save(filepath)
        print("Goodbye MP3 saved.")
    except Exception as e:
        print(f"Goodbye generation failed: {e}")

async def clean_old_mp3s():
    while True:
        await asyncio.sleep(3600 * 6)
        now = time.time()
        for f in os.listdir("static"):
            if f.startswith("answer_") and f.endswith(".mp3"):
                path = os.path.join("static", f)
                try:
                    if os.path.getmtime(path) < now - 3600:
                        os.remove(path)
                        print(f"Cleaned old MP3: {f}")
                except Exception as e:
                    print(f"Cleanup error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up RAG engine...")
    await asyncio.to_thread(init_pipeline)
    print("RAG engine ready.")

    asyncio.create_task(generate_greeting_mp3())
    asyncio.create_task(generate_goodbye_mp3())
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
    prepared_text = prepare_tts_text(text)
    print(f"[TTS PREPARED] {prepared_text}")
    filename = f"answer_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("static", filename)
    communicate = edge_tts.Communicate(prepared_text, "ne-NP-HemkalaNeural")
    await communicate.save(filepath)
    return filepath

async def delete_mp3_after_delay(filepath: str, delay_seconds: int = 300):
    await asyncio.sleep(delay_seconds)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted MP3: {filepath}")
    except Exception as e:
        print(f"Deletion error: {e}")

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.api_route("/voice", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    response = VoiceResponse()
    if os.path.exists("static/greeting.mp3"):
        response.play(GREETING_MP3)
    else:
        response.say("रसुवा, नुवाकोट, धादिङ र चितवनका बाढी प्रभावितहरूको लागि आपतकालीन सहायता सेवामा स्वागत छ। कृपया आफ्नो समस्या वा प्रश्न भन्नुहोस्।")
    response.redirect(LISTEN_URL, method="POST")
    return Response(content=str(response), media_type="application/xml")

@app.api_route("/listen", methods=["GET", "POST"])
async def listen_for_speech(request: Request):
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
    return Response(content=str(response), media_type="application/xml")

@app.api_route("/process_speech", methods=["GET", "POST"])
async def process_speech(request: Request, SpeechResult: str = Form(None), Digits: str = Form(None)):
    form = await request.form()
    call_sid = form.get("CallSid")
    user_input = SpeechResult or Digits
    print(f"[INPUT] {user_input}")

    response = VoiceResponse()
    if not user_input:
        response.redirect(LISTEN_URL, method="POST")
        return Response(content=str(response), media_type="application/xml")

    # Goodbye check (exact short phrases only)
    normalized_input = user_input.strip().lower()
    goodbye_phrases = {"bye", "goodbye", "बिदा", "बाइ", "धन्यवाद", "फेरि भेटौँला"}
    if normalized_input in goodbye_phrases and len(normalized_input) <= 10:
        print("[EXIT] Goodbye triggered.")
        if os.path.exists("static/goodbye.mp3"):
            response.play(GOODBYE_MP3)
        else:
            response.say("धन्यवाद। फेरि भेटौँला।")
        response.hangup()
        return Response(content=str(response), media_type="application/xml")

    # Get answer (single string with follow‑up already appended)
    answer_text = await asyncio.to_thread(answer_user_query, user_input, call_sid)
    print(f"[ANSWER] {answer_text}")

    # Synthesize and play the combined answer + follow‑up as one audio
    try:
        filepath = await synthesize_nepali_speech(answer_text)
        filename = os.path.basename(filepath)
        mp3_url = f"{BASE_URL}/static/{filename}"
        response.play(mp3_url)
        asyncio.create_task(delete_mp3_after_delay(filepath, 300))
    except Exception as e:
        print(f"TTS error: {e}, falling back to <Say>")
        response.say(answer_text)

    # Continue listening
    response.redirect(LISTEN_URL, method="POST")
    return Response(content=str(response), media_type="application/xml")

@app.get("/")
async def root():
    return {"status": "online", "message": "Nepal Flood Emergency Voice Assistant"}

if __name__ == "__main__":
    uvicorn.run("main_rasuwa:app", host="0.0.0.0", port=8000, reload=True)