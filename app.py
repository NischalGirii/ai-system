import os, time, asyncio, tempfile, warnings, pygame, re
import speech_recognition as sr
import streamlit as st
import edge_tts

from src.rag.rag_engine import answer_user_query, is_exit_intent, load_rag_pipeline

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# Pronunciation Dictionary for English Tech Terms
PRONUNCIATION_FIXES = {
    "JavaScript": "जाभास्क्रिप्ट",
    "TypeScript": "टाइपस्क्रिप्ट",
    "Node.js": "नोड जेएस",
    "REST API": "रेस्ट एपीआई",
    "API": "एपीआई",
    "React": "रियाक्ट",
    "MongoDB": "मङ्गो डिबी",
    "PostgreSQL": "पोस्टग्रेस एसक्युएल",
    "RAG": "र्याग",
    "UI": "युआई",
    "UX": "युएक्स",
    "App": "एप",
    "Software": "सफ्टवेयर"
}

st.set_page_config(
    page_title="Nepali Information Service", 
    page_icon="☎️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# ----- TTS VOICE (FEMALE NEPALI) -----
# Changed from male "ne-NP-SagarNeural" to female "ne-NP-HemkalaNeural"
TTS_VOICE = "ne-NP-HemkalaNeural"
TTS_RATE, TTS_VOLUME, TTS_PITCH = "+0%", "+0%", "+0Hz"

IGNORED_UTTERANCES = {"uh", "um", "hmm", "hm", "mm", "mmm", "हम्म", "ह्म्म", "अँ", "ए", "ओ", "आ", "उम्", "हूँ", "हुम्"}

# Initialize Session State
for key, val in [("app_state", "IDLE"), ("needs_greeting", False), ("current_person", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

def set_state(state, needs_greeting=False, current_person=None):
    st.session_state.app_state = state
    st.session_state.needs_greeting = needs_greeting
    st.session_state.current_person = current_person
    st.rerun()

# --- Audio Helper Functions (Hardware Audio via Pygame) ---
def ensure_audio():
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    except Exception as exc:
        print(f"[AUDIO ERROR] {exc}")

def play_ringtone():
    ensure_audio()
    path = "ringtone.mp3" if os.path.exists("ringtone.mp3") else os.path.join("assets", "ringtone.mp3")
    if os.path.exists(path):
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play(loops=-1)
        except Exception as exc:
            print(f"[RINGTONE ERROR] {exc}")

def stop_audio():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as exc:
        print(f"[AUDIO ERROR] {exc}")

async def generate_tts(text, output_path):
    comm = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE, volume=TTS_VOLUME, pitch=TTS_PITCH)
    await comm.save(output_path)

def speak(text):
    if not text or not str(text).strip(): return
    text = str(text).strip()
    
    # Improved regex logic to only swap exact word matches (ignoring case)
    for eng_word, nep_phonetic in PRONUNCIATION_FIXES.items():
        text = re.sub(rf"(?i)\b{re.escape(eng_word)}\b", nep_phonetic, text)

    ensure_audio()
    
    # Safe tempfile management
    fd, output_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    try:
        asyncio.run(generate_tts(text, output_path))
        pygame.mixer.music.stop()
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()
        
        # Block thread while speaking to prevent the microphone from hearing the TTS
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(20)
            
    except Exception as exc:
        print(f"[TTS ERROR] {exc}")
    finally:
        # Guarantee cleanup of the temporary file
        pygame.mixer.music.unload()
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError: pass

# --- Hardware Speech-to-Text Listener ---
def listen():
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.8
    r.phrase_threshold = 0.3
    r.non_speaking_duration = 0.5
    
    try:
        with sr.Microphone() as source:
            print("[MIC] Listening...")
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source, timeout=7.0, phrase_time_limit=15.0)
        
        text = (r.recognize_google(audio, language="ne-NP") or "").strip()
        if len(text) >= 2 and text.lower() not in IGNORED_UTTERANCES:
            print(f"[STT] Valid: {text!r}")
            return text
    except (sr.WaitTimeoutError, sr.UnknownValueError):
        print("[STT] Silence / Speech not understood.")
    except Exception as exc:
        print(f"[STT ERROR] {exc}")
    return ""

# --- UI Styling (Original CSS Preserved) ---
st.markdown("""
<style>
.stApp { background: #0b0d12; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 620px; padding-top: 40px; }
.service-title { text-align: center; color: #f5f5f7; font-size: 30px; font-weight: 700; letter-spacing: -0.8px; margin-bottom: 8px; }
.service-subtitle { text-align: center; color: #777e8a; font-size: 13px; margin-bottom: 40px; }
.call-status { text-align: center; color: #61d68e; font-size: 13px; font-weight: 600; margin-top: 12px; margin-bottom: 20px; }
.call-status.gold { color: #dfb85e; }
.state-icon { text-align: center; font-size: 64px; line-height: 1; margin: 30px 0 15px 0; }
.state-title { text-align: center; color: #f1f2f5; font-size: 25px; font-weight: 650; margin-bottom: 7px; }
.state-description { text-align: center; color: #7a818d; font-size: 13px; line-height: 1.6; margin-bottom: 35px; }
.stButton > button { min-height: 54px; border-radius: 15px; font-weight: 600; }
.start-call button { background: #f0f1f4 !important; color: #0b0d12 !important; border: none !important; }
.end-call button { background: #c93f49 !important; color: white !important; border: none !important; }
.end-call button:hover { background: #d34a54 !important; }
.waveform { display: flex; justify-content: center; align-items: center; gap: 4px; height: 90px; margin: 10px 0 30px 0; }
.bar { width: 4px; border-radius: 10px; background: #9da3af; animation: wave 1s ease-in-out infinite; }
.bar:nth-child(1) { height: 20px; } .bar:nth-child(2) { height: 35px; animation-delay: .1s; } .bar:nth-child(3) { height: 50px; animation-delay: .2s; }
.bar:nth-child(4) { height: 68px; animation-delay: .3s; } .bar:nth-child(5) { height: 82px; animation-delay: .4s; } .bar:nth-child(6) { height: 60px; animation-delay: .3s; }
.bar:nth-child(7) { height: 74px; animation-delay: .2s; } .bar:nth-child(8) { height: 45px; animation-delay: .1s; } .bar:nth-child(9) { height: 28px; }
@keyframes wave { 0%, 100% { transform: scaleY(.45); opacity: .35; } 50% { transform: scaleY(1); opacity: .9; } }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="service-title">Nepali Information Service</div><div class="service-subtitle">Automated telephone information service</div>', unsafe_allow_html=True)

state = st.session_state.app_state

# =====================================================================
# STATE MACHINE LOGIC
# =====================================================================

if state == "IDLE":
    st.markdown('<div class="state-icon">☎️</div><div class="state-title">Ready to call</div><div class="state-description">Ask about people and information available in the connected documents.</div>', unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div class="start-call">', unsafe_allow_html=True)
        if st.button("☎️  Start Call", use_container_width=True):
            set_state("CONNECTING", needs_greeting=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif state == "CONNECTING":
    st.markdown('<div class="state-icon">🔔</div><div class="state-title">Connecting</div><div class="call-status gold">Please wait...</div>', unsafe_allow_html=True)
    play_ringtone()
    time.sleep(0.3)
    try: 
        load_rag_pipeline()
    except Exception as exc: 
        print(f"[PIPELINE ERROR] {exc}")
    finally: 
        stop_audio()
    set_state("CONNECTED", needs_greeting=True)

elif state == "CONNECTED":
    if st.session_state.needs_greeting:
        st.session_state.needs_greeting = False
        st.markdown('<div class="state-icon">☎️</div><div class="state-title">Connected</div><div class="call-status">Call in progress</div>', unsafe_allow_html=True)
        speak("नमस्ते! स्वचालित सूचना सेवामा स्वागत छ। तपाईं के जानकारी चाहनुहुन्छ?")
        st.rerun()

    # Original Waveform UI
    st.markdown('<div class="state-icon">🎙️</div><div class="state-title">Listening</div><div class="state-description">Speak naturally</div>', unsafe_allow_html=True)
    st.markdown('<div class="waveform">' + '<div class="bar"></div>'*9 + '</div>', unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<div class="end-call">', unsafe_allow_html=True)
        if st.button("🔴  End Call", use_container_width=True):
            stop_audio()
            set_state("CALL_ENDED")
        st.markdown('</div>', unsafe_allow_html=True)

    # Hardware listener handles audio capture hands-free
    user_query = listen()
    
    if not user_query:
        time.sleep(0.05)
        st.rerun()

    if is_exit_intent(user_query):
        st.markdown('<div class="state-icon">👋</div><div class="state-title">Ending call</div>', unsafe_allow_html=True)
        response = answer_user_query(user_query)
        if response: speak(response)
        stop_audio()
        set_state("CALL_ENDED")

    st.markdown('<div class="state-icon">◌</div><div class="state-title">One moment</div><div class="call-status gold">Finding the information</div>', unsafe_allow_html=True)
    
    try:
        response = answer_user_query(user_query)
    except Exception as exc:
        print(f"[RAG ERROR] {exc}")
        response = "माफ गर्नुहोस्, अहिले जानकारी प्राप्त गर्न समस्या भयो।"

    if response:
        st.markdown('<div class="state-icon">🔊</div><div class="state-title">Speaking</div>', unsafe_allow_html=True)
        speak(response)
    
    st.rerun()

elif state == "CALL_ENDED":
    st.markdown('<div class="state-icon">📴</div><div class="state-title">Call ended</div><div class="state-description">धन्यवाद। फेरि भेटौँला।</div>', unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div class="start-call">', unsafe_allow_html=True)
        if st.button("☎️  Start New Call", use_container_width=True):
            set_state("CONNECTING", needs_greeting=True)
        st.markdown('</div>', unsafe_allow_html=True)