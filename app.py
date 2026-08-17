# app.py
import os
import warnings
import time
import io
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
import pygame
from src.rag.rag_engine import (
    answer_user_query,
    load_rag_pipeline,
    is_exit_intent,
)

# Suppress logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Nepali Voice RAG Agent", 
    page_icon="🎙️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# AUDIO HELPER FUNCTIONS
# ==========================================
def ensure_mixer_initialized():
    """Safely initializes Pygame audio mixer if stopped."""
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def play_ringtone():
    """Starts looped ringtone in a background audio thread."""
    try:
        ensure_mixer_initialized()
        ringtone_path = "ringtone.mp3"
        if not os.path.exists(ringtone_path):
            ringtone_path = "assets/ringtone.mp3"

        if os.path.exists(ringtone_path):
            pygame.mixer.music.load(ringtone_path)
            pygame.mixer.music.play(-1)
        else:
            print("[AUDIO WARNING] Ringtone file not found")
    except Exception as e:
        print(f"[AUDIO ERROR] Could not play ringtone: {e}")

def stop_ringtone():
    """Stops ringtone playback."""
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as e:
        print(f"[AUDIO ERROR] Stopping ringtone: {e}")

def play_local_audio(text: str):
    """Converts response to TTS and speaks via system speakers."""
    print("--- [TTS START] Synthesizing speech... ---")
    try:
        ensure_mixer_initialized()
        tts = gTTS(text=text, lang="ne")
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        pygame.mixer.music.load(audio_fp)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(15)
            
    except Exception as e:
        print(f"--- [TTS ERROR] {e} ---")

def listen_continuously() -> str:
    """Listens directly to the local microphone with expanded limits to prevent cut-offs."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("\n[MIC ACTIVE] Listening for spoken user input...")
        
        try:
            # Expanded timeout and phrase limit so sentences don't get clipped
            audio_data = recognizer.listen(source, timeout=7.0, phrase_time_limit=15.0)
            print("[STT START] Processing audio with SpeechRecognition...")
            text = recognizer.recognize_google(audio_data, language="ne-NP")
            print(f"[STT COMPLETE] Detected Text: '{text}'")
            return text.strip()
            
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return ""
        except Exception as e:
            print(f"[STT ERROR] {e}")
            return ""

# ==========================================
# MODERN DARK UI (CSS INJECTION)
# ==========================================
st.markdown("""
<style>
    .stApp {
        background: #0d0f1b;
        color: #ffffff;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .header-box {
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }

    .header-subtitle {
        color: #8a8f98;
        font-size: 14px;
        margin-top: 0px;
    }

    .pill-container {
        display: flex;
        justify-content: center;
        margin-bottom: 25px;
    }

    .status-pill-connecting {
        background: rgba(255, 193, 7, 0.12);
        color: #ffca28;
        border: 1px solid rgba(255, 202, 40, 0.4);
        padding: 6px 20px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .status-pill-connected {
        background: rgba(46, 204, 113, 0.12);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.4);
        padding: 6px 20px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    @keyframes pulse-ring-gold {
        0% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.6); }
        70% { box-shadow: 0 0 0 35px rgba(255, 193, 7, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0); }
    }
    
    @keyframes pulse-ring-green {
        0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.6); }
        70% { box-shadow: 0 0 0 35px rgba(46, 204, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }

    .circle-card {
        width: 190px;
        height: 190px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 20px auto 35px auto;
        border: 3px dashed rgba(85, 214, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .gold-theme {
        background: radial-gradient(circle, #b8860b 0%, #4a3500 100%);
        animation: pulse-ring-gold 2s infinite;
    }
    
    .green-theme {
        background: radial-gradient(circle, #27ae60 0%, #114b29 100%);
        animation: pulse-ring-green 2s infinite;
    }
    
    .circle-card h1 { margin: 0; font-size: 46px; }
    .circle-card p { margin: 8px 0 0 0; font-size: 12px; font-weight: 700; letter-spacing: 2px; color: #ffffff; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# THREE-PHASE STATE ENGINE
# ==========================================
if "app_state" not in st.session_state:
    st.session_state.app_state = "START_RING"

# RENDER HEADER
st.markdown("""
<div class="header-box">
    <div class="header-title">Nepali Voice RAG Agent</div>
    <div class="header-subtitle">Real-time hands-free voice assistant (Groq Powered)</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------
# PHASE 1: IMMEDIATE RINGTONE TRIGGER
# ------------------------------------------
if st.session_state.app_state == "START_RING":
    st.markdown("""
        <div class="pill-container">
            <div class="status-pill-connecting">🟡 Loading AI Agent & Connecting...</div>
        </div>
        <div class="circle-card gold-theme">
            <h1>🔔</h1>
            <p>CONNECTING</p>
        </div>
    """, unsafe_allow_html=True)
    
    play_ringtone()
    time.sleep(0.2)
    st.session_state.app_state = "LOADING_MODEL"
    st.rerun()

# ------------------------------------------
# PHASE 2: MODEL LOADING WHILE RINGING
# ------------------------------------------
elif st.session_state.app_state == "LOADING_MODEL":
    st.markdown("""
        <div class="pill-container">
            <div class="status-pill-connecting">🟡 Loading AI Agent & Connecting...</div>
        </div>
        <div class="circle-card gold-theme">
            <h1>🔔</h1>
            <p>CONNECTING</p>
        </div>
    """, unsafe_allow_html=True)
    
    load_rag_pipeline()
    stop_ringtone()
    
    st.session_state.app_state = "CONNECTED"
    st.session_state.needs_greeting = True
    st.rerun()

# ------------------------------------------
# PHASE 3: ACTIVE HANDS-FREE VOICE CALL LOOP
# ------------------------------------------
elif st.session_state.app_state == "CONNECTED":
    
    ui_placeholder = st.empty()
    with ui_placeholder.container():
        st.markdown("""
            <div class="pill-container">
                <div class="status-pill-connected">🟢 Connected — AI Voice Agent</div>
            </div>
            <div class="circle-card green-theme">
                <h1>📞</h1>
                <p>LISTENING</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔴 Hang Up / Restart Call", use_container_width=True):
                stop_ringtone()
                st.session_state.app_state = "START_RING"
                st.session_state.needs_greeting = False
                st.rerun()

    if st.session_state.get("needs_greeting", False):
        st.session_state.needs_greeting = False
        play_local_audio("नमस्ते! कनेक्ट भयो, म तपाईंलाई कसरी सहयोग गर्न सक्छु?")
        st.rerun()
    else:
        user_query = listen_continuously()

        if user_query:
            st.toast(f"🗣️ You said: {user_query}")

    # =====================================================
    # CHECK FOR CALL-END INTENT FIRST
    # =====================================================

    if is_exit_intent(user_query):

        print(
            f"[CALL END] User requested to end conversation: "
            f"{user_query!r}"
        )

        # Goodbye response
        goodbye_response = answer_user_query(
            user_query
        )

        # Speak goodbye
        play_local_audio(
            goodbye_response
        )

        # Stop any active audio
        stop_ringtone()

        # Change state so the listening loop stops
        st.session_state.app_state = "CALL_ENDED"

        st.rerun()

    # =====================================================
    # NORMAL QUERY
    # =====================================================

    with ui_placeholder.container():
        st.markdown("""
            <div class="pill-container">
                <div class="status-pill-connecting">
                    🟡 Thinking / Processing Query...
                </div>
            </div>

            <div class="circle-card gold-theme">
                <h1>🤖</h1>
                <p>THINKING</p>
            </div>
        """, unsafe_allow_html=True)

    # Query RAG + Groq
    ai_response = answer_user_query(
        user_query
    )

    # Speak response
    play_local_audio(
        ai_response
    )

    st.rerun()