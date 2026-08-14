# app.py
import os
import warnings

# Suppress HuggingFace internal logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import time
import io
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
import pygame
from src.rag.rag_engine import answer_user_query

# Initialize Pygame Mixer for local audio playback
pygame.mixer.init()

# Page Configuration
st.set_page_config(page_title="RAG Voice-to-Voice AI", page_icon="🎙️", layout="centered")

st.title("🎙️ RAG Voice-to-Voice AI Assistant")
st.markdown("Interact with your local Nepali PDF documents via a hands-free voice call.")

# ==========================================
# AUDIO HELPER FUNCTIONS (BACKEND)
# ==========================================
def play_local_audio(text: str):
    """Generates TTS and plays it directly through system speakers, blocking until finished."""
    print("--- [TTS START] Converting text response to voice... ---")
    try:
        tts = gTTS(text=text, lang="ne")
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        # Play audio using pygame
        pygame.mixer.music.load(audio_fp)
        pygame.mixer.music.play()
        
        # Block the script until the audio finishes playing so the mic doesn't hear itself
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        print(f"--- [TTS ERROR] {e} ---")

def listen_continuously() -> str:
    """Listens directly to the local microphone without UI interaction."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # Briefly adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("\n[MIC ACTIVE] Listening for your voice...")
        
        try:
            # Listen with a timeout so it doesn't hang forever if there's silence
            audio_data = recognizer.listen(source, timeout=5.0, phrase_time_limit=15.0)
            print("[STT START] Processing speech...")
            text = recognizer.recognize_google(audio_data, language="ne-NP")
            print(f"[STT COMPLETE] You said: '{text}'")
            return text.strip()
            
        except sr.WaitTimeoutError:
            print("[MIC WARNING] No speech detected (Timeout).")
            return ""
        except sr.UnknownValueError:
            print("[STT WARNING] Speech unrecognized or silent.")
            return ""
        except Exception as e:
            print(f"[STT ERROR] {e}")
            return ""

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "call_active" not in st.session_state:
    st.session_state.call_active = False

# ==========================================
# SIDEBAR: PHONE CONNECTION CONTROLS
# ==========================================
st.sidebar.header("📞 Call Controls")
phone_number = st.sidebar.text_input("Enter Phone Number:", value="9800000000")

if not st.session_state.call_active:
    if st.sidebar.button("Call", type="primary"):
        st.session_state.call_active = True
        st.rerun()
else:
    if st.sidebar.button("End Call", type="secondary"):
        st.session_state.call_active = False
        pygame.mixer.music.stop()
        st.rerun()
    st.sidebar.success(f"Connected to: {phone_number}")

# ==========================================
# MAIN INTERFACE: HANDS-FREE CALL LOOP
# ==========================================
if not st.session_state.call_active:
    st.info("👈 Enter a phone number and click **'Call'** in the sidebar to begin the hands-free session.")
else:
    st.markdown("### 📞 Call in Progress...")
    st.markdown("### 🟢 Speak freely into your microphone.")
    
    # Placeholder to show real-time status on the UI
    status_text = st.empty()
    
    # 1. Play Greeting on first connect
    if "greeted" not in st.session_state:
        status_text.info("🔊 Assistant is speaking...")
        play_local_audio("नमस्कार! म तपाईंलाई कसरी सहयोग गर्न सक्छु ?")
        st.session_state.greeted = True
        
    # 2. Continuous Hands-Free Loop
    status_text.success("🎙️ Listening... (Speak now)")
    user_query = listen_continuously()
    
    if user_query:
        status_text.warning(f"**You:** {user_query}\n\n🧠 *Thinking...*")
        
        # Query RAG Pipeline
        ai_response = answer_user_query(user_query, model_name="llama3:latest")
        
        # Play the answer
        status_text.info(f"🔊 Assistant is speaking...\n\n**AI:** {ai_response}")
        play_local_audio(ai_response)
        
    # Force the app to re-run immediately to start listening again
    time.sleep(0.5) 
    st.rerun()