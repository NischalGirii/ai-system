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
from src.rag.rag_engine import answer_user_query

# Page Configuration
st.set_page_config(page_title="RAG Voice-to-Voice AI", page_icon="🎙️", layout="centered")

st.title("🎙️ RAG Voice-to-Voice AI Assistant")
st.markdown("Interact with your local Nepali PDF documents using Hybrid RAG and local LLMs.")


# ==========================================
# AUDIO HELPER FUNCTIONS (STT & TTS)
# ==========================================
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribes audio with Google STT and timeout handling."""
    stt_start = time.time()
    print("\n--- [STT START] Transcribing spoken voice... ---")
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)

        # Transcribe via Google Speech API
        text = recognizer.recognize_google(audio_data, language="ne-NP")
        print(f"--- [STT COMPLETE] Transcribed in {time.time() - stt_start:.2f}s: '{text}' ---")
        return text.strip()

    except sr.UnknownValueError:
        print("--- [STT WARNING] Speech unrecognized or silent ---")
        return ""
    except Exception as e:
        print(f"--- [STT ERROR] {e} ---")
        st.error(f"Audio transcription error: {e}")
        return ""


def text_to_audio_bytes(text: str) -> io.BytesIO:
    """Converts text to audio using gTTS with timing log."""
    tts_start = time.time()
    print("--- [TTS START] Converting text response to voice... ---")
    try:
        tts = gTTS(text=text, lang="ne")
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        print(f"--- [TTS COMPLETE] Generated in {time.time() - tts_start:.2f}s ---")
        return audio_fp
    except Exception as e:
        print(f"--- [TTS ERROR] {e} ---")
        return io.BytesIO()


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "call_active" not in st.session_state:
    st.session_state.call_active = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

# ==========================================
# SIDEBAR: PHONE CONNECTION CONTROLS
# ==========================================
st.sidebar.header("📞 Call Controls")
phone_number = st.sidebar.text_input("Enter Phone Number:", value="9800000000")

if not st.session_state.call_active:
    if st.sidebar.button("Call", type="primary"):
        with st.spinner("Connecting..."):
            time.sleep(0.5)
        st.session_state.call_active = True
        
        greeting_text = "नमस्कार! म तपाईंलाई कसरी सहयोग गर्न सक्छु ?"
        st.session_state.messages.append({"role": "assistant", "content": greeting_text})
        st.rerun()
else:
    if st.sidebar.button("End Call", type="secondary"):
        st.session_state.call_active = False
        st.session_state.messages = []
        st.session_state.last_audio_id = None
        st.rerun()
    st.sidebar.success(f"Connected to: {phone_number}")

# ==========================================
# MAIN INTERFACE: CALL & CHAT SIMULATION
# ==========================================
if not st.session_state.call_active:
    st.info("👈 Enter a phone number and click **'Call'** in the sidebar to begin the simulation.")
else:
    st.markdown("---")
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

    # 1. INPUT WIDGETS
    st.markdown("### 🎙️ Speak your question")
    recorded_audio = st.audio_input("Record your voice query")
    typed_query = st.chat_input("Or type your question here...")

    user_query = ""

    # Priority 1: Handle Text Input
    if typed_query:
        user_query = typed_query

    # Priority 2: Handle Voice Input
    elif recorded_audio:
        audio_bytes = recorded_audio.read()
        audio_id = hash(audio_bytes)

        if st.session_state.last_audio_id != audio_id:
            st.session_state.last_audio_id = audio_id
            with st.spinner("Transcribing your speech..."):
                user_query = transcribe_audio_bytes(audio_bytes)
                
            if not user_query:
                st.warning("आवाज स्पष्ट भएन, कृपया फेरि बोल्नुहोस्। (Audio not clear, please speak again.)")

    # 2. RAG PIPELINE & VOICE OUTPUT GENERATION
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 *Searching documents & generating voice response...*")
            
            # Query hybrid RAG engine using qwen3:4b for fast response
            ai_response = answer_user_query(user_query, model_name="llama3:latest")
            
            # Synthesize audio response
            audio_response = text_to_audio_bytes(ai_response)
            
            message_placeholder.markdown(ai_response)
            st.audio(audio_response, format="audio/mp3", autoplay=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response,
                "audio": audio_response
            })
            st.rerun()