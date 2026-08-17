# =========================================================
# app.py
# Google STT + RAG + Groq GPT-OSS 20B + Edge TTS
# =========================================================

import os
import time
import asyncio
import tempfile
import warnings

import pygame
import speech_recognition as sr
import streamlit as st
import edge_tts

from src.rag.rag_engine import (
    answer_user_query,
    is_exit_intent,
    load_rag_pipeline,
)


# =========================================================
# CONFIGURATION
# =========================================================

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Nepali Information Service",
    page_icon="☎️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =========================================================
# EDGE TTS
# =========================================================

TTS_VOICE = "ne-NP-SagarNeural"

TTS_RATE = "+0%"
TTS_VOLUME = "+0%"
TTS_PITCH = "+0Hz"


# =========================================================
# SESSION STATE
# =========================================================

if "app_state" not in st.session_state:
    st.session_state.app_state = "IDLE"

if "needs_greeting" not in st.session_state:
    st.session_state.needs_greeting = False

if "current_person" not in st.session_state:
    st.session_state.current_person = None


# =========================================================
# AUDIO
# =========================================================

def ensure_audio():

    try:

        if not pygame.mixer.get_init():

            pygame.mixer.init(
                frequency=44100,
                size=-16,
                channels=2,
                buffer=512,
            )

    except Exception as exc:

        print(
            f"[AUDIO ERROR] "
            f"{exc}"
        )


def play_ringtone():

    try:

        ensure_audio()

        path = "ringtone.mp3"

        if not os.path.exists(path):

            path = os.path.join(
                "assets",
                "ringtone.mp3",
            )

        if not os.path.exists(path):

            print(
                "[RINGTONE] File not found."
            )

            return

        pygame.mixer.music.load(
            path
        )

        pygame.mixer.music.play(
            loops=-1
        )

    except Exception as exc:

        print(
            f"[RINGTONE ERROR] "
            f"{exc}"
        )


def stop_audio():

    try:

        if pygame.mixer.get_init():

            pygame.mixer.music.stop()

    except Exception as exc:

        print(
            f"[AUDIO ERROR] "
            f"{exc}"
        )


# =========================================================
# EDGE TTS
# =========================================================

async def generate_tts(
    text: str,
    output_path: str,
):

    communicate = edge_tts.Communicate(
        text=text,
        voice=TTS_VOICE,
        rate=TTS_RATE,
        volume=TTS_VOLUME,
        pitch=TTS_PITCH,
    )

    await communicate.save(
        output_path
    )


def create_tts_audio(
    text: str,
):

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False,
    )

    temp_file.close()

    output_path = temp_file.name

    try:

        asyncio.run(
            generate_tts(
                text,
                output_path,
            )
        )

        return output_path

    except Exception as exc:

        print(
            f"[TTS ERROR] "
            f"{exc}"
        )

        try:

            os.remove(
                output_path
            )

        except OSError:
            pass

        return None


def speak(
    text: str,
):

    if text is None:
        return

    text = str(text).strip()

    if not text:

        print(
            "[TTS] Empty response ignored."
        )

        return

    ensure_audio()

    audio_path = None

    try:

        start = time.perf_counter()

        print(
            f"[TTS] Voice: "
            f"{TTS_VOICE}"
        )

        print(
            f"[TTS] Text: "
            f"{text}"
        )

        audio_path = create_tts_audio(
            text
        )

        if not audio_path:

            return

        print(
            f"[TTS] Generated in "
            f"{time.perf_counter() - start:.3f}s"
        )

        pygame.mixer.music.stop()

        pygame.mixer.music.load(
            audio_path
        )

        pygame.mixer.music.play()

        clock = pygame.time.Clock()

        while pygame.mixer.music.get_busy():

            clock.tick(20)

        print(
            "[TTS] Playback complete."
        )

    except Exception as exc:

        print(
            f"[TTS PLAYBACK ERROR] "
            f"{exc}"
        )

    finally:

        if audio_path:

            try:

                if os.path.exists(
                    audio_path
                ):

                    os.remove(
                        audio_path
                    )

            except OSError:
                pass


# =========================================================
# GOOGLE STT
# =========================================================

IGNORED_UTTERANCES = {
    "uh",
    "um",
    "hmm",
    "hm",
    "mm",
    "mmm",
    "हम्म",
    "ह्म्म",
    "अँ",
    "ए",
    "ओ",
    "आ",
    "उम्",
    "हूँ",
    "हुम्",
}


def listen():

    recognizer = sr.Recognizer()

    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.5

    try:

        with sr.Microphone() as source:

            print(
                "[MIC] Listening..."
            )

            recognizer.adjust_for_ambient_noise(
                source,
                duration=0.3,
            )

            try:

                audio = recognizer.listen(
                    source,
                    timeout=7.0,
                    phrase_time_limit=15.0,
                )

            except sr.WaitTimeoutError:

                print(
                    "[MIC] Silence / timeout. "
                    "No request sent."
                )

                return ""

    except Exception as exc:

        print(
            f"[MIC ERROR] "
            f"{exc}"
        )

        return ""

    try:

        start = time.perf_counter()

        print(
            "[STT] Processing..."
        )

        text = recognizer.recognize_google(
            audio,
            language="ne-NP",
        )

        if text is None:

            return ""

        text = text.strip()

        if not text:

            print(
                "[STT] Empty transcription. "
                "No request sent."
            )

            return ""

        if len(text) < 2:

            print(
                f"[STT] Too short: "
                f"{text!r}"
            )

            return ""

        if text.lower() in IGNORED_UTTERANCES:

            print(
                f"[STT] Noise ignored: "
                f"{text!r}"
            )

            return ""

        print(
            f"[STT] Valid: "
            f"{text!r}"
        )

        print(
            f"[STT] Time: "
            f"{time.perf_counter() - start:.3f}s"
        )

        return text

    except sr.UnknownValueError:

        print(
            "[STT] Speech not understood."
        )

        return ""

    except sr.RequestError as exc:

        print(
            f"[STT ERROR] "
            f"{exc}"
        )

        return ""

    except Exception as exc:

        print(
            f"[STT ERROR] "
            f"{exc}"
        )

        return ""


# =========================================================
# UI
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background: #0b0d12;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    max-width: 620px;
    padding-top: 40px;
}

.service-title {
    text-align: center;
    color: #f5f5f7;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.8px;
    margin-bottom: 8px;
}

.service-subtitle {
    text-align: center;
    color: #777e8a;
    font-size: 13px;
    margin-bottom: 40px;
}

.call-status {
    text-align: center;
    color: #61d68e;
    font-size: 13px;
    font-weight: 600;
    margin-top: 12px;
    margin-bottom: 20px;
}

.call-status.gold {
    color: #dfb85e;
}

.call-status.red {
    color: #df6871;
}

.state-icon {
    text-align: center;
    font-size: 64px;
    line-height: 1;
    margin: 30px 0 15px 0;
}

.state-title {
    text-align: center;
    color: #f1f2f5;
    font-size: 25px;
    font-weight: 650;
    margin-bottom: 7px;
}

.state-description {
    text-align: center;
    color: #7a818d;
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 35px;
}

.stButton > button {
    min-height: 54px;
    border-radius: 15px;
    font-weight: 600;
}

.start-call button {
    background: #f0f1f4 !important;
    color: #0b0d12 !important;
    border: none !important;
}

.end-call button {
    background: #c93f49 !important;
    color: white !important;
    border: none !important;
}

.end-call button:hover {
    background: #d34a54 !important;
}

.waveform {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 4px;
    height: 90px;
    margin: 10px 0 30px 0;
}

.bar {
    width: 4px;
    border-radius: 10px;
    background: #9da3af;
    animation: wave 1s ease-in-out infinite;
}

.bar:nth-child(1) {
    height: 20px;
}

.bar:nth-child(2) {
    height: 35px;
    animation-delay: .1s;
}

.bar:nth-child(3) {
    height: 50px;
    animation-delay: .2s;
}

.bar:nth-child(4) {
    height: 68px;
    animation-delay: .3s;
}

.bar:nth-child(5) {
    height: 82px;
    animation-delay: .4s;
}

.bar:nth-child(6) {
    height: 60px;
    animation-delay: .3s;
}

.bar:nth-child(7) {
    height: 74px;
    animation-delay: .2s;
}

.bar:nth-child(8) {
    height: 45px;
    animation-delay: .1s;
}

.bar:nth-child(9) {
    height: 28px;
}

@keyframes wave {

    0%,
    100% {
        transform: scaleY(.45);
        opacity: .35;
    }

    50% {
        transform: scaleY(1);
        opacity: .9;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="service-title">'
    'Nepali Information Service'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="service-subtitle">'
    'Automated telephone information service'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# IDLE
# =========================================================

if st.session_state.app_state == "IDLE":

    st.markdown(
        '<div class="state-icon">☎️</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="state-title">'
        'Ready to call'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="state-description">'
        'Ask about people and information available '
        'in the connected documents.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1, 1.5, 1]
    )

    with col2:

        st.markdown(
            '<div class="start-call">',
            unsafe_allow_html=True,
        )

        start_call = st.button(
            "☎️  Start Call",
            use_container_width=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    if start_call:

        st.session_state.app_state = (
            "CONNECTING"
        )

        st.session_state.needs_greeting = True

        st.session_state.current_person = None

        st.rerun()


# =========================================================
# CONNECTING
# =========================================================

elif st.session_state.app_state == "CONNECTING":

    st.markdown(
        '<div class="state-icon">🔔</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="state-title">'
        'Connecting'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="call-status gold">'
        'Please wait...'
        '</div>',
        unsafe_allow_html=True,
    )

    play_ringtone()

    time.sleep(0.3)

    try:

        load_rag_pipeline()

    except Exception as exc:

        print(
            f"[PIPELINE ERROR] "
            f"{exc}"
        )

    finally:

        stop_audio()

    st.session_state.app_state = (
        "CONNECTED"
    )

    st.session_state.needs_greeting = True

    st.rerun()


# =========================================================
# CONNECTED
# =========================================================

elif st.session_state.app_state == "CONNECTED":

    # -----------------------------------------------------
    # GREETING
    # -----------------------------------------------------

    if st.session_state.needs_greeting:

        st.session_state.needs_greeting = False

        st.markdown(
            '<div class="state-icon">☎️</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="state-title">'
            'Connected'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="call-status">'
            'Call in progress'
            '</div>',
            unsafe_allow_html=True,
        )

        speak(
            "नमस्ते! स्वचालित सूचना सेवामा स्वागत छ। "
            "तपाईं के जानकारी चाहनुहुन्छ?"
        )

        st.rerun()

    # -----------------------------------------------------
    # LISTENING
    # -----------------------------------------------------

    st.markdown(
        '<div class="state-icon">🎙️</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="state-title">'
        'Listening'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="state-description">'
        'Speak naturally'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="waveform">
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # END CALL
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(
        [1, 1.4, 1]
    )

    with col2:

        st.markdown(
            '<div class="end-call">',
            unsafe_allow_html=True,
        )

        end_call = st.button(
            "🔴  End Call",
            use_container_width=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    if end_call:

        stop_audio()

        st.session_state.app_state = (
            "CALL_ENDED"
        )

        st.session_state.needs_greeting = False

        st.session_state.current_person = None

        st.rerun()

    # -----------------------------------------------------
    # LISTEN
    # -----------------------------------------------------

    user_query = listen()

    # Empty/noisy input never reaches RAG.
    if not user_query:

        print(
            "[CALL] No valid text. "
            "Skipping RAG/Groq."
        )

        time.sleep(0.05)

        st.rerun()

    print(
        f"[CALL] Query: "
        f"{user_query!r}"
    )

    # -----------------------------------------------------
    # EXIT
    # -----------------------------------------------------

    if is_exit_intent(
        user_query
    ):

        st.markdown(
            '<div class="state-icon">👋</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="state-title">'
            'Ending call'
            '</div>',
            unsafe_allow_html=True,
        )

        response = answer_user_query(
            user_query
        )

        if response:

            speak(
                response
            )

        stop_audio()

        st.session_state.app_state = (
            "CALL_ENDED"
        )

        st.session_state.needs_greeting = False

        st.session_state.current_person = None

        st.rerun()

    # -----------------------------------------------------
    # THINKING
    # -----------------------------------------------------

    st.markdown(
        '<div class="state-icon">◌</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="state-title">'
        'One moment'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="call-status gold">'
        'Finding the information'
        '</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # RAG + GROQ
    # -----------------------------------------------------

    start = time.perf_counter()

    try:

        response = answer_user_query(
            user_query
        )

    except Exception as exc:

        print(
            f"[RAG ERROR] "
            f"{exc}"
        )

        response = (
            "माफ गर्नुहोस्, अहिले "
            "जानकारी प्राप्त गर्न समस्या भयो।"
        )

    print(
        f"[RAG] Time: "
        f"{time.perf_counter() - start:.3f}s"
    )

    # -----------------------------------------------------
    # SPEAK
    # -----------------------------------------------------

    if response:

        st.markdown(
            '<div class="state-icon">🔊</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="state-title">'
            'Speaking'
            '</div>',
            unsafe_allow_html=True,
        )

        speak(
            response
        )

    else:

        print(
            "[RAG] Empty response. "
            "TTS skipped."
        )

    # -----------------------------------------------------
    # LISTEN AGAIN
    # -----------------------------------------------------

    st.rerun()


# =========================================================
# CALL ENDED
# =========================================================

elif st.session_state.app_state == "CALL_ENDED":

    st.markdown(
        '<div class="state-icon">📴</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="state-title">'
        'Call ended'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="state-description">'
        'धन्यवाद। फेरि भेटौँला।'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1, 1.5, 1]
    )

    with col2:

        st.markdown(
            '<div class="start-call">',
            unsafe_allow_html=True,
        )

        new_call = st.button(
            "☎️  Start New Call",
            use_container_width=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    if new_call:

        st.session_state.app_state = (
            "CONNECTING"
        )

        st.session_state.needs_greeting = True

        st.session_state.current_person = None

        st.rerun()