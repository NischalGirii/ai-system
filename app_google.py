# app_google.py (Simplified)
import os
import time
import streamlit as st
import live_voice_agent as live

# Configuration
USE_LIVE = True 

st.set_page_config(page_title="विपद् व्यवस्थापन सेवा", page_icon="📘", layout="centered")

# Initialize Session State
if "app_state" not in st.session_state:
    st.session_state.app_state = "IDLE"

def set_state(state):
    st.session_state.app_state = state
    st.rerun()

# UI CSS (Keep your existing styles here...)
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

st.markdown('<div class="service-title">विपद् व्यवस्थापन जानकारी सेवा</div>', unsafe_allow_html=True)

state = st.session_state.app_state

if state == "IDLE":
    st.markdown('<div class="state-icon">📘</div><div class="state-title">कल गर्न तयार</div>', unsafe_allow_html=True)
    if st.button("☎️  कल सुरु गर्नुहोस्", use_container_width=True):
        if live.start_live_session():
            set_state("CONNECTING")
            # We trigger the greeting after a short delay to ensure connection is up
            time.sleep(1)
            live.send_instruction("नमस्ते! तपाईं विपद् व्यवस्थापन जानकारी सेवामा हुनुहुन्छ। म तपाईंलाई कसरी मद्दत गर्न सक्छु?")

elif state == "CONNECTING":
    st.markdown('<div class="state-icon">🔔</div><div class="state-title">जडान हुँदै...</div>', unsafe_allow_html=True)
    time.sleep(2)
    set_state("CONNECTED")

elif state == "CONNECTED":
    st.markdown('<div class="state-icon">🎙️</div><div class="state-title">कल जारी छ</div><div class="state-description">तपाईं बोल्न सक्नुहुन्छ...</div>', unsafe_allow_html=True)
    st.markdown('<div class="waveform">' + '<div class="bar"></div>'*9 + '</div>', unsafe_allow_html=True)

    if st.button("🔴  कल समाप्त गर्नुहोस्", use_container_width=True):
        live.stop_live_session()
        set_state("CALL_ENDED")
    
    time.sleep(0.5)
    st.rerun()

elif state == "CALL_ENDED":
    st.markdown('<div class="state-icon">📴</div><div class="state-title">कल समाप्त भयो</div>', unsafe_allow_html=True)
    if st.button("☎️  नयाँ कल", use_container_width=True):
        set_state("IDLE")