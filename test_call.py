import requests
import time
import pygame
import os
import re
from urllib.parse import urlparse

# Your Cloudflare URL
BASE_URL = "https://apartment-money-encourages-soa.trycloudflare.com"

def play_mp3(url):
    """Download and play an MP3 from a URL using pygame."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print("Failed to fetch MP3")
        return
    # Save to a temp file
    filename = "temp_answer.mp3"
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    # Play
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.quit()
    os.remove(filename)

def ask_question(question):
    """Send a question to /process_speech and return the MP3 URL from TwiML."""
    payload = {"SpeechResult": question, "Digits": ""}
    resp = requests.post(f"{BASE_URL}/process_speech", data=payload)
    if resp.status_code != 200:
        print("Error:", resp.text)
        return None
    # Extract the Play URL from the XML
    match = re.search(r'<Play>(.*?)</Play>', resp.text)
    if match:
        return match.group(1)
    else:
        print("No Play URL found. Response:", resp.text)
        return None

if __name__ == "__main__":
    print("Simulated call to Nepali Information Service. Type 'exit' to quit.")
    while True:
        q = input("\nYour question (in Nepali): ").strip()
        if q.lower() in ("exit", "quit", "bye"):
            break
        if not q:
            continue
        mp3_url = ask_question(q)
        if mp3_url:
            print(f"Playing answer from: {mp3_url}")
            play_mp3(mp3_url)