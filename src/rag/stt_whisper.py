import os
import tempfile
import speech_recognition as sr
from faster_whisper import WhisperModel

# Define the LOCAL path to your downloaded model
# This matches the SAVE_DIR in our download script
LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "models", "whisper-large-v3")

class WhisperSTT:
    def __init__(self, model_path=LOCAL_MODEL_PATH, device="cpu", compute_type="int8"):
        """
        model_path: Now points to the LOCAL FOLDER, not the string 'large-v3'
        """
        print(f"[STT] Loading Whisper from LOCAL path: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Local model not found at {model_path}. Run 'python download_models.py' first!")

        # We pass the folder path directly. 
        # Faster-Whisper will see the files in that folder and load them.
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)
        self.recognizer = sr.Recognizer()
        print("[STT] Local Whisper model loaded successfully.")

   
    def transcribe_mic(self, language="ne"):
        """
        Uses the microphone to record audio, saves it to a temp file, 
        and then uses Whisper to transcribe it.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with sr.Microphone() as source:
                print("[STT] Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("[STT] Listening...")
                # Record audio (max 10 seconds to prevent long pauses)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Save to the temporary wav file
                with open(tmp_path, "wb") as f:
                    f.write(audio.get_wav_data())

            # Perform transcription
            # We set language='ne' (Nepali) to force the model to focus on Nepali
            segments, info = self.model.transcribe(tmp_path, beam_size=5, language=language)
            
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "

            return full_text.strip()

        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            print(f"[STT ERROR] {e}")
            return ""
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

# Singleton instance so we don't reload the model every time a user speaks
_whisper_instance = None

def get_whisper_stt():
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperSTT()
    return _whisper_instance