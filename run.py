import os
import subprocess
import sys

def main():
    # Read the PRODUCTION variable (default to false)
    # PRODUCTION=true  -> Launches Twilio (main.py)
    # PRODUCTION=false -> Launches Streamlit (app_google.py)
    production_mode = os.getenv("PRODUCTION", "false").lower() == "true"

    if production_mode:
        print("\n" + "="*50)
        print("🚀 MODE: PRODUCTION (Twilio Telephony Server)")
        print("Running: uvicorn main:app --host 0.0.0.0 --port 8000")
        print("="*50 + "\n")
        
        # Launch FastAPI/Uvicorn
        try:
            subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"], check=True)
        except KeyboardInterrupt:
            print("\nStopping Production Server...")
        except Exception as e:
            print(f"❌ Error starting production server: {e}")

    else:
        print("\n" + "="*50)
        print("💻 MODE: DEVELOPMENT (Streamlit Web UI)")
        print("Running: streamlit run app_google.py")
        print("="*50 + "\n")
        
        # Launch Streamlit
        try:
            subprocess.run(["streamlit", "run", "app_google.py"], check=True)
        except KeyboardInterrupt:
            print("\nStopping Development Server...")
        except Exception as e:
            print(f"❌ Error starting development server: {e}")

if __name__ == "__main__":
    main()