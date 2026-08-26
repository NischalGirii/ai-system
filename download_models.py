import os
from faster_whisper import WhisperModel

# 1. Define where you want to save the model
# We will put it in a folder named 'models' inside your project
SAVE_DIR = os.path.join(os.getcwd(), "models", "whisper-large-v3")

def download_whisper():
    # Ensure the directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    print(f"Starting download of large-v3 to: {SAVE_DIR}")
    print("This will take a few minutes depending on your internet speed...")

    try:
        # IMPORTANT: We set download_root to our custom folder.
        # We use compute_type='int8' so it downloads the quantized version directly.
        model = WhisperModel(
            "large-v3", 
            device="cpu", 
            compute_type="int8", 
            download_root=SAVE_DIR
        )
        print("\n✅ SUCCESS: Model downloaded and saved locally!")
        print(f"Location: {SAVE_DIR}")
        print("You can now run your project with HF_HUB_OFFLINE=1")

    except Exception as e:
        print(f"\n❌ ERROR during download: {e}")
        print("\nTIP: Make sure your internet is connected and HF_HUB_OFFLINE is NOT set to 1 in your environment.")

if __name__ == "__main__":
    # Ensure we are NOT in offline mode for this script
    if os.getenv("HF_HUB_OFFLINE") == "1":
        print("⚠️ Warning: HF_HUB_OFFLINE is set to 1. Trying to override for download...")
        os.environ["HF_HUB_OFFLINE"] = "0"
        
    download_whisper()