import asyncio
import edge_tts
import os

async def generate_tts(text, filename):
    filepath = os.path.join("static", filename)
    print(f"Generating {filepath}...")
    comm = edge_tts.Communicate(text, "ne-NP-HemkalaNeural", rate="+0%", volume="+0%", pitch="+0Hz")
    await comm.save(filepath)
    print("Done.")

async def main():
    os.makedirs("static", exist_ok=True)
    tasks = [
        generate_tts("नमस्ते! स्वचालित सूचना सेवामा स्वागत छ। तपाईं के जानकारी चाहनुहुन्छ?", "greeting.mp3"),
        generate_tts("मैले तपाईंको कुरा बुझिन। कृपया पुनः भन्नुहोस्।", "retry.mp3"),
        generate_tts("अर्को प्रश्न सोध्नुहोस्।", "prompt_next.mp3"),
        generate_tts("धन्यवाद। फेरि भेटौँला।", "goodbye.mp3"),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())