# 🎙️ Voice‑to‑Voice RAG Assistant (Nepali)

> A production‑ready voice assistant that answers biographical and factual queries in Nepali, using a hybrid RAG pipeline (Vector + BM25) and a Groq LLM, all accessible via phone (Twilio).

---

## 📌 Overview

This project implements a **voice‑in / voice‑out** assistant that can be called via a phone number. It:

- Accepts spoken Nepali queries through a Twilio phone number.
- Transcribes speech using Twilio's built‑in STT.
- Retrieves relevant information from a local knowledge base using a hybrid retrieval system (ChromaDB vector search + BM25).
- Generates a precise, conversational answer in Nepali using Groq's LLM.
- Speaks the answer back using **edge‑tts** (natural‑sounding Nepali voice).
- Supports multi‑turn conversations (follow‑up questions, person disambiguation).

---

## ✨ Key Features

- **📞 Phone Integration:** Works with any phone – just call your Twilio number.
- **🧠 Hybrid RAG:** Combines dense retrieval (ChromaDB) and sparse retrieval (BM25) for high‑accuracy context.
- **🗣️ Natural Nepali TTS:** Uses Azure's `ne‑NP‑HemkalaNeural` voice via `edge‑tts`.
- **💬 Conversational Memory:** Remembers the current person being discussed (e.g., "अर्जुन शर्मा") to handle follow‑up pronouns.
- **🛡️ Resilient:** Graceful fallbacks for API errors, TTS failures, or out‑of‑scope questions.
- **⚡ Fast & Scalable:** Asynchronous FastAPI server handles concurrent calls.

---

## 🛠️ Tech Stack

| Component | Technology / Library |
| :--- | :--- |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) |
| **Voice Gateway** | [Twilio](https://www.twilio.com/) (Voice, Speech‑to‑Text) |
| **Text‑to‑Speech** | [`edge‑tts`](https://github.com/rany2/edge-tts) (Azure Neural TTS) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) |
| **Sparse Retriever** | BM25 (via `rank_bm25`) |
| **LLM** | [Groq](https://groq.com/) (e.g., `openai/gpt-oss-20b` or `llama3-70b-8192`) |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Tunneling** | [ngrok](https://ngrok.com/) (free tier) |
| **Language** | Python 3.9+ |

---

## 🏗️ Architecture & Flow

```mermaid
graph TD
    A[📞 Incoming Call] -->|Twilio Webhook| B[FastAPI /voice]
    B --> C[Gather & Play Greeting]
    C --> D[User Speaks]
    D -->|Twilio STT| E[/process_speech]
    E --> F[RAG Pipeline]
    F --> G[Groq LLM]
    G --> H[Answer Text]
    H --> I[edge‑tts → MP3]
    I --> J[Play MP3 back to caller]
    J --> K[Prompt for next question]
    K --> D
🚀 Getting Started
Prerequisites
Python 3.9+

A Twilio account with a phone number (trial works).

A Groq API key.

ngrok (free account) for exposing your local server.

1. Clone & Install Dependencies
bash
git clone https://github.com/NischalGirii/voice-to-voice
cd voice-to-voice
pip install -r requirements.txt
requirements.txt (example):

text
fastapi
uvicorn[standard]
twilio
edge-tts
groq
langchain-huggingface
langchain-chroma
rank-bm25
python-dotenv
2. Set Up Environment Variables
Create a .env file in the project root:

ini
GROQ_API_KEY=your_groq_api_key_here
3. Prepare Your Knowledge Base
Place your documents (text files) in a directory, then run the indexing script to create:

data/chroma_db/ – Chroma vector store.

data/bm25_retriever.pkl – BM25 index.

4. Run the FastAPI Server
bash
python main.py
The server will start at http://localhost:8000.

5. Expose with ngrok
In a separate terminal:

bash
ngrok http 8000
Copy the HTTPS forwarding URL (e.g., https://xxxx.ngrok-free.dev).

6. Configure Twilio
In your Twilio Console, go to Phone Numbers → Active Numbers → click your number.

Under Voice & Fax, set:

A call comes in → Webhook → URL: https://your-ngrok-url.ngrok-free.dev/voice → HTTP Method: POST.

Save.

Trial account note: If you cannot access the number configuration page, use the "Test a Call" tool in the Twilio Console – enter your ngrok URL + /voice and click Start call.

📞 How to Use
Dial your Twilio phone number.

After the greeting, ask your question in Nepali (e.g., "अर्जुन शर्मा को हुन्?").

Wait for the assistant to answer in natural Nepali.

Continue with follow‑up questions (e.g., "उनको पेशा के हो?").

Say "धन्यवाद" or "बिदा" to end the call.

🧠 RAG Engine Details
Hybrid Retrieval: Combines vector similarity (ChromaDB) and BM25 lexical matching, fused with Reciprocal Rank Fusion (RRF).

Person Tracking: Remembers the current person mentioned to handle pronouns.

Context Building: Limits context to 10k characters and 6 chunks to keep responses concise.

LLM Prompting: Uses a system prompt that instructs the model to answer succinctly in Nepali, based solely on the retrieved context.

⚙️ Configuration
Key settings in rag_engine.py:

Variable	Description	Default
GROQ_MODEL	Groq model to use	openai/gpt-oss-20b
VECTOR_K	Number of vector results	6
BM25_K	Number of BM25 results	6
FINAL_CONTEXT_CHUNKS	Chunks fed to LLM	6
MAX_CONTEXT_CHARS	Max context length	10000
MAX_COMPLETION_TOKENS	Max tokens for answer	512
🛡️ Error Handling & Fallbacks
If Groq is unavailable → returns "माफ गर्नुहोस्, अहिले सूचना सेवा उपलब्ध छैन।"

If no relevant documents → "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"

If TTS fails → falls back to Twilio's <Say> (robotic but functional).

If a user says goodbye → hangs up gracefully.

🧪 Testing Locally Without a Phone
You can simulate a request using curl:

bash
curl -X POST "https://your-ngrok-url.ngrok-free.dev/process_speech" \
  -d "SpeechResult=अर्जुन शर्मा को हुन्" \
  -H "Content-Type: application/x-www-form-urlencoded"
📂 Project Structure
text
.
├── main.py                 # FastAPI application, Twilio endpoints, TTS
├── src/
│   └── rag/
│       └── rag_engine.py   # RAG pipeline (retrieval, LLM, session memory)
├── data/
│   ├── chroma_db/          # Vector store
│   └── bm25_retriever.pkl  # BM25 index
├── static/                 # Generated MP3 files (served statically)
├── .env                    # Environment variables (GROQ_API_KEY)
└── requirements.txt
🔧 Troubleshooting
Issue	Solution
"Application Error" on call	Check ngrok logs; ensure the ngrok-skip-browser-warning header is added (see middleware in main.py).
MP3 not playing	Verify the static URL is absolute (https://your-ngrok-url/static/...) and file exists.
Groq returns 403	Check your API key and network; ensure .env is loaded.
TTS takes too long	Edge‑tts generates fast (~1‑2s). If slower, ensure stable internet.
🚧 Future Improvements
Caching – pre‑generate common answers and greetings.

Streaming TTS – reduce latency by streaming audio chunks.

Multi‑language support – add English fallback.

Call recording / analytics – log interactions for improvement.

📄 License
MIT © Nischal Giri

🙏 Acknowledgements
Twilio for voice and STT.

Groq for fast inference.

edge‑tts for free, high‑quality Nepali TTS.

ngrok for local tunneling.