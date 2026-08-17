# 🎙️ Nepali Information Service (Voice-to-Voice RAG Assistant)

An interactive, ultra-low-latency Voice-to-Voice AI Assistant built with **Streamlit**, **Hybrid RAG** (ChromaDB + BM25), and **Groq**. Designed to process and answer user queries from custom local Nepali text documents through a simulated telephone call interface.

---

## 🚀 Features

* **🎙️ Voice & Text Input:** Transcribes spoken Nepali queries using Google Speech Recognition (`ne-NP`).
* **🔍 Hybrid RAG Retrieval:** Combines dense vector search (ChromaDB with multilingual embeddings) and sparse keyword search (BM25) for high retrieval precision.
* **⚡ Blazing Fast Generation:** Employs the **Groq API** for lightning-fast LLM response synthesis[cite: 2].
* **🔊 Natural Text-to-Speech (TTS):** Converts Nepali text answers into natural audio stream responses using **Edge TTS** (`ne-NP-SagarNeural`) with automated phonetic transliteration for English technical terms.
* **📞 Call Interface Simulation:** Features an interactive phone call UI built natively in Streamlit with dynamic waveform animations[cite: 1].

---

## 🏗️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend / UI** | [Streamlit](https://streamlit.io/)[cite: 1] |
| **Package Manager** | [`uv`](https://github.com/astral-sh/uv) / `pip` |
| **Speech-to-Text (STT)** | `SpeechRecognition` (Google Speech API — `ne-NP`)[cite: 1] |
| **Text-to-Speech (TTS)** | `edge-tts` (Nepali Voice: Sagar)[cite: 1] |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`[cite: 2] |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/)[cite: 2] |
| **Keyword Search** | BM25 (`rank_bm25` / `langchain_community.retrievers`)[cite: 2] |
| **LLM Inference** | [Groq](https://groq.com/) API[cite: 2] |
| **Audio Playback** | `pygame`[cite: 1] |

---

## 📦 Prerequisites

Before getting started, make sure you have the following installed:

1. **Python 3.10+**
2. **Groq API Key**: An active API key from Groq to run LLM inference[cite: 2].
3. **FFmpeg**: Required for audio processing.
   * *Windows:* Install via `winget install FFmpeg` or download from the official site and add it to your system `PATH`.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository


git clone [https://github.com/nischalgirii/ai-system.git](https://github.com/nischalgirii/ai-system.git)
cd ai-system

### 2. Set Up Virtual Environment

Using uv:
uv venv
.\.venv\Scripts\Activate.ps1

Or using standard Python venv:

PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 3. Install Dependencies

Using uv:
uv pip install -r requirements.txt

Or using pip:
pip install -r requirements.txt

### 4. Configure Environment Variables
Create a .env file in the root directory and add your Groq API Key:
GROQ_API_KEY=your_groq_api_key_here

### 📁 Directory Structure
ai-system/
├── app.py                 # Main Streamlit web application & UI/TTS handling[cite: 1]
├── ingest.py              # Text chunking, embedding, and hybrid index builder
├── src/
│   └── rag/
│       └── rag_engine.py  # Cached Hybrid RAG pipeline & Groq integration[cite: 2]
├── data/
│   ├── arjun_bio.txt      # Clean text source for knowledge base
│   ├── chroma_db/         # Persisted vector database (generated)[cite: 2]
│   └── bm25_retriever.pkl # Persisted keyword index (generated)[cite: 2]
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (API Keys)
└── README.md              # Project documentation

### How to Run
1. Ingest Documents (Build the Knowledge Base)
Place your clean text source files inside the data/ folder, then run the ingestion script to build the ChromaDB and BM25 indexes:

    python ingest.py

2. Launch the Web Application
Start the Streamlit server[cite: 1]:

    streamlit run app.py

1. Open http://localhost:8501 in your browser.

2. Click "☎️ Start Call" to initiate the connection and speak naturally into your microphone when prompted[cite: 1].
