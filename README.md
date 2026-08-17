
# 🎙️ Nepali Information Service (Voice-to-Voice RAG Assistant)

An interactive, ultra-low-latency Voice-to-Voice AI Assistant built with **Streamlit**, **Hybrid RAG** (ChromaDB + BM25), and **Groq**. Designed to process and answer user queries from custom local Nepali text documents through a simulated telephone call interface.

---

## 🚀 Features

* **🎙️ Voice & Text Input:** Transcribes spoken Nepali queries using Google Speech Recognition (`ne-NP`).
* **🔍 Hybrid RAG Retrieval:** Combines dense vector search (ChromaDB with multilingual embeddings) and sparse keyword search (BM25) for high retrieval precision.
* **⚡ Blazing Fast Generation:** Employs the **Groq API** for lightning-fast LLM response synthesis.
* **🔊 Natural Text-to-Speech (TTS):** Converts Nepali text answers into natural audio stream responses using **Edge TTS** (`ne-NP-SagarNeural`) with automated phonetic transliteration for English technical terms.
* **📞 Call Interface Simulation:** Features an interactive phone call UI built natively in Streamlit with dynamic waveform animations.

---

## 🏗️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend / UI** | [Streamlit](https://streamlit.io/) |
| **Package Manager** | [`uv`](https://github.com/astral-sh/uv) / `pip` |
| **Speech-to-Text (STT)** | `SpeechRecognition` (Google Speech API — `ne-NP`) |
| **Text-to-Speech (TTS)** | `edge-tts` (Nepali Voice: Sagar) |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) |
| **Keyword Search** | BM25 (`rank_bm25` / `langchain_community.retrievers`) |
| **LLM Inference** | [Groq](https://groq.com/) API |
| **Audio Playback** | `pygame` |

---

## 📦 Prerequisites

Before getting started, make sure you have the following installed:

1. **Python 3.10+**
2. **Groq API Key**: An active API key from Groq to run LLM inference.
3. **FFmpeg**: Required for audio processing.
   * *Windows:* Install via `winget install FFmpeg` or download from the official site and add it to your system `PATH`.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nischalgirii/ai-system.git
cd ai-system
```

### 2. Set Up Virtual Environment

Using uv:
```bash
uv venv
.\.venv\Scripts\Activate.ps1
```

Or using standard Python venv (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Using uv:
```bash
uv pip install -r requirements.txt
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API Key:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

### 📁 Directory Structure
```
ai-system/
├── app.py                 # Main Streamlit web application & UI/TTS handling
├── ingest.py              # Text chunking, embedding, and hybrid index builder
├── src/
│   └── rag/
│       └── rag_engine.py  # Cached Hybrid RAG pipeline & Groq integration
├── data/
│   ├── arjun_bio.txt      # Clean text source for knowledge base
│   ├── chroma_db/         # Persisted vector database (generated)
│   └── bm25_retriever.pkl # Persisted keyword index (generated)
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (API Keys)
└── README.md              # Project documentation
```

### How to Run
1. **Ingest Documents (Build the Knowledge Base)**  
   Place your clean text source files inside the `data/` folder, then run the ingestion script to build the ChromaDB and BM25 indexes:
   ```bash
   python ingest.py
   ```

2. **Launch the Web Application**  
   Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

3. Open http://localhost:8501 in your browser.

4. Click "☎️ Start Call" to initiate the connection and speak naturally into your microphone when prompted.
```
