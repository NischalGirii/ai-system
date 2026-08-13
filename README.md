# 🎙️ RAG Voice-to-Voice AI Assistant (Nepali)

An interactive, low-latency Voice-to-Voice AI Assistant built with **Streamlit**, **Hybrid RAG** (ChromaDB + BM25), and **Ollama**. Designed to process and answer user queries from custom local Nepali PDF documents through a telephone call interface.

---

## 🚀 Features

* **🎙️ Voice & Text Input:** Transcribes spoken Nepali queries using Google Speech Recognition (`ne-NP`) or accepts text inputs.
* **🔍 Hybrid RAG Retrieval:** Combines dense vector search (ChromaDB with multilingual embeddings) and sparse keyword search (BM25) for high retrieval precision (< 0.1s latency).
* **🦙 Local & Private Generation:** Employs **Ollama** (`llama3:latest`) for fast, local LLM response synthesis without data leaking or cloud dependencies.
* **🔊 Text-to-Speech (TTS):** Converts Nepali text answers into audio stream responses using `gTTS`.
* **📞 Call Interface Simulation:** Features a phone call UI built with Streamlit.

---

## 🏗️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend / UI** | [Streamlit](https://streamlit.io/) |
| **Package Manager** | [`uv`](https://github.com/astral-sh/uv) / `pip` |
| **Speech-to-Text (STT)** | `SpeechRecognition` (Google Speech API — `ne-NP`) |
| **Text-to-Speech (TTS)** | `gTTS` (Nepali language model) |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) |
| **Keyword Search** | BM25 (`rank_bm25` / `langchain_community.retrievers`) |
| **Local LLM** | [Ollama](https://ollama.com/) (`llama3:latest`) |
| **Audio Processing** | `pydub` + `ffmpeg` |

---

## 📦 Prerequisites

Before getting started, make sure you have the following installed:

1. **Python 3.10+**
2. **FFmpeg**: Required by `pydub` for processing audio streams.
   * *Windows:* Install via `winget install FFmpeg` or download from the official site and add it to your system `PATH`.
3. **Ollama**: Installed and running locally.
   * Pull the model:
     ```powershell
     ollama pull llama3:latest
     ```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/NischalGirii/ai-system.git
cd ai-system
```

### 2. Set Up Virtual Environment

Using **`uv`** *(Recommended)*:
```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

Or using standard Python `venv`:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Using `uv`:
```powershell
uv pip install -r requirements.txt
```

Or using `pip`:
```powershell
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```text
ai-system/
├── app.py                      # Main Streamlit web application & speech handling
├── src/
│   ├── ingestion/
│   │   ├── extractor.py        # PDF text extraction & OCR preprocessing
│   │   └── embedder.py         # Vector embeddings & BM25 indexing pipeline
│   └── rag/
│       └── rag_engine.py       # Cached Hybrid RAG pipeline & Ollama integration
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules for virtual environments & model caches
└── README.md                   # Project documentation
```

---

## 🏃‍♂️ How to Run

### 1. Ingest Documents (If building index from scratch)
Place your target PDF in the `data/` directory and run the embedding script:
```powershell
python -m src.ingestion.embedder
```

### 2. Launch the Web Application
Start the Streamlit server:
```powershell
streamlit run app.py
```

3. Open `http://localhost:8501` in your browser.
4. Enter a phone number in the sidebar and click **"Call"** to start the voice interaction session.

---

## ⚡ Performance Optimizations

* **Instant Document Retrieval:** Pre-calculated BM25 indices saved via pickle and persisted Chroma DB vectors loaded directly into system RAM using Streamlit's `@st.cache_resource`.
* **Direct Instruction Synthesis:** Utilizes standard instruct-tuned models (`llama3:latest`) to eliminate multi-minute chain-of-thought "thinking" loops on CPU hardware.
* **Strict Memory Isolation:** Heavy model weights, vector embeddings, and local virtual environments (`.venv`) are ignored by Git via `.gitignore` to keep the repository lightweight.
