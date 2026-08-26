# src/rag/rag_engine.py
import os
import pickle
import re
import unicodedata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration (Keep consistent with ingestion.py)
PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_K = 6
BM25_K = 6
MAX_CONTEXT_CHARS = 6000

_embeddings = None
_vector_store = None
_bm25 = None

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()

def init_rag():
    global _embeddings, _vector_store, _bm25
    if _embeddings: return
    
    _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if os.path.exists(PERSIST_DIR):
        _vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
    
    if os.path.exists(BM25_PKL_PATH):
        with open(BM25_PKL_PATH, "rb") as f:
            _bm25 = pickle.load(f)
            _bm25.k = BM25_K

def retrieve_documents(query: str):
    init_rag()
    query = normalize_text(query)
    if not query or not _vector_store or not _bm25:
        return []

    # Parallel retrieval (Simplified for stability)
    vector_docs = _vector_store.similarity_search(query, k=VECTOR_K)
    bm25_docs = _bm25.invoke(query)

    merged = {}
    for r, doc in enumerate(vector_docs, start=1):
        txt = normalize_text(doc.page_content)
        if txt: merged[txt] = {"doc": doc, "vector_rank": r, "bm25_rank": None}

    for r, doc in enumerate(bm25_docs, start=1):
        txt = normalize_text(doc.page_content)
        if txt:
            if txt not in merged:
                merged[txt] = {"doc": doc, "vector_rank": None, "bm25_rank": r}
            else:
                merged[txt]["bm25_rank"] = r

    # RRF Scoring
    RRF_K = 60.0
    fused = []
    for txt, item in merged.items():
        v_score = 1.0 / (RRF_K + item["vector_rank"]) if item["vector_rank"] else 0.0
        b_score = 1.0 / (RRF_K + item["bm25_rank"]) if item["bm25_rank"] else 0.0
        fused.append({"score": v_score + b_score, "text": txt})

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused

def build_context(documents):
    if not documents: return ""
    chunks = []
    total_chars = 0
    for item in documents[:6]: # Take top 6
        txt = item["text"].strip()
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0: break
        piece = txt[:remaining]
        chunks.append(piece)
        total_chars += len(piece)
    return "\n\n---\n\n".join(chunks)