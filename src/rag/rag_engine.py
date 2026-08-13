# src/rag/rag_engine.py
import os
import time
import pickle
import logging
import warnings
import streamlit as st

# Suppress log noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"


@st.cache_resource
def load_rag_pipeline(model_name: str = "llama3:latest"):
    """Caches Embedding Model, Chroma DB, BM25 Index, and Ollama in RAM."""
    print(" [LOG 1/4] Loading RAG Pipeline components into RAM...")
    if not os.path.exists(PERSIST_DIR) or not os.path.exists(BM25_PKL_PATH):
        print(" [ERROR] Chroma DB or BM25 pickle file missing!")
        return None, None, None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"local_files_only": True}
        )
    except Exception:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    with open(BM25_PKL_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
        bm25_retriever.k = 3

    # Use qwen2.5:1.5b for fast local CPU inference with a 20-second timeout
    llm = OllamaLLM(
        model=model_name, 
        temperature=0.3, 
        timeout=20.0,
        keep_alive="30m"
    )

    print(" [LOG 1/4] Pipeline successfully loaded and cached!")
    return vector_store, bm25_retriever, llm


def answer_user_query(query: str, model_name: str = "llama3:latest") -> str:
    """Queries retrievers and LLM with step-by-step timing logs."""
    start_time = time.time()
    
    print(f"\n--- [RAG START] Query: '{query}' ---")
    vector_store, bm25_retriever, llm = load_rag_pipeline(model_name)

    if vector_store is None or bm25_retriever is None:
        return "माफ गर्नुहोला, डाटाबेस वा BM25 इन्डेक्स फेला परेन। कृपया पहिले embedder.py चलाउनुहोस्।"

    # 1. Retrieval
    r_start = time.time()
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_store.similarity_search(query, k=3)

    seen_content = set()
    combined_docs = []
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            combined_docs.append(doc)

    docs = combined_docs[:3]
    print(f" [LOG 2/4] Document Retrieval completed in {time.time() - r_start:.2f} seconds.")

    if not docs:
        return "क्षमा गर्नुहोला, यस विषयमा कागजातमा जानकारी उपलब्ध छैन।"

    context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt = f"""तपाईं एक नम्र र मिठासपूर्ण नेपाली टेलिफोन भोइस असिस्टेन्ट हुनुहुन्छ।
दिइएको सन्दर्भ (Context) को आधारमा मात्र प्राकृतिक र स्पष्ट नेपाली भाषामा उत्तर दिनुहोस्।

सन्दर्भ (Context):
{context_str}

प्रश्न: {query}

नियमहरू:
१. उत्तर १-२ शब्दमा मात्र नदिनुहोस्। पूर्ण र मिठासयुक्त वाक्यमा उत्तर दिनुहोस्।
२. उत्तर दिँदा 'प्राप्त जानकारी अनुसार...' वा 'हाम्रो रेकर्ड अनुसार...' जस्ता वाक्य प्रयोग गर्नुहोस्।
३. यदि सन्दर्भमा उत्तर छैन भने, "क्षमा गर्नुहोला, यस विषयमा कागजातमा जानकारी उपलब्ध छैन" भन्नुहोस्।
"""

    # 2. LLM Generation
    l_start = time.time()
    print(" [LOG 3/4] Sending prompt to Ollama LLM...")
    try:
        response = llm.invoke(prompt)
        print(f" [LOG 3/4] LLM Response generated in {time.time() - l_start:.2f} seconds.")
        print(f"--- [RAG COMPLETE] Total RAG Time: {time.time() - start_time:.2f}s ---\n")
        return response.strip()
    except Exception as e:
        print(f" [ERROR] Ollama invocation failed: {e}")
        return f"Ollama connection error or timeout: {str(e)}"