import os
import pickle
import shutil
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Paths
DATA_PATH = "data/arjun_bio.txt"
PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def build_indices():
    # --- FIX 3: Windows File Locking Prevention ---
    if os.path.exists(PERSIST_DIR):
        try:
            # Gracefully disconnect and delete the collection from memory first
            client = chromadb.PersistentClient(path=PERSIST_DIR)
            client.delete_collection("langchain") 
        except Exception as e:
            print(f"[CLEANUP NOTICE] {e}")
        finally:
            # Now safe to remove the physical directory via PowerShell/OS
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    if os.path.exists(BM25_PKL_PATH):
        os.remove(BM25_PKL_PATH)

    loader = TextLoader(DATA_PATH, encoding="utf-8")
    raw_docs = loader.load()

    summary_doc = Document(
        page_content=(
            "हाम्रो प्रणालीमा हाल अर्जुन शर्माको मात्र जानकारी छ। "
            "अर्जुन शर्मा एक सफ्टवेयर इन्जिनियर तथा प्रविधि परामर्शदाता हुनुहुन्छ।"
        ),
        metadata={"source": "system_summary", "type": "meta"}
    )

    # --- FIX 5: Optimized Nepali Semantic Chunking ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,      # Increased to account for Devanagari token density
        chunk_overlap=150,   # Increased overlap to maintain context across boundaries
        separators=[
            "\n## ", 
            "\n\n", 
            "।",             # Purnabiram (Primary Nepali sentence boundary)
            "?", 
            "!", 
            "\n", 
            " "
        ]
    )
    
    chunks = text_splitter.split_documents(raw_docs)
    chunks.append(summary_doc)
    print(f"Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("ChromaDB vector store created successfully.")

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 6
    with open(BM25_PKL_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 retriever created successfully.")

if __name__ == "__main__":
    build_indices()