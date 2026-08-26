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
DATA_PATH = "data/bipat.txt"
PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def build_indices():
    # --- Clean up old indices ---
    if os.path.exists(PERSIST_DIR):
        try:
            client = chromadb.PersistentClient(path=PERSIST_DIR)
            client.delete_collection("langchain")
        except Exception as e:
            print(f"[CLEANUP NOTICE] {e}")
        finally:
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    if os.path.exists(BM25_PKL_PATH):
        os.remove(BM25_PKL_PATH)

    # --- Load the actual disaster management document ---
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    raw_docs = loader.load()
    full_text = raw_docs[0].page_content if raw_docs else ""

    # --- Create a summary document from the file itself ---
    # You can adjust the length (here we take the first 600 characters)
    summary_text = full_text[:600].strip()
    if summary_text:
        summary_text += "…"  # indicate truncation
    else:
        summary_text = "विपद् व्यवस्थापन सम्बन्धी जानकारी पुस्तिका।"

    summary_doc = Document(
        page_content=summary_text,
        metadata={"source": "bipat_summary", "type": "meta"}
    )

    # --- Split the main document into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
        separators=[
            "\n## ",
            "\n\n",
            "।",             # Purnabiram
            "?",
            "!",
            "\n",
            " "
        ]
    )
    chunks = text_splitter.split_documents(raw_docs)

    # --- Add the summary document as an extra chunk ---
    chunks.append(summary_doc)
    print(f"Total chunks created: {len(chunks)}")

    # --- Build embeddings and vector store ---
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

    # --- Build BM25 retriever ---
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 6
    with open(BM25_PKL_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 retriever created successfully.")

if __name__ == "__main__":
    build_indices()