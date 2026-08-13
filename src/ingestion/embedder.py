# src/ingestion/embedder.py
import os
import shutil
import pickle
from .extractor import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma


def process_and_store_embeddings(
    extracted_pages: list[dict], 
    persist_directory: str = "data/chroma_db",
    bm25_pickle_path: str = "data/bm25_retriever.pkl",
    force_rebuild: bool = True
) -> Chroma:
    """Splits text into chunks, generates vector embeddings, and persists BM25 index to disk."""
    print("\n--- CHECKING VECTOR STORE & BM25 INDEX ---")

    if force_rebuild and os.path.exists(persist_directory):
        print(f"Removing old vector store at: '{persist_directory}'...")
        shutil.rmtree(persist_directory)
    if force_rebuild and os.path.exists(bm25_pickle_path):
        os.remove(bm25_pickle_path)

    os.makedirs(persist_directory, exist_ok=True)

    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print("Creating new chunks, generating embeddings & BM25 index...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", "।", " ", ""]
    )

    all_texts = []
    all_metadatas = []
    langchain_docs = []

    for page in extracted_pages:
        page_num = page["page_number"]
        content = page["content"]
        
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            meta = {"page_number": page_num, "chunk_index": i}
            all_metadatas.append(meta)
            langchain_docs.append(Document(page_content=chunk, metadata=meta))

    print(f"Total Chunks Created: {len(all_texts)}")

    # 1. Store Chroma Vector Store
    vector_store = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=persist_directory
    )

    # 2. Build and Save BM25 Retriever to disk (.pkl)
    print("Building and persisting BM25 retriever index to disk...")
    bm25_retriever = BM25Retriever.from_documents(langchain_docs)
    bm25_retriever.k = 3

    with open(bm25_pickle_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"Embeddings saved to '{persist_directory}'!")
    print(f"BM25 index saved to '{bm25_pickle_path}'!")
    return vector_store


if __name__ == "__main__":
    target_pdf = "data/नमूना_व्यक्तिगत_जीवनी_RAG_DATA_ONLY.pdf"
    
    try:
        results = extract_text_from_pdf(target_pdf, force_ocr=True)
        vector_store = process_and_store_embeddings(
            results, 
            persist_directory="data/chroma_db",
            bm25_pickle_path="data/bm25_retriever.pkl",
            force_rebuild=True
        )
        print("\nEmbedding & BM25 creation finished successfully!")
            
    except Exception as e:
        print(f"Error during pipeline execution: {e}")