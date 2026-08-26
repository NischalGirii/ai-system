# src/rag/rag_tools.py
from src.rag.rag_google import retrieve_documents, build_context

def search_knowledge_base(query: str) -> str:
    """
    Searches the local disaster management database for information.
    Use this when the user asks about disaster procedures, contact numbers, or safety rules.
    """
    print(f"[TOOL] Gemini is searching for: {query}")
    documents = retrieve_documents(query)
    if not documents:
        return "No information found in the database regarding this topic."
    return build_context(documents)