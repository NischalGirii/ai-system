# ingestion.py
import os
import uuid
import shutil
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import re

DATA_PATH = "data/rasuwa_nuwakot_dhading_chitwan_flood_emergency_nepali.txt"
PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ------------------------------------------------------------------
# Parse the emergency TXT into sections
# ------------------------------------------------------------------
def parse_emergency_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by section markers: lines of "====" and a title line
    # Pattern: "============================================================\n\d+\. .*?\n============================================================"
    # We'll use a regex to capture the title and the content until the next section marker.
    sections = []
    # First, find all section headers (lines with number and dot)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("==") and i+1 < len(lines):
            # next line might be the title
            title_line = lines[i+1].strip()
            if re.match(r"^\d+\. .+", title_line):
                # we have a title; find the end of this section: next "====" line or EOF
                start_idx = i
                i += 2
                content_lines = []
                while i < len(lines):
                    if lines[i].strip().startswith("=="):
                        break
                    content_lines.append(lines[i])
                    i += 1
                # If we stopped at a marker, we have consumed it; the loop will continue with that marker
                # But we need to not skip the marker for the next iteration; we already advanced i to the marker line.
                # Actually we want to rewind i by 1 to process the marker in the next iteration.
                # Simpler: we can just collect content until we encounter a line that starts with "==".
                # We'll handle this differently: use regex on entire text.
                pass
        i += 1

    # Let's use a simpler approach: split by the pattern of section header.
    # Pattern: "============================================================" then a line with number. then "============================================================"
    # But the content may have nested "=="? No.
    # We'll find all occurrences of the section title pattern.
    # We'll split the text by the marker lines.
    parts = re.split(r"={4,}\n", text)
    # parts[0] might be empty or a header? The first part before first ==== is the title? Actually file starts with title.
    # Better: find all section headers with regex.
    headers = re.finditer(r"(?m)^={4,}\n(\d+\. .+?)\n={4,}$", text)
    # We'll iterate through headers and extract content between them.
    matches = list(headers)
    sections = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx+1].start() if idx+1 < len(matches) else len(text)
        content = text[start:end].strip()
        # Determine district if present
        district = None
        if "रसुवा" in title or "रसुवा" in content[:200]:
            district = "Rasuwa"
        elif "नुवाकोट" in title or "नुवाकोट" in content[:200]:
            district = "Nuwakot"
        elif "धादिङ" in title or "धादिङ" in content[:200]:
            district = "Dhading"
        elif "चितवन" in title or "चितवन" in content[:200]:
            district = "Chitwan"
        # Determine category
        category = "general"
        if "आपतकालीन सम्पर्क" in title or "नम्बर" in title:
            category = "emergency_contacts"
        elif "सबैभन्दा पहिले" in title:
            category = "immediate_safety"
        elif "उद्धार" in title:
            category = "rescue"
        elif "हराएको" in title:
            category = "missing_person"
        elif "राहत" in title:
            category = "relief"
        elif "स्वास्थ्य" in title or "घाइते" in title:
            category = "health"
        elif "सडक" in title or "पुल" in title:
            category = "road_bridge"
        elif "सञ्चार" in title:
            category = "communication"
        elif "बालबालिका" in title or "वृद्ध" in title:
            category = "vulnerable"
        elif "जोखिम" in title:
            category = "post_flood_risks"
        elif "छिटो सूची" in title:
            category = "quick_numbers"

        sections.append({
            "title": title,
            "content": content,
            "section_number": idx+1,
            "district": district,
            "category": category,
            "source": os.path.basename(DATA_PATH)
        })
    return sections

def build_indices():
    # Clean up old indices
    for path in [PERSIST_DIR, BM25_PKL_PATH]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)

    # Parse emergency TXT
    sections = parse_emergency_sections(DATA_PATH)
    if not sections:
        print("No sections found in emergency TXT.")
        return

    # Create LangChain Documents
    docs = []
    for sec in sections:
        # Build a document with metadata
        doc = Document(
            page_content=sec["content"],
            metadata={
                "source": sec["source"],
                "section_title": sec["title"],
                "section_number": sec["section_number"],
                "district": sec["district"] or "national",
                "category": sec["category"],
                "document_type": "emergency_information",
                "domain": "flood_emergency",
                "priority": "P1" if "नम्बर" in sec["title"] or "उद्धार" in sec["title"] else "P2"
            }
        )
        docs.append(doc)

    # Splitting: we want to keep each section as a single chunk, because sections are short.
    # But we can also split very long sections if needed, but for this TXT they are not huge.
    # We'll use a splitter with large chunk size to preserve boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "।", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    # But if we want to preserve section boundaries, we can just use the documents directly without splitting.
    # However, some sections are long (like the rescue info list). We'll split but with large chunk size.
    # To be safe, we'll use a splitter that respects section boundaries by adding a separator that is unlikely.
    # We'll just use the docs as chunks and not split further, because they are already under ~2000 chars.
    # We'll check max length.
    max_len = max(len(doc.page_content) for doc in docs)
    print(f"Max section length: {max_len} chars. Using whole sections as chunks.")
    # Use the docs directly as chunks.
    chunks = docs

    # Generate deterministic IDs to avoid duplicates
    namespace = uuid.NAMESPACE_DNS
    for doc in chunks:
        # Use source + section_title + content hash
        content_hash = uuid.uuid5(namespace, doc.page_content + doc.metadata["source"] + doc.metadata["section_title"]).hex
        doc.metadata["doc_id"] = content_hash

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    # ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"ChromaDB created with {len(chunks)} chunks.")

    # BM25
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 6
    with open(BM25_PKL_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 retriever saved.")

if __name__ == "__main__":
    import pickle
    build_indices()