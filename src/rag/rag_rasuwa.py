# src/rag/rag_rasuwa.py
import os
import re
import pickle
import threading
import logging
import warnings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

os.environ.update({
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
})
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

from groq import Groq

# ------------------------------------------------------------------
# Configuration – optimised for speed
# ------------------------------------------------------------------
PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMERGENCY_DOC_PATH = os.getenv(
    "EMERGENCY_DOC_PATH",
    "data/rasuwa_nuwakot_dhading_chitwan_flood_emergency_nepali.txt"
)

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 100          # ← drastically reduced for speed

# ------------------------------------------------------------------
# Parse and cache emergency sections
# ------------------------------------------------------------------
_sections_cache = None
_sections_mtime = None
_sections_lock = threading.Lock()

def parse_emergency_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    headers = re.finditer(r"(?m)^={4,}\n(\d+\. .+?)\n={4,}$", text)
    matches = list(headers)
    sections = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx+1].start() if idx+1 < len(matches) else len(text)
        content = text[start:end].strip()
        district = None
        if "रसुवा" in title or "रसुवा" in content[:200]:
            district = "Rasuwa"
        elif "नुवाकोट" in title or "नुवाकोट" in content[:200]:
            district = "Nuwakot"
        elif "धादिङ" in title or "धादिङ" in content[:200]:
            district = "Dhading"
        elif "चितवन" in title or "चितवन" in content[:200]:
            district = "Chitwan"
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
            "source": os.path.basename(file_path)
        })
    return sections

def load_emergency_sections():
    global _sections_cache, _sections_mtime
    try:
        mtime = os.path.getmtime(EMERGENCY_DOC_PATH)
    except OSError:
        return []
    with _sections_lock:
        if _sections_cache is None or _sections_mtime != mtime:
            _sections_cache = parse_emergency_sections(EMERGENCY_DOC_PATH)
            _sections_mtime = mtime
            print(f"[EMERGENCY] Loaded {len(_sections_cache)} sections.")
        return _sections_cache

# ------------------------------------------------------------------
# Deterministic number lookup
# ------------------------------------------------------------------
def get_deterministic_answer(query, sections):
    q = query.lower()
    districts = ["Rasuwa", "Nuwakot", "Dhading", "Chitwan", "रसुवा", "नुवाकोट", "धादिङ", "चितवन"]
    for d in districts:
        d_lower = d.lower()
        if d_lower in q:
            for sec in sections:
                if sec["district"] and sec["district"].lower() == d_lower:
                    numbers = re.findall(r"[\d\-]+", sec["content"])
                    if numbers:
                        lines = sec["content"].splitlines()
                        for line in lines:
                            if any(num in line for num in numbers):
                                return line.strip()
    if any(word in q for word in ["प्रहरी", "police"]):
        return "नेपाल प्रहरी आपतकालीन नम्बर १०० हो।"
    if any(word in q for word in ["एम्बुलेन्स", "ambulance"]):
        return "एम्बुलेन्स सेवाको नम्बर १०२ हो।"
    if any(word in q for word in ["राष्ट्रिय विपद्", "neoc"]):
        return "राष्ट्रिय विपद् जोखिम न्यूनीकरण तथा व्यवस्थापन केन्द्रको टोल-फ्री नम्बर ११४९ हो।"
    return None

# ------------------------------------------------------------------
# Emergency context builder
# ------------------------------------------------------------------
class EmergencyContextBuilder:
    def __init__(self, sections):
        self.sections = sections

    def get_relevant_sections(self, query):
        q = query.lower()
        scored = []
        for sec in self.sections:
            score = 0
            if sec["district"] and sec["district"].lower() in q:
                score += 15
            cat_keywords = {
                "immediate_safety": ["बाढी", "पानी", "बढिरहेको", "के गर्ने", "सुरक्षित", "उद्धार", "पहिले"],
                "emergency_contacts": ["नम्बर", "फोन", "सम्पर्क", "contact", "number"],
                "rescue": ["उद्धार", "rescue", "फस्यो", "फसेको"],
                "missing_person": ["हराएको", "missing"],
                "relief": ["राहत", "खाना", "पानी", "आश्रय"],
                "health": ["घाइते", "स्वास्थ्य", "एम्बुलेन्स"],
                "road_bridge": ["सडक", "पुल", "बाटो"],
                "communication": ["मोबाइल", "नेटवर्क", "सञ्चार", "बन्द"],
                "vulnerable": ["बालबालिका", "वृद्ध", "अशक्त"],
                "post_flood_risks": ["जोखिम", "पहिरो"],
                "quick_numbers": ["नम्बर", "सूची"]
            }
            for cat, words in cat_keywords.items():
                if sec["category"] == cat:
                    if any(w in q for w in words):
                        if cat == "immediate_safety":
                            score += 15
                        else:
                            score += 10
            content_words = set(sec["content"].lower().split())
            query_words = set(q.split())
            overlap = len(query_words.intersection(content_words))
            score += overlap * 2
            if sec["title"].lower() in q:
                score += 5
            scored.append((sec, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, sc in scored if sc > 0][:5]

    def build_context(self, query):
        sections = self.get_relevant_sections(query)
        if not sections:
            return ""
        parts = []
        for sec in sections:
            parts.append(f"**{sec['title']}**\n{sec['content']}")
        return "\n\n---\n\n".join(parts)

# ------------------------------------------------------------------
# RAG pipeline (fallback)
# ------------------------------------------------------------------
_embeddings = None
_vector_store = None
_bm25 = None
_groq = None
_pipeline_loaded = False

def init_pipeline():
    global _embeddings, _vector_store, _bm25, _groq, _pipeline_loaded
    if _pipeline_loaded:
        return
    try:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True}
        )

    if os.path.exists(PERSIST_DIR):
        _vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)

    if os.path.exists(BM25_PKL_PATH):
        with open(BM25_PKL_PATH, "rb") as f:
            _bm25 = pickle.load(f)
            _bm25.k = 6

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            _groq = Groq(api_key=api_key)
            print("[GROQ] Client initialized.")
        except Exception as e:
            print(f"[GROQ ERROR] {e}")
    else:
        print("[WARNING] GROQ_API_KEY not set.")
    _pipeline_loaded = True

def retrieve_documents(query):
    if not _vector_store or not _bm25:
        return []
    try:
        vector_docs = _vector_store.similarity_search(query, k=4)
        bm25_docs = _bm25.invoke(query)
    except Exception as e:
        print(f"[RETRIEVAL ERROR] {e}")
        return []
    seen = set()
    combined = []
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined

def build_context_from_docs(docs):
    if not docs:
        return ""
    parts = []
    for doc in docs[:3]:
        parts.append(doc.page_content)
    return "\n\n---\n\n".join(parts)

# ------------------------------------------------------------------
# Normalization helper
# ------------------------------------------------------------------
def normalize_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

# ------------------------------------------------------------------
# Main answer function – returns a single string (no tuple)
# ------------------------------------------------------------------
def answer_user_query(query: str, call_sid: Optional[str] = None) -> str:
    """
    Returns a concise answer with a short follow‑up prompt appended.
    """
    try:
        if 'à¤µ' in query or 'à¤¿' in query:
            query = query.encode('latin-1').decode('utf-8')
    except Exception:
        pass
    query = normalize_text(query)
    print(f"[QUERY] {query}")

    if len(query) < 2:
        return "कृपया फेरि भन्नुहोस्।"

    sections = load_emergency_sections()
    if not sections:
        return "क्षमा गर्नुहोस्, आपतकालीन जानकारी उपलब्ध छैन।"

    # Deterministic answer
    det = get_deterministic_answer(query, sections)
    if det:
        return det + " के अर्को प्रश्न सोध्नुहुन्छ?"

    # Build context
    builder = EmergencyContextBuilder(sections)
    context = builder.build_context(query)
    if not context:
        init_pipeline()
        docs = retrieve_documents(query)
        if docs:
            context = build_context_from_docs(docs)
        else:
            return "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"

    # Ensure Groq is initialised
    init_pipeline()
    groq_client = _groq
    if not groq_client:
        return "क्षमा गर्नुहोस्, सूचना सेवा उपलब्ध छैन।"

    system_prompt = f"""
तपाईं नेपाली बाढी आपतकालीन सूचना सहायक हुनुहुन्छ। 
तपाईंले आपतकालीन जानकारी स्रोतबाट मात्र जवाफ दिनुपर्छ।
स्रोतमा नभएको कुनै पनि जानकारी आफैं नबनाउनुहोस्।
आपतकालीन नम्बरको प्रश्नमा स्रोतको सही नम्बर प्रयोग गर्नुहोस्।

**महत्त्वपूर्ण:** 
- जवाफ **अत्यन्त संक्षिप्त** राख्नुहोस् – अधिकतम २ वाक्य वा ३० शब्द।
- यदि प्रश्न सुरक्षा कार्यको बारेमा हो भने, सबैभन्दा महत्त्वपूर्ण २–३ बुँदा मात्र दिनुहोस्।
- लामो सूची नदिनुहोस्; संक्षिप्त र स्पष्ट जवाफ दिनुहोस्।

स्रोत:
{context}
"""
    try:
        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            reasoning_effort=GROQ_REASONING_EFFORT,
            include_reasoning=False,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            stream=False,
        )
        answer = res.choices[0].message.content.strip()
        if not answer:
            return "क्षमा गर्नुहोस्, जवाफ उत्पन्न गर्न सकिएन।"
        # Clean up
        answer = re.sub(r"[*_#>`]", "", answer)
        answer = re.sub(r"\s+", " ", answer)
        if answer and answer[-1] not in ("।", "?", "!", ".", "…"):
            answer += "।"
        # Append a short follow‑up prompt (no separate TTS)
        answer += " के अर्को प्रश्न सोध्नुहुन्छ?"
        return answer
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "क्षमा गर्नुहोस्, अहिले सेवामा समस्या छ।"

# ------------------------------------------------------------------
# Initialize at module load
# ------------------------------------------------------------------
init_pipeline()
load_emergency_sections()