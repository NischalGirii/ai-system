import os
import time
import pickle
import logging
import warnings
import unicodedata
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

os.environ.update({
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
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

# =========================================================
# CONFIGURATION
# =========================================================
PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GROQ_MODEL = "openai/gpt-oss-20b"           # Your chosen model
GROQ_REASONING_EFFORT = "low"

VECTOR_K = 6
BM25_K = 6
FINAL_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 10000
MAX_COMPLETION_TOKENS = 512

NO_INFO = "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"
DATABASE_ERROR = "माफ गर्नुहोस्, अहिले जानकारी प्रणालीमा समस्या देखिएको छ।"
GROQ_UNAVAILABLE = "माफ गर्नुहोस्, अहिले सूचना सेवा उपलब्ध छैन।"
SERVER_ERROR = "माफ गर्नुहोस्, अहिले सर्भरमा समस्या देखिएको छ।"
GOODBYE_RESPONSE = "धन्यवाद। फेरि भेटौँला।"

EXIT_PHRASES = {"bye", "bye bye", "goodbye", "good bye", "बिदा", "बाइ", "बाइ बाइ", "धन्यवाद", "फेरि भेटौँला"}
DISCOVERY_PHRASES = {"whose information", "कसको जानकारी", "को को हुनुहुन्छ"}
SERVICE_PERSPECTIVE_TERMS = {"तपाईंलाई", "तपाईलाई", "तपाईँलाई"}
TECHNICAL_TERMS = {"technical", "engineer", "सफ्टवेयर", "इन्जिनियर"}
PERSON_PRONOUNS = {"उनी", "उहाँ", "उनको", "उहाँको", "he", "she"}
QUESTION_WORDS = {"who", "is", "are", "को", "के", "कहाँ", "कहिले"}
PERSON_ALIASES = {"arjun sharma": "अर्जुन शर्मा", "arjun": "अर्जुन शर्मा", "अर्जुन शर्मा": "अर्जुन शर्मा", "अर्जुन": "अर्जुन शर्मा"}

# =========================================================
# SESSION MEMORY (per call)
# =========================================================
call_state = {"current_person": None}

def get_current_person():
    return call_state.get("current_person", None)

def set_current_person(name):
    call_state["current_person"] = name if name else None

# =========================================================
# TEXT PROCESSING
# =========================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()

def is_exit_intent(query: str) -> bool:
    q = normalize_text(query).lower()
    return bool(q) and any(phrase in q for phrase in EXIT_PHRASES)

def detect_person(query: str):
    q = normalize_text(query).lower()
    for alias, canonical in PERSON_ALIASES.items():
        if alias in q:
            return canonical
    return None

def analyze_intent(query: str) -> dict:
    q = normalize_text(query).lower()
    has_perspective = any(term in q for term in SERVICE_PERSPECTIVE_TERMS)
    service_quest = has_perspective and any(t in q for t in {"जानकारी", "कसको", "who", "know"})
    is_discovery = any(p in q for p in DISCOVERY_PHRASES) or service_quest
    
    return {
        "discovery": is_discovery,
        "technical": (is_discovery and any(t in q for t in TECHNICAL_TERMS)),
        "service_perspective": service_quest,
        "followup": any(w in q for w in PERSON_PRONOUNS),
        "intro": any(p in q for p in ["who is", "परिचय", "बारेमा बताउनुहोस्"]),
    }

def resolve_followup_query(query: str) -> str:
    person = get_current_person()
    if not person: return query
    q = normalize_text(query).lower()
    if any(word in q for word in PERSON_PRONOUNS):
        return f"{person} {query}"
    return query

def expand_entity_query(query: str) -> str:
    orig = normalize_text(query)
    additions = [canonical for alias, canonical in PERSON_ALIASES.items() if alias in orig.lower()]
    if additions:
        return f"{orig} {' '.join(dict.fromkeys(additions))}".strip()
    return orig

def clean_bm25_query(query: str) -> str:
    q = normalize_text(query)
    cleaned_tokens = [t.lower().strip(".,!?;:'\"()[]{}") for t in q.split() if t and t not in QUESTION_WORDS]
    return " ".join(cleaned_tokens) or q

# =========================================================
# PIPELINE CACHING (global)
# =========================================================
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
            _bm25.k = BM25_K

    # ---- Load Groq API key from environment ----
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            client = Groq(api_key=api_key)
            # Set the client even if the model check fails; we'll let the API call fail later.
            _groq = client
            # Optionally verify model availability:
            # if GROQ_MODEL in {m.id for m in client.models.list().data}:
            #     _groq = client
        except Exception as exc:
            print(f"[GROQ INIT ERROR] {exc}")
    else:
        print("[WARNING] GROQ_API_KEY not set in environment.")

    _pipeline_loaded = True

def load_rag_pipeline():
    init_pipeline()
    return _vector_store, _bm25, _groq

# =========================================================
# RETRIEVAL & CONTEXT
# =========================================================
def retrieve_documents(query: str):
    query = normalize_text(query)
    if len(query) < 2: return []

    vector_store, bm25, _ = load_rag_pipeline()
    if not vector_store or not bm25: return []

    resolved = resolve_followup_query(query)
    expanded = expand_entity_query(resolved)
    lexical = clean_bm25_query(expanded)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            v_fut = executor.submit(vector_store.similarity_search, expanded, VECTOR_K)
            b_fut = executor.submit(bm25.invoke, lexical)
            vector_docs = v_fut.result()
            bm25_docs = b_fut.result()
    except Exception as exc:
        print(f"[RETRIEVAL ERROR] {exc}")
        return []

    merged = {}
    for r, doc in enumerate(vector_docs, start=1):
        txt = normalize_text(doc.page_content)
        if txt:
            merged[txt] = {"doc": doc, "vector_rank": r, "bm25_rank": None}

    for r, doc in enumerate(bm25_docs, start=1):
        txt = normalize_text(doc.page_content)
        if txt:
            if txt not in merged:
                merged[txt] = {"doc": doc, "vector_rank": None, "bm25_rank": r}
            else:
                merged[txt]["bm25_rank"] = r

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
    for item in documents[:FINAL_CONTEXT_CHUNKS]:
        txt = item["text"].strip()
        if not txt: continue
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0: break
        piece = txt[:remaining]
        chunks.append(piece)
        total_chars += len(piece)
    return "\n\n---\n\n".join(chunks)

def correct_service_perspective(answer: str, query: str) -> str:
    if not answer or not analyze_intent(query)["discovery"]:
        return answer.strip() if answer else ""
    corrected = answer.strip()
    for old in SERVICE_PERSPECTIVE_TERMS:
        corrected = corrected.replace(old, "मलाई")
    corrected = re.sub(r"^\s*(तपाईंलाई|तपाईलाई|तपाईँलाई)\s+", "मलाई ", corrected)
    return corrected.strip()

def cleanup_answer(answer: str) -> str:
    if not answer: return ""
    answer = answer.strip()
    answer = re.sub(r"^[`\"']+|[`\"']+$", "", answer).strip()
    answer = re.sub(r"^(उत्तर|Answer)\s*[:：]\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"[*_#>`]", "", answer).strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer

# =========================================================
# MAIN ANSWER FUNCTION
# =========================================================
def answer_user_query(query: str) -> str:
    query = normalize_text(query)
    if len(query) < 2:
        return ""

    if is_exit_intent(query):
        set_current_person(None)
        return GOODBYE_RESPONSE

    detected_person = detect_person(query)
    if detected_person:
        set_current_person(detected_person)

    vector_store, bm25, groq = load_rag_pipeline()
    if not vector_store or not bm25:
        return DATABASE_ERROR
    if not groq:
        return GROQ_UNAVAILABLE

    documents = retrieve_documents(query)
    if not documents:
        return NO_INFO

    context = build_context(documents)
    if not context:
        return NO_INFO

    task = "प्रयोगकर्ताको प्रश्नको सिधा, छोटो र conversational नेपाली उत्तर दिनुहोस्।"
    current_person = get_current_person() or "कुनै निश्चित व्यक्ति छैन"

    system_prompt = f"""
    तपाईं "अर्जुन शर्मा व्यक्तिगत सूचना सेवा" का नेपाली voice assistant हुनुहुन्छ। छोटो उत्तर दिनुहोस्।
    हाल सम्झिएको व्यक्ति: {current_person}
    कार्य: {task}
    सन्दर्भ: {context}
    """

    try:
        res = groq.chat.completions.create(
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

        choices = getattr(res, "choices", None)
        if not choices:
            return SERVER_ERROR

        msg = choices[0].message
        ans = str(getattr(msg, "content", "") or "").strip()

        if not ans:
            raw = getattr(msg, "model_dump", lambda: {})() or vars(msg)
            ans = str(raw.get("content") or raw.get("text") or "").strip()

        ans = cleanup_answer(ans)
        ans = correct_service_perspective(ans, query)

        if ans and len(ans) > 25 and ans[-1] not in ("।", "?", "!", ".", "…"):
            ans += "।"

        return ans or SERVER_ERROR

    except Exception as exc:
        print(f"[GROQ ERROR] {exc}")
        return SERVER_ERROR