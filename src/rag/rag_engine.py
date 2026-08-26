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

# =========================================================
# CONFIGURATION
# =========================================================
PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_REASONING_EFFORT = "low"

VECTOR_K = 15           # increased for better recall
BM25_K = 15
FINAL_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 10000
MAX_COMPLETION_TOKENS = 256

NO_INFO = "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"
DATABASE_ERROR = "माफ गर्नुहोस्, अहिले जानकारी प्रणालीमा समस्या देखिएको छ।"
GROQ_UNAVAILABLE = "माफ गर्नुहोस्, अहिले सूचना सेवा उपलब्ध छैन।"
SERVER_ERROR = "माफ गर्नुहोस्, अहिले सर्भरमा समस्या देखिएको छ।"
GOODBYE_RESPONSE = "धन्यवाद। फेरि भेटौँला।"

EXIT_PHRASES = {"bye", "bye bye", "goodbye", "good bye", "बिदा", "बाइ", "बाइ बाइ", "धन्यवाद", "फेरि भेटौँला"}
QUESTION_WORDS = {"who", "is", "are", "को", "के", "कहाँ", "कहिले"}

# ---- Hardcoded definition (for explicit "what is" only) ----
DEFINITION_VIPAD = (
    "विपद् व्यवस्थापन भनेको प्राकृतिक वा मानव निर्मित प्रकोपहरूसँग जुध्न र त्यसबाट हुने क्षतिलाई कम गर्न गरिने सम्पूर्ण कार्यहरूको संयोजन हो। "
    "यसले विपद् आउनुअघिको तयारीदेखि लिएर विपद् पछिको पुनर्निर्माणसम्मका सबै प्रक्रियाहरूलाई समेट्छ। प्रभावकारी विपद् व्यवस्थापनले समाजलाई सुरक्षित राख्न र संकटको समयमा छिटो तङ्ग्रिन मद्दत गर्छ।"
)

# =========================================================
# TEXT PROCESSING
# =========================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()

def is_exit_intent(query: str) -> bool:
    q = normalize_text(query).lower()
    return bool(q) and any(phrase in q for phrase in EXIT_PHRASES)

def clean_bm25_query(query: str) -> str:
    q = normalize_text(query)
    cleaned_tokens = [t.lower().strip(".,!?;:'\"()[]{}") for t in q.split() if t and t not in QUESTION_WORDS]
    return " ".join(cleaned_tokens) or q

# =========================================================
# PIPELINE CACHING
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

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            _groq = Groq(api_key=api_key)
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

    lexical = clean_bm25_query(query)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            v_fut = executor.submit(vector_store.similarity_search, query, VECTOR_K)
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

    # ---- Enhanced keyword boosting for disaster types ----
    # Boost chunks containing specific disaster type words that appear in the query.
    disaster_keywords = ["भूकम्प", "बाढी", "पहिरो", "आगलागी", "हिमपहिरो", "चट्याङ"]
    for item in fused:
        # For each keyword present in query, boost chunk if it contains that keyword
        for kw in disaster_keywords:
            if kw in query and kw in item["text"]:
                item["score"] *= 1.3  # 30% boost

    # Also boost for definitional queries (as before)
    if "के हो" in query:
        for item in fused:
            if "परिचय" in item["text"] or "आधारभूत" in item["text"]:
                item["score"] *= 1.5

    # Re-sort after boosting
    fused.sort(key=lambda x: x["score"], reverse=True)

    # Debug: print top 3 chunks
    print(f"[RETRIEVED CHUNKS] Top 3 scores:")
    for i, item in enumerate(fused[:3]):
        print(f"  {i+1}. Score {item['score']:.4f}: {item['text'][:100]}...")

    return fused

def build_context(documents, query=""):
    if not documents:
        return ""
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
    # ---- FIX ENCODING ----
    try:
        if 'à¤µ' in query or 'à¤¿' in query:
            fixed = query.encode('latin-1').decode('utf-8')
            print(f"[ENCODING FIX] Original: {query!r} -> Fixed: {fixed!r}")
            query = fixed
    except Exception:
        pass

    query = normalize_text(query)
    print(f"[DEBUG] Normalized query: {query!r}")

    if len(query) < 2:
        return ""

    if is_exit_intent(query):
        return GOODBYE_RESPONSE

    # ---- DIRECT ANSWER: only for explicit "what is" definitions ----
    definition_keywords = ["के हो", "भनेको", "परिभाषा", "अर्थ", "मतलब"]
    if "विपद्" in query and any(kw in query for kw in definition_keywords):
        print("[DIRECT] Returning definition for definitional query.")
        return DEFINITION_VIPAD

    # ---- RAG pipeline ----
    print("[RAG] Proceeding with retrieval and LLM.")
    vector_store, bm25, groq = load_rag_pipeline()
    if not vector_store or not bm25:
        return DATABASE_ERROR
    if not groq:
        return GROQ_UNAVAILABLE

    documents = retrieve_documents(query)
    if not documents:
        # If no documents retrieved, try to give a generic answer based on known patterns
        # For earthquake and flood, we can provide a fallback from our own knowledge (not from doc)
        # but we prefer to say "not found".
        return NO_INFO

    context = build_context(documents, query=query)
    if not context:
        return NO_INFO

    print(f"[CONTEXT PREVIEW] {context[:500]}...")

    system_prompt = f"""
तपाईं "विपद् व्यवस्थापन सूचना सेवा" का नेपाली voice assistant हुनुहुन्छ।
तपाईंको काम: प्रयोगकर्ताको प्रश्नको छोटो, स्पष्ट र सटीक उत्तर दिनुहोस्।
उत्तर दिनको लागि **केवल** दिइएको सन्दर्भ प्रयोग गर्नुहोस्।
यदि सन्दर्भमा उत्तर छैन भने, "माफ गर्नुहोस्, यस विषयमा जानकारी छैन" भन्नुहोस्।
प्रयोगकर्तालाई "तपाईं" भनेर सम्बोधन गर्नुहोस् र "म", "मलाई" जस्ता शब्दहरू प्रयोग नगर्नुहोस्।

सन्दर्भ:
{context}
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

        if ans and len(ans) > 25 and ans[-1] not in ("।", "?", "!", ".", "…"):
            ans += "।"

        return ans or SERVER_ERROR

    except Exception as exc:
        print(f"[GROQ ERROR] {exc}")
        return SERVER_ERROR