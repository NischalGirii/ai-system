import os, time, pickle, logging, warnings, unicodedata, re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
os.environ.update({"TRANSFORMERS_VERBOSITY": "error", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

from groq import Groq

# Configuration Constants
PERSIST_DIR = "data/chroma_db"
BM25_PKL_PATH = "data/bm25_retriever.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_REASONING_EFFORT = "low"
VECTOR_K = BM25_K = FINAL_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 10000
MAX_COMPLETION_TOKENS = 512

# Standard System Responses
NO_INFO = "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"
DATABASE_ERROR = "माफ गर्नुहोस्, अहिले जानकारी प्रणालीमा समस्या देखिएको छ।"
GROQ_UNAVAILABLE = "माफ गर्नुहोस्, अहिले सूचना सेवा उपलब्ध छैन।"
SERVER_ERROR = "माफ गर्नुहोस्, अहिले सर्भरमा समस्या देखिएको छ।"
GOODBYE_RESPONSE = "धन्यवाद। फेरि भेटौँला।"

EXIT_PHRASES = {
    "bye", "bye bye", "goodbye", "good bye", "see you", "see you later", "thanks", "thank you",
    "thank you very much", "thanks for the conversation", "thank you for the conversation",
    "thanks for talking to me", "thank you for talking to me", "thanks for your help",
    "thank you for your help", "बिदा", "बाइ", "बाइ बाइ", "धन्यवाद", "धेरै धन्यवाद",
    "कुरा गरेकोमा धन्यवाद", "सहयोगको लागि धन्यवाद", "सहयोगका लागि धन्यवाद", "अब जान्छु", "फेरि भेटौँला"
}

DISCOVERY_PHRASES = {
    "whose information", "whose information do you have", "who do you know", "do you know anyone",
    "do you have anyone", "who are available", "who is available", "who are there", "who is there",
    "who do we have", "what people do you know", "which people do you know", "कसको जानकारी",
    "कसको बारेमा जानकारी", "कसको बारेमा थाहा", "कसको जानकारी छ", "कस-कसको जानकारी", "कसको बारेमा",
    "कसको विवरण", "को को हुनुहुन्छ", "को को छन्", "को-को हुनुहुन्छ", "कस-कसको", "को को",
    "हाम्रोमा को", "हाम्रोमा कसको", "हाम्रोमा को-को", "तपाईंलाई", "तपाईलाई", "तपाईँलाई",
    "तपाईं लाई", "तपाई लाई", "तपाईँ लाई", "tapai lai", "tapailai"
}

SERVICE_PERSPECTIVE_TERMS = {"तपाईंलाई", "तपाईलाई", "तपाईँलाई", "तपाईं लाई", "तपाई लाई", "तपाईँ लाई", "tapai lai", "tapailai"}

TECHNICAL_TERMS = {
    "technical", "technician", "engineer", "engineering", "developer", "software", "programmer",
    "technology", "tech", "it", "ai", "technical person", "प्राविधिक", "इन्जिनियर", "इन्जिनियरिङ",
    "सफ्टवेयर", "डेभलपर", "प्रोग्रामर", "प्रविधि", "प्राविधिक व्यक्ति", "आईटी", "कम्प्युटर"
}

PERSON_PRONOUNS = {"उनी", "उहाँ", "उनको", "उनका", "उनकी", "उनले", "उनलाई", "उहाँको", "उहाँका", "उहाँकी", "उहाँले", "उहाँलाई", "his", "her", "their", "he", "she", "they"}

QUESTION_WORDS = {
    "who", "is", "are", "was", "were", "what", "where", "when", "why", "how", "which", "tell", "me",
    "about", "please", "the", "a", "an", "in", "of", "to", "for", "on", "does", "did", "has", "have",
    "do", "we", "any", "को", "का", "की", "हो", "हुन्", "हुनुहुन्छ", "के", "कुन", "कहाँ", "कहिले",
    "किन", "कसरी", "कति", "बारे", "बारेमा", "बताउनुहोस्", "भन्नुहोस्", "गर्नुहोस्", "छ", "छन्",
    "हाम्रो", "हाम्रोमा", "कुनै", "व्यक्ति"
}

# Session Memory Helpers
get_current_person = lambda: st.session_state.get("current_person", None)
def set_current_person(name): st.session_state.current_person = name if name else None

# Text Processing Helpers
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()

def is_exit_intent(query: str) -> bool:
    q = normalize_text(query).lower()
    return bool(q) and any(phrase in q for phrase in EXIT_PHRASES)

def analyze_intent(query: str) -> dict:
    q = normalize_text(query).lower()
    has_perspective = any(term in q for term in SERVICE_PERSPECTIVE_TERMS)
    service_quest = has_perspective and any(t in q for t in {"जानकारी", "जानकार", "थाहा", "कसको", "क-कसको", "कस-कसको", "को", "who", "know", "information"})
    
    is_discovery = (
        any(p in q for p in DISCOVERY_PHRASES) or
        any(k in q for k in ["do we have", "is there", "are there"]) or
        ("हाम्रोमा" in q and any(w in q for w in ["को", "कुनै", "व्यक्ति"])) or
        service_quest
    )
    
    return {
        "discovery": is_discovery,
        "technical": is_discovery and any(t in q for t in TECHNICAL_TERMS),
        "service_perspective": service_quest,
        "followup": any(w in q for w in PERSON_PRONOUNS),
        "intro": any(p in q for p in ["who is", "who's", "who was", "tell me about", "can you tell me about", "introduce", "what is", "को हुन्", "को हो", "को हुनुहुन्छ", "को रहेछन्", "परिचय", "परिचय दिनुहोस्", "बारेमा बताउनुहोस्", "बारेमा भन्नुहोस्"])
    }

def resolve_followup_query(query: str) -> str:
    person = get_current_person()
    q = normalize_text(query).lower()
    if person and any(word in q for word in PERSON_PRONOUNS):
        return f"{person} {query}"
    return query

def expand_entity_query(query: str) -> str:
    orig = normalize_text(query)
    aliases = {"arjun sharma": "अर्जुन शर्मा", "arjun": "अर्जुन शर्मा"}
    additions = [canon for alias, canon in aliases.items() if alias in orig.lower()]
    return f"{orig} {' '.join(dict.fromkeys(additions))}".strip() if additions else orig

def clean_bm25_query(query: str) -> str:
    q = normalize_text(query)
    cleaned = " ".join([t.lower().strip(".,!?;:'\"()[]{}") for t in q.split() if t.lower().strip(".,!?;:'\"()[]{}") not in QUESTION_WORDS])
    return cleaned or q

# Model & Store Resource Loaders
@st.cache_resource(show_spinner=False)
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu", "local_files_only": True}, encode_kwargs={"normalize_embeddings": True})
    except Exception:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})

@st.cache_resource(show_spinner=False)
def load_vector_store():
    if not os.path.exists(PERSIST_DIR): return None
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=load_embeddings())

@st.cache_resource(show_spinner=False)
def load_bm25():
    if not os.path.exists(BM25_PKL_PATH): return None
    with open(BM25_PKL_PATH, "rb") as f:
        bm25 = pickle.load(f)
    bm25.k = BM25_K
    return bm25

@st.cache_resource(show_spinner=False)
def load_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return None
    try:
        client = Groq(api_key=api_key)
        if GROQ_MODEL in {m.id for m in client.models.list().data}:
            return client
    except Exception as exc:
        print(f"[GROQ INIT ERROR] {exc}")
    return None

@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    return load_vector_store(), load_bm25(), load_groq()

# Hybrid Document Retrieval (Reciprocal Rank Fusion)
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
            vector_docs, bm25_docs = v_fut.result(), b_fut.result()
    except Exception as exc:
        print(f"[RETRIEVAL ERROR] {exc}")
        return []

    merged = {}
    for r, doc in enumerate(vector_docs, start=1):
        txt = normalize_text(doc.page_content)
        if txt: merged[txt] = {"doc": doc, "vector_rank": r, "bm25_rank": None}

    for r, doc in enumerate(bm25_docs, start=1):
        txt = normalize_text(doc.page_content)
        if txt:
            if txt not in merged: merged[txt] = {"doc": doc, "vector_rank": None, "bm25_rank": r}
            else: merged[txt]["bm25_rank"] = r

    RRF_K = 60.0
    fused = []
    for txt, item in merged.items():
        sc = (1.0 / (RRF_K + item["vector_rank"]) if item["vector_rank"] else 0.0) + \
             (1.0 / (RRF_K + item["bm25_rank"]) if item["bm25_rank"] else 0.0)
        fused.append({"score": sc, "text": txt, "doc": item["doc"], "vector_rank": item["vector_rank"], "bm25_rank": item["bm25_rank"]})

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused

def build_context(documents):
    if not documents: return ""
    chunks, total_chars = [], 0
    for item in documents[:FINAL_CONTEXT_CHUNKS]:
        txt = item["text"].strip()
        if not txt: continue
        rem = MAX_CONTEXT_CHARS - total_chars
        if rem <= 0: break
        chunks.append(txt[:rem])
        total_chars += len(txt[:rem])
    return "\n\n---\n\n".join(chunks)

def correct_service_perspective(answer: str, query: str) -> str:
    if not answer or not analyze_intent(query)["discovery"]: return answer.strip()
    corrected = answer.strip()
    for old in SERVICE_PERSPECTIVE_TERMS:
        corrected = corrected.replace(old, "मलाई")
    return re.sub(r"^\s*(तपाईंलाई|तपाईलाई|तपाईँलाई)\s+", "मलाई ", corrected).strip()

# Main Query Execution Pipeline
# =========================================================
# MAIN ANSWER (Fixed Discovery Dump)
# =========================================================

def answer_user_query(query: str) -> str:
    query = normalize_text(query)
    if len(query) < 2: return ""
    
    if is_exit_intent(query):
        set_current_person(None)
        return GOODBYE_RESPONSE

    vector_store, bm25, groq = load_rag_pipeline()
    if not vector_store or not bm25: return DATABASE_ERROR
    if not groq: return GROQ_UNAVAILABLE

    documents = retrieve_documents(query)
    if not documents: return NO_INFO

    intents = analyze_intent(query)
    context = build_context(documents)
    if not context: return NO_INFO

    # ---------------------------------------------------------
    # TASK INSTRUCTIONS (Strict Scope Limits)
    # ---------------------------------------------------------
    if intents["technical"]:
        task = "प्रयोगकर्ताले प्राविधिक व्यक्ति खोजिरहेको छ। केवल नाम र भूमिका मात्र बताउनुहोस् (जस्तै: 'मलाई अर्जुन शर्माको बारेमा थाहा छ, उहाँ सफ्टवेयर इन्जिनियर हुनुहुन्छ।')। विस्तृत जीवनी नदिनुहोस्।"
    elif intents["service_perspective"] or intents["discovery"]:
        task = "प्रयोगकर्ताले कसको जानकारी छ भनेर सोधिरहेको छ। केवल नाम र पेशा/भूमिका मात्र बताउनुहोस् (जस्तै: 'मलाई अर्जुन शर्माको बारेमा जानकारी छ, उहाँ सफ्टवेयर इन्जिनियर हुनुहुन्छ।')। उहाँको जन्म, शिक्षा, परिवार वा रुचि जस्ता विवरणहरू पटक्कै नदिनुहोस्।"
    elif intents["intro"]:
        task = "प्रयोगकर्ताले व्यक्तिको परिचय मागेको छ। नाम, पेशा र १-२ मुख्य तथ्यबाट छोटो परिचय दिनुहोस्।"
    elif intents["followup"]:
        task = "प्रयोगकर्ता अघिल्लो व्यक्तिबारे थप प्रश्न गर्दैछ। सोधिएको विशिष्ट कुराको मात्र उत्तर दिनुहोस्।"
    else:
        task = "प्रयोगकर्ताको प्रश्नको सिधा र छोटो उत्तर दिनुहोस्।"

    current_person = get_current_person() or "कुनै निश्चित व्यक्ति छैन"
    
    # ---------------------------------------------------------
    # SYSTEM PROMPT (Strict Overview Boundaries)
    # ---------------------------------------------------------
    system_prompt = f"""तपाईं फोनमा कुरा गरिरहेको एक सहयोगी व्यक्ति हुनुहुन्छ।

    कडा नियमहरू:
    1. डोमेन बाहिरको बन्देज (STRICT DOMAIN LIMIT): तपाईंको काम केवल दिइएको 'सन्दर्भ' (Context) भित्रका व्यक्तिहरूको व्यक्तिगत र व्यावसायिक जानकारी दिनु मात्र हो। सामान्य ज्ञान, देशको राजधानी, भूगोल, गणित, वा इतिहास जस्ता विषयमा कहिल्यै उत्तर नदिनुहोस्। यदि प्रश्न सन्दर्भका व्यक्तिहरूसँग सीधै सम्बन्धित छैन भने, सीधै "{NO_INFO}" भन्नुहोस्।
    2. तपाईं सन्दर्भमा उल्लेख गरिएको व्यक्ति हुनुहुन्न। कहिल्यै पनि 'म [व्यक्तिको नाम] हुँ' नभन्नुहोस्।
    3. उत्तर दिँदा तेस्रो पुरुष ('उहाँ', 'उहाँको') प्रयोग गर्नुहोस्।
    4. 'कसको जानकारी छ?' जस्ता प्रश्नमा उपलब्ध व्यक्तिको नाम र भूमिका मात्र भन्नुहोस्। नसोधिएका विस्तृत विवरण नदिनुहोस्।
    5. 'प्राप्त जानकारी अनुसार', 'रेकर्ड अनुसार', 'डेटाबेसमा', वा 'कागजात अनुसार' जस्ता शब्दहरू कहिल्यै प्रयोग नगर्नुहोस्। मानौं तपाईंलाई यो कुरा पहिल्यै थाहा छ।
    6. सन्दर्भमा उत्तर स्पष्ट नभए अनुमान नगर्नुहोस्, सीधै "{NO_INFO}" भन्नुहोस्।

    हालको सम्झिएको व्यक्ति: {current_person}
    यस प्रश्नको उद्देश्य: {task}

    सन्दर्भ:
    {context}"""

    try:
        res = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            temperature=0.0,
            reasoning_effort=GROQ_REASONING_EFFORT,
            include_reasoning=False,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            stream=False
        )
        choices = getattr(res, "choices", None)
        if not choices: return SERVER_ERROR

        msg = choices[0].message
        ans = str(getattr(msg, "content", "") or "").strip()
        
        if not ans:
            raw = getattr(msg, "model_dump", lambda: {})() or vars(msg)
            ans = str(raw.get("content") or raw.get("text") or "").strip()
        if not ans: return SERVER_ERROR

        ans = re.sub(r"^[`\"']+|[`\"']+$", "", ans).strip()
        ans = re.sub(r"^(उत्तर|Answer)\s*[:：]\s*", "", ans, flags=re.IGNORECASE).strip()
        ans = correct_service_perspective(ans, query)

        if len(ans) > 25 and ans[-1] not in ("।", "?", "!", ".", "…"):
            ans += "।"

        return ans
    except Exception as exc:
        print(f"[GROQ ERROR] {exc}")
        return SERVER_ERROR