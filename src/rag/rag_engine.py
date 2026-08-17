# =========================================================
# src/rag/rag_engine.py
# Generic Multi-Person Voice RAG Information Service
# =========================================================

import os
import time
import pickle
import logging
import warnings
import unicodedata
import re
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import streamlit as st


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.CRITICAL)


# =========================================================
# IMPORTS
# =========================================================

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

# Must be the same model used to create Chroma.
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Groq generation model.
GROQ_MODEL = "openai/gpt-oss-20b"

# GPT-OSS reasoning.
GROQ_REASONING_EFFORT = "low"

# Retrieval.
VECTOR_K = 6
BM25_K = 6

# Context.
FINAL_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 10000

# Enough room to finish answers.
MAX_COMPLETION_TOKENS = 512


# =========================================================
# STANDARD RESPONSES
# =========================================================

NO_INFO = (
    "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"
)

DATABASE_ERROR = (
    "माफ गर्नुहोस्, अहिले जानकारी प्रणालीमा समस्या देखिएको छ।"
)

GROQ_UNAVAILABLE = (
    "माफ गर्नुहोस्, अहिले सूचना सेवा उपलब्ध छैन।"
)

SERVER_ERROR = (
    "माफ गर्नुहोस्, अहिले सर्भरमा समस्या देखिएको छ।"
)

GOODBYE_RESPONSE = (
    "धन्यवाद। फेरि भेटौँला।"
)


# =========================================================
# EXIT PHRASES
# =========================================================

EXIT_PHRASES = {
    # English
    "bye",
    "bye bye",
    "goodbye",
    "good bye",
    "see you",
    "see you later",
    "thanks",
    "thank you",
    "thank you very much",
    "thanks for the conversation",
    "thank you for the conversation",
    "thanks for talking to me",
    "thank you for talking to me",
    "thanks for your help",
    "thank you for your help",

    # Nepali
    "बिदा",
    "बाइ",
    "बाइ बाइ",
    "धन्यवाद",
    "धेरै धन्यवाद",
    "कुरा गरेकोमा धन्यवाद",
    "सहयोगको लागि धन्यवाद",
    "सहयोगका लागि धन्यवाद",
    "अब जान्छु",
    "फेरि भेटौँला",
}


# =========================================================
# DISCOVERY PHRASES
# =========================================================

DISCOVERY_PHRASES = {
    # English
    "whose information",
    "whose information do you have",
    "who do you know",
    "do you know anyone",
    "do you have anyone",
    "who are available",
    "who is available",
    "who are there",
    "who is there",
    "who do we have",
    "what people do you know",
    "which people do you know",

    # Nepali
    "कसको जानकारी",
    "कसको बारेमा जानकारी",
    "कसको बारेमा थाहा",
    "कसको जानकारी छ",
    "कस-कसको जानकारी",
    "कसको बारेमा",
    "कसको विवरण",
    "को को हुनुहुन्छ",
    "को को छन्",
    "को-को हुनुहुन्छ",
    "कस-कसको",
    "को को",
    "हाम्रोमा को",
    "हाम्रोमा कसको",
    "हाम्रोमा को-को",

    # STT variants
    "तपाईंलाई",
    "तपाईलाई",
    "तपाईँलाई",
    "तपाईं लाई",
    "तपाई लाई",
    "तपाईँ लाई",
    "tapai lai",
    "tapailai",
}


# =========================================================
# SERVICE PERSPECTIVE TERMS
# =========================================================

SERVICE_PERSPECTIVE_TERMS = {
    "तपाईंलाई",
    "तपाईलाई",
    "तपाईँलाई",
    "तपाईं लाई",
    "तपाई लाई",
    "तपाईँ लाई",

    "tapai lai",
    "tapailai",
}


# =========================================================
# TECHNICAL DISCOVERY TERMS
# =========================================================

TECHNICAL_TERMS = {
    "technical",
    "technician",
    "engineer",
    "engineering",
    "developer",
    "software",
    "programmer",
    "technology",
    "tech",
    "it",
    "ai",
    "technical person",

    "प्राविधिक",
    "इन्जिनियर",
    "इन्जिनियरिङ",
    "सफ्टवेयर",
    "डेभलपर",
    "प्रोग्रामर",
    "प्रविधि",
    "प्राविधिक व्यक्ति",
    "आईटी",
    "कम्प्युटर",
}


# =========================================================
# PERSON REFERENCE WORDS
# =========================================================

PERSON_PRONOUNS = {
    "उनी",
    "उहाँ",
    "उनको",
    "उनका",
    "उनकी",
    "उनले",
    "उनलाई",
    "उहाँको",
    "उहाँका",
    "उहाँकी",
    "उहाँले",
    "उहाँलाई",

    "his",
    "her",
    "their",
    "he",
    "she",
    "they",
}


# =========================================================
# SESSION MEMORY
# =========================================================

def get_current_person():
    return st.session_state.get(
        "current_person",
        None,
    )


def set_current_person(name):
    if name:
        st.session_state.current_person = name
    else:
        st.session_state.current_person = None


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


normalize_nepali = normalize_text


# =========================================================
# EXIT DETECTION
# =========================================================

def is_exit_intent(query: str) -> bool:

    q = normalize_text(query).lower()

    if not q:
        return False

    if q in EXIT_PHRASES:
        return True

    for phrase in EXIT_PHRASES:
        if phrase in q:
            return True

    return False


# =========================================================
# SERVICE-PERSPECTIVE QUESTION
# =========================================================

def is_service_perspective_question(
    query: str,
) -> bool:

    q = normalize_text(query).lower()

    has_perspective = any(
        phrase.lower() in q
        for phrase in SERVICE_PERSPECTIVE_TERMS
    )

    if not has_perspective:
        return False

    information_terms = {
        "जानकारी",
        "जानकार",
        "थाहा",
        "कसको",
        "क-कसको",
        "कस-कसको",
        "को",
        "who",
        "know",
        "information",
    }

    return any(
        term in q
        for term in information_terms
    )


# =========================================================
# DISCOVERY DETECTION
# =========================================================

def is_discovery_question(query: str) -> bool:

    q = normalize_text(query).lower()

    if not q:
        return False

    for phrase in DISCOVERY_PHRASES:

        if phrase.lower() in q:
            return True

    if (
        "do we have" in q
        or "is there" in q
        or "are there" in q
    ):
        return True

    if (
        "हाम्रोमा" in q
        and (
            "को" in q
            or "कुनै" in q
            or "व्यक्ति" in q
        )
    ):
        return True

    if is_service_perspective_question(q):
        return True

    return False


# =========================================================
# TECHNICAL DISCOVERY
# =========================================================

def is_technical_discovery(query: str) -> bool:

    q = normalize_text(query).lower()

    if not is_discovery_question(q):
        return False

    return any(
        term in q
        for term in TECHNICAL_TERMS
    )


# =========================================================
# FOLLOW-UP DETECTION
# =========================================================

def is_person_followup(query: str) -> bool:

    q = normalize_text(query).lower()

    return any(
        word in q
        for word in PERSON_PRONOUNS
    )


# =========================================================
# PERSON INTRODUCTION DETECTION
# =========================================================

def is_person_intro(query: str) -> bool:

    q = normalize_text(query).lower()

    patterns = [
        "who is",
        "who's",
        "who was",
        "tell me about",
        "can you tell me about",
        "introduce",
        "what is",

        "को हुन्",
        "को हो",
        "को हुनुहुन्छ",
        "को रहेछन्",
        "परिचय",
        "परिचय दिनुहोस्",
        "बारेमा बताउनुहोस्",
        "बारेमा भन्नुहोस्",
    ]

    return any(
        pattern in q
        for pattern in patterns
    )


# =========================================================
# FOLLOW-UP RESOLUTION
# =========================================================

def resolve_followup_query(
    query: str,
) -> str:

    current_person = get_current_person()

    if not current_person:
        return query

    if not is_person_followup(query):
        return query

    resolved = (
        f"{current_person} {query}"
    )

    print(
        f"[MEMORY] Resolved follow-up: "
        f"{resolved}"
    )

    return resolved


# =========================================================
# ENTITY EXPANSION
# =========================================================

def expand_entity_query(
    query: str,
) -> str:

    original = normalize_text(query)

    if not original:
        return ""

    aliases = {
        "arjun sharma": "अर्जुन शर्मा",
        "arjun": "अर्जुन शर्मा",
    }

    lower = original.lower()

    additions = []

    for alias, canonical in aliases.items():

        if alias in lower:
            additions.append(canonical)

    if additions:

        unique = list(
            dict.fromkeys(additions)
        )

        return (
            f"{original} "
            f"{' '.join(unique)}"
        )

    return original


# =========================================================
# BM25 QUERY CLEANING
# =========================================================

QUESTION_WORDS = {
    "who",
    "is",
    "are",
    "was",
    "were",
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "tell",
    "me",
    "about",
    "please",
    "the",
    "a",
    "an",
    "in",
    "of",
    "to",
    "for",
    "on",
    "does",
    "did",
    "has",
    "have",
    "do",
    "we",
    "any",

    "को",
    "का",
    "की",
    "हो",
    "हुन्",
    "हुनुहुन्छ",
    "के",
    "कुन",
    "कहाँ",
    "कहिले",
    "किन",
    "कसरी",
    "कति",
    "बारे",
    "बारेमा",
    "बताउनुहोस्",
    "भन्नुहोस्",
    "गर्नुहोस्",
    "छ",
    "छन्",
    "हाम्रो",
    "हाम्रोमा",
    "कुनै",
    "व्यक्ति",
}


def clean_bm25_query(
    query: str,
) -> str:

    query = normalize_text(query)

    if not query:
        return ""

    result = []

    for token in query.split():

        token = token.lower().strip(
            ".,!?;:'\"()[]{}"
        )

        if not token:
            continue

        if token in QUESTION_WORDS:
            continue

        result.append(token)

    cleaned = " ".join(result)

    return (
        cleaned
        if cleaned
        else query
    )


# =========================================================
# EMBEDDINGS
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():

    print(
        "[INIT] Loading embedding model..."
    )

    try:

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu",
                "local_files_only": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        print(
            "[INIT] Embedding model loaded."
        )

        return embeddings

    except Exception as exc:

        print(
            f"[WARN] Local embedding load failed: {exc}"
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        return embeddings


# =========================================================
# CHROMA
# =========================================================

@st.cache_resource(show_spinner=False)
def load_vector_store():

    if not os.path.exists(
        PERSIST_DIR
    ):

        print(
            f"[ERROR] Chroma database missing: "
            f"{PERSIST_DIR}"
        )

        return None

    embeddings = load_embeddings()

    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    print(
        "[INIT] Chroma ready."
    )

    return vector_store


# =========================================================
# BM25
# =========================================================

@st.cache_resource(show_spinner=False)
def load_bm25():

    if not os.path.exists(
        BM25_PKL_PATH
    ):

        print(
            f"[ERROR] BM25 index missing: "
            f"{BM25_PKL_PATH}"
        )

        return None

    with open(
        BM25_PKL_PATH,
        "rb",
    ) as file:

        bm25 = pickle.load(file)

    bm25.k = BM25_K

    print(
        "[INIT] BM25 ready."
    )

    return bm25


# =========================================================
# GROQ
# =========================================================

@st.cache_resource(show_spinner=False)
def load_groq():

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        print(
            "[GROQ ERROR] "
            "GROQ_API_KEY is not set."
        )

        return None

    print(
        "[GROQ] API key detected."
    )

    try:

        client = Groq(
            api_key=api_key
        )

        # ---------------------------------------------
        # Verify that the current API key can see
        # the configured model.
        # ---------------------------------------------

        models = client.models.list()

        available_models = {
            model.id
            for model in models.data
        }

        print(
            f"[GROQ] Checking model: "
            f"{GROQ_MODEL}"
        )

        if GROQ_MODEL not in available_models:

            print(
                f"[GROQ ERROR] "
                f"{GROQ_MODEL} is not available "
                f"for this API key/project."
            )

            print(
                "[GROQ] Available models:"
            )

            for model_id in sorted(
                available_models
            ):
                print(
                    f"  - {model_id}"
                )

            return None

        print(
            f"[GROQ] Model available: "
            f"{GROQ_MODEL}"
        )

        return client

    except Exception as exc:

        print(
            f"[GROQ INIT ERROR] "
            f"{exc}"
        )

        return None


# =========================================================
# COMPLETE PIPELINE
# =========================================================

@st.cache_resource(show_spinner=False)
def load_rag_pipeline():

    print(
        "[INIT] Loading RAG pipeline..."
    )

    vector_store = load_vector_store()
    bm25 = load_bm25()
    groq_client = load_groq()

    print(
        "[PIPELINE STATUS] "
        f"vector_store={vector_store is not None}, "
        f"bm25={bm25 is not None}, "
        f"groq={groq_client is not None}"
    )

    return (
        vector_store,
        bm25,
        groq_client,
    )


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_documents(
    query: str,
):

    if query is None:

        print(
            "[RAG] None query ignored."
        )

        return []

    query = normalize_text(
        query
    )

    if not query:

        print(
            "[RAG] Empty query ignored."
        )

        return []

    if len(query) < 2:

        print(
            f"[RAG] Query too short: "
            f"{query!r}"
        )

        return []

    vector_store, bm25, _ = (
        load_rag_pipeline()
    )

    if (
        vector_store is None
        or bm25 is None
    ):

        print(
            "[RETRIEVAL] "
            "Vector/BM25 unavailable."
        )

        return []

    start = time.perf_counter()

    resolved_query = (
        resolve_followup_query(
            query
        )
    )

    expanded_query = (
        expand_entity_query(
            resolved_query
        )
    )

    semantic_query = expanded_query

    lexical_query = (
        clean_bm25_query(
            expanded_query
        )
    )

    print(
        f"[QUERY] Original: "
        f"{query}"
    )

    if resolved_query != query:

        print(
            f"[QUERY RESOLVED] "
            f"{resolved_query}"
        )

    print(
        f"[QUERY BM25] "
        f"{lexical_query}"
    )

    # -----------------------------------------------------
    # Parallel retrieval
    # -----------------------------------------------------

    try:

        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:

            vector_future = executor.submit(
                vector_store.similarity_search,
                semantic_query,
                VECTOR_K,
            )

            bm25_future = executor.submit(
                bm25.invoke,
                lexical_query,
            )

            vector_docs = (
                vector_future.result()
            )

            bm25_docs = (
                bm25_future.result()
            )

    except Exception as exc:

        print(
            f"[RETRIEVAL ERROR] "
            f"{exc}"
        )

        return []

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    merged = {}

    for rank, doc in enumerate(
        vector_docs,
        start=1,
    ):

        text = normalize_text(
            doc.page_content
        )

        if not text:
            continue

        merged.setdefault(
            text,
            {
                "doc": doc,
                "vector_rank": rank,
                "bm25_rank": None,
            },
        )

    for rank, doc in enumerate(
        bm25_docs,
        start=1,
    ):

        text = normalize_text(
            doc.page_content
        )

        if not text:
            continue

        if text not in merged:

            merged[text] = {
                "doc": doc,
                "vector_rank": None,
                "bm25_rank": rank,
            }

        else:

            merged[text][
                "bm25_rank"
            ] = rank

    # -----------------------------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------------------------

    RRF_K = 60.0

    fused = []

    for text, item in merged.items():

        score = 0.0

        if item["vector_rank"] is not None:

            score += (
                1.0
                / (
                    RRF_K
                    + item["vector_rank"]
                )
            )

        if item["bm25_rank"] is not None:

            score += (
                1.0
                / (
                    RRF_K
                    + item["bm25_rank"]
                )
            )

        fused.append(
            {
                "score": score,
                "text": text,
                "doc": item["doc"],
                "vector_rank": item["vector_rank"],
                "bm25_rank": item["bm25_rank"],
            }
        )

    fused.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print(
        "[RETRIEVAL] "
        f"vector={len(vector_docs)} "
        f"bm25={len(bm25_docs)} "
        f"unique={len(fused)} "
        f"time={time.perf_counter() - start:.3f}s"
    )

    for index, item in enumerate(
        fused[
            :FINAL_CONTEXT_CHUNKS
        ],
        start=1,
    ):

        preview = (
            item["text"][:160]
            .replace("\n", " ")
        )

        print(
            f"[RESULT {index}] "
            f"rrf={item['score']:.6f} "
            f"vector={item['vector_rank']} "
            f"bm25={item['bm25_rank']} "
            f"{preview}"
        )

    return fused


# =========================================================
# CONTEXT BUILDER
# =========================================================

def build_context(
    documents,
):

    if not documents:
        return ""

    chunks = []
    total_chars = 0

    for item in documents[
        :FINAL_CONTEXT_CHUNKS
    ]:

        text = item["text"].strip()

        if not text:
            continue

        remaining = (
            MAX_CONTEXT_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        text = text[
            :remaining
        ]

        chunks.append(
            text
        )

        total_chars += len(
            text
        )

    return "\n\n---\n\n".join(
        chunks
    )


# =========================================================
# SERVICE PERSPECTIVE CORRECTION
# =========================================================

def correct_service_perspective(
    answer: str,
    query: str,
) -> str:

    if not answer:
        return ""

    answer = answer.strip()

    if not is_discovery_question(
        query
    ):
        return answer

    replacements = [
        ("तपाईंलाई", "मलाई"),
        ("तपाईलाई", "मलाई"),
        ("तपाईँलाई", "मलाई"),
        ("तपाईं लाई", "मलाई"),
        ("तपाई लाई", "मलाई"),
        ("तपाईँ लाई", "मलाई"),
        ("tapai lai", "malai"),
        ("tapailai", "malai"),
    ]

    corrected = answer

    for old, new in replacements:

        corrected = corrected.replace(
            old,
            new,
        )

    # Beginning of answer.
    corrected = re.sub(
        r"^\s*तपाईंलाई\s+",
        "मलाई ",
        corrected,
    )

    corrected = re.sub(
        r"^\s*तपाईलाई\s+",
        "मलाई ",
        corrected,
    )

    corrected = re.sub(
        r"^\s*तपाईँलाई\s+",
        "मलाई ",
        corrected,
    )

    return corrected.strip()


# =========================================================
# MAIN ANSWER
# =========================================================

def answer_user_query(
    query: str,
) -> str:

    total_start = time.perf_counter()

    # =====================================================
    # ABSOLUTE EMPTY INPUT PROTECTION
    # =====================================================

    if query is None:

        print(
            "[RAG] None query ignored. "
            "No retrieval. No Groq."
        )

        return ""

    query = normalize_text(
        query
    )

    if not query:

        print(
            "[RAG] Blank query ignored. "
            "No retrieval. No Groq."
        )

        return ""

    if len(query) < 2:

        print(
            f"[RAG] Too-short query ignored: "
            f"{query!r}"
        )

        return ""

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"[RAG START] {query!r}"
    )

    print(
        "=" * 70
    )

    # =====================================================
    # EXIT
    # =====================================================

    if is_exit_intent(
        query
    ):

        set_current_person(
            None
        )

        return GOODBYE_RESPONSE

    # =====================================================
    # LOAD PIPELINE
    # =====================================================

    (
        vector_store,
        bm25,
        groq,
    ) = load_rag_pipeline()

    print(
        "[PIPELINE STATUS] "
        f"vector_store={vector_store is not None}, "
        f"bm25={bm25 is not None}, "
        f"groq={groq is not None}"
    )

    if (
        vector_store is None
        or bm25 is None
    ):

        return DATABASE_ERROR

    if groq is None:

        print(
            "[RAG] Groq client unavailable."
        )

        return GROQ_UNAVAILABLE

    # =====================================================
    # RETRIEVAL
    # =====================================================

    retrieval_start = (
        time.perf_counter()
    )

    documents = retrieve_documents(
        query
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    print(
        f"[TIMING] Retrieval: "
        f"{retrieval_time:.3f}s"
    )

    if not documents:

        return NO_INFO

    # =====================================================
    # INTENT
    # =====================================================

    discovery = (
        is_discovery_question(
            query
        )
    )

    technical_discovery = (
        is_technical_discovery(
            query
        )
    )

    followup = (
        is_person_followup(
            query
        )
    )

    intro = (
        is_person_intro(
            query
        )
    )

    service_perspective = (
        is_service_perspective_question(
            query
        )
    )

    print(
        "[INTENT] "
        f"discovery={discovery}, "
        f"technical={technical_discovery}, "
        f"followup={followup}, "
        f"intro={intro}, "
        f"service_perspective={service_perspective}"
    )

    # =====================================================
    # CONTEXT
    # =====================================================

    context = build_context(
        documents
    )

    if not context:

        return NO_INFO

    # =====================================================
    # TASK
    # =====================================================

    if technical_discovery:

        task_instruction = """
प्रयोगकर्ताले प्राविधिक वा प्रविधिसम्बन्धी
क्षेत्रमा पर्ने व्यक्ति खोजिरहेको छ।

सन्दर्भमा प्राविधिक काम, सफ्टवेयर, इन्जिनियरिङ,
प्रोग्रामिङ, प्रविधि वा सम्बन्धित सीप भएका
व्यक्तिहरू पहिचान गर्नुहोस्।

मिल्ने व्यक्तिको नाम र भूमिका बताउनुहोस्।

एक जना मात्र मिलेमा एक जनाको नाम बताउनुहोस्।
धेरै जना मिलेमा छोटकरीमा सबैको नाम बताउनुहोस्।

उत्तरमा "तपाईंलाई" प्रयोग नगर्नुहोस्।
"""

    elif service_perspective:

        task_instruction = """
प्रयोगकर्ताले सेवासँग उपलब्ध जानकारीबारे सोधिरहेको छ।

उत्तर सेवाको पहिलो व्यक्तिको दृष्टिकोणबाट दिनुहोस्।

अत्यन्त महत्वपूर्ण:

"तपाईंलाई", "तपाईलाई", "तपाईँलाई",
"तपाईं लाई", "तपाई लाई", "tapai lai",
"tapailai" प्रयोग नगर्नुहोस्।

सधैं "मलाई" प्रयोग गर्नुहोस्।

उदाहरण:

प्रश्न:
"तपाईंलाई कस-कसको बारेमा जानकारी छ?"

सही:
"मलाई अर्जुन शर्मा, उहाँकी बहिनी तथा
उहाँका आमाबुबाको बारेमा जानकारी छ।"

गलत:
"तपाईंलाई अर्जुन शर्मा..."

सन्दर्भमा उल्लेख भएका सान्दर्भिक व्यक्तिहरू,
परिवारका सदस्यहरू वा सम्बन्धित व्यक्तिहरूको
जानकारी प्रश्नसँग सम्बन्धित भए उनीहरूको
नाम/सम्बन्ध स्पष्ट रूपमा बताउनुहोस्।
"""

    elif discovery:

        task_instruction = """
प्रयोगकर्ताले हाम्रो जानकारीमा रहेका व्यक्ति
वा सम्बन्धित व्यक्ति खोजिरहेको छ।

सन्दर्भमा उपलब्ध व्यक्तिहरू पहिचान गर्नुहोस्।

"कसको जानकारी छ?" जस्तो प्रश्न भए व्यक्तिहरूको
नाम स्पष्ट रूपमा बताउनुहोस्।

यदि सन्दर्भमा परिवारका सदस्य वा अन्य व्यक्तिहरूको
बारेमा छुट्टै जानकारी उपलब्ध छ भने प्रश्नले
त्यो मागेको अवस्थामा उनीहरूलाई पनि समावेश गर्नुहोस्।

कुनै विशेष विषय वा क्षेत्र उल्लेख गरिएको छ भने
त्यससँग सम्बन्धित व्यक्तिलाई प्राथमिकता दिनुहोस्।

उत्तर सेवाको दृष्टिकोणबाट दिनुहोस्।
"""

    elif intro:

        task_instruction = """
प्रयोगकर्ताले कुनै व्यक्तिको परिचय मागेको छ।

सन्दर्भमा भएको नाम, पेशा, स्थान, अनुभव र अन्य
मुख्य परिचयात्मक तथ्यबाट छोटो तर पूर्ण परिचय दिनुहोस्।

नाम मात्र दोहोर्याएर उत्तर नदिनुहोस्।
"""

    elif followup:

        task_instruction = """
प्रयोगकर्ता अघिल्लो कुराकानीमा उल्लेख भएको
व्यक्तिबारे थप प्रश्न गरिरहेको छ।

त्यही व्यक्तिसँग सम्बन्धित सन्दर्भबाट
सीधै प्रश्नको उत्तर दिनुहोस्।
"""

    else:

        task_instruction = """
प्रयोगकर्ताको प्रश्नको सिधा उत्तर दिनुहोस्।

सम्बन्धित context chunks का तथ्यहरू जोडेर
पूर्ण तर संक्षिप्त उत्तर बनाउनुहोस्।
"""

    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

    current_person = (
        get_current_person()
    )

    system_prompt = f"""
तपाईं एक स्वचालित टेलिफोन सूचना सेवाको
उत्तर जनरेटर हुनुहुन्छ।

तपाईंको ज्ञानको एकमात्र स्रोत तलको सन्दर्भ हो।

कडा नियम:

१. सन्दर्भ बाहिरको सामान्य ज्ञान प्रयोग नगर्नुहोस्।

२. आफ्नो प्रशिक्षणबाट आएको तथ्य थपेर उत्तर नदिनुहोस्।

३. अनुमान गरेर नयाँ तथ्य नबनाउनुहोस्।

४. सन्दर्भमा उत्तर भेटिएमा सिधै उत्तर दिनुहोस्।

५. सन्दर्भमा पर्याप्त जानकारी नभएमा:
"माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"
भन्नुहोस्।

६. प्रयोगकर्ताले कुनै व्यक्तिको बारेमा सोधेमा
त्यही व्यक्तिको जानकारी दिनुहोस्।

७. आफूलाई AI, chatbot, voice assistant,
virtual assistant वा language model भनेर
परिचय नदिनुहोस्।

८. "Who is Arjun Sharma?" अथवा
"अर्जुन शर्मा को हुन्?" जस्ता प्रश्नमा
अर्जुन शर्माको परिचय दिनुहोस्,
सेवाको परिचय होइन।

९. सेवासँग उपलब्ध जानकारीबारे प्रश्न हुँदा
सेवाको पहिलो व्यक्तिको दृष्टिकोणबाट उत्तर दिनुहोस्।

१०. विशेष रूपमा:

प्रश्न:
"तपाईंलाई कस-कसको बारेमा जानकारी छ?"

उत्तर:
"मलाई अर्जुन शर्मा, उहाँकी बहिनी तथा
उहाँका आमाबुबाको बारेमा जानकारी छ।"

कहिल्यै:
"तपाईंलाई अर्जुन शर्मा..."
नभन्नुहोस्।

११. "तपाईंलाई", "तपाईलाई", "तपाईँलाई",
"तपाईं लाई", "तपाई लाई", "tapai lai",
"tapailai" प्रयोगकर्ताको प्रश्नमा भए पनि
उत्तरमा ती शब्दहरू नक्कल नगर्नुहोस्।

१२. सेवाको दृष्टिकोणबाट "मलाई" प्रयोग गर्नुहोस्।

१३. "कसको जानकारी छ?", "हाम्रोमा को-को हुनुहुन्छ?"
जस्ता प्रश्नमा सन्दर्भमा उपलब्ध व्यक्तिहरूको
नाम बताउनुहोस्।

१४. "कुनै प्राविधिक व्यक्ति हुनुहुन्छ?",
"कुनै इन्जिनियर हुनुहुन्छ?" जस्ता प्रश्नमा
सन्दर्भमा मिल्ने व्यक्तिको नाम र भूमिका बताउनुहोस्।

१५. "उनी", "उहाँ", "उनको", "उनले" जस्ता
सर्वनामलाई कुराकानीमा सम्झिएको व्यक्तिसँग
जोड्नुहोस्।

१६. उत्तर केवल नेपाली देवनागरीमा दिनुहोस्।

१७. फोनमा बोल्न प्राकृतिक सुनिने भाषा प्रयोग गर्नुहोस्।

१८. सामान्य प्रश्नमा १ देखि ४ वटा छोटा वाक्य पर्याप्त छन्।

१९. उत्तरको अन्तिम वाक्य पूरा गर्नुहोस्।
अधुरो वाक्यमा कहिल्यै रोक्नु हुँदैन।

२०. अनावश्यक भूमिका वा लामो व्याख्या नगर्नुहोस्।

२१. "प्राप्त जानकारी अनुसार",
"कागजात अनुसार",
"डाटा अनुसार"
जस्ता औपचारिक वाक्यांश प्रयोग नगर्नुहोस्।

२२. सन्दर्भमा नभएको तथ्य कहिल्यै नबनाउनुहोस्।

२३. प्रश्नको उत्तर दिँदा प्रश्नको विषयमै केन्द्रित
रहनुहोस्। आफू, यो सेवा वा मोडेलको परिचयतर्फ
विषय नबदल्नुहोस्।

हालको कुराकानीमा सम्झिएको व्यक्ति:
{current_person or "कुनै निश्चित व्यक्ति छैन"}

यस प्रश्नको उद्देश्य:
{task_instruction}

सन्दर्भ:
==================================================
{context}
==================================================
"""

    # =====================================================
    # GROQ / GPT-OSS 20B
    # =====================================================

    llm_start = time.perf_counter()

    try:

        response = (
            groq
            .chat
            .completions
            .create(
                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],

                temperature=0.0,

                reasoning_effort=(
                    GROQ_REASONING_EFFORT
                ),

                include_reasoning=False,

                max_completion_tokens=(
                    MAX_COMPLETION_TOKENS
                ),

                stream=False,
            )
        )

        # =================================================
        # SAFE RESPONSE EXTRACTION
        # =================================================

        choices = getattr(
            response,
            "choices",
            None,
        )

        if not choices:

            print(
                "[GROQ ERROR] "
                "No choices returned."
            )

            return SERVER_ERROR

        message = choices[0].message

        content = getattr(
            message,
            "content",
            None,
        )

        reasoning = getattr(
            message,
            "reasoning",
            None,
        )

        print(
            "[GROQ DEBUG] "
            f"content_present={bool(content)} "
            f"reasoning_present={bool(reasoning)}"
        )

        if reasoning:

            print(
                "[GROQ DEBUG] "
                f"Reasoning length: "
                f"{len(reasoning)}"
            )

        answer = ""

        if content:

            answer = str(
                content
            ).strip()

        # -------------------------------------------------
        # Defensive model_dump fallback
        # -------------------------------------------------

        if not answer:

            print(
                "[GROQ] Final content empty."
            )

            raw_dict = {}

            try:

                raw_dict = (
                    message.model_dump()
                )

            except Exception:

                try:

                    raw_dict = vars(
                        message
                    )

                except Exception:

                    raw_dict = {}

            possible_content = (
                raw_dict.get(
                    "content"
                )
                or raw_dict.get(
                    "text"
                )
            )

            if possible_content:

                answer = str(
                    possible_content
                ).strip()

        if not answer:

            print(
                "[GROQ ERROR] "
                "Model returned no final answer."
            )

            return SERVER_ERROR

        # =================================================
        # CLEAN
        # =================================================

        answer = re.sub(
            r"^[`\"']+|[`\"']+$",
            "",
            answer,
        ).strip()

        answer = re.sub(
            r"^(उत्तर|Answer)\s*[:：]\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()

        # =================================================
        # SERVICE PERSPECTIVE CORRECTION
        # =================================================

        before_perspective = answer

        answer = correct_service_perspective(
            answer,
            query,
        )

        if answer != before_perspective:

            print(
                "[PERSPECTIVE] "
                "Corrected response perspective."
            )

        # =================================================
        # TERMINAL PUNCTUATION
        # =================================================

        if (
            len(answer) > 25
            and answer[-1] not in (
                "।",
                "?",
                "!",
                ".",
                "…",
            )
        ):

            answer += "।"

        # =================================================
        # TIMING
        # =================================================

        llm_time = (
            time.perf_counter()
            - llm_start
        )

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[MODEL] {GROQ_MODEL}"
        )

        print(
            f"[TIMING] LLM: "
            f"{llm_time:.3f}s"
        )

        print(
            f"[TIMING] TOTAL: "
            f"{total_time:.3f}s"
        )

        print(
            f"[ANSWER] {answer}"
        )

        print(
            "=" * 70
        )

        return answer

    except Exception as exc:

        print(
            f"[GROQ ERROR] {exc}"
        )

        return SERVER_ERROR


# =========================================================
# DEBUG
# =========================================================

def debug_query(
    query: str,
):

    print(
        "\n"
        + "#" * 70
    )

    print(
        f"QUERY: {query!r}"
    )

    print(
        f"MODEL: {GROQ_MODEL}"
    )

    print(
        f"DISCOVERY: "
        f"{is_discovery_question(query)}"
    )

    print(
        f"SERVICE PERSPECTIVE: "
        f"{is_service_perspective_question(query)}"
    )

    print(
        f"TECHNICAL: "
        f"{is_technical_discovery(query)}"
    )

    print(
        f"FOLLOWUP: "
        f"{is_person_followup(query)}"
    )

    print(
        f"INTRO: "
        f"{is_person_intro(query)}"
    )

    results = retrieve_documents(
        query
    )

    print(
        f"RESULTS: {len(results)}"
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- RESULT {index} ---"
        )

        print(
            f"RRF: "
            f"{item['score']:.6f}"
        )

        print(
            f"VECTOR: "
            f"{item['vector_rank']}"
        )

        print(
            f"BM25: "
            f"{item['bm25_rank']}"
        )

        print(
            item["text"]
        )

    print(
        "#" * 70
    )