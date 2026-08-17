import os
import time
import pickle
import logging
import warnings
import unicodedata
import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st


# =========================================================
# ENVIRONMENT
# =========================================================

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

# IMPORTANT:
# This must be the SAME model that was used to build Chroma.
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

GROQ_MODEL = "llama-3.1-8b-instant"

# Retrieval candidates.
VECTOR_K = 6
BM25_K = 6

# Final context chunks sent to Groq.
FINAL_CONTEXT_CHUNKS = 5

# Maximum context size.
MAX_CONTEXT_CHARS = 9000

# Maximum generated answer size.
MAX_RESPONSE_TOKENS = 120


# =========================================================
# STANDARD RESPONSES
# =========================================================

NO_INFO = (
    "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"
)

DATABASE_ERROR = (
    "माफ गर्नुहोस्, अहिले जानकारी प्रणालीमा समस्या देखिएको छ।"
)

SERVER_ERROR = (
    "माफ गर्नुहोस्, अहिले सर्भरमा समस्या देखिएको छ।"
)

GOODBYE_RESPONSE = (
    "धन्यवाद। फेरि भेटौँला।"
)


# =========================================================
# KNOWLEDGE DOMAIN
# =========================================================
#
# This assistant is currently a CLOSED-DOMAIN biography
# assistant about one person.
#
# The important distinction is:
#
#   "Is the question about Arjun?"
#
# NOT:
#
#   "Do the exact words of the question occur in a chunk?"
#
# This is what prevents:
#
#   "Which is the tallest mountain in Nepal?"
#
# from being answered with the model's outside knowledge.
# =========================================================

ENTITY_ALIASES = {
    # English / Romanized
    "arjun sharma": "अर्जुन शर्मा",
    "arjun": "अर्जुन शर्मा",

    # Nepali
    "अर्जुन शर्मा": "अर्जुन शर्मा",
    "अर्जुन": "अर्जुन शर्मा",
}


# Words that indicate the user is referring back to the
# already-known person.
PERSON_REFERENCE_WORDS = {
    "उनी",
    "उहाँ",
    "उनको",
    "उनका",
    "उनकी",
    "उनले",
    "उहाँको",
    "उहाँका",
    "उहाँकी",
    "अर्जुनको",
    "अर्जुनका",
    "अर्जुनकी",
    "अर्जुनले",
    "अर्जुनलाई",
    "अर्जुनसँग",
    "अर्जुनबाट",
}


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
# NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """
    Unicode normalization + whitespace cleanup.
    """

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


# Keep old function name available if other files import it.
normalize_nepali = normalize_text


# =========================================================
# EXIT INTENT
# =========================================================

def is_exit_intent(query: str) -> bool:
    """
    Detect conversation-ending phrases.
    """

    query = normalize_text(query).lower()

    if not query:
        return False

    if query in EXIT_PHRASES:
        return True

    for phrase in EXIT_PHRASES:
        if phrase in query:
            return True

    return False


# =========================================================
# ENTITY QUERY EXPANSION
# =========================================================

def expand_entity_query(query: str) -> str:
    """
    Add the Nepali canonical entity form when the user
    speaks in English/Romanized form.

    Example:

        Who is Arjun Sharma?
        ->
        Who is Arjun Sharma? अर्जुन शर्मा
    """

    original = normalize_text(query)

    if not original:
        return ""

    lower = original.lower()

    additions = []

    for alias, canonical in ENTITY_ALIASES.items():

        if alias in lower:
            additions.append(canonical)

    if additions:

        unique = list(dict.fromkeys(additions))

        return (
            f"{original} "
            f"{' '.join(unique)}"
        )

    return original


# =========================================================
# DETECT WHETHER QUESTION IS ABOUT ARJUN
# =========================================================

def is_arjun_question(query: str) -> bool:
    """
    Determine whether a question belongs to the current
    knowledge domain.

    The knowledge base is currently about Arjun Sharma.

    Examples that return True:

        अर्जुनको पेशा के हो?
        अर्जुनको जन्म कहाँ भएको हो?
        उनको मनपर्ने खाना के हो?
        Who is Arjun Sharma?
        What are Arjun's technical skills?

    Examples that return False:

        नेपालको सबैभन्दा अग्लो हिमाल कुन हो?
        आज काठमाडौंको मौसम कस्तो छ?
        नेपालको राष्ट्रपति को हुनुहुन्छ?
    """

    query = normalize_text(query)

    if not query:
        return False

    lower = query.lower()

    # -----------------------------------------------------
    # Direct entity reference
    # -----------------------------------------------------

    for alias in ENTITY_ALIASES:

        if alias.lower() in lower:
            return True

    # -----------------------------------------------------
    # Nepali person reference
    # -----------------------------------------------------

    for word in PERSON_REFERENCE_WORDS:

        if word in query:
            return True

    return False


# =========================================================
# ENGLISH QUESTION NORMALIZATION FOR BM25
# =========================================================

QUESTION_WORDS = {
    # English
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

    # Nepali
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
}


def clean_bm25_query(query: str) -> str:
    """
    Remove common question words while preserving the
    important subject/topic words.
    """

    query = normalize_text(query)

    if not query:
        return ""

    tokens = query.split()

    cleaned = []

    for token in tokens:

        token = token.lower().strip(
            ".,!?;:'\"()[]{}"
        )

        if not token:
            continue

        if token in QUESTION_WORDS:
            continue

        cleaned.append(token)

    result = " ".join(cleaned)

    return result if result else query


# =========================================================
# LOAD EMBEDDINGS
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
            "[INIT] Embedding model loaded locally."
        )

        return embeddings

    except Exception as exc:

        print(
            "[WARN] Local embedding load failed:"
            f" {exc}"
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        print(
            "[INIT] Embedding model loaded."
        )

        return embeddings


# =========================================================
# LOAD CHROMA
# =========================================================

@st.cache_resource(show_spinner=False)
def load_vector_store():

    if not os.path.exists(PERSIST_DIR):

        print(
            "[ERROR] Chroma database missing:"
            f" {PERSIST_DIR}"
        )

        return None

    embeddings = load_embeddings()

    print(
        "[INIT] Opening Chroma database..."
    )

    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    print(
        "[INIT] Chroma database ready."
    )

    return vector_store


# =========================================================
# LOAD BM25
# =========================================================

@st.cache_resource(show_spinner=False)
def load_bm25():

    if not os.path.exists(BM25_PKL_PATH):

        print(
            "[ERROR] BM25 index missing:"
            f" {BM25_PKL_PATH}"
        )

        return None

    print(
        "[INIT] Loading BM25 index..."
    )

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
# LOAD GROQ
# =========================================================

@st.cache_resource(show_spinner=False)
def load_groq():

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        print(
            "[ERROR] GROQ_API_KEY missing."
        )

        return None

    print(
        "[INIT] Initializing Groq..."
    )

    client = Groq(
        api_key=api_key
    )

    print(
        "[INIT] Groq ready."
    )

    return client


# =========================================================
# LOAD COMPLETE PIPELINE
# =========================================================

@st.cache_resource(show_spinner=False)
def load_rag_pipeline():

    print(
        "[INIT] Loading RAG pipeline..."
    )

    vector_store = load_vector_store()
    bm25 = load_bm25()
    groq_client = load_groq()

    if vector_store is None:
        print(
            "[ERROR] Vector store unavailable."
        )

    if bm25 is None:
        print(
            "[ERROR] BM25 unavailable."
        )

    if groq_client is None:
        print(
            "[ERROR] Groq unavailable."
        )

    if (
        vector_store is not None
        and bm25 is not None
        and groq_client is not None
    ):

        print(
            "[INIT] RAG pipeline ready."
        )

    return (
        vector_store,
        bm25,
        groq_client,
    )


# =========================================================
# HYBRID RETRIEVAL
# =========================================================

def retrieve_documents(
    query: str,
):
    """
    BM25 + vector retrieval in parallel.

    We use rank fusion because BM25 scores and vector
    distances are not on the same numerical scale.
    """

    vector_store, bm25, _ = (
        load_rag_pipeline()
    )

    if (
        vector_store is None
        or bm25 is None
    ):
        return []

    query = normalize_text(query)

    if not query:
        return []

    start = time.perf_counter()

    # -----------------------------------------------------
    # Expand entity
    # -----------------------------------------------------

    expanded_query = (
        expand_entity_query(query)
    )

    semantic_query = expanded_query

    lexical_query = clean_bm25_query(
        expanded_query
    )

    print(
        f"[QUERY] Original: {query}"
    )

    if expanded_query != query:

        print(
            f"[QUERY] Expanded: {expanded_query}"
        )

    print(
        f"[QUERY] BM25: {lexical_query}"
    )

    # -----------------------------------------------------
    # Parallel retrieval
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MERGE
    # -----------------------------------------------------

    merged = {}

    # Vector results
    for rank, doc in enumerate(
        vector_docs,
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
                "vector_rank": rank,
                "bm25_rank": None,
            }

    # BM25 results
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
    # RRF
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
                "vector_rank": item[
                    "vector_rank"
                ],
                "bm25_rank": item[
                    "bm25_rank"
                ],
            }
        )

    fused.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        "[RETRIEVAL] "
        f"vector={len(vector_docs)} "
        f"bm25={len(bm25_docs)} "
        f"unique={len(fused)} "
        f"time={elapsed:.3f}s"
    )

    # Debug output
    for index, item in enumerate(
        fused[:FINAL_CONTEXT_CHUNKS],
        start=1,
    ):

        preview = (
            item["text"][:150]
            .replace("\n", " ")
        )

        print(
            f"[RETRIEVAL {index}] "
            f"rrf={item['score']:.6f} "
            f"vector={item['vector_rank']} "
            f"bm25={item['bm25_rank']} "
            f"text={preview}"
        )

    return fused


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    documents,
) -> str:
    """
    Keep several relevant chunks instead of only one.
    """

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

        if len(text) > remaining:

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
# MAIN ANSWER FUNCTION
# =========================================================

def answer_user_query(
    query: str,
) -> str:
    """
    Closed-domain conversational RAG.

    Important behavior:

        About Arjun
            -> retrieve + answer

        Not about Arjun
            -> NO_INFO

        Goodbye
            -> goodbye immediately
    """

    total_start = (
        time.perf_counter()
    )

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

    query = normalize_text(
        query
    )

    if not query:
        return NO_INFO

    # =====================================================
    # CALL END
    # =====================================================

    if is_exit_intent(query):

        print(
            "[INTENT] EXIT"
        )

        return GOODBYE_RESPONSE

    # =====================================================
    # DOMAIN GATE
    # =====================================================
    #
    # This is intentionally performed BEFORE retrieval.
    #
    # It prevents Groq from answering general-world
    # questions using its training knowledge.
    # =====================================================

    if not is_arjun_question(query):

        print(
            "[DOMAIN] OUTSIDE KNOWLEDGE BASE"
        )

        return NO_INFO

    print(
        "[DOMAIN] ARJUN QUESTION"
    )

    # =====================================================
    # LOAD PIPELINE
    # =====================================================

    (
        vector_store,
        bm25,
        groq_client,
    ) = load_rag_pipeline()

    if (
        vector_store is None
        or bm25 is None
    ):

        return DATABASE_ERROR

    if groq_client is None:

        return SERVER_ERROR

    # =====================================================
    # RETRIEVE
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

        print(
            "[RAG] No documents retrieved."
        )

        return NO_INFO

    # =====================================================
    # CONTEXT
    # =====================================================

    context = build_context(
        documents
    )

    if not context:

        return NO_INFO

    print(
        f"[CONTEXT] "
        f"{len(context)} characters"
    )

    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

    system_prompt = f"""
तपाईं अर्जुन शर्माको व्यक्तिगत जीवनीमा आधारित
नेपाली फोन भोइस असिस्टेन्ट हुनुहुन्छ।

अत्यन्त महत्वपूर्ण:

तलको सन्दर्भ नै तपाईंको सम्पूर्ण ज्ञान हो।

तपाईंले बाहिरी संसारको ज्ञान प्रयोग गर्न पाउनुहुन्न।
आफ्नो प्रशिक्षणबाट आएको तथ्य प्रयोग नगर्नुहोस्।
अनुमान गरेर नयाँ तथ्य नबनाउनुहोस्।

सन्दर्भमा भएको तथ्यहरूलाई मात्र प्रयोग गर्नुहोस्।

प्रश्नको उत्तर सन्दर्भमा छ भने त्यसलाई
प्राकृतिक नेपाली भाषामा स्पष्ट रूपमा उत्तर दिनुहोस्।

प्रश्न सन्दर्भमा छैन भने:

"माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"

मात्र भन्नुहोस्।

विशेष निर्देशन:

१. उत्तर केवल नेपाली देवनागरी लिपिमा दिनुहोस्।

२. प्रश्नको अर्थ बुझ्नुहोस्; प्रश्नका शब्दहरू
   हुबहु सन्दर्भमा नहुन पनि सक्छन्।

३. समान अर्थ भएका शब्दहरू बुझ्नुहोस्।

   उदाहरण:
   "पेशा" = काम / पेशागत विवरण
   "जन्म कहाँ" = जन्म स्थान
   "कुन विषयमा स्नातक" = स्नातक विषय
   "कहिले पूरा गरे" = प्राप्त गरेको वर्ष
   "सीप" = प्राविधिक सीप
   "मनपर्ने खाना" = मनपर्ने कुराहरू
   "पहिले कुन कम्पनी" = प्रारम्भिक/पूर्व कम्पनी
   "प्रमुख उपलब्धि" = प्रमुख उपलब्धिहरू
   "भविष्यको लक्ष्य" = भविष्यका लक्ष्य

४. फरक context chunks मा एउटै प्रश्नसँग
   सम्बन्धित तथ्यहरू छन् भने तिनीहरूलाई जोड्नुहोस्।

५. "को हुन्?" भन्ने प्रश्नमा केवल नाम नदोहोर्याई
   छोटो परिचय दिनुहोस्।

६. "उनको", "उनले", "उनी", "उहाँ" जस्ता
   शब्दहरू यहाँ अर्जुन शर्मालाई जनाउँछन्।

७. उत्तर १–३ प्राकृतिक वाक्यमा राख्नुहोस्,
   जबसम्म थप विवरण आवश्यक हुँदैन।

८. "प्राप्त जानकारी अनुसार", "कागजात अनुसार",
   "डाटा अनुसार" जस्ता वाक्यांश प्रयोग नगर्नुहोस्।

९. फोनमा बोल्दा प्राकृतिक सुनिने भाषा प्रयोग गर्नुहोस्।

१०. सन्दर्भमा नभएको कुनै तथ्य कहिल्यै नबनाउनुहोस्।

सन्दर्भ:
==================================================
{context}
==================================================
"""

    # =====================================================
    # GROQ
    # =====================================================

    llm_start = (
        time.perf_counter()
    )

    try:

        response = (
            groq_client
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

                max_tokens=MAX_RESPONSE_TOKENS,
            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        if not answer:

            return SERVER_ERROR

        llm_time = (
            time.perf_counter()
            - llm_start
        )

        total_time = (
            time.perf_counter()
            - total_start
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
# DEBUG FUNCTION
# =========================================================

def debug_query(
    query: str,
):
    """
    Debug domain detection and retrieval.

    Example:

        debug_query(
            "अर्जुनको पेशा के हो?"
        )

        debug_query(
            "Which is the tallest mountain in Nepal?"
        )
    """

    print(
        "\n"
        + "#" * 70
    )

    print(
        f"QUERY: {query}"
    )

    print(
        f"EXIT INTENT: "
        f"{is_exit_intent(query)}"
    )

    print(
        f"ARJUN DOMAIN: "
        f"{is_arjun_question(query)}"
    )

    if not is_arjun_question(query):

        print(
            "RESULT: OUTSIDE KNOWLEDGE BASE"
        )

        print(
            "#" * 70
        )

        return

    results = retrieve_documents(
        query
    )

    print(
        f"RESULT COUNT: {len(results)}"
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- RESULT {index} ---"
        )

        print(
            f"RRF: {item['score']:.6f}"
        )

        print(
            f"VECTOR RANK: "
            f"{item['vector_rank']}"
        )

        print(
            f"BM25 RANK: "
            f"{item['bm25_rank']}"
        )

        print(
            item["text"]
        )

    print(
        "#" * 70
    )