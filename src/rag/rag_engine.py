import os
import time
import pickle
import logging
import warnings
import unicodedata
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
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

VECTOR_K = 15
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

# =========================================================
# PRONUNCIATION HELPERS
# =========================================================
ENGLISH_TO_NEPALI = {
    "Disaster Management": "विपद् व्यवस्थापन",
    "Mitigation": "न्यूनीकरण",
    "Preparedness": "पूर्वतयारी",
    "Response": "प्रतिकार्य",
    "Recovery": "पुनर्लाभ",
    "First Aid": "प्राथमिक उपचार",
    "Search and Rescue": "खोज र उद्धार",
    "Early Warning": "पूर्वसूचना",
    "GIS": "जीआईएस",
    "AI": "एआई",
    "Drone": "ड्रोन",
    "VHF": "भीएचएफ",
    "UHF": "यूएचएफ",
    "NDRRMA": "एनडीआरआरएमए",
    "GPS": "जीपीएस",
    "Risk": "जोखिम",
    "Hazard": "प्रकोप",
}

def pronounce_english_terms(text: str) -> str:
    if not text:
        return text
    for eng, nep in sorted(ENGLISH_TO_NEPALI.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(eng) + r'\b'
        text = re.sub(pattern, nep, text, flags=re.IGNORECASE)
    return text

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
# TOPIC DEFINITIONS
# =========================================================
TOPIC_SUMMARY = (
    "विपद् व्यवस्थापन भनेको प्राकृतिक वा मानव निर्मित प्रकोपहरूसँग जुध्न र त्यसबाट हुने क्षतिलाई कम गर्न गरिने सम्पूर्ण कार्यहरूको संयोजन हो। "
    "यसले विपद् आउनुअघिको तयारीदेखि लिएर विपद् पछिको पुनर्निर्माणसम्मका सबै प्रक्रियाहरूलाई समेट्छ। प्रभावकारी विपद् व्यवस्थापनले समाजलाई सुरक्षित राख्न र संकटको समयमा छिटो तङ्ग्रिन मद्दत गर्छ।"
)

SUBTOPICS = [
    {
        "name": "प्रकार",
        "keywords": ["प्रकार", "types", "किसिम", "विपद्का प्रकार"],
        "answer": (
            "नेपालमा मुख्यतया दुई प्रकारका विपद्हरू देखा पर्छन्।\n"
            "प्राकृतिक विपद्अन्तर्गत भूकम्प, बाढी, पहिरो, आगलागी, हिमपहिरो र चट्याङ पर्दछन्।\n"
            "मानव निर्मित विपद्अन्तर्गत सडक दुर्घटना, औद्योगिक दुर्घटना, आगजनी र वातावरणीय प्रदूषण पर्दछन्।\n"
            "भौगोलिक बनावटका कारण नेपाल प्राकृतिक विपद्को उच्च जोखिममा रहेको छ।"
        )
    },
    {
        "name": "चरण",
        "keywords": ["चरण", "phases", "व्यवस्थापन चक्र", "चार चरण"],
        "answer": (
            "विपद् व्यवस्थापनलाई मुख्यतया चार चरणमा विभाजन गरिन्छ:\n"
            "१. जोखिम न्यूनीकरण (Mitigation): विपद् हुनुअघि नै सम्भावित क्षतिलाई कम गर्ने दीर्घकालीन उपायहरू (जस्तै: भूकम्प प्रतिरोधी घर निर्माण)।\n"
            "२. पूर्वतयारी (Preparedness): आपत्कालीन अवस्थाका लागि स्रोत, साधन र जनशक्तिको व्यवस्था गर्ने कार्य।\n"
            "३. प्रतिकार्य/उद्धार (Response): विपद्को समयमा तत्काल गरिने खोज, उद्धार र राहत वितरण कार्य।\n"
            "४. पुनर्लाभ/पुनर्स्थापना (Recovery): प्रभावित समुदाय र भौतिक संरचनालाई पुरानै वा अझ राम्रो अवस्थामा फर्काउने कार्य।"
        )
    },
    {
        "name": "सीप",
        "keywords": ["सीप", "प्रविधि", "skill", "technology", "प्रविधिहरू"],
        "answer": None
    },
    {
        "name": "चुनौती",
        "keywords": ["चुनौती", "challenges", "समस्या", "कठिनाई"],
        "answer": (
            "नेपालले भोग्नुपरेको सबैभन्दा ठूलो चुनौती भौगोलिक विकटता हो, जसले उद्धार कार्यलाई ढिलो बनाउँछ।\n"
            "यसका साथै जनचेतनाको कमी, पूर्वतयारीका लागि पर्याप्त बजेटको अभाव, असुरक्षित बस्ती विकास र सरोकारवाला निकायहरूबीच द्रुत समन्वयको कमीले गर्दा विपद्का समयमा प्रभावकारी काम गर्न कठिनाइ हुने गरेको छ।"
        )
    },
    {
        "name": "सम्पर्क",
        "keywords": ["सम्पर्क", "फोन", "नम्बर", "आपत्कालीन", "emergency", "contact"],
        "answer": (
            "महत्त्वपूर्ण आपत्कालीन सम्पर्क नम्बरहरू:\n"
            "नेपाल प्रहरी: १००\n"
            "दमकल (आगलागी नियन्त्रण): १०१\n"
            "एम्बुलेन्स सेवा: १०२\n"
            "ट्राफिक प्रहरी: १०३\n"
            "राष्ट्रिय आपत्कालीन कार्य सञ्चालन केन्द्र: ११४९"
        )
    },
    {
        "name": "भविष्य",
        "keywords": ["भविष्य", "दिशा", "प्रविधि", "future", "AI", "ड्रोन"],
        "answer": (
            "विपद् जोखिम न्यूनीकरणका लागि भविष्यमा पूर्वसूचना प्रणालीलाई थप आधुनिक र प्रविधिमैत्री बनाउनु आवश्यक छ।\n"
            "कृत्रिम बुद्धिमत्ता (AI), ड्रोन प्रविधि, र स्याटेलाइट इमेजरीको प्रयोग गरी विपद्को भविष्यवाणी गर्ने र जोखिमयुक्त क्षेत्रका नागरिकहरूलाई समयमै सुरक्षित स्थानमा सार्ने प्रणाली विकास गर्नु राष्ट्रको दीर्घकालीन लक्ष्य हुनुपर्छ।"
        )
    }
]

SKILLS_DATA = [
    {
        "name": "प्राथमिक उपचार र जीवन रक्षा सीप",
        "aliases": ["प्राथमिक उपचार", "जीवन रक्षा", "first aid", "प्राथमिक"],
        "detail": "प्राथमिक उपचार भनेको आपत्कालीन अवस्थामा घाइतेको ज्यान बचाउन गरिने तत्काल सहायता हो। यसले रगत रोक्ने, घाउमा पट्टी बाँध्ने, सास फेर्न सहयोग गर्ने, र मुटुको धड्कन जाँच्ने जस्ता क्रियाकलाप समेट्छ। जीवन रक्षा सीपले पानीमा डुब्ने, आगोमा परेका, वा भग्नावशेषमा थिचिएका व्यक्तिहरूलाई सुरक्षित निकाल्ने तरिका सिकाउँछ।"
    },
    {
        "name": "खोज तथा उद्धार प्रविधि",
        "aliases": ["खोज", "उद्धार", "search", "rescue", "खोज उद्धार"],
        "detail": "खोज तथा उद्धार प्रविधिले विपद्को समयमा हराएका वा फसेका मानिसहरूलाई पत्ता लगाई सुरक्षित निकाल्ने विधि सिकाउँछ। यसमा भग्नावशेषमा खोजी गर्ने, विशेष उपकरण (जस्तै: लाइफ ज्याकेट, रस्सी, क्यामेरा, ड्रोन) प्रयोग गर्ने, र उद्धार टोलीको सुरक्षा प्रोटोकल समावेश हुन्छ।"
    },
    {
        "name": "पूर्वसूचना प्रणाली सञ्चालन",
        "aliases": ["पूर्वसूचना", "early warning", "सूचना प्रणाली", "warning"],
        "detail": "पूर्वसूचना प्रणालीले मौसमी, भूकम्पीय, वा अन्य प्राकृतिक जोखिमको भविष्यवाणी गरी समयमै सचेत गराउँछ। यसको सञ्चालनमा रेडियो, टेलिभिजन, मोबाइल एप, र साइरन जस्ता माध्यमबाट सन्देश प्रसारण गर्ने, र प्राप्त डाटाको विश्लेषण गरी उपयुक्त कारबाहीको निर्देशन दिने कार्य पर्दछ।"
    },
    {
        "name": "आपत्कालीन सञ्चार उपकरणहरू",
        "aliases": ["सञ्चार", "VHF", "UHF", "रेडियो", "स्याटेलाइट", "communication"],
        "detail": "विपद्को समयमा नियमित फोन लाइन अवरुद्ध हुन सक्छ, त्यसैले VHF/UHF रेडियो, स्याटेलाइट फोन, र मेसेन्जर उपकरणहरू प्रयोग गरिन्छ। यी उपकरणहरूले टोलीबीच द्रुत सन्दान गर्न, उद्धार समन्वय गर्न, र पीडितहरूको अवस्थाबारे सूचना आदानप्रदान गर्न मद्दत गर्छन्। सञ्चालनका लागि ब्याट्री, चार्जिङ व्यवस्था, र ब्याकअप फ्रिक्वेन्सीको ज्ञान आवश्यक हुन्छ।"
    },
    {
        "name": "विपद् नक्सांकन र भूगोलीय सूचना प्रणाली",
        "aliases": ["नक्सांकन", "GIS", "भौगोलिक", "mapping"],
        "detail": "GIS प्रविधिले विपद् जोखिम क्षेत्रको नक्सा तयार गरी प्रभावित वस्ती, सुरक्षित आश्रयस्थल, उद्धार मार्ग, र स्रोतहरूको अवस्था चिन्न सहयोग गर्छ। यसले उपग्रह तस्बिर, भौगोलिक डाटा, र जनसङ्ख्या विवरणलाई एकीकृत गरी निर्णयकर्तालाई प्रभावकारी योजना बनाउन मार्गदर्शन गर्छ।"
    }
]

SKILLS_LIST_TEXT = (
    "आवश्यक सीप तथा प्रविधिहरू निम्न छन्:\n"
    "१. प्राथमिक उपचार र जीवन रक्षा सीप\n"
    "२. खोज तथा उद्धार प्रविधि\n"
    "३. पूर्वसूचना प्रणाली सञ्चालन\n"
    "४. आपत्कालीन सञ्चार उपकरणहरू (VHF/UHF रेडियो, स्याटेलाइट फोन)\n"
    "५. विपद् नक्सांकन र भौगोलिक सूचना प्रणाली (GIS)\n\n"
    "यी मध्ये तपाईं कुन सीपको बारेमा थप जानकारी चाहनुहुन्छ? कृपया नम्बर वा नाम भन्नुहोस्।"
)

# =========================================================
# SESSION MANAGEMENT & TOPIC HANDLING
# =========================================================
_session_store: Dict[str, Dict[str, Any]] = {}
_session_lock = threading.Lock()

ALL_TOPICS = ["प्रकार", "चरण", "सीप", "चुनौती", "सम्पर्क", "भविष्य"]

def _list_available_topics(exclude: Optional[str] = None) -> str:
    if exclude:
        topics = [t for t in ALL_TOPICS if t != exclude]
    else:
        topics = ALL_TOPICS
    if not topics:
        return "तपाईं अरू कुनै पक्षको बारेमा सोध्न सक्नुहुन्छ।"
    if len(topics) == 1:
        return f"तपाईं {topics[0]} पक्षको बारेमा सोध्न सक्नुहुन्छ।"
    all_str = ", ".join(topics[:-1]) + " वा " + topics[-1]
    return f"तपाईं {all_str} पक्षहरूको बारेमा सोध्न सक्नुहुन्छ।"

def _is_general_disaster_query(query: str) -> bool:
    q = normalize_text(query).lower()
    if "विपद्" not in q and "disaster" not in q:
        return False
    for sub in SUBTOPICS:
        for kw in sub["keywords"]:
            if kw in q:
                return False
    if q.strip() in ["विपद्", "विपद", "disaster"]:
        return True
    general_indicators = [
        "के हो", "भनेको", "परिभाषा", "अर्थ", "मतलब",
        "जानकारी", "बारे", "भन", "बताउ", "के छ",
        "what is", "about", "tell me"
    ]
    return any(ind in q for ind in general_indicators)

def _match_subtopic(query: str) -> Optional[Dict]:
    q = normalize_text(query).lower()
    for sub in SUBTOPICS:
        if sub["name"] == "सीپ":
            continue
        for kw in sub["keywords"]:
            if kw in q:
                return sub
    return None

def _match_skill(query: str) -> Optional[Dict]:
    q = normalize_text(query).lower()
    number_map = {"१":1, "1":1, "एक":1, "पहिलो":1, "first":1,
                  "२":2, "2":2, "दुई":2, "दोस्रो":2, "second":2,
                  "३":3, "3":3, "तीन":3, "तेस्रो":3, "third":3,
                  "४":4, "4":4, "चार":4, "चौथो":4, "fourth":4,
                  "५":5, "5":5, "पाँच":5, "पाँचौँ":5, "fifth":5}
    for token in q.split():
        if token in number_map:
            idx = number_map[token] - 1
            if 0 <= idx < len(SKILLS_DATA):
                return SKILLS_DATA[idx]
    for skill in SKILLS_DATA:
        for alias in skill["aliases"]:
            if alias in q:
                return skill
    return None

def _is_what_else_query(query: str) -> bool:
    q = normalize_text(query).lower()
    patterns = [r'\bअरु\b', r'\bके के\b', r'\bके छ\b']
    return any(re.search(p, q) for p in patterns)

def handle_topic_query(query: str, call_sid: str) -> Optional[str]:
    with _session_lock:
        session = _session_store.get(call_sid, {})
        followup = session.get("dm_followup", False)
        last_topic = session.get("last_topic", None)

        if "सीप" in normalize_text(query).lower() or "skill" in normalize_text(query).lower():
            session["dm_followup"] = True
            session["last_topic"] = "सीप"
            _session_store[call_sid] = session
            return pronounce_english_terms(SKILLS_LIST_TEXT)

        if followup:
            matched_skill = _match_skill(query)
            if matched_skill:
                if last_topic == matched_skill["name"]:
                    return "तपाईंले यो सीपको बारेमा पहिले नै सोधिसक्नुभयो। के अरू कुनै सीप वा अर्को पक्ष (प्रकार, चरण, चुनौती, सम्पर्क, भविष्य) सोध्न चाहनुहुन्छ?"
                session["last_topic"] = matched_skill["name"]
                _session_store[call_sid] = session
                detail = pronounce_english_terms(matched_skill["detail"])
                return detail + "। के अरू कुनै सीप वा अर्को पक्षको बारेमा जान्न चाहनुहुन्छ?"

            matched_sub = _match_subtopic(query)
            if matched_sub:
                if last_topic == matched_sub["name"]:
                    others = _list_available_topics(exclude=matched_sub["name"])
                    return f"तपाईंले '{matched_sub['name']}' को बारेमा पहिले नै सोधिसक्नुभयो। {others}"
                session["last_topic"] = matched_sub["name"]
                _session_store[call_sid] = session
                answer = pronounce_english_terms(matched_sub["answer"])
                followup_prompt = _list_available_topics(exclude=matched_sub["name"])
                return answer + "। " + followup_prompt + " तपाईं कुन पक्षको बारेमा जान्न चाहनुहुन्छ?"

            if _is_what_else_query(query):
                exclude = last_topic if last_topic else None
                return _list_available_topics(exclude=exclude) + " तपाईं कुन पक्षको बारेमा जान्न चाहनुहुन्छ?"

            session.pop("dm_followup", None)
            session.pop("last_topic", None)
            _session_store[call_sid] = session
            return None

        else:
            if _is_general_disaster_query(query):
                session["dm_followup"] = True
                session["last_topic"] = None
                _session_store[call_sid] = session
                prompt = (
                    "तपाईं विपद् व्यवस्थापनको प्रकार, चरण, आवश्यक सीप, चुनौती, आपत्कालीन सम्पर्क, वा भविष्यको दिशा जस्ता पक्षमा थप जानकारी सोध्न सक्नुहुन्छ। "
                    "कृपया भन्नुहोस् तपाईं कुनको बारेमा जान्न चाहनुहुन्छ?"
                )
                return pronounce_english_terms(TOPIC_SUMMARY) + "। " + prompt

            matched_sub = _match_subtopic(query)
            if matched_sub:
                if last_topic == matched_sub["name"]:
                    others = _list_available_topics(exclude=matched_sub["name"])
                    return f"तपाईंले '{matched_sub['name']}' को बारेमा पहिले नै सोधिसक्नुभयो। {others}"
                session["dm_followup"] = True
                session["last_topic"] = matched_sub["name"]
                _session_store[call_sid] = session
                answer = pronounce_english_terms(matched_sub["answer"])
                followup_prompt = _list_available_topics(exclude=matched_sub["name"])
                return answer + "। " + followup_prompt + " तपाईं कुन पक्षको बारेमा जान्न चाहनुहुन्छ?"

            if _is_what_else_query(query):
                return _list_available_topics() + " तपाईं कुन पक्षको बारेमा जान्न चाहनुहुन्छ?"

            return None

# =========================================================
# PIPELINE CACHING (unchanged)
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

    disaster_keywords = ["भूकम्प", "बाढी", "पहिरो", "आगलागी", "हिमपहिरो", "चट्याङ"]
    for item in fused:
        for kw in disaster_keywords:
            if kw in query and kw in item["text"]:
                item["score"] *= 1.3

    if "के हो" in query:
        for item in fused:
            if "परिचय" in item["text"] or "आधारभूत" in item["text"]:
                item["score"] *= 1.5

    fused.sort(key=lambda x: x["score"], reverse=True)

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
    # Remove duplicate consecutive words (e.g., "न्यूनीकरण न्यूनीकरण")
    words = answer.split()
    if len(words) > 1:
        cleaned = []
        for i, w in enumerate(words):
            if i == 0 or w != words[i-1]:
                cleaned.append(w)
        answer = " ".join(cleaned)
    return answer

# =========================================================
# MAIN ANSWER FUNCTION
# =========================================================
def answer_user_query(query: str, call_sid: Optional[str] = None) -> str:
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

    # ---- HANDLE TOPIC-BASED CONVERSATION ----
    if call_sid:
        topic_answer = handle_topic_query(query, call_sid)
        if topic_answer is not None:
            return pronounce_english_terms(topic_answer)

    # ---- DIRECT ANSWER (fallback) ----
    definition_keywords = ["के हो", "भनेको", "परिभाषा", "अर्थ", "मतलब"]
    if "विपद्" in query and any(kw in query for kw in definition_keywords):
        answer = TOPIC_SUMMARY + "। तपाईं प्रकार, चरण, सीप, चुनौती, सम्पर्क, वा भविष्यको बारेमा सोध्न सक्नुहुन्छ।"
        return pronounce_english_terms(answer)

    # ---- RAG PIPELINE ----
    print("[RAG] Proceeding with retrieval and LLM.")
    vector_store, bm25, groq = load_rag_pipeline()
    if not vector_store or not bm25:
        return DATABASE_ERROR
    if not groq:
        return GROQ_UNAVAILABLE

    documents = retrieve_documents(query)
    if not documents:
        return NO_INFO

    context = build_context(documents, query=query)
    if not context:
        return NO_INFO

    print(f"[CONTEXT PREVIEW] {context[:500]}...")

    # ---- Enhanced system prompt for comparisons ----
    system_prompt = f"""
तपाईं "विपद् व्यवस्थापन सूचना सेवा" का नेपाली voice assistant हुनुहुन्छ।
तपाईंको काम: प्रयोगकर्ताको प्रश्नको छोटो, स्पष्ट र सटीक उत्तर दिनुहोस्।
उत्तर दिनको लागि **केवल** दिइएको सन्दर्भ प्रयोग गर्नुहोस्।
यदि प्रश्नले दुई वा बढी वस्तुहरूको तुलना माग्छ भने, तिनीहरूको भिन्नता र समानता स्पष्ट रूपमा देखाउनुहोस्।
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

        ans = pronounce_english_terms(ans)
        return ans or SERVER_ERROR

    except Exception as exc:
        print(f"[GROQ ERROR] {exc}")
        return SERVER_ERROR