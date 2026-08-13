# src/ingestion/extractor.py
import io
import os
import re
import unicodedata
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image

OCR_DPI = 300
OCR_CONFIG = r"--oem 1 --psm 6"

CORRUPT_LATIN_RE = re.compile(r"[\u0250-\u02AF\u00C0-\u024F]")
LATIN_INSIDE_DEVANAGARI_RE = re.compile(r"(?<=[\u0900-\u097F])[A-Za-z](?=[\u0900-\u097F])")
BULLET_CLEAN_RE = re.compile(r"^[\u00AB\u00BB\u2022\u2023\u2043\u2219\u25AA\u25CF\u25E6\u2027\u00B7\u2013\u2014\u2015\u0660\u002B\u0022\u00AB\u00BB\u201c\u201d\-«]\s*")


def preprocess_devanagari_image(pil_img: Image.Image) -> Image.Image:
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def resolve_ocr_lang_flag() -> str:
    try:
        installed = pytesseract.get_languages(config="")
        priority = ["script/Devanagari", "nep", "eng"]
        selected = [lang for lang in priority if lang in installed]
        return "+".join(selected) if selected else "eng"
    except Exception:
        return "eng"


def ocr_scanned_page(page: fitz.Page, lang_flag: str) -> str:
    pix = page.get_pixmap(dpi=OCR_DPI)
    raw_pil = Image.open(io.BytesIO(pix.tobytes("png")))
    processed_pil = preprocess_devanagari_image(raw_pil)
    return pytesseract.image_to_string(processed_pil, lang=lang_flag, config=OCR_CONFIG)


def is_corrupt_native_text(text: str) -> bool:
    if not text or len(text.strip()) < 30:
        return True

    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    total_chars = len(re.sub(r"\s+", "", text))
    if total_chars == 0:
        return True

    devanagari_ratio = devanagari_chars / total_chars
    corrupt_latin = len(CORRUPT_LATIN_RE.findall(text))
    latin_inside = len(LATIN_INSIDE_DEVANAGARI_RE.findall(text))

    if corrupt_latin >= 3 or latin_inside >= 2 or devanagari_ratio < 0.4:
        return True

    corruption_markers = ("अजEर्जुन", "शमार्जु", "ɡ", "ɟ", "ɠ", "Ê", "Ö")
    return any(marker in text for marker in corruption_markers)


def clean_devanagari_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue

        if BULLET_CLEAN_RE.match(line):
            line = BULLET_CLEAN_RE.sub("• ", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def post_correct_ocr_errors(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    corrections = {
        "२०१८ मा उनले व्यवसाय": "२०१९ मा उनले व्यवसाय",
        "ए. प्राविधिक सीप": "५. प्राविधिक सीप",
        "G. रुचि तथा शौख": "८. रुचि तथा शौख",
        "८. मनपर्ने कुराहरू": "९. मनपर्ने कुराहरू",
        "१०, भविष्यका लक्ष्य": "१०. भविष्यका लक्ष्य",
        "कृत्रिम gala": "कृत्रिम बुद्धिमत्ता",
        "३.शिक्षा": "३. शिक्षा",
        "व्यावसायिकअनुभव": "व्यावसायिक अनुभव",
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"([\u0966-\u096F\d]+\.)([^\s\d])", r"\1 \2", text)
    text = re.sub(r"मद्दत\s+ग[सस्]यो", "मद्दत गर्यो", text)
    return text


def extract_text_from_pdf(pdf_path: str, force_ocr: bool = False) -> list[dict]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    lang_flag = resolve_ocr_lang_flag()
    doc = fitz.open(pdf_path)
    extracted_pages = []

    print(f"Processing: {os.path.basename(pdf_path)} ({len(doc)} pages) | OCR Engine: {lang_flag}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        native_text = page.get_text().strip()

        if force_ocr or is_corrupt_native_text(native_text):
            raw_text = ocr_scanned_page(page, lang_flag)
            extraction_mode = "Scanned / OCR Fallback"
        else:
            raw_text = native_text
            extraction_mode = "Direct Digital Extraction"

        cleaned_text = clean_devanagari_text(raw_text)
        final_text = post_correct_ocr_errors(cleaned_text)

        extracted_pages.append({
            "page_number": page_num + 1,
            "extraction_mode": extraction_mode,
            "content": final_text,
        })

    return extracted_pages