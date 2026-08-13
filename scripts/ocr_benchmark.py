"""Benchmark OCR settings for Nepali PDF extraction."""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fitz
import pytesseract
from PIL import Image
from ingestion.extractor import preprocess_devanagari_image

pdf = ROOT / "data" / "नमूना_व्यक्तिगत_जीवनी_RAG_DATA_ONLY.pdf"
doc = fitz.open(pdf)
page = doc[0]

# Ground-truth snippets we expect (from visually correct document)
CHECKS = [
    "अर्जुन शर्मा",
    "२०४८ साल चैत २ गते",
    "सफ्टवेयर इन्जिनियर",
    "हिमालयन इन्स्टिच्युट",
    "ब्लुपिक सोलुसन्स",
    "५. प्राविधिक सीप",
    "कृत्रिम बुद्धिमत्ता",
]

configs = [
    ("300dpi nep+eng psm6", 300, "nep+eng", r"--oem 1 --psm 6", True),
    ("400dpi nep+eng psm6", 400, "nep+eng", r"--oem 1 --psm 6", True),
    ("300dpi nep only psm6", 300, "nep", r"--oem 1 --psm 6", True),
    ("300dpi nep+eng psm3", 300, "nep+eng", r"--oem 1 --psm 3", True),
    ("300dpi nep+eng psm4", 300, "nep+eng", r"--oem 1 --psm 4", True),
    ("300dpi nep+eng no preprocess", 300, "nep+eng", r"--oem 1 --psm 6", False),
    ("600dpi nep+eng psm6", 600, "nep+eng", r"--oem 1 --psm 6", True),
]

lines = []
for name, dpi, lang, cfg, preprocess in configs:
    pix = page.get_pixmap(dpi=dpi)
    pil = Image.open(io.BytesIO(pix.tobytes("png")))
    if preprocess:
        pil = preprocess_devanagari_image(pil)
    text = pytesseract.image_to_string(pil, lang=lang, config=cfg)
    hits = sum(1 for c in CHECKS if c in text)
    lines.append(f"\n=== {name} | score {hits}/{len(CHECKS)} ===")
    lines.append(text[:1200])

out = ROOT / "scripts" / "ocr_benchmark.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}")
