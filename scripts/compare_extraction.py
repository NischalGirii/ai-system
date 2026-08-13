"""Compare native PDF text vs OCR for accuracy diagnosis."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import fitz
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.extractor import (
    extract_text_from_pdf,
    is_valid_devanagari_text,
    clean_devanagari_text,
)

pdf = ROOT / "data" / "नमूना_व्यक्तिगत_जीवनी_RAG_DATA_ONLY.pdf"
out = ROOT / "scripts" / "extraction_comparison.txt"

lines = []
doc = fitz.open(pdf)

for i in range(len(doc)):
    native = doc[i].get_text().strip()
    valid = is_valid_devanagari_text(native)
    lines.append(f"=== PAGE {i+1} ===")
    lines.append(f"Native valid: {valid} | chars: {len(native)}")
    lines.append("--- NATIVE RAW ---")
    lines.append(native)
    lines.append("--- NATIVE CLEANED ---")
    lines.append(clean_devanagari_text(native))
    lines.append("")

lines.append("\n\n========== OCR (force_ocr=True) ==========\n")
for page in extract_text_from_pdf(str(pdf), force_ocr=True):
    lines.append(f"--- PAGE {page['page_number']} [{page['extraction_mode']}] ---")
    lines.append(page["content"])
    lines.append("")

lines.append("\n\n========== AUTO (force_ocr=False) ==========\n")
for page in extract_text_from_pdf(str(pdf), force_ocr=False):
    lines.append(f"--- PAGE {page['page_number']} [{page['extraction_mode']}] ---")
    lines.append(page["content"])
    lines.append("")

out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote comparison to {out}")
