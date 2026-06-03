import time
import sys
import os
import fitz  # PyMuPDF
from PIL import Image
from surya.recognition import RecognitionPredictor
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
#  CONFIG — change this to your PDF path
# ─────────────────────────────────────────────
INPUT_PDF = "input1.pdf"
DPI = 150        # increase to 200/300 for better accuracy on small text
# ─────────────────────────────────────────────


def pdf_to_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def block_to_text(block):
    soup = BeautifulSoup(block.html, "html.parser")
    label = block.label

    if label == "Table":
        rows = soup.find_all("tr")
        col_data = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            col_data.append([c.get_text(strip=True) for c in cells])

        if not col_data:
            return ""

        col_widths = [
            max(len(row[i]) for row in col_data if i < len(row))
            for i in range(max(len(row) for row in col_data))
        ]

        lines = []
        for i, row in enumerate(col_data):
            padded = [cell.ljust(col_widths[j]) for j, cell in enumerate(row)]
            lines.append("  ".join(padded))
            if i == 0:
                lines.append("-" * (sum(col_widths) + len(col_widths) * 2))
        return "\n".join(lines)

    elif label == "ListGroup":
        items = soup.find_all("li")
        lines = []
        for i, item in enumerate(items, 1):
            for br in item.find_all("br"):
                br.replace_with("\n    ")
            lines.append(f"{i}. {item.get_text(strip=True)}")
        return "\n".join(lines)

    else:
        for br in soup.find_all("br"):
            br.replace_with("\n")
        return soup.get_text(strip=True)


# ── Validate input ──────────────────────────────────────────────────────────
if not os.path.isfile(INPUT_PDF):
    print(f"ERROR: File not found — {INPUT_PDF}")
    sys.exit(1)

if not INPUT_PDF.lower().endswith(".pdf"):
    print("ERROR: Input file must be a .pdf")
    sys.exit(1)

# ── Run OCR ─────────────────────────────────────────────────────────────────
print("Loading models...")
recognition_predictor = RecognitionPredictor()

print(f"Converting PDF to images at {DPI} DPI...")
images = pdf_to_images(INPUT_PDF, dpi=DPI)
print(f"Found {len(images)} page(s)")

print("Starting OCR extraction...")
start_time = time.time()

predictions = recognition_predictor(images)

elapsed = time.time() - start_time
print(f"Extraction completed in: {elapsed:.2f} seconds")

# ── Save TXT ─────────────────────────────────────────────────────────────────
output_name = os.path.splitext(INPUT_PDF)[0] + "_ocr.txt"
with open(output_name, "w", encoding="utf-8") as f:
    for page_num, page in enumerate(predictions):
        f.write(f"{'='*60}\n  Page {page_num + 1}\n{'='*60}\n\n")
        for block in sorted(page.blocks, key=lambda b: b.reading_order):
            if not block.skipped and not block.error and block.html.strip():
                text = block_to_text(block)
                if text:
                    f.write(text + "\n\n")

print(f"Output saved to: {output_name}")