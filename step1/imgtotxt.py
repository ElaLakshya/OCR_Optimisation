import time
import sys
import os
from PIL import Image
from surya.recognition import RecognitionPredictor
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
#  CONFIG — change this to your image path
# ─────────────────────────────────────────────
INPUT_IMAGE = "input1.jpeg"   # supports .jpg, .jpeg, .png
# ─────────────────────────────────────────────


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
if not os.path.isfile(INPUT_IMAGE):
    print(f"ERROR: File not found — {INPUT_IMAGE}")
    sys.exit(1)

ext = os.path.splitext(INPUT_IMAGE)[1].lower()
if ext not in (".jpg", ".jpeg", ".png"):
    print(f"ERROR: Unsupported format '{ext}'. Use .jpg, .jpeg, or .png")
    sys.exit(1)

# ── Run OCR ─────────────────────────────────────────────────────────────────
print("Loading models...")
recognition_predictor = RecognitionPredictor()

print(f"Opening image: {INPUT_IMAGE}")
image = Image.open(INPUT_IMAGE).convert("RGB")

print("Starting OCR extraction...")
start_time = time.time()

predictions = recognition_predictor([image])

elapsed = time.time() - start_time
print(f"Extraction completed in: {elapsed:.2f} seconds")

# ── Save TXT ─────────────────────────────────────────────────────────────────
output_name = os.path.splitext(INPUT_IMAGE)[0] + "_ocr.txt"
with open(output_name, "w", encoding="utf-8") as f:
    for page in predictions:
        for block in sorted(page.blocks, key=lambda b: b.reading_order):
            if not block.skipped and not block.error and block.html.strip():
                text = block_to_text(block)
                if text:
                    f.write(text + "\n\n")

print(f"Output saved to: {output_name}")