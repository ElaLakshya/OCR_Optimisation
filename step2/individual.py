import time
import sys
import os
import json
import fitz
from PIL import Image

from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
INPUT_PDF = "input1.pdf"
DPI = 150
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

if not os.path.isfile(INPUT_PDF):
    print(f"ERROR: File not found — {INPUT_PDF}")
    sys.exit(1)

print("==========================================================")
print("             SURYA PIPELINE TIMING BENCHMARK              ")
print("==========================================================\n")
total_start = time.time()

# ── Stage 1: Model Loading ────────────────────────────────────────────────────
print("  [1/6] Loading models...")
t0 = time.time()
det_predictor = DetectionPredictor()
lay_predictor = LayoutPredictor()
rec_predictor = RecognitionPredictor()
tab_predictor = TableRecPredictor()
images = pdf_to_images(INPUT_PDF, dpi=DPI)
load_time = time.time() - t0
print(f"        Done in {load_time:.2f}s ({len(images)} page(s) loaded)\n")

# ── Stage 2: Text Line Detection ─────────────────────────────────────────────
print("  [2/6] Text Line Detection...")
t0 = time.time()
det_results = det_predictor(images)
det_time = time.time() - t0
print(f"        Done in {det_time:.2f}s\n")

# ── Stage 3: Layout Analysis ──────────────────────────────────────────────────
print("  [3/6] Layout Analysis...")
t0 = time.time()
lay_results = lay_predictor(images)
lay_time = time.time() - t0
print(f"        Done in {lay_time:.2f}s\n")

# ── Stage 4: Table Recognition ────────────────────────────────────────────────
print("  [4/6] Table Structure Recognition...")
table_images = []
table_page_map = []  # track which page each table crop came from
for page_idx, (img, lay) in enumerate(zip(images, lay_results)):
    for block in lay.bboxes:
        if block.label == "Table":
            x0, y0, x1, y1 = [int(v) for v in block.bbox]
            table_images.append(img.crop((x0, y0, x1, y1)))
            table_page_map.append(page_idx)

t0 = time.time()
if table_images:
    tab_results = tab_predictor(table_images)
    tab_time = time.time() - t0
    print(f"        Done in {tab_time:.2f}s ({len(table_images)} table(s) found)\n")
else:
    tab_results = []
    tab_time = 0.0
    print(f"        No tables found, skipping (0.00s)\n")

# ── Stage 5: JSON Write ───────────────────────────────────────────────────────
print("  [5/6] JSON Write...")
t0 = time.time()

base_name = os.path.splitext(INPUT_PDF)[0]
json_output = []

for page_idx, (det, lay) in enumerate(zip(det_results, lay_results)):
    page_data = {
        "page": page_idx + 1,
        "text_lines": [
            {
                "bbox": line.bbox,
                "confidence": getattr(line, "confidence", None),
            }
            for line in det.bboxes
        ],
        "layout_blocks": [
            {
                "bbox": block.bbox,
                "label": block.label,
                "reading_order": getattr(block, "reading_order", None),
            }
            for block in lay.bboxes
        ],
        "tables": []
    }
    json_output.append(page_data)

# Attach table results back to their pages
for table_idx, (tab_res, page_idx) in enumerate(zip(tab_results, table_page_map)):
    json_output[page_idx]["tables"].append({
        "table_index": table_idx,
        "cells": [
            {
                "bbox": cell.bbox,
                "text": getattr(cell, "text", ""),
                "row": getattr(cell, "row_id", None),
                "col": getattr(cell, "col_id", None),
            }
            for cell in getattr(tab_res, "cells", [])
        ]
    })

json_path = base_name + "_structure.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

json_time = time.time() - t0
print(f"        Done in {json_time:.2f}s → {json_path}\n")

# ── Stage 6: OCR Recognition ──────────────────────────────────────────────────
print("  [6/6] OCR Text Recognition...")
t0 = time.time()
rec_results = rec_predictor(images, lay_results)
rec_time = time.time() - t0
print(f"        Done in {rec_time:.2f}s\n")

total_time = time.time() - total_start

# ── Summary ───────────────────────────────────────────────────────────────────
print("==========================================================")
print("                      SUMMARY                             ")
print("==========================================================")
print(f"  Stage 1 — Model Loading          : {load_time:.2f}s")
print(f"  Stage 2 — Text Line Detection    : {det_time:.2f}s")
print(f"  Stage 3 — Layout Analysis        : {lay_time:.2f}s")
print(f"  Stage 4 — Table Recognition      : {tab_time:.2f}s")
print(f"  Stage 5 — JSON Write             : {json_time:.2f}s")
print(f"  Stage 6 — OCR Recognition        : {rec_time:.2f}s")
print(f"  ─────────────────────────────────────────────────────")
print(f"  TOTAL                            : {total_time:.2f}s")
print("==========================================================")

# ── Save timing log ───────────────────────────────────────────────────────────
timing_file = base_name + "_timing.txt"
with open(timing_file, "w", encoding="utf-8") as f:
    f.write("SURYA PIPELINE TIMING BENCHMARK\n")
    f.write("==========================================================\n")
    f.write(f"  Input file : {INPUT_PDF}\n")
    f.write(f"  Pages      : {len(images)}\n")
    f.write(f"  DPI        : {DPI}\n")
    f.write("==========================================================\n")
    f.write(f"  Stage 1 — Model Loading          : {load_time:.2f}s\n")
    f.write(f"  Stage 2 — Text Line Detection    : {det_time:.2f}s\n")
    f.write(f"  Stage 3 — Layout Analysis        : {lay_time:.2f}s\n")
    f.write(f"  Stage 4 — Table Recognition      : {tab_time:.2f}s\n")
    f.write(f"  Stage 5 — JSON Write             : {json_time:.2f}s\n")
    f.write(f"  Stage 6 — OCR Recognition        : {rec_time:.2f}s\n")
    f.write(f"  ─────────────────────────────────────────────────────\n")
    f.write(f"  TOTAL                            : {total_time:.2f}s\n")
    f.write("==========================================================\n")
print(f"Timing saved to: {timing_file}")

# ── Save extracted text ───────────────────────────────────────────────────────
text_file = base_name + "_ocr.txt"
with open(text_file, "w", encoding="utf-8") as f:
    for page_num, page in enumerate(rec_results):
        f.write(f"{'='*60}\n  Page {page_num + 1}\n{'='*60}\n\n")
        for block in sorted(page.blocks, key=lambda b: b.reading_order):
            if not block.skipped and not block.error and block.html.strip():
                text = block_to_text(block)
                if text:
                    f.write(text + "\n\n")
print(f"Extracted text saved to: {text_file}")