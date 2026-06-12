"""
benchmark_hybrid.py

Benchmarks the hybrid pipeline against the full Surya baseline.

Hybrid pipeline:
    Stage 1 — Model Loading          : Surya models (baseline) / PP-DocLayoutV3 ONNX (hybrid)
    Stage 2 — Text Line Detection    : Surya DetectionPredictor  (unchanged)
    Stage 3 — Layout Analysis        : PP-DocLayoutV3 ONNX       (SWAPPED)
    Stage 4 — Table Recognition      : Surya TableRecPredictor   (unchanged)
    Stage 5 — JSON Write             : same logic as individual.py
    Stage 6 — OCR Recognition        : Surya RecognitionPredictor(unchanged)

Run from Surya venv:
    python benchmark_hybrid.py
"""

import time
import sys
import os
import json
import fitz
from PIL import Image
from bs4 import BeautifulSoup

from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor
from pp_layout_backend import PPLayoutPredictor

sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_PDF = "Pumplet01.jpg"
DPI       = 150
PP_MODEL = os.path.join(os.getcwd(), "models", "PP-DocLayoutV3.onnx")
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

def write_json(det_results, lay_results, tab_results, tab_page_map, images, output_path):
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
    for table_idx, (tab_res, page_idx) in enumerate(zip(tab_results, tab_page_map)):
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

def run_table_rec(tab_predictor, images, lay_results):
    table_images = []
    table_page_map = []
    for page_idx, (img, lay) in enumerate(zip(images, lay_results)):
        for block in lay.bboxes:
            if block.label == "Table":
                x0, y0, x1, y1 = [int(v) for v in block.bbox]
                table_images.append(img.crop((x0, y0, x1, y1)))
                table_page_map.append(page_idx)
    if table_images:
        tab_results = tab_predictor(table_images)
        return tab_results, table_page_map, len(table_images)
    return [], [], 0

if not os.path.isfile(INPUT_PDF):
    print(f"ERROR: File not found — {INPUT_PDF}")
    sys.exit(1)

base_name = os.path.splitext(INPUT_PDF)[0]



# ══════════════════════════════════════════════════════════════════════════════
#  RUN 1 — SURYA BASELINE (all 6 stages)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  RUN 1 — SURYA BASELINE")
print("=" * 60)

# Stage 1 — Model Loading
print("  [1/6] Model Loading...")
t0 = time.time()
det_predictor = DetectionPredictor()
lay_predictor = LayoutPredictor()
rec_predictor = RecognitionPredictor()
tab_predictor = TableRecPredictor()
images = pdf_to_images(INPUT_PDF, dpi=DPI)
base_load_time = time.time() - t0
print(f"        {base_load_time:.2f}s  ({len(images)} page(s) loaded)\n")

# Stage 2 — Text Line Detection
print("  [2/6] Text Line Detection...")
t0 = time.time()
det_results = det_predictor(images)
base_det_time = time.time() - t0
print(f"        {base_det_time:.2f}s\n")

# Stage 3 — Layout Analysis
print("  [3/6] Layout Analysis (Surya)...")
t0 = time.time()
base_lay_results = lay_predictor(images)
base_lay_time = time.time() - t0
print(f"        {base_lay_time:.2f}s  ({sum(len(r.bboxes) for r in base_lay_results)} blocks)\n")

# Stage 4 — Table Recognition
print("  [4/6] Table Recognition...")
t0 = time.time()
base_tab_results, base_tab_map, base_n_tables = run_table_rec(tab_predictor, images, base_lay_results)
base_tab_time = time.time() - t0
print(f"        {base_tab_time:.2f}s  ({base_n_tables} table(s))\n")

# Stage 5 — JSON Write
print("  [5/6] JSON Write...")
t0 = time.time()
write_json(det_results, base_lay_results, base_tab_results, base_tab_map, images,
           base_name + "_baseline_structure.json")
base_json_time = time.time() - t0
print(f"        {base_json_time:.2f}s\n")

# Stage 6 — OCR Recognition
print("  [6/6] OCR Recognition...")
t0 = time.time()
base_rec_results = rec_predictor(images, base_lay_results)
base_rec_time = time.time() - t0
print(f"        {base_rec_time:.2f}s\n")

base_total = base_load_time + base_det_time + base_lay_time + base_tab_time + base_json_time + base_rec_time

# Save baseline OCR output
with open(base_name + "_baseline_ocr.txt", "w", encoding="utf-8") as f:
    for page_num, page in enumerate(base_rec_results):
        f.write(f"{'='*60}\n  Page {page_num + 1}\n{'='*60}\n\n")
        for block in sorted(page.blocks, key=lambda b: b.reading_order):
            if not block.skipped and not block.error and block.html.strip():
                text = block_to_text(block)
                if text:
                    f.write(text + "\n\n")
print(f"  Baseline OCR saved to: {base_name}_baseline_ocr.txt\n")

# ══════════════════════════════════════════════════════════════════════════════
#  RUN 2 — HYBRID PIPELINE (all 6 stages)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  RUN 2 — HYBRID (PP-DocLayoutV3 layout swap)")
print("=" * 60)

# Stage 1 — Model Loading (PP-DocLayoutV3 only; other models reused from baseline)
print("  [1/6] Model Loading (PP-DocLayoutV3 ONNX)...")
t0 = time.time()
pp_predictor = PPLayoutPredictor(PP_MODEL)
# Force load now so timing is accurate
pp_predictor._load()
hybrid_load_time = time.time() - t0
print(f"        {hybrid_load_time:.2f}s\n")

# Stage 2 — Text Line Detection (reuse baseline results, same predictor)
print("  [2/6] Text Line Detection (reusing baseline results)...")
print(f"        {base_det_time:.2f}s  (same as baseline)\n")
hybrid_det_time = base_det_time

# Stage 3 — Layout Analysis (PP-DocLayoutV3)
print("  [3/6] Layout Analysis (PP-DocLayoutV3)...")
t0 = time.time()
hybrid_lay_results = pp_predictor(images)
hybrid_lay_time = time.time() - t0
print(f"        {hybrid_lay_time:.2f}s  ({sum(len(r.bboxes) for r in hybrid_lay_results)} blocks)\n")

# Stage 4 — Table Recognition
print("  [4/6] Table Recognition...")
t0 = time.time()
hybrid_tab_results, hybrid_tab_map, hybrid_n_tables = run_table_rec(tab_predictor, images, hybrid_lay_results)
hybrid_tab_time = time.time() - t0
print(f"        {hybrid_tab_time:.2f}s  ({hybrid_n_tables} table(s))\n")

# Stage 5 — JSON Write
print("  [5/6] JSON Write...")
t0 = time.time()
write_json(det_results, hybrid_lay_results, hybrid_tab_results, hybrid_tab_map, images,
           base_name + "_hybrid_structure.json")
hybrid_json_time = time.time() - t0
print(f"        {hybrid_json_time:.2f}s\n")

# Stage 6 — OCR Recognition
print("  [6/6] OCR Recognition...")
t0 = time.time()
hybrid_rec_results = rec_predictor(images, hybrid_lay_results)
hybrid_rec_time = time.time() - t0
print(f"        {hybrid_rec_time:.2f}s\n")

hybrid_total = hybrid_load_time + hybrid_det_time + hybrid_lay_time + hybrid_tab_time + hybrid_json_time + hybrid_rec_time

# Save hybrid OCR output
with open(base_name + "_hybrid_ocr.txt", "w", encoding="utf-8") as f:
    for page_num, page in enumerate(hybrid_rec_results):
        f.write(f"{'='*60}\n  Page {page_num + 1}\n{'='*60}\n\n")
        for block in sorted(page.blocks, key=lambda b: b.reading_order):
            if not block.skipped and not block.error and block.html.strip():
                text = block_to_text(block)
                if text:
                    f.write(text + "\n\n")
print(f"  Hybrid OCR saved to: {base_name}_hybrid_ocr.txt\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
saved     = base_total - hybrid_total
pct_saved = (saved / base_total * 100) if base_total > 0 else 0.0

print("=" * 60)
print("  COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Stage':<30} {'Baseline':>10} {'Hybrid':>10} {'Saved':>10}")
print(f"  {'-'*60}")
print(f"  {'Model Loading':<30} {base_load_time:>9.2f}s {hybrid_load_time:>9.2f}s {base_load_time - hybrid_load_time:>8.2f}s")
print(f"  {'Text Line Detection':<30} {base_det_time:>9.2f}s {hybrid_det_time:>9.2f}s {'—':>9}")
print(f"  {'Layout Analysis':<30} {base_lay_time:>9.2f}s {hybrid_lay_time:>9.2f}s {base_lay_time - hybrid_lay_time:>8.2f}s")
print(f"  {'Table Recognition':<30} {base_tab_time:>9.2f}s {hybrid_tab_time:>9.2f}s {base_tab_time - hybrid_tab_time:>8.2f}s")
print(f"  {'JSON Write':<30} {base_json_time:>9.2f}s {hybrid_json_time:>9.2f}s {'—':>9}")
print(f"  {'OCR Recognition':<30} {base_rec_time:>9.2f}s {hybrid_rec_time:>9.2f}s {'—':>9}")
print(f"  {'-'*60}")
print(f"  {'TOTAL':<30} {base_total:>9.2f}s {hybrid_total:>9.2f}s {saved:>8.2f}s")
print(f"  {'Improvement':<30} {'':>10} {'':>10} {pct_saved:>8.1f}%")
print("=" * 60)
print(f"\n  Layout speedup : {base_lay_time / hybrid_lay_time:.1f}x faster")
print(f"  Layout blocks  : Surya={sum(len(r.bboxes) for r in base_lay_results)}  PP-DocLayout={sum(len(r.bboxes) for r in hybrid_lay_results)}")
print(f"\n  Compare OCR output quality:")
print(f"    Baseline : {base_name}_baseline_ocr.txt")
print(f"    Hybrid   : {base_name}_hybrid_ocr.txt")
print("=" * 60)

# Save timing log
timing_path = base_name + "_hybrid_timing.txt"
with open(timing_path, "w", encoding="utf-8") as f:
    f.write("HYBRID PIPELINE BENCHMARK\n")
    f.write("=" * 60 + "\n")
    f.write(f"  Input : {INPUT_PDF}\n")
    f.write(f"  Pages : {len(images)}\n")
    f.write(f"  DPI   : {DPI}\n")
    f.write("=" * 60 + "\n")
    f.write(f"  {'Stage':<30} {'Baseline':>10} {'Hybrid':>10} {'Saved':>10}\n")
    f.write(f"  {'-'*60}\n")
    f.write(f"  {'Model Loading':<30} {base_load_time:>9.2f}s {hybrid_load_time:>9.2f}s {base_load_time - hybrid_load_time:>8.2f}s\n")
    f.write(f"  {'Text Line Detection':<30} {base_det_time:>9.2f}s {hybrid_det_time:>9.2f}s {'—':>9}\n")
    f.write(f"  {'Layout Analysis':<30} {base_lay_time:>9.2f}s {hybrid_lay_time:>9.2f}s {base_lay_time - hybrid_lay_time:>8.2f}s\n")
    f.write(f"  {'Table Recognition':<30} {base_tab_time:>9.2f}s {hybrid_tab_time:>9.2f}s {base_tab_time - hybrid_tab_time:>8.2f}s\n")
    f.write(f"  {'JSON Write':<30} {base_json_time:>9.2f}s {hybrid_json_time:>9.2f}s {'—':>9}\n")
    f.write(f"  {'OCR Recognition':<30} {base_rec_time:>9.2f}s {hybrid_rec_time:>9.2f}s {'—':>9}\n")
    f.write(f"  {'-'*60}\n")
    f.write(f"  {'TOTAL':<30} {base_total:>9.2f}s {hybrid_total:>9.2f}s {saved:>8.2f}s\n")
    f.write(f"  {'Improvement':<30} {'':>10} {'':>10} {pct_saved:>8.1f}%\n")
    f.write("=" * 60 + "\n")
print(f"\nTiming saved to: {timing_path}")