"""
test_ppdoclayout.py
Tests PP-DocLayoutV3 ONNX layout detection standalone,
then benchmarks it against Surya's LayoutPredictor.

Run from Surya venv:
    python test_ppdoclayout.py
"""

import sys
import time
import os
import fitz
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

INPUT_PDF   = "input1.pdf"
MODEL_PATH  = "./models/PP-DocLayoutV3.onnx"
DPI         = 150


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


if not os.path.isfile(INPUT_PDF):
    print(f"ERROR: {INPUT_PDF} not found")
    sys.exit(1)

# ── Download model if not present ─────────────────────────────────────────────
if not os.path.isfile(MODEL_PATH):
    print("Downloading PP-DocLayoutV3 ONNX model...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="alex-dinh/PP-DocLayoutV3-ONNX",
        filename="PP-DocLayoutV3.onnx",
        local_dir="./models",
    )
    print("Download complete.")

print("Converting PDF to images...")
images = pdf_to_images(INPUT_PDF, dpi=DPI)
print(f"Found {len(images)} page(s)\n")

# ── TEST 1: PP-DocLayoutV3 ONNX ───────────────────────────────────────────────
print("=" * 60)
print("  TEST 1 — PP-DocLayoutV3 ONNX Layout")
print("=" * 60)

from pp_layout_backend import PPLayoutPredictor

from pp_layout_backend import PPLayoutPredictor, CONF_THRESH
print(f"Confidence threshold: {CONF_THRESH}")
pp_predictor = PPLayoutPredictor(MODEL_PATH)

t = time.time()
pp_results = pp_predictor(images)
pp_time = time.time() - t

print(f"\n  Time: {pp_time:.3f}s")
print(f"  Blocks detected: {sum(len(r.bboxes) for r in pp_results)}")
print("\n  Detected blocks:")
for r in pp_results:
    for box in r.bboxes:
        print(f"    [{box.label}] raw={box.raw_label!r:25s} conf={box.confidence:.2f} bbox={[round(v,1) for v in box.bbox]}")

# ── TEST 2: Surya LayoutPredictor ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST 2 — Surya LayoutPredictor (baseline)")
print("=" * 60)
from surya.layout import LayoutPredictor

surya_lay = LayoutPredictor()

t = time.time()
surya_results = surya_lay(images)
surya_time = time.time() - t

print(f"\n  Time: {surya_time:.3f}s")
print(f"  Blocks detected: {sum(len(r.bboxes) for r in surya_results)}")
print("\n  Detected blocks:")
for r in surya_results:
    for box in r.bboxes:
        print(f"    [{box.label}] bbox={[round(v,1) for v in box.bbox]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LAYOUT COMPARISON SUMMARY")
print("=" * 60)
print(f"  PP-DocLayoutV3 ONNX : {pp_time:.3f}s")
print(f"  Surya LayoutPredictor: {surya_time:.3f}s")
if pp_time < surya_time:
    print(f"  Speedup: {surya_time/pp_time:.1f}x faster with PP-DocLayoutV3")
else:
    print(f"  Slowdown: {pp_time/surya_time:.1f}x slower with PP-DocLayoutV3")
print("=" * 60)