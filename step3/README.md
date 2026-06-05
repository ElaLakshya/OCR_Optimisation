# Step 3 — Layout Analysis Pipeline Swap (Surya → PP-DocLayoutV3 ONNX)

## Overview

Step 2 showed that Layout Analysis consumed ~41% of total pipeline runtime (65s out of 160s), entirely due to Surya's VLM-backed layout model. Step 3 replaces that single stage with a fast CNN-based alternative — **PP-DocLayoutV3 ONNX** — while keeping every other Surya stage intact.

This is a surgical swap: the overall pipeline architecture is unchanged. Only the layout predictor is replaced.

---

## Why Layout Analysis Was Targeted

| Stage | Time | Replaceable? |
|---|---|---|
| Text Line Detection | 4s | Not needed — already fast |
| Layout Analysis | 65s | ✅ Yes — pure detection task, no language understanding needed |
| Table Recognition | 10s | Potential future target |
| OCR Recognition | 80s | ❌ Hard constraint — Surya's VLM kept intentionally |

OCR Recognition was deliberately left unchanged. The goal of this project is to preserve Surya's OCR accuracy while reducing total pipeline time. Replacing the OCR model would risk accuracy loss, especially on mixed Hindi/English documents.

Layout analysis, by contrast, is a purely geometric task — it finds and labels rectangular regions on a page. It has no dependency on language, script, or text content. This makes it safe to swap independently.

---

## Why PP-DocLayoutV3 ONNX

- Pure CNN object detector (RT-DETR architecture) — no VLM, no llama.cpp
- Pre-converted ONNX model available: `alex-dinh/PP-DocLayoutV3-ONNX` on HuggingFace
- No PaddleX or Paddle framework dependency — runs on pure `onnxruntime + numpy + opencv`
- 23 layout categories that map cleanly onto Surya's label schema
- Runs entirely on CPU — compatible with the Intel Iris Xe hardware constraint

---

## How the Swap Works

Surya's `RecognitionPredictor` accepts a `layout_results` parameter — a list of `LayoutResult` objects, one per page. By building a drop-in predictor that produces `LayoutResult` objects matching Surya's exact schema, we can substitute the layout stage without touching any other part of the pipeline.

```
BEFORE:  DetectionPredictor → LayoutPredictor (VLM) → TableRecPredictor → RecognitionPredictor
AFTER:   DetectionPredictor → PPLayoutPredictor (ONNX CNN) → TableRecPredictor → RecognitionPredictor
```

`pp_layout_backend.py` implements `PPLayoutPredictor` as a drop-in replacement. It imports and constructs Surya's real `LayoutBox` and `LayoutResult` Pydantic models directly from `surya.layout.schema`, ensuring full schema compatibility with downstream stages.

---

## Label Mapping

PP-DocLayoutV3 uses its own 23-category label set. These are mapped to Surya's canonical labels:

```python
LABEL_MAP = {
    "text":             "Text",
    "paragraph_title":  "SectionHeader",
    "document_title":   "SectionHeader",
    "table":            "Table",
    "figure":           "Picture",
    "image":            "Picture",
    "figure_caption":   "Caption",
    "table_caption":    "Caption",
    "footnotes":        "Footnote",
    "header":           "PageHeader",
    "footer":           "PageFooter",
    "formula":          "Equation",
    "algorithm":        "Code",
    ...
}
```

---

## Table Detection Problem and Fix

### The Problem

During testing on a CBSE marksheet (`input1.pdf`), PP-DocLayoutV3 misclassified the marks table as `raw_label='image'` (mapped to `Picture`). This is a known limitation of models trained primarily on academic papers — a dense ruled table rendered from PDF at 150 DPI can visually resemble a structured graphic.

This was a critical bug: blocks labelled `Picture` are passed to Surya's recognition stage as images to skip, so the entire table content would have been lost.

### How It Was Identified

A debug print of `raw_label` was added to the test script to inspect what PP-DocLayoutV3 actually assigned to each block before label mapping:

```
[Picture] raw='image' conf=0.97 bbox=[39.5, 621.3, 1196.8, 864.1]
```

The block spanning nearly the full page width with 97% confidence was being called `image`.

### The Fix — Geometric Heuristic

Since layout analysis is purely geometric, a three-signal heuristic was used to distinguish wide ruled tables from actual images:

```python
def _looks_like_table(det, orig_w, orig_h):
    w = x1 - x0
    h = y1 - y0
    aspect     = w / h          # tables are wide and shallow
    width_frac = w / orig_w     # tables span most of page width
    area_frac  = (w * h) / (orig_w * orig_h)

    return aspect > 2.5 and width_frac > 0.7 and area_frac > 0.08
```

This is applied only to blocks where `raw_label in ("image", "figure")` — it never affects blocks the model already labelled correctly. A real photograph rarely spans 70%+ of page width and has an aspect ratio above 2.5 simultaneously.

---

## Token Budget Fix

Surya's `RecognitionPredictor` uses `box.count` to calculate a token budget for each block:

```python
max_tokens = image_token_budget(box.count, ceiling=2048)
# count=0   → max_tokens=100
# count=50  → max_tokens=150
# count=300 → max_tokens=400
# count=900 → max_tokens=1000
```

PP-DocLayoutV3 does not produce a token estimate. An initial hardcoded `count=1` caused the marks table to be truncated after the header row (only 101 tokens available).

The fix uses a geometric area estimate, with a hard override for Table blocks:

```python
def _estimate_count(x0, y0, x1, y1, page_w, page_h):
    area_frac = ((x1 - x0) * (y1 - y0)) / (page_w * page_h)
    if area_frac < 0.05:   return 50
    elif area_frac < 0.15: return 300
    elif area_frac < 0.30: return 600
    else:                  return 900

# Table blocks always get maximum budget
if det["label"] == "Table":
    count = 900
```

---

## Results

| Metric | Surya Baseline | Hybrid |
|---|---|---|
| Layout Analysis | 65.14s | 2.72s |
| Layout Speedup | — | **24x faster** |
| Total Pipeline | 160.60s | 93.71s |
| Total Improvement | — | **41.6% faster** |
| OCR Output Quality | Reference | Equivalent |

The marks table, all subject scores, personal details, and result legend are all present and correct in the hybrid output.

---

## Files

| File | Purpose |
|---|---|
| `pp_layout_backend.py` | Drop-in `PPLayoutPredictor` — the layout swap implementation |
| `test_ppdoclayout.py` | Standalone test: compares PP-DocLayoutV3 vs Surya layout side by side |
| `benchmark_hybrid.py` | Full pipeline benchmark: runs Surya baseline then hybrid, saves timing + OCR output for both |

### What `benchmark_hybrid.py` produces

- Console comparison table (per-stage timings for both pipelines)
- `input1_baseline_ocr.txt` — OCR output from full Surya pipeline
- `input1_hybrid_ocr.txt` — OCR output from hybrid pipeline
- `input1_hybrid_timing.txt` — timing log saved to disk

---

## How to Run

```bash
# Test layout detection only
python test_ppdoclayout.py

# Full hybrid vs baseline benchmark
python benchmark_hybrid.py
```

Requires the PP-DocLayoutV3 ONNX model at `./models/PP-DocLayoutV3.onnx`. Download with:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='alex-dinh/PP-DocLayoutV3-ONNX', filename='PP-DocLayoutV3.onnx', local_dir='./models')"
```

---

## Requirements

```
onnxruntime
opencv-python
numpy
huggingface_hub
```

All other dependencies are inherited from the Surya virtual environment.