# Step 3 — Layout Analysis Pipeline Swap + Hybrid Web App

## Overview

Step 3 has two parts:

1. **Layout Analysis Swap** — Surya's slow VLM-based layout model is replaced with PP-DocLayoutV3 ONNX (a fast CNN), achieving a 24x speedup on layout analysis and 41.6% total pipeline improvement.

2. **Hybrid Web Application** — A FastAPI + React web app that combines Docling (for digital text extraction) with the optimised Surya OCR pipeline (for image regions and garbled text).

---

## Part 1 — Layout Analysis Swap

### Why Layout Was Targeted

From the Step 2 benchmark, Layout Analysis consumed 65s (41% of total runtime) using Surya's VLM backend. It was the largest replaceable bottleneck — layout detection is a purely geometric task with no language dependency, making it safe to swap independently.

### PP-DocLayoutV3 ONNX

- Pure CNN object detector (RT-DETR architecture) — no VLM, no llama.cpp
- Pre-converted ONNX — runs on pure `onnxruntime + numpy + opencv`
- 23 layout categories mapping to Surya's label schema
- Runs on CPU — compatible with Intel Iris Xe hardware

### Table Misclassification Fix

PP-DocLayoutV3 misclassified wide ruled tables as `image` on the CBSE marksheet. Fixed with a geometric heuristic:

```python
def _looks_like_table(x0, y0, x1, y1, orig_w, orig_h):
    aspect     = (x1-x0) / (y1-y0)
    width_frac = (x1-x0) / orig_w
    area_frac  = ((x1-x0) * (y1-y0)) / (orig_w * orig_h)
    return aspect > 2.5 and width_frac > 0.7 and area_frac > 0.08
```

### Token Budget Fix

Surya's `RecognitionPredictor` uses `box.count` for token budgeting. PP-DocLayoutV3 doesn't produce this value so we estimate it geometrically, with Table blocks always getting `count=900` → `max_tokens=1000`.

### Results

| Stage | Surya Baseline | Hybrid | Saved |
|---|---|---|---|
| Layout Analysis | 65.14s | 2.72s | 62.42s |
| Total Pipeline | 160.60s | 93.71s | 66.89s |
| Improvement | — | — | **41.6%** |

---

## Part 2 — Hybrid Web Application

### Architecture

```
PDF uploaded
    │
    ▼
Docling (do_ocr=False)
    ├── Clean digital text → extracted directly
    ├── Garbled text (custom font encoding) → crop → Surya OCR
    ├── Pictures → SHA-256 cache → PP-DocLayoutV3 → Skip/Table/OCR
    └── Tables → Docling native HTML
                 → if garbled headers → Surya TableRecPredictor
                 → if empty → RapidTable fallback
    │
    ▼
Merge all blocks by position → render HTML → convert to PDF (wkhtmltopdf)

Image uploaded (JPG/PNG)
    │
    ▼
PP-DocLayoutV3 layout → Surya OCR + RapidTable → render HTML → PDF
```

### SHA-256 Image Cache

Every image region is hashed before classification. Identical images (e.g. a logo on every page) are processed once and cached — subsequent pages get the result instantly.

Two-layer cache:
- **Layer 1** — in-memory dict (microseconds)
- **Layer 2** — SQLite on disk (persists across runs)

### Garbled Hindi Detection

PDFs with custom font encoding store Hindi as ASCII lookalikes. Detection uses word-level analysis:

```python
_GARBLED_CHARS = set(";\\/æøåÆØÅçèéêëìíîïðñòóôõöùúûüýþÿ'{}")

def _word_is_garbled(word):
    if any(c in _GARBLED_CHARS for c in word):
        return True
    # Unpronounceable consonant cluster
    if re.search(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}', word):
        return True
    return False
```

If >10% of words in a block are garbled, the block is cropped and sent to Surya OCR.

### Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Python |
| Frontend | React (create-react-app) |
| Digital text extraction | Docling 2.98.0 |
| Layout classification | PP-DocLayoutV3 ONNX |
| OCR | Surya RecognitionPredictor (VLM via llama-server) |
| Table recognition (PDF) | Surya TableRecPredictor |
| Table recognition (images) | RapidTable (SLANet ONNX) |
| PDF output | wkhtmltopdf via pdfkit |
| Image cache | Python dict + SQLite |

---

## Files

| File | Purpose |
|---|---|
| `pp_layout_backend.py` | PP-DocLayoutV3 drop-in for Surya's LayoutPredictor |
| `test_ppdoclayout.py` | Layout comparison test (PP-DocLayout vs Surya) |
| `benchmark_hybrid.py` | Full pipeline benchmark — baseline vs hybrid |
| `webapp/backend/main.py` | FastAPI app — upload, status, result, download endpoints |
| `webapp/backend/pipeline.py` | Main OCR pipeline orchestrator |
| `webapp/backend/image_cache.py` | SHA-256 image cache (memory + SQLite) |
| `webapp/backend/layout.py` | PP-DocLayoutV3 single-image classifier |
| `webapp/backend/table_rec.py` | RapidTable wrapper (swappable) |
| `webapp/frontend/src/App.js` | React UI — upload, progress, output display |

---

## How to Run

### Backend
```powershell
cd webapp\backend
.\venv_webapp\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

### Frontend
```powershell
cd webapp\frontend
npm start
```

Open `http://localhost:3000`

### Requirements (webapp venv)
```
fastapi
uvicorn
python-multipart
docling
onnxruntime
opencv-python
numpy
Pillow
beautifulsoup4
pymupdf
rapid-table
pdfkit
surya-ocr (local source install)
```

### External dependencies
- **wkhtmltopdf** — for PDF generation. Install from https://wkhtmltopdf.org/downloads.html
- **PP-DocLayoutV3.onnx** — download from `alex-dinh/PP-DocLayoutV3-ONNX` on HuggingFace, place at `Surya_OCR/surya/models/PP-DocLayoutV3.onnx`
- **llama-server.exe** — Vulkan build, place at `Surya_OCR/surya/llama_cpp/llama-server.exe`

---

## Known Limitations

- PDFs with custom Hindi font encoding will have garbled table column headers. Body text and standalone Hindi paragraphs are correctly routed to Surya OCR. This is a PDF encoding issue — the font glyph mapping is private and cannot be reversed without the original font file.
- PDF output requires wkhtmltopdf installed separately on Windows.
- First request after server start is slower — Surya models load on startup (~30s).