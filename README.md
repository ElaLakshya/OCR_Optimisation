# Surya OCR Pipeline Optimisation

The goal of this project is to benchmark and optimise each stage of the [Surya OCR](https://github.com/VikParuchuri/surya) document understanding pipeline by evaluating and selecting the best-performing model for each step, and ultimately delivering a web application for document extraction.

Surya processes a document through up to 5 sequential deep-learning pipelines. This project analyses each pipeline individually — measuring speed, accuracy, and resource usage — to identify bottlenecks and explore better model alternatives.

---

## The 5-Step Surya Pipeline

```
[Input Document Image]
         │
         ▼
 1. Text Line Detection ───► 2. Layout Analysis
         │                           │
         ▼                           ▼
 3. Reading Order Detection  4. Table Structure Recognition
         │
         ▼
 5. Text / LaTeX Recognition ───► [Final Structured Output]
```

| Step | Pipeline | Description |
|---|---|---|
| 1 | **Text Line Detection** | Locates bounding polygons around every line of text on the page |
| 2 | **Layout Analysis** | Classifies page regions into semantic zones: Text, Table, Picture, Formula, Header, Footer, Caption |
| 3 | **Reading Order Prediction** | Computes correct 0-indexed reading sequence across columns and layout blocks |
| 4 | **Table Structure Recognition** | Identifies rows, columns, and cell boundaries; outputs structured HTML or schema |
| 5 | **Text & LaTeX Recognition** | Transcribes visual line segments into text strings or LaTeX notation across 90+ languages |

---

## Project Structure

```
/
├── .gitignore
├── README.md                        ← you are here
│
├── step1/                           ← Baseline extraction pipeline
│   ├── imgtohtml.py                 Image → formatted HTML
│   ├── imgtotxt.py                  Image → plain text
│   ├── pdftohtml.py                 PDF → formatted HTML
│   ├── pdftotext.py                 PDF → plain text
│   ├── README.md
│   └── requirements.txt
│
├── step2/                           ← Per-pipeline timing benchmark
│   ├── individual.py                Times each Surya stage separately
│   └── README.md
│
├── step3/                           ← Layout swap + hybrid pipeline + BitNet AI
│   ├── pp_layout_backend.py         PP-DocLayoutV3 drop-in for Surya layout
│   ├── benchmark_hybrid.py          Baseline vs hybrid benchmark
│   ├── test_ppdoclayout.py          Layout comparison test
│   ├── SLANet_accuracy.py           SLANet table recognition evaluation
│   ├── requirements.txt
│   └── README.md
│
└── webapp/                          ← Web application (FastAPI + React)
    ├── README.md
    ├── backend/
    │   ├── main.py                  FastAPI app (OCR + BitNet LLM integration)
    │   ├── pipeline.py              OCR pipeline orchestrator
    │   ├── image_cache.py           dHash (perceptual) image cache
    │   ├── layout.py                PP-DocLayoutV3 classifier
    │   └── table_rec.py             RapidTable wrapper
    └── frontend/
        └── src/
            └── App.js               React UI with LLM action buttons
```

---

## Steps

### Step 1 — Baseline Extraction Pipeline
**Folder:** `step1/`

Establishes the baseline: running the full Surya pipeline end-to-end on image and PDF inputs, measuring total extraction time, and saving output as plain text or formatted HTML. Supports Hindi (Devanagari) and English.

### Step 2 — Per-Pipeline Timing Benchmark
**Folder:** `step2/`

Instruments each Surya stage individually to measure time taken per stage:

| Stage | Time | % of Total |
|---|---|---|
| Text Line Detection | 4.14s | 2.6% |
| Layout Analysis | 65.14s | 40.6% |
| Table Recognition | 10.59s | 6.6% |
| OCR Recognition | 80.73s | 50.3% |
| **Total** | **160.60s** | 100% |

Layout Analysis (41%) and OCR Recognition (50%) identified as the two main bottlenecks.

### Step 3 — Layout Analysis Swap + Hybrid Pipeline
**Folder:** `step3/`

Replaces Surya's VLM-based Layout Analysis with **PP-DocLayoutV3 ONNX** (a fast CNN detector):

| Metric | Surya Baseline | Hybrid |
|---|---|---|
| Layout Analysis | 65.14s | 2.72s |
| Total Pipeline | 160.60s | 93.71s |
| Improvement | — | **41.6% faster** |

Key fixes: table misclassification heuristic, token budget estimation, Pydantic schema compatibility.

### Step 4 — Web Application & AI Document Intelligence
**Folder:** `webapp/`

A FastAPI + React web app combining document extraction with local generative AI:

- **Docling** for fast digital text extraction from PDFs
- **PP-DocLayoutV3** for image region classification
- **Surya OCR** for garbled Hindi text and image regions
- **dHash (Difference Hash) Cache** to perceptually skip duplicate image regions (logos, seals)
- **Local BitNet 1.58b LLM** integrated natively via C++ subprocess to classify documents and generate strict, structured summaries completely offline
- **wkhtmltopdf** for PDF output

Supports PDF and image (JPG/PNG) input. Output as HTML and PDF.

---

## Hardware

All benchmarks and development on:
- CPU: Intel Core i7-1355U (13th Gen)
- RAM: 16GB
- GPU: Intel Iris Xe (integrated, no CUDA)
- OS: Windows 11
- Inference: llama.cpp Vulkan build / Native C++ CPU Inference

---

## Running the Web App

```powershell
# Terminal 1 — Backend
cd webapp\backend
.\venv_webapp\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd webapp\frontend
npm start
```

Open `http://localhost:3000` — see `webapp/README.md` for full setup instructions.

---

## Setup

See `step1/README.md` for Surya installation. See `webapp/README.md` for webapp setup including wkhtmltopdf and model downloads. See `step3/README.md` for the BitNet build process.

---

## Credits

- OCR engine: [Surya](https://github.com/VikParuchuri/surya) by Vik Paruchuri — licensed under GPL-3.0
- Layout model: [PP-DocLayoutV3](https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX) — ONNX export by alex-dinh
- Digital extraction: [Docling](https://github.com/docling-project/docling) by IBM
- Inference backends: [llama.cpp](https://github.com/ggml-org/llama.cpp) & Microsoft Visual Studio 2022 (MSVC)
- LLM Foundation: [BitNet b1.58 2B](https://github.com/microsoft/BitNet) by Microsoft