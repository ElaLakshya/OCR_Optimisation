# Surya OCR Pipeline Optimisation

The goal of this project is to benchmark and optimise each stage of the [Surya OCR](https://github.com/VikParuchuri/surya) document understanding pipeline by evaluating and selecting the best-performing model for each step.

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
├── README.md                  ← you are here
│
├── step1/                     ← Baseline extraction pipeline
│   ├── imgtohtml.py           Image → formatted HTML
│   ├── imgtotxt.py            Image → plain text
│   ├── pdftohtml.py           PDF → formatted HTML
│   ├── pdftotext.py           PDF → plain text
│   ├── README.md
│   └── requirements.txt
│
└── step2/                     ← (coming soon) Per-pipeline timing benchmark
```

---

## Steps

### Step 1 — Baseline Extraction Pipeline
**Folder:** `step1/`

Establishes the baseline: running the full Surya pipeline end-to-end on image and PDF inputs, measuring total extraction time, and saving output as plain text or formatted HTML.

Supports Hindi (Devanagari) and English. HTML output uses Google Noto fonts for correct multilingual rendering.

### Step 2 — Per-Pipeline Timing Benchmark *(in progress)*
**Folder:** `step2/`

Instruments each of the 5 Surya pipelines individually to measure how long each stage takes. This isolates where time is being spent across:
- Text Line Detection
- Layout Analysis
- Reading Order Prediction
- Table Structure Recognition
- Text & LaTeX Recognition

Results will identify the bottleneck stages that are the best candidates for model optimisation.

### Step 3+ — Model Optimisation *(planned)*
Each bottleneck pipeline will be tested with alternative models to find faster or more accurate replacements while maintaining output quality.

---

## Setup

See `step1/README.md` for full installation instructions including Python environment setup and the required `llama-server` binary.

---

## Credits

- OCR engine: [Surya](https://github.com/VikParuchuri/surya) by Vik Paruchuri — licensed under GPL-3.0
- Inference backend: [llama.cpp](https://github.com/ggml-org/llama.cpp)
