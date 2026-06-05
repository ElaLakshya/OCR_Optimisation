# Step 2 — Surya OCR Pipeline Timing Benchmark

## Overview

Step 1 gave us a single end-to-end timestamp for the full Surya OCR pipeline. Step 2 breaks that pipeline into individual stages and times each one separately, so we can identify exactly where the bottlenecks are.

---

## Why This Was Needed

The end-to-end time from Step 1 told us the pipeline was slow, but not *which part* was slow. Before attempting any optimisation, we needed a breakdown — otherwise we might waste effort speeding up a stage that only contributes 2% of total runtime.

---

## The 6 Stages

Surya's OCR pipeline is not a single monolithic process. It is a sequence of models and operations that run one after another. `individual.py` times each of these stages separately:

| Stage | What it does |
|---|---|
| 1. Model Loading | Loads all four Surya predictors into memory |
| 2. Text Line Detection | Finds every line of text on the page using Surya's CNN detector |
| 3. Layout Analysis | Segments the page into regions (Text, Table, Picture, etc.) and assigns reading order |
| 4. Table Recognition | Crops each Table region and runs Surya's table structure model on it |
| 5. JSON Write | Serialises detection and layout results to disk for inspection |
| 6. OCR Recognition | Runs the VLM-backed OCR model on each layout block to extract text |

> **Note on Reading Order:** Reading order detection is embedded inside the Recognition stage in this version of Surya and is not exposed as a separately callable stage. It is therefore not timed independently.

---

## How Timing Works

Each stage is wrapped in a `time.time()` call:

```python
t0 = time.time()
lay_results = lay_predictor(images)
lay_time = time.time() - t0
```

This measures wall-clock time for that stage only. Model loading is separated from inference so that loading overhead does not contaminate the inference timings.

---

## Baseline Results

**Input:** `input1.pdf` (1 page, CBSE marksheet, printed Hindi + English), DPI = 150

| Stage | Time | % of Total |
|---|---|---|
| Model Loading | ~0.36s | — |
| Text Line Detection | ~4.14s | 2.6% |
| Layout Analysis | ~65.14s | 40.6% |
| Table Recognition | ~10.59s | 6.6% |
| JSON Write | ~0.01s | <0.1% |
| OCR Recognition | ~80.73s | 50.3% |
| **TOTAL (excl. loading)** | **~160.6s** | 100% |

### Key Finding

Two stages dominate total runtime:
- **OCR Recognition (80.73s, 50%)** — uses Surya's VLM backend (llama.cpp via Vulkan)
- **Layout Analysis (65.14s, 41%)** — also uses the VLM backend

Text Line Detection (4s) and Table Recognition (10s) are secondary. JSON Write is negligible.

This breakdown directly informed the optimisation strategy in Step 3.

---

## Files

| File | Purpose |
|---|---|
| `individual.py` | Main benchmarking script — times all 6 stages and saves results |

### What `individual.py` produces

- Console output with per-stage timings
- `input1_timing.txt` — timing log saved to disk
- `input1_structure.json` — layout and detection results in JSON format
- `input1_ocr.txt` — extracted text output

---

## How to Run

Activate the Surya virtual environment, then:

```bash
python individual.py
```

Ensure `input1.pdf` is present in the same directory. DPI and input filename are configurable at the top of the script.

---

## Requirements

Same as Step 1 — Surya OCR installed in the active virtual environment. No additional dependencies.