"""
main.py
FastAPI backend for the OCR webapp.
Integrated with local BitNet-b1.58-2B-4T CPU Inference for:
- /summarize/{job_id} — strict 3-line FIR summary
- /classify/{job_id}  — Valid FIR / Invalid FIR classification
"""

from __future__ import annotations

import os
import uuid
import threading
import time
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from bs4 import BeautifulSoup

from pipeline import OCRPipeline

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="OCR Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ───────────────────────────────────────────────────────────────
_THIS_DIR  = Path(__file__).parent
UPLOAD_DIR = _THIS_DIR / "uploads"
OUTPUT_DIR = _THIS_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Global pipeline instance ──────────────────────────────────────────────────
_pipeline = OCRPipeline()

# ── BitNet paths ──────────────────────────────────────────────────────────────
_BITNET_DIR   = Path(r"C:\Users\ElaYTurbo\Desktop\TurboFile\Projects\BitNet")
_BITNET_BIN   = _BITNET_DIR / "build" / "bin" / "Release"
_LLAMA_CLI    = _BITNET_BIN / "llama-cli.exe"
_BITNET_MODEL = _BITNET_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"

# Safety limit: keeps prompt + generation comfortably under the 2048 context window
_MAX_DOC_CHARS = 6000

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)

def _run_job(job_id: str, file_path: str, file_type: str):
    try:
        _update_job(job_id, status="running", stage="Starting", progress=0)

        def progress_callback(stage: str, pct: int):
            _update_job(job_id, stage=stage, progress=pct)

        if file_type == "pdf":
            result = _pipeline.process_pdf(file_path, progress_callback)
        else:
            result = _pipeline.process_image(file_path, progress_callback)

        html_path = OUTPUT_DIR / f"{job_id}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(result["html"])

        # Convert HTML to PDF using pdfkit
        pdf_path = OUTPUT_DIR / f"{job_id}.pdf"
        try:
            import pdfkit
            config = pdfkit.configuration(
                wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
            )
            options = {
                'encoding': 'UTF-8',
                'enable-local-file-access': None,
                'quiet': '',
            }
            pdfkit.from_string(result["html"], str(pdf_path), configuration=config, options=options)
            print("[pdfkit] PDF generated successfully")
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                pdf_path = None
        except Exception as e:
            import traceback
            print(f"[pdfkit] PDF generation failed: {e}")
            print(traceback.format_exc())
            pdf_path = None

        _update_job(
            job_id,
            status="done",
            stage="Done",
            progress=100,
            html=result["html"],
            html_path=str(html_path),
            pdf_path=str(pdf_path) if pdf_path and pdf_path.exists() else None,
            stats=result["stats"],
        )

    except Exception as e:
        import traceback
        _update_job(
            job_id,
            status="error",
            stage="Error",
            progress=0,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        print(traceback.format_exc())
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

def _get_clean_document_text(job: dict) -> str:
    """Strip HTML tags and truncate to a safe length for BitNet's context window."""
    clean_text = BeautifulSoup(job["html"], "html.parser").get_text(separator="\n", strip=True)
    if len(clean_text) > _MAX_DOC_CHARS:
        clean_text = clean_text[:_MAX_DOC_CHARS] + "\n... [DOCUMENT TRUNCATED DUE TO LENGTH LIMIT] ..."
    return clean_text

def _run_bitnet(job_id: str, prompt: str, n_tokens: int = 400) -> str:
    """
    Writes the prompt to a temp file and invokes the compiled BitNet llama-cli
    binary. Always cleans up the temp file, even on failure.
    """
    prompt_file = OUTPUT_DIR / f"{job_id}_prompt.txt"
    try:
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        command = [
            str(_LLAMA_CLI),
            "-m", str(_BITNET_MODEL),
            "-f", str(prompt_file),
            "-n", str(n_tokens),
            "--temp", "0.1",
            "-c", "2048",
            "-b", "128",  # lower batch size avoids CPU segfaults on this hardware
        ]

        print(f"[BitNet] Triggering native C++ CPU inference from {_BITNET_BIN}...")

        # cwd must be the binary's directory so Windows resolves ggml.dll correctly
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=str(_BITNET_BIN),
            check=True,
        )
        return result.stdout

    except subprocess.CalledProcessError as e:
        print("[BitNet] Subprocess error stdout:\n", e.stdout)
        print("[BitNet] Subprocess error stderr:\n", e.stderr)
        raise HTTPException(status_code=500, detail="BitNet inference failed during execution.")
    finally:
        try:
            os.remove(prompt_file)
        except Exception:
            pass

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    print("[Startup] Pre-loading Surya predictors...")
    try:
        _pipeline._load_surya()
        print("[Startup] Surya ready.")
    except Exception as e:
        print(f"[Startup] WARNING: Surya pre-load failed: {e}")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    filename  = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        file_type = "pdf"
    elif extension in (".jpg", ".jpeg", ".png"):
        file_type = "image"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Upload PDF, JPG, or PNG."
        )

    job_id    = str(uuid.uuid4())
    save_path = str(UPLOAD_DIR / f"{job_id}{extension}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with _jobs_lock:
        _jobs[job_id] = {
            "status":     "queued",
            "stage":      "Queued",
            "progress":   0,
            "html":       None,
            "html_path":  None,
            "pdf_path":   None,
            "stats":      None,
            "error":      None,
            "created_at": time.time(),
            "filename":   filename,
            "file_type":  file_type,
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, save_path, file_type),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "filename": filename, "file_type": file_type}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id":   job_id,
        "status":   job["status"],
        "stage":    job["stage"],
        "progress": job["progress"],
        "error":    job.get("error"),
        "filename": job.get("filename"),
    }

@app.get("/result/{job_id}")
def get_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Unknown error"))
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job still processing")

    return {
        "job_id":   job_id,
        "html":     job["html"],
        "has_pdf":  job["pdf_path"] is not None,
        "stats":    job["stats"],
        "filename": job["filename"],
    }

# ── BitNet: 3-line FIR Summary ───────────────────────────────────────────────
@app.post("/summarize/{job_id}")
def summarize_document(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None or job["status"] != "done" or not job.get("html"):
        raise HTTPException(status_code=400, detail="Job not ready for summarization")

    try:
        clean_text = _get_clean_document_text(job)

        # 1. THE PROMPT CONSTRAINT: Demand a single paragraph and 3 sentences
        prompt = f"""You are a helpful assistant that summarizes text. Extract exactly three facts from this document. Fill in the three numbered lines below based on the text.
CRITICAL RULE: DO NOT invent unproven information or outcomes.

DOCUMENT TEXT:
{clean_text}

3-LINE SUMMARY:
1."""

        # We lower the token limit to 100 since 3 lines will never need more than that
        output = _run_bitnet(job_id, prompt, n_tokens=100)

        # 2. BULLETPROOF EXTRACTION: Split safely at the header
        if "3-LINE SUMMARY:" in output:
            # Take everything AFTER "3-LINE SUMMARY:"
            summary_raw = output.split("3-LINE SUMMARY:")[-1].strip()
            
            # Since the prompt ended with "1.", summary_raw will already start with "1."
            # If for some reason it doesn't, we add it back safely.
            if not summary_raw.startswith("1."):
                summary_raw = "1. " + summary_raw
                
            summary_text = summary_raw
        else:
            summary_text = output.strip()

        # 3. THE PYTHON GUILLOTINE: Strictly enforce 3 lines maximum
        clean_lines = [line.strip() for line in summary_text.split('\n') if line.strip()]
        summary_text = '\n'.join(clean_lines[:3])

        return {"summary": summary_text}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("[BitNet] Unknown error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ── BitNet: Document Type + Investigation Status Classification ──────────────
# NOTE: "Not an FIR" is a document-type check (is this even a police case
# document at all). "Suspect Identified" / "Case Open" report investigation
# status for documents that ARE FIRs — NOT a judgment on whether the
# underlying complaint is legitimate. An "open/untraced" result is a normal,
# common outcome for genuine crimes and does not mean the FIR was fake,
# rejected, or without merit — it only reflects whether the document
# currently shows a named suspect tied to supporting evidence.
@app.post("/classify/{job_id}")
def classify_document(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None or job["status"] != "done" or not job.get("html"):
        raise HTTPException(status_code=400, detail="Job not ready for classification")

    try:
        clean_text = _get_clean_document_text(job)

        prompt = f"""You are a helpful assistant reviewing a document. First check whether this document is a police FIR / case diary at all. Then, only if it is, determine the current investigation status based only on what is stated in the text.

Respond with exactly one of these three labels, followed by a one-sentence reason:
- "Not an FIR" — the document is not a police FIR or case diary (e.g. it is an unrelated letter, certificate, invoice, or other document type).
- "Suspect Identified" — the document is an FIR, a specific named individual is identified as a suspect, AND there is stated evidence connecting them to the offense (e.g. confession, forensic match, recovered stolen property, witness identification).
- "Case Open" — the document is an FIR, but no named suspect is tied to supporting evidence (this includes: no suspect found, suspects were questioned but not linked by evidence, or the case is recorded as untraced/pending).

Do not judge whether the complaint itself is genuine or whether it should be filed — only report the document type, and if it is an FIR, what it states about suspect identification and evidence.

DOCUMENT TEXT:
{clean_text}

STATUS:"""

        output = _run_bitnet(job_id, prompt, n_tokens=60)

        if "STATUS:" in output:
            label = output.split("STATUS:")[-1].strip()
        else:
            label = output.strip()

        # Normalize to one of the three expected labels where possible.
        # Order matters: check "not an fir" before the "suspect identified" /
        # "case open" checks, since a non-FIR document should never fall
        # through to an investigation-status label.
        label_lower = label.lower()
        if "not an fir" in label_lower or "not a fir" in label_lower:
            normalized = "Not an FIR"
        elif "suspect identified" in label_lower:
            normalized = "Suspect Identified"
        elif "case open" in label_lower or "untraced" in label_lower:
            normalized = "Case Open"
        else:
            normalized = label  # fall back to raw model output

        return {"status": normalized, "raw_output": label}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("[BitNet] Unknown error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{job_id}/html")
def download_html(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job or not job.get("html_path") or not os.path.exists(job["html_path"]):
        raise HTTPException(status_code=404, detail="HTML not found")

    filename = Path(job["filename"]).stem + "_ocr.html"
    return FileResponse(job["html_path"], media_type="text/html", filename=filename)

@app.get("/download/{job_id}/pdf")
def download_pdf(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job or not job.get("pdf_path") or not os.path.exists(job["pdf_path"]):
        raise HTTPException(status_code=404, detail="PDF not found or generation failed")

    filename = Path(job["filename"]).stem + "_ocr.pdf"
    return FileResponse(job["pdf_path"], media_type="application/pdf", filename=filename)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "jobs":   len(_jobs),
        "surya_loaded": _pipeline._surya_loaded,
    }