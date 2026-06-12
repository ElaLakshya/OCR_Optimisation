"""
main.py

FastAPI backend for the OCR webapp.
"""

from __future__ import annotations

import os
from unittest import result
import uuid
import threading
import time
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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

        # Convert HTML to PDF using pure-Python xhtml2pdf compiler
        # Convert HTML to PDF using pure-Python xhtml2pdf compiler
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


# ── Startup: pre-load Surya so first request doesn't pay loading cost ─────────
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