# Webapp — OCR Pipeline Web Application

A local web application that extracts text from PDFs and images using a hybrid pipeline combining Docling (digital text) and Surya OCR (image regions), with local BitNet LLM integration for document summarization and classification.

## Quick Start

### 1. Start the backend
```powershell
cd webapp\backend
.\venv_webapp\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

### 2. Start the frontend
```powershell
cd webapp\frontend
npm start
```

### 3. Open browser
Navigate to `http://localhost:3000`

## Usage

1. Drag and drop or click to upload a PDF, JPG, or PNG
2. Watch the progress bar — stages take 30-120 seconds depending on document complexity
3. When done, preview the output in the browser
4. Download as HTML or PDF
5. Click **Summarize** for a strict 3-line factual summary of the document, generated locally by BitNet
6. Click **Classify** to categorize the document as `Valid FIR`, `Invalid FIR`, or `Not an FIR`

Both AI buttons run entirely offline via a locally compiled BitNet binary — no external API calls, no data leaves the machine.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload file, returns `job_id` |
| GET | `/status/{job_id}` | Poll progress |
| GET | `/result/{job_id}` | Get HTML output when done |
| POST | `/summarize/{job_id}` | Generate a strict 3-line summary via local BitNet LLM |
| POST | `/classify/{job_id}` | Classify document as Valid FIR / Invalid FIR / Not an FIR via local BitNet LLM |
| GET | `/download/{job_id}/html` | Download HTML file |
| GET | `/download/{job_id}/pdf` | Download PDF file |
| GET | `/health` | Server health check |

## Setup Requirements

### Python (webapp venv)
```powershell
cd webapp\backend
python -m venv venv_webapp
.\venv_webapp\Scripts\Activate.ps1
pip install fastapi uvicorn python-multipart docling onnxruntime opencv-python numpy Pillow beautifulsoup4 pymupdf rapid-table pdfkit
# Install Surya as local source
cd ..\..\Surya_OCR\surya
pip install -e .
```

**Note:** `webapp/backend/pipeline.py` imports `pp_layout_backend.py` (used for image-only uploads). If this repo is kept separate from the `step3/` project, copy `pp_layout_backend.py` directly into `webapp/backend/` so the import resolves — it is not included in this repo by default.

### Node.js
Download from https://nodejs.org — LTS version

```powershell
cd webapp\frontend
npm install
```

### wkhtmltopdf (PDF generation)
Download Windows 64-bit installer from https://wkhtmltopdf.org/downloads.html

### Models
- PP-DocLayoutV3.onnx — place at `Surya_OCR/surya/models/PP-DocLayoutV3.onnx`
- llama-server.exe — place at `Surya_OCR/surya/llama_cpp/llama-server.exe`
- Set environment variable: `$env:LLAMA_CPP_BINARY = "<path>\llama-server.exe"`

## Local LLM Setup (BitNet)

BitNet is not a pip package — it must be cloned and compiled separately with Visual Studio 2022 (Desktop development with C++ workload + C++ Clang tools for Windows individual component), run from a **Developer PowerShell for VS 2022**:

```powershell
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet

conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp
pip install -r requirements.txt

huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

### Fixing the BitNet Compilation Error

If the `setup_env.py` build fails, check `logs\compile.log`. On Windows, the Clang compiler is exceptionally strict and will often throw a `const`-correctness compilation error or complain about missing standard C++ libraries.

To patch the source code:

1. Open the specific `.cpp` or `.h` source file flagged in the `compile.log` error output.
2. **For missing definitions** (like `std::chrono` errors): add `#include <chrono>` at the very top of the file along with the other `#include` statements. MSVC does not implicitly bundle these the way GCC does.
3. **For const type-casting conflicts**: locate the line causing the error (typically a conflict between `const char*` and a non-const `char*`). Explicitly update the variable definition to `const`, or wrap the pointer in `const_cast<char*>()` to satisfy the Clang compiler.
4. Save the file and re-run the `python setup_env.py ...` command.

### Verifying the Build

Once the build completes successfully, verify the binaries were generated:

```powershell
dir build\bin\Release
```

You should see `llama-cli.exe` and `ggml.dll`.

Finally, update these path constants in `webapp/backend/main.py` to point to your local BitNet clone:

```python
_BITNET_DIR   = Path(r"C:\path\to\your\BitNet")
_BITNET_BIN   = _BITNET_DIR / "build" / "bin" / "Release"
_LLAMA_CLI    = _BITNET_BIN / "llama-cli.exe"
_BITNET_MODEL = _BITNET_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
```