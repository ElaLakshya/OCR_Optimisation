# Webapp — OCR Pipeline Web Application

A local web application that extracts text from PDFs and images using a hybrid pipeline combining Docling (digital text) and Surya OCR (image regions).

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

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload file, returns `job_id` |
| GET | `/status/{job_id}` | Poll progress |
| GET | `/result/{job_id}` | Get HTML output when done |
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