# Surya OCR Pipeline

A set of Python scripts to extract text from images and PDFs using [Surya OCR](https://github.com/VikParuchuri/surya). Supports Hindi (Devanagari) and English, with output in plain text or formatted HTML.

---

## Scripts

| Script | Input | Output |
|---|---|---|
| `imgtohtml.py` | `.jpg / .jpeg / .png` | `_ocr.html` |
| `imgtotxt.py` | `.jpg / .jpeg / .png` | `_ocr.txt` |
| `pdftohtml.py` | `.pdf` | `_ocr.html` |
| `pdftotext.py` | `.pdf` | `_ocr.txt` |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `surya-ocr` — the OCR engine
- `pymupdf` — for reading PDFs
- `beautifulsoup4` — for parsing HTML output from Surya
- `pillow` — for image handling

> On first run, Surya will automatically download model weights (~1.5 GB) from Hugging Face. This only happens once and is cached for future runs.

### 4. Download llama-server binary

Surya requires the `llama-server` binary from [llama.cpp](https://github.com/ggml-org/llama.cpp/releases). Download the right build for your system:

| System | Build to download |
|---|---|
| Intel/AMD with integrated GPU | `llama-bXXXX-bin-win-vulkan-x64.zip` |
| NVIDIA GPU | `llama-bXXXX-bin-win-cuda-12.x-x64.zip` |
| CPU only | `llama-bXXXX-bin-win-cpu-x64.zip` |

Extract the zip and place `llama-server.exe` somewhere on your machine.

### 5. Set the binary path

Run this in PowerShell before using any script:
```powershell
$env:LLAMA_CPP_BINARY = "C:\path\to\llama-server.exe"
```

To set it permanently so you never have to type it again, add that line to your PowerShell profile:
```powershell
notepad $PROFILE
```

---

## Usage

Open whichever script you want to use and change the input file path at the top:

```python
INPUT_IMAGE = "your_file.jpeg"   # for image scripts
INPUT_PDF   = "your_file.pdf"    # for PDF scripts
```

Then run:
```powershell
python imgtohtml.py
```

Output is saved in the same folder as the input file, with `_ocr` appended to the name (e.g. `document_ocr.html`).

---

## Notes

- HTML output renders Hindi and multilingual text correctly using Google Noto fonts — open in any browser
- OCR speed on CPU: ~2 min/page. Vulkan build is faster on Intel/AMD integrated graphics
- PDF scripts convert each page to an image internally before running OCR

---

## Credits

- OCR engine: [Surya](https://github.com/VikParuchuri/surya) by Vik Paruchuri — licensed under GPL-3.0
- Inference backend: [llama.cpp](https://github.com/ggml-org/llama.cpp)
