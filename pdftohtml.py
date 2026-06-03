import time
import sys
import os
import fitz  # PyMuPDF
from PIL import Image
from surya.recognition import RecognitionPredictor
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
#  CONFIG — change this to your PDF path
# ─────────────────────────────────────────────
INPUT_PDF = "input1.pdf"
DPI = 150        # increase to 200/300 for better accuracy on small text
# ─────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OCR Output — {filename}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600&family=Noto+Sans+Devanagari:wght@400;600&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Noto Sans', 'Noto Sans Devanagari', sans-serif;
    background: #f4f6f8;
    color: #1a1a2e;
    padding: 40px 20px;
    font-size: 14px;
    line-height: 1.7;
  }}

  .document {{
    max-width: 900px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    padding: 48px 56px;
  }}

  .doc-title {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #aaa;
    text-transform: uppercase;
    margin-bottom: 32px;
    padding-bottom: 16px;
    border-bottom: 1px solid #e0e0e0;
  }}

  .page-divider {{
    text-align: center;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #aaa;
    text-transform: uppercase;
    margin: 40px 0 32px;
    border-top: 1px solid #e0e0e0;
    padding-top: 16px;
  }}

  .block {{ margin-bottom: 18px; }}

  .block-SectionHeader {{
    text-align: center;
    font-weight: 600;
    font-size: 15px;
    line-height: 2;
    margin-bottom: 24px;
    color: #111;
  }}

  .block-SectionHeader.note {{
    text-align: left;
    font-size: 13px;
    margin-bottom: 8px;
    color: #555;
  }}

  .block-Text {{ padding: 2px 0; color: #222; }}
  .block-Text.result {{
    font-weight: 600;
    font-size: 15px;
    margin-top: 12px;
    color: #1a5c2a;
  }}

  .block-Table {{
    width: 100%;
    margin: 20px 0;
    border-collapse: collapse;
    font-size: 13px;
  }}

  .block-Table th {{
    background: #1a1a2e;
    color: #fff;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    line-height: 1.4;
  }}

  .block-Table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #e8e8e8;
    vertical-align: middle;
  }}

  .block-Table tr:nth-child(even) td {{ background: #f9f9fb; }}
  .block-Table tr:last-child td {{ border-bottom: none; }}

  .block-ListGroup {{ padding-left: 20px; color: #333; }}
  .block-ListGroup li {{ margin-bottom: 6px; line-height: 1.6; }}

  .footer {{
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #e0e0e0;
    font-size: 11px;
    color: #aaa;
    text-align: right;
  }}
</style>
</head>
<body>
<div class="document">
  <div class="doc-title">OCR Output — {filename}</div>
  {content}
  <div class="footer">Extracted in {elapsed:.2f}s using Surya OCR</div>
</div>
</body>
</html>"""


def pdf_to_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def block_to_html(block):
    label = block.label
    soup = BeautifulSoup(block.html, "html.parser")

    if label == "Table":
        table = soup.find("table")
        if table:
            table["class"] = "block-Table"
            if table.get("border"):
                del table["border"]
            return str(table)
        return ""

    elif label == "ListGroup":
        ol = soup.find("ol") or soup.find("ul")
        if ol:
            tag = "ol" if soup.find("ol") else "ul"
            items = ol.find_all("li")
            html_items = ""
            for item in items:
                for br in item.find_all("br"):
                    br.replace_with(" | ")
                html_items += f"<li>{item.decode_contents().strip()}</li>\n"
            return f'<{tag} class="block-ListGroup">\n{html_items}</{tag}>'
        return ""

    elif label == "SectionHeader":
        for br in soup.find_all("br"):
            br.replace_with("\n")
        lines = soup.get_text().strip().split("\n")
        lines_html = "<br/>".join(l.strip() for l in lines if l.strip())
        css_class = "block-SectionHeader note" if "Note" in soup.get_text() else "block-SectionHeader"
        return f'<div class="{css_class}">{lines_html}</div>'

    else:
        for br in soup.find_all("br"):
            br.replace_with("\n")
        text = soup.get_text().strip()
        css_extra = " result" if "RESULT" in text.upper() else ""
        text_html = "<br/>".join(line.strip() for line in text.split("\n") if line.strip())
        return f'<div class="block-Text{css_extra}">{text_html}</div>'


# ── Validate input ──────────────────────────────────────────────────────────
if not os.path.isfile(INPUT_PDF):
    print(f"ERROR: File not found — {INPUT_PDF}")
    sys.exit(1)

if not INPUT_PDF.lower().endswith(".pdf"):
    print("ERROR: Input file must be a .pdf")
    sys.exit(1)

# ── Run OCR ─────────────────────────────────────────────────────────────────
print("Loading models...")
recognition_predictor = RecognitionPredictor()

print(f"Converting PDF to images at {DPI} DPI...")
images = pdf_to_images(INPUT_PDF, dpi=DPI)
print(f"Found {len(images)} page(s)")

print("Starting OCR extraction...")
start_time = time.time()

predictions = recognition_predictor(images)

elapsed = time.time() - start_time
print(f"Extraction completed in: {elapsed:.2f} seconds")

# ── Build HTML ───────────────────────────────────────────────────────────────
content_parts = []
for page_num, page in enumerate(predictions):
    if page_num > 0:
        content_parts.append(f'<div class="page-divider">Page {page_num + 1}</div>')
    for block in sorted(page.blocks, key=lambda b: b.reading_order):
        if not block.skipped and not block.error and block.html.strip():
            html_block = block_to_html(block)
            if html_block:
                content_parts.append(f'<div class="block">{html_block}</div>')

filename = os.path.basename(INPUT_PDF)
final_html = HTML_TEMPLATE.format(
    filename=filename,
    content="\n".join(content_parts),
    elapsed=elapsed
)

# ── Save output ──────────────────────────────────────────────────────────────
output_name = os.path.splitext(INPUT_PDF)[0] + "_ocr.html"
with open(output_name, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"Output saved to: {output_name}")
print("Open it in any browser to view the formatted result.")