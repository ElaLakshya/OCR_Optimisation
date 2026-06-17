"""
pipeline.py

Main OCR pipeline orchestrator for the webapp.

Two modes:
    A) PDF  → Docling extracts digital text, garbled blocks sent to Surya OCR,
              tables with garbled headers use Surya TableRecPredictor + Cell OCR
    B) Image → PP-DocLayoutV3 layout + Surya OCR directly

NOTE: This checkpoint fixes three bugs from the dHash/300-DPI revision:
    1. image_cache.py now compares dHashes via Hamming distance 
       so near-identical signature crops actually deduplicate.
    2. _surya_table_to_html now extracts cell text from BlockOCRResult.html
       (via BeautifulSoup) instead of a non-existent .text attribute.
    3. Per-cell table OCR now passes a LayoutBox/LayoutResult per cell so
       Surya uses fast block-mode instead of slow full-page mode per cell,
       and all cells are batched into a single _rec_predictor call.

Known NOT fixed here (separate issue): Docling/DocLayNet misclassifying
form-style header boxes as TABLE on documents like test1.pdf, which
scrambles reading order. That is a layout-detection issue independent of
the Hindi-garbled-table problem these fixes target.
"""
from __future__ import annotations

import os
os.environ["RECOGNITION_MAX_PAGES"] = "1"
os.environ["RECOGNITION_IMAGE_SIZE"] = "896"

import sys
import time
import io
from typing import Optional
from PIL import Image

from image_cache import ImageCache
from layout import LayoutClassifier
from table_rec import TableRecognizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PP_MODEL_PATH = os.path.normpath(os.path.join(
    _THIS_DIR, "..", "..", "Surya_OCR", "surya", "models", "PP-DocLayoutV3.onnx"
))

_SURYA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "Surya_OCR", "surya"))
if _SURYA_DIR not in sys.path:
    sys.path.insert(0, _SURYA_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  GARBLED TEXT DETECTION — word level
# ══════════════════════════════════════════════════════════════════════════════
_GARBLED_CHARS = set(";\\/æøåÆØÅçèéêëìíîïðñòóôõöùúûüýþÿ'{}")

def _word_is_garbled(word: str) -> bool:
    import re
    if any(c in _GARBLED_CHARS for c in word):
        return True
    # Unpronounceable consonant cluster — 4+ consonants in a row
    if re.search(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}', word):
        return True
    return False

def _is_garbled(text: str) -> bool:
    import re
    clean = re.sub(r'<[^>]+>', '', text).strip()
    words = [w for w in clean.split() if len(w) > 1]
    if len(words) < 2:
        return False
    garbled = sum(1 for w in words if _word_is_garbled(w))
    return (garbled / len(words)) > 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE REGION PRE-FILTER
# ══════════════════════════════════════════════════════════════════════════════
def _should_skip_region(image: Image.Image) -> bool:
    w, h = image.size
    if w < 50 or h < 20:
        return True
    aspect = w / h if h > 0 else 1
    if 0.8 <= aspect <= 1.2 and w < 250 and h < 250:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE HTML VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def _table_html_has_content(html: str) -> bool:
    if not html or not html.strip():
        return False
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        cells = soup.find_all(["td", "th"])
        non_empty = [c for c in cells if c.get_text(strip=True)]
        return len(non_empty) >= 2
    except Exception:
        return bool(html.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class OCRPipeline:

    def __init__(self):
        self._layout_classifier   = LayoutClassifier(PP_MODEL_PATH)
        self._table_recognizer    = TableRecognizer()   # RapidTable — for image input
        self._surya_tab_predictor = None                # Surya TableRecPredictor — for PDFs
        self._surya_loaded        = False
        self._det_predictor       = None
        self._rec_predictor       = None

    def _load_surya(self):
        if self._surya_loaded:
            return
        print("[Pipeline] Loading Surya predictors...")
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.table_rec import TableRecPredictor
        self._det_predictor       = DetectionPredictor()
        self._rec_predictor       = RecognitionPredictor()
        self._surya_tab_predictor = TableRecPredictor()
        self._surya_loaded        = True
        print("[Pipeline] Surya predictors loaded.")

    def _ocr_crop(self, image: Image.Image) -> str:
        """Run Surya OCR on a single image crop using fast block mode."""
        self._load_surya()
        from surya.layout.schema import LayoutBox, LayoutResult
        import torch

        w, h = image.size
        box = LayoutBox(
            polygon=[0, 0, w, h],
            label="Text",
            raw_label="text",
            position=0,
            confidence=1.0,
            count=300,
        )
        layout = LayoutResult(
            bboxes=[box],
            image_bbox=[0.0, 0.0, float(w), float(h)],
        )

        with torch.inference_mode():
            results = self._rec_predictor([image], [layout])

        if not results or not results[0].blocks:
            return ""

        html_parts = []
        for block in sorted(results[0].blocks, key=lambda b: b.reading_order):
            if not block.skipped and not block.error and block.html.strip():
                html_parts.append(block.html)
        return "\n".join(html_parts)

    def _surya_table_to_html(self, table_result, base_table_img: Image.Image) -> str:
        """
        Convert Surya TableRecPredictor cell layout into HTML, running Surya
        OCR on each cell crop to recover correct Hindi/Devanagari text that
        Docling's native export got from a garbled font encoding.
        """
        self._load_surya()
        import torch
        from bs4 import BeautifulSoup
        from surya.layout.schema import LayoutBox, LayoutResult

        cells = getattr(table_result, 'cells', [])
        if not cells:
            return ""

        max_row = max((getattr(c, 'row_id', 0) for c in cells), default=0)
        max_col = max((getattr(c, 'col_id', 0) for c in cells), default=0)

        cell_crops   = []
        cell_layouts = []
        cell_coords  = []

        for cell in cells:
            r = getattr(cell, 'row_id', 0)
            c = getattr(cell, 'col_id', 0)
            x0, y0, x1, y1 = [int(v) for v in cell.bbox]

            # Padding preserves Hindi matras at cell edges
            pad = 6
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1 = min(base_table_img.width,  x1 + pad)
            cy1 = min(base_table_img.height, y1 + pad)
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            crop = base_table_img.crop((cx0, cy0, cx1, cy1))
            cw, ch = crop.size

            box = LayoutBox(
                polygon=[0, 0, cw, ch],
                label="Text",
                raw_label="text",
                position=0,
                confidence=1.0,
                count=150,  # small cells — modest token budget is plenty
            )
            layout = LayoutResult(
                bboxes=[box],
                image_bbox=[0.0, 0.0, float(cw), float(ch)],
            )

            cell_crops.append(crop)
            cell_layouts.append(layout)
            cell_coords.append((r, c))

        grid = {}
        if cell_crops:
            with torch.inference_mode():
                # Single batched call across all cells, each using block mode
                rec_results = self._rec_predictor(cell_crops, cell_layouts)

            for idx, res in enumerate(rec_results):
                r, c = cell_coords[idx]
                text_parts = []
                if res and res.blocks:
                    for b in sorted(res.blocks, key=lambda x: x.reading_order):
                        if not b.skipped and not b.error and b.html.strip():
                            text_parts.append(
                                BeautifulSoup(b.html, "html.parser").get_text(strip=True)
                            )
                grid[(r, c)] = " ".join(text_parts).strip()

        rows = []
        for r in range(max_row + 1):
            cols = []
            for c in range(max_col + 1):
                text = grid.get((r, c), '')
                tag = 'th' if r == 0 else 'td'
                cols.append(f'<{tag}>{text}</{tag}>')
            rows.append(f'<tr>{"".join(cols)}</tr>')

        return f'<table>{"".join(rows)}</table>'

    def _crop_pdf_region(self, page_obj, x0, y0, x1, y1, dpi=300, padding=12):
        """
        Crop a region from a PDF page as PIL Image at 300 DPI for high-accuracy OCR.
        padding (in points) avoids cutting off matras/diacritics at edges.
        """
        import fitz
        pw = page_obj.rect.width
        ph = page_obj.rect.height

        pad_pts = padding * 72 / dpi
        cx0 = max(0, x0 - pad_pts)
        cy0 = max(0, y0 - pad_pts)
        cx1 = min(pw, x1 + pad_pts)
        cy1 = min(ph, y1 + pad_pts)

        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        clip = fitz.Rect(cx0, cy0, cx1, cy1)
        pix  = page_obj.get_pixmap(matrix=mat, clip=clip)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # ══════════════════════════════════════════════════════════════════════════
    #  PDF PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def process_pdf(self, pdf_path: str, progress_callback=None) -> dict:
        cache       = ImageCache()
        stats       = {"stages": {}, "cache_hits": 0, "cache_misses": 0}
        pages       = []
        total_start = time.time()

        def _progress(stage, pct):
            if progress_callback:
                progress_callback(stage, pct)
            print(f"[Pipeline] {stage} ({pct}%)")

        # ── Stage 1: Docling extraction ───────────────────────────────────────
        _progress("Extracting digital text", 5)
        t0 = time.time()

        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import DocItemLabel
        import fitz

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr                  = False
        pipeline_options.do_table_structure      = True
        pipeline_options.generate_picture_images = False
        pipeline_options.generate_page_images    = False

        converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(pdf_path)
        doc    = result.document
        stats["stages"]["docling"] = round(time.time() - t0, 2)
        _progress("Digital text extracted", 20)

        # ── Stage 2: Build page blocks ────────────────────────────────────────
        _progress("Analysing document structure", 25)
        t0 = time.time()

        pdf_doc     = fitz.open(pdf_path)
        n_pages     = len(pdf_doc)
        page_blocks = [[] for _ in range(n_pages)]
        pos_counter = [0] * n_pages

        for item, level in doc.iterate_items():
            if not hasattr(item, 'prov') or not item.prov:
                continue

            for prov in item.prov:
                page_no  = prov.page_no - 1
                bbox_obj = prov.bbox
                if bbox_obj is None:
                    continue

                page_obj = pdf_doc[page_no]
                ph       = page_obj.rect.height

                x0 = bbox_obj.l
                y0 = ph - bbox_obj.t
                x1 = bbox_obj.r
                y1 = ph - bbox_obj.b
                if y0 > y1: y0, y1 = y1, y0
                if x0 > x1: x0, x1 = x1, x0

                bbox = [x0, y0, x1, y1]
                pos  = pos_counter[page_no]
                pos_counter[page_no] += 1
                label = item.label

                # ── Picture → cache + classify + OCR ─────────────────────────
                if label == DocItemLabel.PICTURE:
                    try:
                        img = self._crop_pdf_region(page_obj, x0, y0, x1, y1)
                        page_blocks[page_no].append({
                            "type": "image_region", "image": img,
                            "page": page_no, "bbox": bbox,
                            "position": pos, "content": None,
                        })
                    except Exception as e:
                        print(f"[Pipeline] Picture crop failed: {e}")

                # ── Table ─────────────────────────────────────────────────────
                elif label == DocItemLabel.TABLE:
                    try:
                        tbl_html = item.export_to_html(doc)
                    except TypeError:
                        try:
                            tbl_html = item.export_to_html()
                        except Exception:
                            tbl_html = ""
                    except Exception:
                        tbl_html = ""

                    if not _table_html_has_content(tbl_html):
                        try:
                            img = self._crop_pdf_region(page_obj, x0, y0, x1, y1)
                            tbl_html, _ = self._table_recognizer.recognize(img)
                            tbl_html = tbl_html or ""
                        except Exception as e:
                            print(f"[Pipeline] RapidTable fallback failed: {e}")
                            tbl_html = ""

                    elif _is_garbled(tbl_html):
                        print(f"[Pipeline] Table garbled — running Surya TableRecPredictor + Cell OCR")
                        try:
                            self._load_surya()
                            img = self._crop_pdf_region(page_obj, x0, y0, x1, y1)
                            import torch
                            with torch.inference_mode():
                                tab_results = self._surya_tab_predictor([img])
                            if tab_results and getattr(tab_results[0], 'cells', []):
                                tbl_html = self._surya_table_to_html(tab_results[0], img)
                        except Exception as e:
                            print(f"[Pipeline] Surya TableRecPredictor failed: {e}")

                    page_blocks[page_no].append({
                        "type": "table", "page": page_no, "bbox": bbox,
                        "position": pos, "content": tbl_html,
                    })

                # ── List items ────────────────────────────────────────────────
                elif label == DocItemLabel.LIST_ITEM:
                    raw_text = ""
                    if hasattr(item, 'orig') and item.orig:
                        raw_text = item.orig
                    elif hasattr(item, 'text') and item.text:
                        raw_text = item.text

                    html = f"<ul><li>{raw_text}</li></ul>"
                    page_blocks[page_no].append({
                        "type": "list", "page": page_no, "bbox": bbox,
                        "position": pos, "content": html,
                    })

                # ── Text → check garbled, route accordingly ───────────────────
                else:
                    raw_text = ""
                    if hasattr(item, 'orig') and item.orig:
                        raw_text = item.orig
                    elif hasattr(item, 'text') and item.text:
                        raw_text = item.text

                    try:
                        exported = item.export_to_html() if hasattr(item, 'export_to_html') else ""
                    except Exception:
                        exported = ""

                    if _is_garbled(raw_text) or _is_garbled(exported):
                        print(f"[Pipeline] Garbled: {repr(raw_text[:40])}")
                        try:
                            img = self._crop_pdf_region(page_obj, x0, y0, x1, y1)
                            page_blocks[page_no].append({
                                "type": "image_region", "image": img,
                                "page": page_no, "bbox": bbox,
                                "position": pos, "content": None,
                            })
                        except Exception as e:
                            print(f"[Pipeline] Garbled crop failed: {e}")
                    else:
                        html = exported if exported.strip() else f"<p>{raw_text}</p>"
                        if html.strip():
                            page_blocks[page_no].append({
                                "type": "text", "page": page_no, "bbox": bbox,
                                "position": pos, "content": html,
                            })

        pdf_doc.close()
        stats["stages"]["structure_analysis"] = round(time.time() - t0, 2)
        _progress("Structure analysis complete", 40)

        # ── Stage 3: Process image regions with cache ─────────────────────────
        t0 = time.time()
        _progress("Running OCR on image regions", 45)

        n_images = sum(
            1 for page in page_blocks for b in page
            if b["type"] == "image_region"
        )

        blocks_to_ocr = []
        images_to_ocr = []

        for page_no, blocks in enumerate(page_blocks):
            for block in blocks:
                if block["type"] != "image_region":
                    continue

                img = block["image"]

                # Cache check (dHash + Hamming distance)
                cached = cache.get(img)
                if cached is not None:
                    stats["cache_hits"] += 1
                    block["content"] = "" if cache.is_skip(cached) else cached
                    continue

                # Pre-filter: QR codes, tiny icons
                if _should_skip_region(img):
                    cache.mark_skip(img)
                    block["content"] = ""
                    print(f"[Pipeline] Pre-filtered: size={img.size}")
                    continue

                # Classify with PP-DocLayoutV3
                label = self._layout_classifier.classify(img)
                print(f"[Pipeline] Classified: size={img.size} → {label}")

                if label == "Skip":
                    cache.mark_skip(img)
                    block["content"] = ""
                elif label == "Table":
                    html, _ = self._table_recognizer.recognize(img)
                    cache.set(img, html or "")
                    block["content"] = html or ""
                else:
                    blocks_to_ocr.append(block)
                    images_to_ocr.append(img)
                    stats["cache_misses"] += 1

        # OCR each non-cached, non-skipped image region using fast block mode
        if images_to_ocr:
            self._load_surya()
            print(f"[Pipeline] OCR on {len(images_to_ocr)} image region(s)...")

            import gc
            gc.collect()

            for idx, img in enumerate(images_to_ocr):
                final_html = self._ocr_crop(img)
                blocks_to_ocr[idx]["content"] = final_html
                cache.set(images_to_ocr[idx], final_html)
                print(f"[Pipeline] OCR {idx+1}/{len(images_to_ocr)} done")

            gc.collect()

        stats["stages"]["ocr"] = round(time.time() - t0, 2)
        _progress("OCR complete", 85)

        # ── Stage 4: Assemble output ──────────────────────────────────────────
        for page_no, blocks in enumerate(page_blocks):
            pages.append({
                "page":   page_no + 1,
                "blocks": sorted(blocks, key=lambda b: b["position"]),
            })

        stats["total"]           = round(time.time() - total_start, 2)
        stats["n_pages"]         = n_pages
        stats["n_image_regions"] = n_images

        _progress("Rendering output", 90)
        html = _render_html(pages)
        _progress("Done", 100)

        return {"html": html, "pages": pages, "stats": stats}

    # ══════════════════════════════════════════════════════════════════════════
    #  IMAGE PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def process_image(self, image_path: str, progress_callback=None) -> dict:
        stats       = {"stages": {}, "cache_hits": 0, "cache_misses": 0}
        total_start = time.time()

        def _progress(stage, pct):
            if progress_callback:
                progress_callback(stage, pct)
            print(f"[Pipeline] {stage} ({pct}%)")

        _progress("Loading image", 5)
        image = Image.open(image_path).convert("RGB")

        _progress("Analysing layout", 10)
        t0 = time.time()
        from pp_layout_backend import PPLayoutPredictor
        lay_predictor = PPLayoutPredictor(PP_MODEL_PATH)
        lay_results   = lay_predictor([image])
        stats["stages"]["layout"] = round(time.time() - t0, 2)
        _progress("Layout complete", 30)

        _progress("Recognising tables", 35)
        t0 = time.time()
        blocks_out = []

        for pos, box in enumerate(lay_results[0].bboxes):
            x0, y0, x1, y1 = [int(v) for v in box.bbox]
            crop = image.crop((x0, y0, x1, y1))

            if box.label == "Table":
                tbl_html, _ = self._table_recognizer.recognize(crop)
                blocks_out.append({
                    "type": "table", "content": tbl_html or "",
                    "bbox": box.bbox, "position": pos,
                })
            elif box.label == "Picture":
                blocks_out.append({
                    "type": "skip", "content": "",
                    "bbox": box.bbox, "position": pos,
                })
            else:
                blocks_out.append({
                    "type": "text_block", "content": None,
                    "bbox": box.bbox, "position": pos,
                })

        stats["stages"]["table_rec"] = round(time.time() - t0, 2)

        _progress("Running OCR", 50)
        t0 = time.time()
        self._load_surya()

        import torch
        with torch.inference_mode():
            rec_results = self._rec_predictor([image], lay_results)
        stats["stages"]["ocr"] = round(time.time() - t0, 2)
        _progress("OCR complete", 85)

        if rec_results:
            for block in rec_results[0].blocks:
                if not block.skipped and not block.error and block.html.strip():
                    for b in blocks_out:
                        if b["type"] == "text_block" and b["position"] == block.reading_order:
                            b["content"] = block.html
                            break

        clean_blocks = [b for b in blocks_out if b["type"] != "skip"]
        pages = [{"page": 1, "blocks": sorted(clean_blocks, key=lambda b: b["position"])}]

        stats["total"]           = round(time.time() - total_start, 2)
        stats["n_pages"]         = 1
        stats["n_image_regions"] = 0

        _progress("Rendering output", 90)
        html = _render_html(pages)
        _progress("Done", 100)

        return {"html": html, "pages": pages, "stats": stats}


# ══════════════════════════════════════════════════════════════════════════════
# HTML RENDERER
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# HTML RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def _render_html(pages: list) -> str:
    html_parts = ["""
<!DOCTYPE html>
<html lang="hi">
<head>
<meta charset="UTF-8">
<style>
  body {
    font-family: 'Mangal', Arial, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    line-height: 1.5;
    font-size: 11pt;
    color: #222222;
  }
  
  /* FIX 2: Let wkhtmltopdf paginate naturally to stop blank overflow pages */
  .page { margin-bottom: 30px; }
  
  .page-header { font-size: 9pt; color: #888888; margin-bottom: 10px; border-bottom: 1px solid #e5e5e5; padding-bottom: 4px; }
  
  /* FIX 1: Left align text to stop weird justified white-spaces */
  .text-block { margin-bottom: 10px; text-align: left; }
  
  .table-block { margin-top: 15px; margin-bottom: 20px; width: 100%; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 10pt;
    table-layout: auto;
  }
  td, th {
    border: 1px solid #666666;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
    word-break: break-word;
  }
  th { background-color: #f2f2f2; font-weight: bold; }
  
  /* FIX 3: Remove default black dots & add a hanging indent for alignment */
  ul, ol { 
    margin: 4px 0; 
    padding-left: 20px; 
    list-style-type: none; 
  }
  li { 
    margin-bottom: 6px; 
    text-indent: -18px; /* Pulls the '1.' or 'i.' to the left */
    padding-left: 18px; /* Pushes the subsequent lines to the right */
  }
</style>
</head>
<body>
"""]
    for page in pages:
        html_parts.append(f'<div class="page"><div class="page-header">Page {page["page"]}</div>')
        for block in page["blocks"]:
            content = block.get("content") or ""
            if not content.strip():
                continue

            # Skip Docling hallucinated picture descriptions (img with no src)
            if '<img' in content and 'src=' not in content:
                continue

            if block["type"] == "table":
                html_parts.append(f'<div class="table-block">{content}</div>')
            elif block["type"] == "list":
                html_parts.append(f'<div>{content}</div>')
            else:
                html_parts.append(f'<div class="text-block">{content}</div>')
        html_parts.append('</div>')

    html_parts.append("</body></html>")
    return "\n".join(html_parts)