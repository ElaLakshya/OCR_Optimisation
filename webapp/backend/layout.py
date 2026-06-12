"""
layout.py

PP-DocLayoutV3 ONNX wrapper for the webapp pipeline.
Classifies image regions cropped from Docling's flagged blocks.
"""

from __future__ import annotations

import os
from typing import Optional
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# ── Model path — absolute, resolved at import time ────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    _THIS_DIR, "..", "..", "Surya_OCR", "surya", "models", "PP-DocLayoutV3.onnx"
)

# ── Label index → name ────────────────────────────────────────────────────────
IDX_TO_LABEL = [
    "text", "paragraph_title", "document_title", "table", "figure",
    "figure_caption", "table_caption", "footnotes", "header", "footer",
    "formula", "algorithm", "references", "aside_text", "seal",
    "chart", "chart_caption", "abstract", "content_list", "index",
    "date_place", "image",
]

# ── Labels that should be skipped entirely (no OCR) ───────────────────────────
SKIP_RAW_LABELS = {"seal", "chart", "figure_caption", "table_caption",
                   "chart_caption", "header", "footer"}

# ── Confidence threshold ──────────────────────────────────────────────────────
CONF_THRESH = 0.3
INPUT_SIZE  = (800, 800)


def _preprocess_letterbox(image_bgr: np.ndarray):
    """
    Pads the image with white borders to keep its aspect ratio intact 
    before resizing to 800x800. Prevents network distortion and hangs.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    
    # Calculate scale factor matching the longest edge
    scale = min(INPUT_SIZE[0] / orig_h, INPUT_SIZE[1] / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    
    # Resize keeping aspect ratio intact
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a solid white 800x800 canvas
    canvas = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), 255, dtype=np.uint8)
    
    # Center the resized crop on the canvas
    top = (INPUT_SIZE[0] - new_h) // 2
    left = (INPUT_SIZE[1] - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    
    # Normalize channels
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    
    # Scales relative to original sizes
    scale_h = new_h / orig_h
    scale_w = new_w / orig_w
    
    return blob, scale_h, scale_w, top, left


def _looks_like_table(x0, y0, x1, y1, orig_w, orig_h) -> bool:
    """Geometric heuristic to catch tables misclassified as images."""
    w = x1 - x0
    h = y1 - y0
    if h == 0:
        return False
    aspect     = w / h
    width_frac = w / orig_w
    area_frac  = (w * h) / (orig_w * orig_h)
    return aspect > 2.5 and width_frac > 0.7 and area_frac > 0.08


class LayoutClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = os.path.normpath(model_path)
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"PP-DocLayoutV3 ONNX model not found at: {self.model_path}"
            )
        print(f"[LayoutClassifier] Loading model from {self.model_path}...")
        
        # FIX: Force strict single-thread execution to stop deadlocks inside FastAPI background workers
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print("[LayoutClassifier] Model loaded cleanly.")

    def classify(self, image: Image.Image) -> str:
        self._load()

        # Catch miniature/empty images before processing
        if image.size[0] < 5 or image.size[1] < 5:
            return "Text"

        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = bgr.shape[:2]
        
        # Preprocess with letterboxing instead of forceful stretching
        blob, sh, sw, pad_top, pad_left = _preprocess_letterbox(bgr)

        input_feed = {
            "im_shape":     np.array([[800, 800]], dtype=np.float32),
            "image":        blob[np.newaxis, ...],
            "scale_factor": np.array([[sh, sw]], dtype=np.float32),
        }

        output_names = [o.name for o in self._session.get_outputs()]
        
        try:
            raw_output = self._session.run(output_names, input_feed)[0]
        except Exception as e:
            print(f"[LayoutClassifier WARNING] ONNX graph execution error: {e}. Falling back to Text.")
            return "Text"

        if raw_output is None or len(raw_output) == 0:
            return "Text"

        best_label = "text"
        best_score = 0.0
        best_bbox  = (0.0, 0.0, 0.0, 0.0)

        for row in raw_output:
            label_idx = int(row[0])
            score     = float(row[1])
            
            # Map back removing pad shifts
            x0 = (float(row[2]) - pad_left) / sw
            y0 = (float(row[3]) - pad_top) / sh
            x1 = (float(row[4]) - pad_left) / sw
            y1 = (float(row[5]) - pad_top) / sh

            if score < CONF_THRESH:
                continue
            if score > best_score:
                best_score = score
                best_label = IDX_TO_LABEL[label_idx] if label_idx < len(IDX_TO_LABEL) else "text"
                best_bbox  = (x0, y0, x1, y1)

        if best_label in SKIP_RAW_LABELS:
            return "Skip"

        if best_label == "table":
            return "Table"

        if best_label in ("image", "figure"):
            if _looks_like_table(*best_bbox, orig_w, orig_h):
                return "Table"
            return "Skip"

        return "Text"