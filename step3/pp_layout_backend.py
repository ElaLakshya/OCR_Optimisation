"""
pp_layout_backend.py

Drop-in replacement for Surya's LayoutPredictor using PP-DocLayoutV3 ONNX.
No PaddleX, no Paddle framework — pure onnxruntime + numpy + opencv.

Usage:
    from pp_layout_backend import PPLayoutPredictor
    lay_predictor = PPLayoutPredictor()
    lay_results = lay_predictor(images)   # same call as Surya's LayoutPredictor

Requirements:
    pip install onnxruntime opencv-python numpy
    Download model:
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='alex-dinh/PP-DocLayoutV3-ONNX', filename='PP-DocLayoutV3.onnx', local_dir='./models')"
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from surya.layout.schema import LayoutBox, LayoutResult

# ── Label mapping: PP-DocLayoutV3 → Surya canonical labels ───────────────────
LABEL_MAP = {
    "text":             "Text",
    "paragraph_title":  "SectionHeader",
    "document_title":   "SectionHeader",
    "table":            "Table",
    "figure":           "Picture",
    "image":            "Picture",
    "figure_caption":   "Caption",
    "table_caption":    "Caption",
    "footnotes":        "Footnote",
    "header":           "PageHeader",
    "footer":           "PageFooter",
    "formula":          "Equation",
    "algorithm":        "Code",
    "references":       "Text",
    "aside_text":       "Text",
    "seal":             "Picture",
    "chart":            "Picture",
    "chart_caption":    "Caption",
    "abstract":         "Text",
    "content_list":     "Text",
    "index":            "Text",
    "date_place":       "Text",
}

# PP-DocLayoutV3 class index → label name
IDX_TO_LABEL = [
    "text", "paragraph_title", "document_title", "table", "figure",
    "figure_caption", "table_caption", "footnotes", "header", "footer",
    "formula", "algorithm", "references", "aside_text", "seal",
    "chart", "chart_caption", "abstract", "content_list", "index",
    "date_place", "image",
]

DEFAULT_MODEL_PATH = "./models/PP-DocLayoutV3.onnx"
INPUT_SIZE = (800, 800)
CONF_THRESH = 0.3   # lowered from 0.5 — raise if too many false positives

def _pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _preprocess(image_bgr: np.ndarray) -> tuple:
    orig_h, orig_w = image_bgr.shape[:2]
    target_h, target_w = INPUT_SIZE
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # 1/255 only — no ImageNet normalization
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC → CHW
    return blob, scale_h, scale_w


def _bbox_to_polygon(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _postprocess(raw_output: np.ndarray, n_images: int,
                 orig_sizes: List[tuple]) -> List[List[dict]]:
    """
    Parse flat ONNX output into per-image detection lists.

    Actual output columns (7 cols):
        [label_idx, score, x0, y0, x1, y1, read_order]
    """
    results = [[] for _ in range(n_images)]

    if raw_output is None or len(raw_output) == 0:
        return results

    for row in raw_output:
        # Correct column order confirmed from debug output
        label_idx = int(row[0])
        score     = float(row[1])
        x0        = float(row[2])
        y0        = float(row[3])
        x1        = float(row[4])
        y1        = float(row[5])
        img_idx   = 0   # single image per call

        if score < CONF_THRESH:
            continue

        # Clamp to image bounds
        orig_w, orig_h = orig_sizes[img_idx]
        x0 = max(0.0, min(x0, orig_w))
        y0 = max(0.0, min(y0, orig_h))
        x1 = max(0.0, min(x1, orig_w))
        y1 = max(0.0, min(y1, orig_h))

        if x1 <= x0 or y1 <= y0:
            continue

        raw_label = IDX_TO_LABEL[label_idx] if label_idx < len(IDX_TO_LABEL) else "text"

        results[img_idx].append({
            "raw_label": raw_label,
            "label":     LABEL_MAP.get(raw_label, "Text"),
            "score":     score,
            "bbox":      [x0, y0, x1, y1],
        })

    return results


def _looks_like_table(det: dict, orig_w: float, orig_h: float) -> bool:
    """
    Heuristic to distinguish wide ruled tables misclassified as images/figures.
    A real table tends to be very wide (>70% page width), shallow relative to
    its width (aspect > 2.5), and large enough to be meaningful (>8% page area).
    Only applied to blocks the model labelled as 'image' or 'figure'.
    """
    x0, y0, x1, y1 = det["bbox"]
    w = x1 - x0
    h = y1 - y0
    if h == 0:
        return False
    aspect     = w / h
    width_frac = w / orig_w
    area_frac  = (w * h) / (orig_w * orig_h)
    return aspect > 2.5 and width_frac > 0.7 and area_frac > 0.08

def _estimate_count(x0: float, y0: float, x1: float, y1: float,
                    page_w: float, page_h: float) -> int:
    """
    Estimate token count for a block based on its area fraction.
    Surya uses multiples of 50. We approximate:
      - Small blocks  (<5% page)  -> 50  tokens
      - Medium blocks (5-15%)     -> 150 tokens
      - Large blocks  (15-30%)    -> 300 tokens
      - Very large    (>30%)      -> 500 tokens
    These map to max_tokens of 150/250/400/600 via image_token_budget.
    """
    def _estimate_count(x0: float, y0: float, x1: float, y1: float,
                    page_w: float, page_h: float) -> int:
        area_frac = ((x1 - x0) * (y1 - y0)) / (page_w * page_h)
        if area_frac < 0.05:
            return 50
        elif area_frac < 0.15:
            return 300
        elif area_frac < 0.30:
            return 600
        else:
            return 900
class PPLayoutPredictor:
    """
    Drop-in replacement for Surya's LayoutPredictor.
    Uses PP-DocLayoutV3 ONNX model via onnxruntime.
    Call signature matches Surya exactly: lay_results = lay_predictor(images)
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"PP-DocLayoutV3 ONNX model not found at: {self.model_path}\n"
                f"Download with:\n"
                f"  python -c \"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download(repo_id='alex-dinh/PP-DocLayoutV3-ONNX', "
                f"filename='PP-DocLayoutV3.onnx', local_dir='./models')\""
            )
        print(f"[PPLayoutPredictor] Loading ONNX model from {self.model_path}...")
        self._session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        print("[PPLayoutPredictor] Model loaded.")

    def __call__(
        self,
        images: List[Image.Image],
        target_image_sizes=None,
        max_tokens=None,
    ) -> List[LayoutResult]:
        self._load()

        if not images:
            return []

        results: List[LayoutResult] = []

        # Process one image at a time to keep img_idx simple
        for img in images:
            bgr = _pil_to_cv2(img)
            orig_w, orig_h = img.width, img.height
            blob, sh, sw = _preprocess(bgr)

            input_blob = blob[np.newaxis, ...]               # (1,3,800,800)
            shape_arr  = np.array([[800, 800]], dtype=np.float32)   # (1,2)
            scale_arr  = np.array([[sh, sw]],  dtype=np.float32)    # (1,2)

            input_feed = {
                "im_shape":     shape_arr,
                "image":        input_blob,
                "scale_factor": scale_arr,
            }

            output_names = [o.name for o in self._session.get_outputs()]
            raw_output = self._session.run(output_names, input_feed)[0]

            per_image = _postprocess(raw_output, 1, [(orig_w, orig_h)])
            dets = per_image[0]

            # Sort by top-to-bottom, left-to-right reading order
            dets_sorted = sorted(dets, key=lambda d: (d["bbox"][1], d["bbox"][0]))

            # ── Table/image disambiguation ────────────────────────────────
            for det in dets_sorted:
                if det["raw_label"] in ("image", "figure") and det["label"] == "Picture":
                    if _looks_like_table(det, orig_w, orig_h):
                        det["label"] = "Table"
            # ─────────────────────────────────────────────────────────────

            boxes: List[LayoutBox] = []
            for pos, det in enumerate(dets_sorted):
                x0, y0, x1, y1 = det["bbox"]
                count = _estimate_count(x0, y0, x1, y1, orig_w, orig_h)
                if count is None:
                    count = 50
                    # Tables need maximum token budget — always override
                if det["label"] == "Table":
                    count = 900
                boxes.append(LayoutBox(
                    polygon=[x0, y0, x1, y1],
                    label=det["label"],
                    raw_label=det["raw_label"],
                    position=pos,
                    confidence=det["score"],
                    count=count,
                ))

            results.append(LayoutResult(
                bboxes=boxes,
                image_bbox=[0.0, 0.0, float(orig_w), float(orig_h)],
            ))
        return results