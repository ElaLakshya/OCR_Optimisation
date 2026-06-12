"""
table_rec.py

Swappable table recognition module using RapidTable (SLANet ONNX).

To swap to a different table recognition model later:
    - Keep the TableRecognizer class interface identical
    - Replace the internals of __init__ and recognize()
    - Everything else in the pipeline will work unchanged

Usage:
    rec = TableRecognizer()
    html, elapsed = rec.recognize(pil_image)
"""

from __future__ import annotations

import time
from typing import Optional, Tuple
from PIL import Image
import numpy as np


class TableRecognizer:
    """
    Wraps RapidTable for drop-in table structure recognition.
    Input  : PIL.Image crop of a table region
    Output : HTML string of the table + time taken
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from rapid_table import RapidTable
        # RapidTable downloads its ONNX model automatically on first run
        self._model = RapidTable()

    def recognize(self, image: Image.Image) -> Tuple[str, float]:
        """
        Run table structure recognition on a cropped table image.

        Args:
            image: PIL Image of the table region (RGB)

        Returns:
            (html_string, elapsed_seconds)
            html_string is empty string on failure
        """
        self._load()

        try:
            t0 = time.time()
            # RapidTable accepts numpy array (BGR) or PIL image
            img_array = np.array(image.convert("RGB"))
            output = self._model(img_array)
            elapsed = time.time() - t0
            if output is None or not output.pred_htmls:
                return "", elapsed
            return output.pred_htmls[0], elapsed

        except Exception as e:
            print(f"[TableRecognizer] Error: {e}")
            return "", 0.0

    def recognize_batch(self, images: list) -> list:
        """
        Recognize multiple table images.
        Returns list of (html_string, elapsed_seconds) tuples.
        """
        return [self.recognize(img) for img in images]