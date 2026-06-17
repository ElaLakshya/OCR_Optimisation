"""
image_cache.py

Two-layer visual image cache:
    Layer 1 — in-memory dict (microseconds, per-session)
    Layer 2 — SQLite on disk (milliseconds, persistent across runs)

Uses a 16x16 Difference Hash (dHash) + Hamming-distance comparison to catch
visually identical signatures, logos, and stamps even with microscopic
pixel-level variations between pages (anti-aliasing, sub-pixel crop offsets).

IMPORTANT: dHash alone is not enough — two crops of "the same" image at
slightly different bbox positions can produce hashes that differ by a few
bits. We therefore compare via Hamming distance against a threshold, not
exact string equality.
"""
from __future__ import annotations

import io
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH  = os.path.join(_THIS_DIR, "image_hash_cache.db")

# dHash size: 16x16 -> 256 bits -> 64 hex chars
_HASH_SIZE = 16
_HASH_BITS = _HASH_SIZE * _HASH_SIZE  # 256

# Two dHashes within this Hamming distance are considered "the same image".
# 256 bits total; ~10 bits of difference tolerates compression / rendering
# noise while still rejecting genuinely different images (which typically
# differ by 80+ bits).
_HAMMING_THRESHOLD = 10


def _init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_cache (
            hash        TEXT PRIMARY KEY,
            result      TEXT NOT NULL,
            label       TEXT NOT NULL DEFAULT 'text',
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON image_cache(hash)")
    conn.commit()


def _hamming_distance(hex1: str, hex2: str) -> int:
    """Hamming distance between two equal-length hex-encoded bit strings."""
    if len(hex1) != len(hex2):
        return _HASH_BITS  # treat mismatched lengths as maximally different
    b1 = bin(int(hex1, 16))[2:].zfill(_HASH_BITS)
    b2 = bin(int(hex2, 16))[2:].zfill(_HASH_BITS)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))


class ImageCache:
    SKIP = "__SKIP__"

    def __init__(self, db_path: str = _DB_PATH, hamming_threshold: int = _HAMMING_THRESHOLD):
        self._store: dict[str, Any] = {}
        self._lock  = threading.Lock()
        self._db_path = db_path
        self._threshold = hamming_threshold
        self._conn: Optional[sqlite3.Connection] = None
        self._init_sqlite()

    def _init_sqlite(self):
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            _init_db(self._conn)
            print(f"[ImageCache] SQLite cache at: {self._db_path}")
        except Exception as e:
            print(f"[ImageCache] SQLite init failed, using memory only: {e}")
            self._conn = None

    def _hash(self, image: Image.Image) -> str:
        """
        Computes a 16x16 perceptual Difference Hash (dHash).
        Returns a 64-char hex string.
        """
        img_gray = image.convert("L").resize(
            (_HASH_SIZE + 1, _HASH_SIZE), Image.Resampling.LANCZOS
        )
        pixels = list(img_gray.getdata())

        diff = []
        for row in range(_HASH_SIZE):
            for col in range(_HASH_SIZE):
                left  = pixels[row * (_HASH_SIZE + 1) + col]
                right = pixels[row * (_HASH_SIZE + 1) + col + 1]
                diff.append('1' if left > right else '0')

        decimal_value = int(''.join(diff), 2)
        return format(decimal_value, f'0{_HASH_BITS // 4}x')

    def get(self, image: Image.Image) -> Optional[Any]:
        """
        Returns cached result for a visually similar image (within Hamming
        threshold), or None if nothing close enough has been seen.
        """
        h = self._hash(image)

        # Layer 1 — memory (scan for nearest match)
        with self._lock:
            for stored_hash, result in self._store.items():
                if _hamming_distance(h, stored_hash) <= self._threshold:
                    return result

        # Layer 2 — SQLite (scan for nearest match, promote on hit)
        if self._conn:
            try:
                with self._lock:
                    rows = self._conn.execute(
                        "SELECT hash, result FROM image_cache"
                    ).fetchall()
                for stored_hash, result in rows:
                    if _hamming_distance(h, stored_hash) <= self._threshold:
                        with self._lock:
                            self._store[h] = result
                        return result
            except Exception as e:
                print(f"[ImageCache] SQLite read error: {e}")

        return None

    def set(self, image: Image.Image, result: Any, label: str = "text") -> None:
        """Store result in both memory and SQLite, keyed by this image's dHash."""
        h = self._hash(image)

        with self._lock:
            self._store[h] = result

        if self._conn:
            try:
                with self._lock:
                    self._conn.execute(
                        """INSERT OR REPLACE INTO image_cache
                           (hash, result, label, created_at)
                           VALUES (?, ?, ?, ?)""",
                        (h, str(result), label, time.time())
                    )
                    self._conn.commit()
            except Exception as e:
                print(f"[ImageCache] SQLite write error: {e}")

    def mark_skip(self, image: Image.Image) -> None:
        self.set(image, self.SKIP, label="skip")

    def is_skip(self, result: Any) -> bool:
        return result == self.SKIP

    def size(self) -> int:
        return len(self._store)

    def clear_all(self) -> None:
        with self._lock:
            self._store.clear()
        if self._conn:
            try:
                with self._lock:
                    self._conn.execute("DELETE FROM image_cache")
                    self._conn.commit()
            except Exception:
                pass

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None