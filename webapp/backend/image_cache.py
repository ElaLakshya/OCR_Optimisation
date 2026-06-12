"""
image_cache.py

Two-layer SHA-256 image cache:
    Layer 1 — in-memory dict (microseconds, per-session)
    Layer 2 — SQLite on disk (milliseconds, persistent across runs)

On cache hit: returns instantly from memory or disk
On cache miss: caller runs OCR, stores result in both layers

Usage:
    cache = ImageCache()
    result = cache.get(pil_image)
    if result is None:
        result = run_ocr(pil_image)
        cache.set(pil_image, result)
"""

from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from PIL import Image

# SQLite DB stored next to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH  = os.path.join(_THIS_DIR, "image_hash_cache.db")


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


class ImageCache:
    """
    Two-layer cache: in-memory dict + SQLite persistence.
    Thread-safe via a per-instance lock.
    """

    SKIP = "__SKIP__"

    def __init__(self, db_path: str = _DB_PATH):
        self._store: dict[str, Any] = {}
        self._lock  = threading.Lock()
        self._db_path = db_path
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
        """SHA-256 of raw PNG pixel bytes — stable across runs."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def get(self, image: Image.Image) -> Optional[Any]:
        """
        Check cache for this image.
        Returns cached result or None if not seen before.
        Check is_skip(result) to detect skip-flagged images.
        """
        h = self._hash(image)

        # Layer 1 — memory
        with self._lock:
            if h in self._store:
                return self._store[h]

        # Layer 2 — SQLite
        if self._conn:
            try:
                with self._lock:
                    row = self._conn.execute(
                        "SELECT result FROM image_cache WHERE hash = ?", (h,)
                    ).fetchone()
                if row:
                    result = row[0]
                    # Promote to memory layer
                    with self._lock:
                        self._store[h] = result
                    return result
            except Exception as e:
                print(f"[ImageCache] SQLite read error: {e}")

        return None

    def set(self, image: Image.Image, result: Any, label: str = "text") -> None:
        """Store result in both memory and SQLite."""
        import time
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
        """Mark image as logo/seal/decorative — skip OCR permanently."""
        self.set(image, self.SKIP, label="skip")

    def is_skip(self, result: Any) -> bool:
        return result == self.SKIP

    def size(self) -> int:
        """Total entries in memory layer."""
        return len(self._store)

    def db_size(self) -> int:
        """Total entries in SQLite layer."""
        if self._conn:
            try:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM image_cache"
                ).fetchone()
                return row[0] if row else 0
            except Exception:
                pass
        return 0

    def clear_memory(self) -> None:
        """Clear in-memory layer only — SQLite persists."""
        with self._lock:
            self._store.clear()

    def clear_all(self) -> None:
        """Clear both memory and SQLite — full reset."""
        with self._lock:
            self._store.clear()
        if self._conn:
            try:
                with self._lock:
                    self._conn.execute("DELETE FROM image_cache")
                    self._conn.commit()
            except Exception as e:
                print(f"[ImageCache] SQLite clear error: {e}")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None