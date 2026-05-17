"""
catalog.py

Loads the FAISS index + metadata from disk.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
ITEMS_PATH = DATA_DIR / "items.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K_HARD_CAP = 10

# lazy load model so startup is faster
_query_model: SentenceTransformer | None = None


def _get_query_model() -> SentenceTransformer:
    global _query_model

    if _query_model is None:
        log.info("Loading embedding model...")
        _query_model = SentenceTransformer(EMBED_MODEL)

    return _query_model


class SHLCatalog:

    def __init__(self) -> None:

        # load item metadata
        if not ITEMS_PATH.exists():
            raise FileNotFoundError(
                f"Missing items file: {ITEMS_PATH}"
            )

        self.items: list[dict[str, Any]] = json.loads(
            ITEMS_PATH.read_text()
        )

        log.info("Loaded %d items", len(self.items))

        # load faiss index
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Missing FAISS index: {INDEX_PATH}"
            )

        self._index = faiss.read_index(str(INDEX_PATH))

        log.info(
            "Index loaded with %d vectors",
            self._index.ntotal
        )

        # quick lookups
        self._name_to_idx = {
            it["name"].lower(): i
            for i, it in enumerate(self.items)
        }

        self._url_to_idx = {
            it["url"].rstrip("/"): i
            for i, it in enumerate(self.items)
        }

    def search(self, query: str, top_k: int = TOP_K_HARD_CAP) -> list[dict]:

        # dont let top_k go too high
        top_k = min(top_k, TOP_K_HARD_CAP, len(self.items))

        model = _get_query_model()

        vec = model.encode(
            [query],
            show_progress_bar=False
        ).astype(np.float32)

        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            item = dict(self.items[idx])
            item["_score"] = float(score)

            results.append(item)

        return results

    def get_by_name(self, name: str) -> dict | None:

        idx = self._name_to_idx.get(name.lower())

        if idx is None:
            return None

        return dict(self.items[idx])

    def get_by_url(self, url: str) -> dict | None:

        clean_url = url.rstrip("/")

        idx = self._url_to_idx.get(clean_url)

        if idx is not None:
            return dict(self.items[idx])

        # fallback search if map misses somthing
        for item in self.items:

            if item["url"].rstrip("/") == clean_url:
                return dict(item)

        return None

    def all_names(self) -> list[str]:
        return [it["name"] for it in self.items]

    def format_for_context(
        self,
        items: list[dict],
        max_items: int = 10
    ) -> str:

        lines = []

        for i, it in enumerate(items[:max_items], 1):

            lines.append(
                f"{i}. [{it['name']}]({it['url']})\n"
                f"   Types: {', '.join(it['keys']) or 'N/A'} | "
                f"Levels: {', '.join(it['job_levels']) or 'N/A'} | "
                f"Duration: {it['duration'] or 'N/A'}\n"
                f"   {it['description'][:180]}"
            )

        return "\n\n".join(lines)