"""
catalog.py

loads the SHL catalog from the json url or local cache,
creates FAISS index and gives helper funcs for search + lookup
"""

from __future__ import annotations
import re
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)

CACHE_PATH = Path(__file__).parent / "data" / "catalog.json"

EMBED_MODEL = "all-MiniLM-L6-v2"

TOP_K_HARD_CAP = 10


def _download_catalog() -> list[dict]:
    """download catalog and save localy"""

    log.info("Downloading catalog from %s", CATALOG_URL)

    req = urllib.request.Request(
        CATALOG_URL,
        headers={"User-Agent": "Mozilla/5.0 SHL-Agent/1.0"},
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")

    # removing weird chars
    raw = re.sub(r"[\x00-\x1F\x7F]", "", raw)

    data = json.loads(raw)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    CACHE_PATH.write_text(json.dumps(data, indent=2))

    log.info("Cached %d items → %s", len(data), CACHE_PATH)

    return data


def _load_catalog() -> list[dict]:

    if CACHE_PATH.exists():
        log.info("Loading catalog from cache %s", CACHE_PATH)
        return json.loads(CACHE_PATH.read_text())

    return _download_catalog()


def _make_embedding_text(item: dict) -> str:
    """makes text for embeddings"""

    parts: list[str] = [item.get("name", "")]

    desc = item.get("description", "")

    if desc:
        parts.append(desc)

    keys = item.get("keys", [])

    if keys:
        parts.append("Test types: " + ", ".join(keys))

    levels = item.get("job_levels", [])

    if levels:
        parts.append("Job levels: " + ", ".join(levels))

    duration = item.get("duration", "")

    if duration:
        parts.append(f"Duration: {duration}")

    langs = item.get("languages", [])

    if langs:
        parts.append("Languages: " + ", ".join(langs))

    return " | ".join(parts)


class SHLCatalog:
    """
    in memory catalog + FAISS index

    Example:
        catalog = SHLCatalog()
        results = catalog.search("Java developer stakeholder management", top_k=5)
    """

    def __init__(self) -> None:

        raw = _load_catalog()

        # storing items
        self.items: list[dict[str, Any]] = []

        for item in raw:

            self.items.append(
                {
                    "entity_id": item.get("entity_id", ""),
                    "name": item.get("name", ""),
                    "url": item.get("link", ""),
                    "description": item.get("description", ""),
                    "keys": item.get("keys", []),
                    "job_levels": item.get("job_levels", []),
                    "languages": item.get("languages", []),
                    "duration": item.get("duration", ""),
                    "remote": item.get("remote", ""),
                    "adaptive": item.get("adaptive", ""),

                    # saved text for embeddings
                    "_embed_text": _make_embedding_text(item),
                }
            )

        log.info("Loaded %d assessments", len(self.items))

        # loading embedding model
        log.info("Loading embedding model %s …", EMBED_MODEL)

        self._model = SentenceTransformer(EMBED_MODEL)

        texts = [it["_embed_text"] for it in self.items]

        log.info("Embedding %d texts …", len(texts))

        vecs = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False
        )

        vecs = vecs.astype(np.float32)

        # normalize vecs
        faiss.normalize_L2(vecs)

        self._index = faiss.IndexFlatIP(vecs.shape[1])

        self._index.add(vecs)

        log.info(
            "FAISS index built (%d vectors, dim=%d)",
            *vecs.shape
        )

        # mapping names to indexes
        self._name_to_idx: dict[str, int] = {
            it["name"].lower(): i
            for i, it in enumerate(self.items)
        }

    # public funcs

    def search(
        self,
        query: str,
        top_k: int = TOP_K_HARD_CAP
    ) -> list[dict]:

        """
        semantic search
        returns top matching items
        """

        top_k = min(
            top_k,
            TOP_K_HARD_CAP,
            len(self.items)
        )

        vec = self._model.encode(
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
        """exact name search"""

        idx = self._name_to_idx.get(name.lower())

        return dict(self.items[idx]) if idx is not None else None

    def get_by_url(self, url: str) -> dict | None:
        """url lookup"""

        for item in self.items:

            if item["url"].rstrip("/") == url.rstrip("/"):
                return dict(item)

        return None

    def all_names(self) -> list[str]:

        return [it["name"] for it in self.items]

    def format_for_context(
        self,
        items: list[dict],
        max_items: int = 10
    ) -> str:

        """
        formats items for llm context
        """

        lines: list[str] = []

        for i, it in enumerate(items[:max_items], 1):

            lines.append(
                f"{i}. [{it['name']}]({it['url']})\n"
                f"   Types: {', '.join(it['keys']) or 'N/A'} | "
                f"Levels: {', '.join(it['job_levels']) or 'N/A'} | "
                f"Duration: {it['duration'] or 'N/A'} | "
                f"Remote: {it['remote']} | Adaptive: {it['adaptive']}\n"
                f"   {it['description'][:180]}"
                f"{'…' if len(it['description']) > 180 else ''}"
            )

        return "\n\n".join(lines)