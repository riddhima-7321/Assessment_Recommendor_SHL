"""
indexing.py

Run once locally 

Builds:
- faiss index
- cleaned metadata
- embedding text cache
"""

import json
import logging
import re
import urllib.request
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

log = logging.getLogger(__name__)

CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)

EMBED_MODEL = "all-MiniLM-L6-v2"

DATA_DIR = Path(__file__).parent / "data"

# output files
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH = DATA_DIR / "index.faiss"
ITEMS_PATH = DATA_DIR / "items.json"
TEXTS_PATH = DATA_DIR / "embed_texts.npy"


def _download_catalog() -> list[dict]:

    log.info("Downloading catalog...")

    req = urllib.request.Request(
        CATALOG_URL,
        headers={
            "User-Agent": "Mozilla/5.0 SHL-Agent/1.0"
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode(
            "utf-8",
            errors="ignore"
        )

    
    raw = re.sub(r"[\x00-\x1F\x7F]", "", raw)

    data = json.loads(raw)

    log.info("Downloaded %d items", len(data))

    return data


def _load_or_download_catalog() -> list[dict]:

    if CATALOG_PATH.exists():

        log.info("Using cached catalog")

        return json.loads(
            CATALOG_PATH.read_text()
        )

    return _download_catalog()


def _make_embedding_text(item: dict) -> str:

    parts = [item.get("name", "")]

    desc = item.get("description", "")

    if desc:
        parts.append(desc)

    keys = item.get("keys", [])

    if keys:
        parts.append(
            "Test types: " + ", ".join(keys)
        )

    levels = item.get("job_levels", [])

    if levels:
        parts.append(
            "Job levels: " + ", ".join(levels)
        )

    duration = item.get("duration", "")

    if duration:
        parts.append(f"Duration: {duration}")

    langs = item.get("languages", [])

    if langs:
        parts.append(
            "Languages: " + ", ".join(langs)
        )

    return " | ".join(parts)


def _normalise_item(item: dict) -> dict:

    return {
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
    }


def build():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # load catalog first
    raw = _load_or_download_catalog()

    # save raw catalog locally
    if not CATALOG_PATH.exists():

        CATALOG_PATH.write_text(
            json.dumps(raw, indent=2)
        )

        log.info("Saved catalog")

    # clean item structure
    items = [_normalise_item(r) for r in raw]

    log.info(
        "Normalised %d items",
        len(items)
    )

    # create embedding text
    texts = [
        _make_embedding_text(r)
        for r in raw
    ]

    # load embedding model
    log.info("Loading model...")

    model = SentenceTransformer(
        EMBED_MODEL
    )

    log.info(
        "Embedding %d texts",
        len(texts)
    )

    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # normalize for cosine simlarity
    faiss.normalize_L2(vecs)

    # build index
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(vecs)

    log.info(
        "Index ready with %d vectors",
        index.ntotal
    )

    # save files
    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    ITEMS_PATH.write_text(
        json.dumps(items, indent=2)
    )

    np.save(
        str(TEXTS_PATH),
        np.array(texts)
    )

    log.info("Saved all output files")

    print("\nDone building index")
    print(f"Items: {len(items)}")
    print(f"Vector dim: {dim}")


if __name__ == "__main__":
    build()