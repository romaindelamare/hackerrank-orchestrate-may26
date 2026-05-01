"""Builds the ChromaDB vector index from the markdown corpus.

Idempotent: the same chunk ID (path#section_index) overwrites on re-run.
Use `--reindex` from main.py to wipe and rebuild from scratch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from code.config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, EMBEDDING_MODEL
from code.retrieval.chunker import MarkdownChunker
from code.retrieval.interfaces import IChunker

_BATCH_SIZE = 128


def _make_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )


def _chunk_id(source_file: str, section_idx: int) -> str:
    digest = hashlib.sha1(f"{source_file}#{section_idx}".encode("utf-8")).hexdigest()[:16]
    return f"{digest}"


def get_or_create_collection():
    """Create the persistent client+collection (no rebuild)."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_make_embedding_fn(),
    )


def build_index(chunker: IChunker | None = None, *, reset: bool = False) -> int:
    """Walk DATA_DIR, chunk every .md, upsert to ChromaDB. Returns chunk count."""
    chunker = chunker or MarkdownChunker()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_make_embedding_fn(),
    )

    md_files = sorted(DATA_DIR.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under {DATA_DIR}")

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    total = 0

    for path in md_files:
        for idx, chunk in enumerate(chunker.chunk(path)):
            ids.append(_chunk_id(chunk.source_file, idx))
            docs.append(chunk.text)
            metas.append({
                "company": chunk.company,
                "product_area": chunk.product_area,
                "source_file": chunk.source_file,
                "source_url": chunk.source_url,
            })
            if len(ids) >= _BATCH_SIZE:
                collection.upsert(ids=ids, documents=docs, metadatas=metas)
                total += len(ids)
                ids, docs, metas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        total += len(ids)

    return total


def collection_size() -> int:
    """Number of chunks currently in the collection (0 if missing)."""
    try:
        return get_or_create_collection().count()
    except Exception:
        return 0


def _walk_md(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))
