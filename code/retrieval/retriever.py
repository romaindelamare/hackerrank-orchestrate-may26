"""ChromaDB-backed retriever (concrete IRetriever implementation)."""

from __future__ import annotations

from code.config import VALID_COMPANIES
from code.retrieval.indexer import get_or_create_collection
from code.schemas.ticket import Chunk


class ChromaRetriever:
    """Top-K semantic retriever filtered by company metadata."""

    def __init__(self) -> None:
        self._collection = get_or_create_collection()

    def retrieve(
        self,
        query: str,
        company: str | None = None,
        n_results: int = 8,
    ) -> list[Chunk]:
        where = self._build_where(company)
        result = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return self._unpack(result)

    @staticmethod
    def _build_where(company: str | None) -> dict | None:
        if not company:
            return None
        c = company.strip().lower()
        if c not in VALID_COMPANIES:
            return None
        return {"company": c}

    @staticmethod
    def _unpack(result: dict) -> list[Chunk]:
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        chunks: list[Chunk] = []
        for text, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            chunks.append(
                Chunk(
                    text=text,
                    company=meta.get("company", "unknown"),
                    product_area=meta.get("product_area", "general"),
                    source_file=meta.get("source_file", ""),
                    source_url=meta.get("source_url", ""),
                    score=float(1.0 - dist) if dist is not None else 0.0,
                )
            )
        return chunks
