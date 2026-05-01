"""Abstract retrieval interfaces (Dependency Inversion).

Nodes depend on these protocols, not on ChromaDB or any specific chunker.
Swap the implementation (Pinecone, Weaviate, BM25) without touching nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

from code.schemas.ticket import Chunk


class IChunker(Protocol):
    """Splits a markdown file into retrievable chunks with metadata."""

    def chunk(self, path: Path) -> Iterable[Chunk]:
        ...


class IRetriever(Protocol):
    """Returns the most relevant chunks for a query, optionally scoped by company."""

    def retrieve(
        self,
        query: str,
        company: str | None = None,
        n_results: int = 8,
    ) -> list[Chunk]:
        ...
