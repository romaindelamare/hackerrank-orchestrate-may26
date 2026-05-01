"""Semantic similarity cache backed by a dedicated ChromaDB collection.

Uses the same embedding model as the retrieval index so no extra downloads
are needed. Results are stored as JSON in ChromaDB metadata and looked up
by cosine similarity against the incoming ticket text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from code.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    SEMANTIC_CACHE_COLLECTION,
    SEMANTIC_CACHE_THRESHOLD,
)


class SemanticCache:
    """Cache LLM results keyed by ticket semantic content."""

    def __init__(
        self,
        db_path: Path = CHROMA_DIR,
        collection_name: str = SEMANTIC_CACHE_COLLECTION,
        model_name: str = EMBEDDING_MODEL,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
    ) -> None:
        self._threshold = threshold
        self._collection_name = collection_name
        self._db_path = db_path
        self._model_name = model_name
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        # cosine space so distances map cleanly to 1-dist = similarity
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def get(self, ticket_text: str) -> dict | None:
        """Return cached result dict if a similar ticket exists, else None."""
        if self._collection.count() == 0:
            return None
        result = self._collection.query(query_texts=[ticket_text], n_results=1)
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        if not distances or not metadatas:
            return None
        similarity = 1.0 - distances[0]
        if similarity >= self._threshold:
            payload = metadatas[0].get("payload")
            return json.loads(payload) if payload else None
        return None

    def set(self, ticket_text: str, result_dict: dict) -> None:
        """Store a result dict for the given ticket text."""
        cache_id = hashlib.sha256(ticket_text.encode("utf-8")).hexdigest()[:24]
        self._collection.upsert(
            ids=[cache_id],
            documents=[ticket_text],
            metadatas=[{"payload": json.dumps(result_dict)}],
        )

    def reset(self) -> None:
        """Clear all cached results by deleting and recreating the collection."""
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
