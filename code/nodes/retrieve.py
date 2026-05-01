"""retrieve_node — semantic search over the support corpus."""

from __future__ import annotations

from code.config import RETRIEVAL_TOP_K, VALID_COMPANIES
from code.nodes.base import Node
from code.retrieval.interfaces import IRetriever
from code.schemas.state import TicketState


class RetrieveNode(Node):
    """Pulls top-K corpus chunks for the ticket, scoped by inferred company."""

    def __init__(self, retriever: IRetriever) -> None:
        self._retriever = retriever

    def __call__(self, state: TicketState) -> dict:
        query = self._build_query(state)
        company = state.get("inferred_company", "none")
        scope = company if company in VALID_COMPANIES else None
        chunks = self._retriever.retrieve(
            query=query,
            company=scope,
            n_results=RETRIEVAL_TOP_K,
        )
        return {"retrieved_chunks": [c.model_dump() for c in chunks]}

    @staticmethod
    def _build_query(state: TicketState) -> str:
        subject = (state.get("subject") or "").strip()
        issue = (state.get("issue") or "").strip()
        return f"{subject}\n{issue}".strip() or issue or subject
