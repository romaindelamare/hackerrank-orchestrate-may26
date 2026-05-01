"""triage_router — pure conditional edge (no LLM).

Returns the name of the next node. LangGraph wires this to either the
escalate or reply branch based on the risk flags populated by classify_node.
"""

from __future__ import annotations

from code.schemas.state import TicketState

ESCALATE = "escalate"
REPLY = "reply"


def triage_router(state: TicketState) -> str:
    """Escalate when any risk flag is present OR when retrieval is empty."""
    if state.get("risk_flags"):
        return ESCALATE
    if not state.get("retrieved_chunks"):
        return ESCALATE
    return REPLY
