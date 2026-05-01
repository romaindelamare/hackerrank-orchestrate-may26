"""escalate_node — compose an escalation message for high-risk tickets."""

from __future__ import annotations

from code.llm.interfaces import ILLMClient
from code.llm.prompts import (
    ESCALATE_PROMPT_TEMPLATE,
    ESCALATE_SYSTEM,
    build_context_block,
)
from code.nodes.base import Node
from code.schemas.state import TicketState
from code.schemas.ticket import Chunk, TicketOutput


class EscalateNode(Node):
    """LLM call that produces an empathetic, non-committal escalation reply."""

    def __init__(self, llm: ILLMClient) -> None:
        self._llm = llm

    def __call__(self, state: TicketState) -> dict:
        chunks = [Chunk(**c) for c in state.get("retrieved_chunks", [])]
        prompt = ESCALATE_PROMPT_TEMPLATE.format(
            subject=state.get("subject", "") or "(none)",
            company=state.get("inferred_company", "none"),
            risk_flags=", ".join(state.get("risk_flags", [])) or "(none — empty retrieval)",
            issue=state.get("issue", ""),
            context=build_context_block(chunks),
        )
        try:
            output = self._llm.generate_structured(
                prompt, TicketOutput, system=ESCALATE_SYSTEM
            )
            # Force status='escalated' regardless of what the LLM returned.
            return {
                "status": "escalated",
                "product_area": output.product_area,
                "response": output.response,
                "justification": output.justification,
                "request_type": output.request_type,
            }
        except Exception as err:
            return self._fallback(state, str(err))

    @staticmethod
    def _fallback(state: TicketState, reason: str) -> dict:
        flags = state.get("risk_flags", [])
        flags_str = ", ".join(flags) if flags else "uncovered_topic"
        return {
            "status": "escalated",
            "product_area": "general",
            "response": (
                "Thank you for reaching out. Your request needs a human support "
                "specialist to review it carefully. Someone from the team will "
                "follow up with you directly."
            ),
            "justification": f"Escalated due to: {flags_str}. Requires human review for appropriate handling.",
            "request_type": "product_issue",
        }
