"""reply_node — produce a grounded answer from retrieved corpus chunks."""

from __future__ import annotations

from code.llm.interfaces import ILLMClient
from code.llm.prompts import REPLY_PROMPT_TEMPLATE, REPLY_SYSTEM, build_context_block
from code.agents.triage.nodes.base import Node
from code.schemas.state import TicketState
from code.schemas.ticket import Chunk, TicketOutput


class ReplyNode(Node):
    """LLM call that yields a TicketOutput grounded in retrieved passages."""

    def __init__(self, llm: ILLMClient) -> None:
        self._llm = llm

    def __call__(self, state: TicketState) -> dict:
        chunks = [Chunk(**c) for c in state.get("retrieved_chunks", [])]
        prompt = REPLY_PROMPT_TEMPLATE.format(
            subject=state.get("subject", "") or "(none)",
            company=state.get("inferred_company", "none"),
            issue=state.get("issue", ""),
            context=build_context_block(chunks),
        )
        try:
            output = self._llm.generate_structured(
                prompt, TicketOutput, system=REPLY_SYSTEM
            )
            return self._dump(output)
        except Exception as err:
            return self._fallback(str(err))

    @staticmethod
    def _dump(output: TicketOutput) -> dict:
        return {
            "status": output.status,
            "product_area": output.product_area,
            "response": output.response,
            "justification": output.justification,
            "request_type": output.request_type,
        }

    @staticmethod
    def _fallback(reason: str) -> dict:
        return {
            "status": "escalated",
            "product_area": "general",
            "response": (
                "We weren't able to generate a confident answer for this ticket. "
                "A human support specialist will follow up with you shortly."
            ),
            "justification": "Unable to generate a confident reply from available documentation. Escalated for human review.",
            "request_type": "invalid",
        }
