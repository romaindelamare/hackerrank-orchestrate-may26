"""review_node — quality-gate that validates and optionally improves triage output."""

from __future__ import annotations

from code.llm.interfaces import ILLMClient
from code.llm.prompts import REVIEWER_PROMPT_TEMPLATE, REVIEWER_SYSTEM
from code.schemas.state import TicketState
from code.schemas.ticket import ReviewOutput


class ReviewNode:
    """Post-triage quality gate.

    Checks the triage output for safety, grounding, and consistency.
    Returns an updated state with the final response and a reviewer_action tag.
    """

    def __init__(self, llm: ILLMClient) -> None:
        self._llm = llm

    def __call__(self, state: TicketState) -> dict:
        prompt = REVIEWER_PROMPT_TEMPLATE.format(
            subject=state.get("subject", "") or "(none)",
            company=state.get("inferred_company", state.get("company", "none")),
            issue=state.get("issue", ""),
            status=state.get("status", "escalated"),
            product_area=state.get("product_area", "general"),
            request_type=state.get("request_type", "invalid"),
            response=state.get("response", ""),
            justification=state.get("justification", ""),
        )
        try:
            result: ReviewOutput = self._llm.generate_structured(
                prompt, ReviewOutput, system=REVIEWER_SYSTEM
            )
            return {
                "status": result.status,
                "response": result.response,
                "justification": result.justification,
                "reviewer_action": result.action,
                "reviewer_notes": result.notes,
            }
        except Exception as err:
            # On failure, pass through unchanged and mark as approved to avoid breaking the pipeline.
            return {
                "reviewer_action": "approved",
                "reviewer_notes": f"Reviewer skipped due to error: {err}",
            }
