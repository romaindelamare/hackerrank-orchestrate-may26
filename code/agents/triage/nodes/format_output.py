"""format_output_node — final validation & normalization of the answer.

This is a defensive layer: reply_node and escalate_node already produce
TicketOutput-shaped dicts, but if either failed silently we want to catch
that here before writing the CSV.
"""

from __future__ import annotations

from pydantic import ValidationError

from code.agents.triage.nodes.base import Node
from code.schemas.state import TicketState
from code.schemas.ticket import TicketOutput


class FormatOutputNode(Node):
    """Coerces and validates the final TicketOutput. Never raises."""

    def __call__(self, state: TicketState) -> dict:
        candidate = {
            "status": state.get("status") or "escalated",
            "product_area": (state.get("product_area") or "general").lower().strip(),
            "response": state.get("response") or "",
            "justification": state.get("justification") or "",
            "request_type": state.get("request_type") or "invalid",
        }
        try:
            validated = TicketOutput.model_validate(candidate)
            return validated.model_dump()
        except ValidationError as err:
            return {
                "status": "escalated",
                "product_area": "general",
                "response": (
                    "Your request needs additional review by a human specialist. "
                    "We'll follow up shortly."
                ),
                "justification": "Final validation required. Escalated for human review.",
                "request_type": "invalid",
            }
