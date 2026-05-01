"""Pydantic models for ticket I/O and retrieval chunks.

`TicketOutput` is the contract the agent must satisfy for each row of
the output CSV. Gemini is instructed to return JSON matching this schema
via `response_schema`, so the validation step is a single `model_validate`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["replied", "escalated"]
RequestType = Literal["product_issue", "feature_request", "bug", "invalid"]


class TicketInput(BaseModel):
    """One row of the input CSV."""

    issue: str
    subject: str = ""
    company: str = "None"


class TicketOutput(BaseModel):
    """One row of the output CSV.

    Fields and allowed values are dictated by the evaluation rubric.
    """

    status: Status = Field(
        ...,
        description="'replied' if the agent answers directly; 'escalated' otherwise.",
    )
    product_area: str = Field(
        ...,
        description="Most relevant support category (e.g. 'screen', 'billing').",
    )
    response: str = Field(
        ...,
        description="User-facing answer grounded strictly in the support corpus.",
    )
    justification: str = Field(
        ...,
        description="Concise reasoning trace pointing at corpus evidence.",
    )
    request_type: RequestType = Field(
        ...,
        description="Best-fit classification of the request.",
    )


class ReviewOutput(BaseModel):
    """Output of the reviewer agent pass over a triage result."""

    action: Literal["approved", "refined", "escalated"] = Field(
        ...,
        description="approved=no changes needed; refined=response improved; escalated=safety issue found.",
    )
    response: str = Field(..., description="Final user-facing response (may be unchanged).")
    justification: str = Field(..., description="Final justification (may be unchanged).")
    status: Status = Field(..., description="Final status after review.")
    notes: str = Field(
        ...,
        description="Brief explanation of what was changed or why it was approved as-is.",
    )


class Chunk(BaseModel):
    """A retrievable section of a corpus document."""

    text: str
    company: str
    product_area: str
    source_file: str
    source_url: str = ""
    score: float = 0.0

    def to_context_block(self) -> str:
        """Render a chunk as a labeled passage for LLM context."""
        header = f"[{self.company}/{self.product_area} | {self.source_file}]"
        return f"{header}\n{self.text}".strip()
