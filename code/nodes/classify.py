"""classify_node — resolve company and detect risk flags."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from code.config import ESCALATE_TRIGGERS, VALID_COMPANIES
from code.llm.interfaces import ILLMClient
from code.llm.prompts import CLASSIFY_PROMPT_TEMPLATE, CLASSIFY_SYSTEM
from code.nodes.base import Node
from code.schemas.state import TicketState


class _ClassifyResult(BaseModel):
    inferred_company: Literal["hackerrank", "claude", "visa", "none"]
    risk_flags: list[str] = Field(default_factory=list)


class ClassifyNode(Node):
    """Single-call LLM classification: company + risk flags."""

    def __init__(self, llm: ILLMClient) -> None:
        self._llm = llm

    def __call__(self, state: TicketState) -> dict:
        normalized = self._normalize_company(state.get("company", "None"))
        if normalized in VALID_COMPANIES:
            # Even when company is known, we still ask the LLM for risk flags.
            result = self._classify(state, hint=normalized)
            inferred = normalized
        else:
            result = self._classify(state, hint="none")
            inferred = result.inferred_company

        risk_flags = [f for f in result.risk_flags if f in ESCALATE_TRIGGERS]
        return {"inferred_company": inferred, "risk_flags": risk_flags}

    def _classify(self, state: TicketState, *, hint: str) -> _ClassifyResult:
        prompt = CLASSIFY_PROMPT_TEMPLATE.format(
            subject=state.get("subject", "") or "(none)",
            company=hint,
            issue=state.get("issue", ""),
        )
        try:
            return self._llm.generate_structured(
                prompt, _ClassifyResult, system=CLASSIFY_SYSTEM
            )
        except Exception:
            # Soft fallback: never crash the graph on classification.
            return _ClassifyResult(inferred_company="none", risk_flags=[])

    @staticmethod
    def _normalize_company(raw: str) -> str:
        if not raw:
            return "none"
        c = raw.strip().lower()
        return c if c in VALID_COMPANIES else "none"
