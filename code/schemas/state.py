"""LangGraph shared state.

A single TypedDict carries all data through the graph. Each node reads
the slots it needs and returns a partial dict that LangGraph merges back.
"""

from __future__ import annotations

from typing import TypedDict


class TicketState(TypedDict, total=False):
    # ---- Inputs (set by main.py before invoking the graph) ---------------
    issue: str
    subject: str
    company: str  # raw value from CSV: "HackerRank" | "Claude" | "Visa" | "None"

    # ---- Set by classify_node -------------------------------------------
    inferred_company: str  # normalized lower-case: "hackerrank"|"claude"|"visa"|"none"
    risk_flags: list[str]  # subset of config.ESCALATE_TRIGGERS

    # ---- Set by retrieve_node -------------------------------------------
    retrieved_chunks: list[dict]  # serialized Chunk dicts

    # ---- Set by reply_node / escalate_node ------------------------------
    raw_llm_output: str  # JSON string returned by Gemini

    # ---- Set by format_node (final answer) ------------------------------
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str

    # ---- Set by reviewer_node (post-triage quality gate) ----------------
    reviewer_action: str   # "approved" | "refined" | "escalated"
    reviewer_notes: str    # brief explanation of the review decision
