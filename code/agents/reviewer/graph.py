"""Reviewer agent graph.

Single-node graph: START -> review -> END
Takes the full TicketState after triage and runs a quality-gate check.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from code.agents.reviewer.nodes.review import ReviewNode
from code.llm.interfaces import ILLMClient
from code.schemas.state import TicketState


def build_reviewer_graph(llm: ILLMClient):
    """Wire the reviewer node and return a compiled graph."""
    review = ReviewNode(llm)

    g = StateGraph(TicketState)
    g.add_node("review", review)
    g.add_edge(START, "review")
    g.add_edge("review", END)

    return g.compile()
