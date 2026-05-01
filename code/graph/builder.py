"""Compile the support-triage StateGraph.

  START -> classify -> retrieve -> [router]
                                       |
                            +----------+----------+
                            |                     |
                       escalate              reply
                            |                     |
                            +----------+----------+
                                       |
                                  format -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from code.llm.interfaces import ILLMClient
from code.nodes import (
    ClassifyNode,
    EscalateNode,
    FormatOutputNode,
    ReplyNode,
    RetrieveNode,
    triage_router,
)
from code.nodes.router import ESCALATE, REPLY
from code.retrieval.interfaces import IRetriever
from code.schemas.state import TicketState


def build_graph(llm: ILLMClient, retriever: IRetriever):
    """Wire nodes + edges and return a compiled graph.

    Dependencies are injected here, the single composition root.
    Nodes themselves never instantiate their collaborators.
    """
    classify = ClassifyNode(llm)
    retrieve = RetrieveNode(retriever)
    reply = ReplyNode(llm)
    escalate = EscalateNode(llm)
    fmt = FormatOutputNode()

    g = StateGraph(TicketState)

    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("reply", reply)
    g.add_node("escalate", escalate)
    g.add_node("format", fmt)

    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_conditional_edges(
        "retrieve",
        triage_router,
        {REPLY: "reply", ESCALATE: "escalate"},
    )
    g.add_edge("reply", "format")
    g.add_edge("escalate", "format")
    g.add_edge("format", END)

    return g.compile()
