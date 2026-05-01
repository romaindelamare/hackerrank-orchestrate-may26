"""Common node interface (Liskov-substitutable).

Every node is a callable that takes the shared TicketState and returns a
partial dict with whatever slots it updates. LangGraph merges the partial
back into the running state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from code.schemas.state import TicketState


class Node(ABC):
    """Abstract base for every graph node.

    Subclasses receive their dependencies via __init__ (constructor injection)
    and implement __call__ for the per-state work. Keeping nodes as classes
    rather than free functions makes dependency wiring explicit and unit
    testing trivial.
    """

    @abstractmethod
    def __call__(self, state: TicketState) -> dict:
        """Run the node and return the partial state update."""
        raise NotImplementedError
