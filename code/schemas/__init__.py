"""Typed contracts shared across the agent (state + I/O models)."""

from code.schemas.state import TicketState
from code.schemas.ticket import Chunk, TicketInput, TicketOutput

__all__ = ["TicketState", "Chunk", "TicketInput", "TicketOutput"]
