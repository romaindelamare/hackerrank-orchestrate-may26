"""LLM client protocol.

Nodes depend on this interface; the concrete Gemini implementation lives
in `gemini_client.py`. Swap providers by writing another class that
implements the same two methods.
"""

from __future__ import annotations

from typing import Protocol, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ILLMClient(Protocol):
    def generate_text(self, prompt: str, *, system: str | None = None) -> str:
        """Plain text completion. Used for short classifications."""
        ...

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: str | None = None,
    ) -> T:
        """Structured output validated against a Pydantic schema."""
        ...
