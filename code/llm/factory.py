"""LLM client factory.

Maps a LLMConfig (provider + model) to the correct concrete ILLMClient.
Add a new branch here when adding a new provider.
"""

from __future__ import annotations

from code.config import LLMConfig
from code.llm.interfaces import ILLMClient


def create_llm_client(cfg: LLMConfig) -> ILLMClient:
    """Instantiate and return the appropriate LLM client for *cfg*.

    Raises ValueError for unknown providers so misconfiguration surfaces early.
    """
    provider = cfg.provider.lower()

    if provider == "mistral":
        from code.llm.clients.mistral_client import MistralClient
        return MistralClient(model=cfg.model)

    if provider == "anthropic":
        from code.llm.clients.claude_client import ClaudeClient
        return ClaudeClient(model=cfg.model)

    if provider == "openai":
        from code.llm.clients.openai_client import OpenAIClient
        return OpenAIClient(model=cfg.model)

    if provider == "google" or provider == "gemini":
        from code.llm.clients.gemini_client import GeminiClient
        return GeminiClient(model=cfg.model)

    raise ValueError(
        f"Unknown LLM provider '{cfg.provider}'. "
        f"Supported: 'mistral', 'anthropic', 'openai', 'google'."
    )
