"""Anthropic Claude client wrapper (concrete ILLMClient).

Uses the Anthropic Python SDK. Structured output is achieved by instructing
the model to return strict JSON and validating with Pydantic; retries on
validation failure up to LLM_MAX_RETRIES times.
"""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from code.config import LLM_MAX_RETRIES, LLM_TEMPERATURE
from code.llm.backoff import call_with_backoff

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MAX_TOKENS = 2048


class ClaudeClient:
    """Thin Anthropic wrapper that returns text or schema-validated objects."""

    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to code/.env or export it in your shell."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    # ------------------------------------------------------------------ text
    def generate_text(self, prompt: str, *, system: str | None = None) -> str:
        kwargs: dict = dict(
            model=self._model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        response = call_with_backoff(lambda: self._client.messages.create(**kwargs), provider="anthropic")
        return response.content[0].text if response.content else ""

    # ------------------------------------------------------------ structured
    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: str | None = None,
    ) -> T:
        json_schema = schema.model_json_schema()
        schema_hint = (
            f"\n\nYou MUST respond with valid JSON that matches this schema exactly:\n"
            f"{json.dumps(json_schema, indent=2)}\n"
            f"Return ONLY the JSON object — no markdown, no commentary."
        )

        augmented_system = (system or "") + schema_hint
        last_err: Exception | None = None
        attempt_prompt = prompt

        for attempt in range(LLM_MAX_RETRIES + 1):
            kwargs: dict = dict(
                model=self._model,
                max_tokens=_DEFAULT_MAX_TOKENS,
                system=augmented_system,
                messages=[{"role": "user", "content": attempt_prompt}],
            )
            try:
                response = call_with_backoff(lambda: self._client.messages.create(**kwargs), provider="anthropic")
                raw = (response.content[0].text if response.content else "").strip()
                # Strip accidental markdown fences
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                return schema.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError, ValueError) as err:
                last_err = err
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation with this error:\n{err}\n"
                    f"Return ONLY valid JSON matching the required schema."
                )

        raise RuntimeError(
            f"Claude failed to produce valid {schema.__name__} JSON after "
            f"{LLM_MAX_RETRIES + 1} attempts: {last_err}"
        )
