"""OpenAI client wrapper (concrete ILLMClient).

Uses the openai Python SDK with structured output via response_format=json_schema
(supported on gpt-4o and newer). Falls back to JSON-mode + Pydantic retry on
older models that don't support the strict schema parameter.
"""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from code.config import LLM_MAX_RETRIES, LLM_TEMPERATURE
from code.llm.backoff import call_with_backoff

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MAX_TOKENS = 2048

# Models that support the strict JSON schema response_format (gpt-4o family and newer).
_STRICT_SCHEMA_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
}


class OpenAIClient:
    """Thin OpenAI wrapper that returns text or schema-validated objects."""

    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to code/.env or export it in your shell."
            )
        self._client = OpenAI(api_key=key)
        self._model = model
        self._use_strict_schema = model in _STRICT_SCHEMA_MODELS

    # ------------------------------------------------------------------ text
    def generate_text(self, prompt: str, *, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = call_with_backoff(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=_DEFAULT_MAX_TOKENS,
            ),
            provider="openai",
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------ structured
    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: str | None = None,
    ) -> T:
        if self._use_strict_schema:
            return self._structured_strict(prompt, schema, system=system)
        return self._structured_json_mode(prompt, schema, system=system)

    def _structured_strict(self, prompt: str, schema: Type[T], *, system: str | None) -> T:
        """Use OpenAI's native structured output (gpt-4o+). Single attempt — model guarantees schema."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = call_with_backoff(
            lambda: self._client.beta.chat.completions.parse(
                model=self._model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=_DEFAULT_MAX_TOKENS,
                response_format=schema,
            ),
            provider="openai",
        )
        return response.choices[0].message.parsed

    def _structured_json_mode(self, prompt: str, schema: Type[T], *, system: str | None) -> T:
        """JSON mode + Pydantic retry for models without strict schema support."""
        json_schema = schema.model_json_schema()
        schema_hint = (
            f"\n\nYou MUST respond with valid JSON that matches this schema exactly:\n"
            f"{json.dumps(json_schema, indent=2)}\n"
            f"Return ONLY the JSON object — no markdown, no commentary."
        )
        augmented_system = (system or "") + schema_hint
        last_err: Exception | None = None
        attempt_prompt = prompt

        for _ in range(LLM_MAX_RETRIES + 1):
            messages = []
            if augmented_system:
                messages.append({"role": "system", "content": augmented_system})
            messages.append({"role": "user", "content": attempt_prompt})

            try:
                response = call_with_backoff(
                    lambda: self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=_DEFAULT_MAX_TOKENS,
                        response_format={"type": "json_object"},
                    ),
                    provider="openai",
                )
                raw = (response.choices[0].message.content or "").strip()
                return schema.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError, ValueError) as err:
                last_err = err
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation with this error:\n{err}\n"
                    f"Return ONLY valid JSON matching the required schema."
                )

        raise RuntimeError(
            f"OpenAI failed to produce valid {schema.__name__} JSON after "
            f"{LLM_MAX_RETRIES + 1} attempts: {last_err}"
        )
