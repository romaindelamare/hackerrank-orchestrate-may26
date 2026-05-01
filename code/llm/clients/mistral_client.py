"""Mistral AI client wrapper (concrete ILLMClient).

Uses the Mistral SDK v2.x with structured output via JSON mode and JSON schemas.
We validate output with `model_validate_json` to ensure schema compliance.
"""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from mistralai.client import Mistral
from pydantic import BaseModel, ValidationError

from code.config import LLM_MAX_RETRIES, LLM_TEMPERATURE
from code.llm.backoff import call_with_backoff

T = TypeVar("T", bound=BaseModel)


class MistralClient:
    """Thin Mistral wrapper that returns text or schema-validated objects."""

    def __init__(self, *, model: str = "mistral-small-latest", api_key: str | None = None) -> None:
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Copy code/.env.example to code/.env "
                "and fill it in, or export the variable in your shell."
            )
        self._client = Mistral(api_key=key)
        self._model = model

    # ------------------------------------------------------------------ text
    def generate_text(self, prompt: str, *, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return call_with_backoff(
            lambda: self._client.chat.complete(
                model=self._model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
            ),
            provider="mistral",
        ).choices[0].message.content or ""

    # ------------------------------------------------------------ structured
    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: str | None = None,
    ) -> T:
        last_err: Exception | None = None
        attempt_prompt = prompt

        # Convert Pydantic schema to JSON schema for Mistral
        json_schema = schema.model_json_schema()

        for attempt in range(LLM_MAX_RETRIES + 1):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": attempt_prompt})

            try:
                response = call_with_backoff(
                    lambda: self._client.chat.complete(
                        model=self._model,
                        messages=messages,
                        temperature=LLM_TEMPERATURE,
                        response_format={
                            "type": "json_object",
                            "schema": json_schema,
                        },
                    ),
                    provider="mistral",
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
            f"Mistral failed to produce valid {schema.__name__} JSON after "
            f"{LLM_MAX_RETRIES + 1} attempts: {last_err}"
        )
