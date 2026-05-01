"""Google Gemini client wrapper (concrete ILLMClient).

Uses the google-genai Python SDK (google-generativeai >= 0.8 / google-genai).
Structured output uses Gemini's native response_schema for models that support
it (gemini-1.5-pro, gemini-2.0-flash, etc.); falls back to JSON prompt + retry
for older models.
"""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from code.config import LLM_MAX_RETRIES, LLM_TEMPERATURE
from code.llm.backoff import call_with_backoff

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MAX_TOKENS = 2048

# Models with reliable response_schema support.
_SCHEMA_CAPABLE_MODELS = {
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
}


class GeminiClient:
    """Thin Google Gemini wrapper that returns text or schema-validated objects."""

    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to code/.env or export it in your shell."
            )
        self._client = genai.Client(api_key=key)
        self._model = model
        self._use_response_schema = any(
            model.startswith(prefix) for prefix in _SCHEMA_CAPABLE_MODELS
        )

    # ------------------------------------------------------------------ text
    def generate_text(self, prompt: str, *, system: str | None = None) -> str:
        config = genai_types.GenerateContentConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=_DEFAULT_MAX_TOKENS,
            system_instruction=system,
        )
        response = call_with_backoff(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            ),
            provider="gemini",
        )
        return response.text or ""

    # ------------------------------------------------------------ structured
    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: str | None = None,
    ) -> T:
        if self._use_response_schema:
            return self._structured_schema(prompt, schema, system=system)
        return self._structured_json_mode(prompt, schema, system=system)

    def _structured_schema(self, prompt: str, schema: Type[T], *, system: str | None) -> T:
        """Use Gemini's native response_schema for guaranteed JSON output."""
        config = genai_types.GenerateContentConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=_DEFAULT_MAX_TOKENS,
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
        )
        last_err: Exception | None = None
        attempt_prompt = prompt

        for _ in range(LLM_MAX_RETRIES + 1):
            try:
                response = call_with_backoff(
                    lambda: self._client.models.generate_content(
                        model=self._model,
                        contents=attempt_prompt,
                        config=config,
                    ),
                    provider="gemini",
                )
                raw = (response.text or "").strip()
                return schema.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError, ValueError) as err:
                last_err = err
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation:\n{err}\n"
                    f"Return ONLY valid JSON matching the required schema."
                )

        raise RuntimeError(
            f"Gemini failed to produce valid {schema.__name__} JSON after "
            f"{LLM_MAX_RETRIES + 1} attempts: {last_err}"
        )

    def _structured_json_mode(self, prompt: str, schema: Type[T], *, system: str | None) -> T:
        """Prompt-based JSON extraction for models without response_schema support."""
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
            config = genai_types.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=_DEFAULT_MAX_TOKENS,
                system_instruction=augmented_system,
                response_mime_type="application/json",
            )
            try:
                response = call_with_backoff(
                    lambda: self._client.models.generate_content(
                        model=self._model,
                        contents=attempt_prompt,
                        config=config,
                    ),
                    provider="gemini",
                )
                raw = (response.text or "").strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                return schema.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError, ValueError) as err:
                last_err = err
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation:\n{err}\n"
                    f"Return ONLY valid JSON matching the required schema."
                )

        raise RuntimeError(
            f"Gemini failed to produce valid {schema.__name__} JSON after "
            f"{LLM_MAX_RETRIES + 1} attempts: {last_err}"
        )
