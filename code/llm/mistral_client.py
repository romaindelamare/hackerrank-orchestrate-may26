"""Mistral AI client wrapper (concrete ILLMClient).

Uses the Mistral SDK v2.x with structured output via JSON mode and JSON schemas.
We validate output with `model_validate_json` to ensure schema compliance.
Includes global rate limiter to handle API limits.
"""

from __future__ import annotations

import json
import os
import time
from typing import Type, TypeVar

from mistralai.client import Mistral
from pydantic import BaseModel, ValidationError

from code.config import MISTRAL_MODEL, LLM_MAX_RETRIES, LLM_TEMPERATURE

T = TypeVar("T", bound=BaseModel)

# Global rate limiter state: tracks when we hit a rate limit and backs off all subsequent calls
_rate_limit_backoff_until = 0.0
_rate_limit_delay = 0.0
_last_rate_limit_time = 0.0


class MistralClient:
    """Thin Mistral wrapper that returns text or schema-validated objects."""

    def __init__(self, *, model: str = MISTRAL_MODEL, api_key: str | None = None) -> None:
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

        return self._call_with_backoff(
            lambda: self._client.chat.complete(
                model=self._model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
            )
        ).choices[0].message.content or ""

    def _call_with_backoff(self, fn):
        """Call fn with global rate limiting. On 429, exponentially back off all future calls.
        Decay backoff after 30s of successful requests."""
        global _rate_limit_backoff_until, _rate_limit_delay, _last_rate_limit_time

        now = time.time()

        # Decay backoff if we haven't hit a rate limit in 30 seconds
        if _last_rate_limit_time > 0 and (now - _last_rate_limit_time) > 30:
            _rate_limit_delay = max(0, _rate_limit_delay * 0.5)
            _last_rate_limit_time = now
            if _rate_limit_delay < 0.5:
                _rate_limit_delay = 0.0

        # Wait if we're in a backoff period
        if now < _rate_limit_backoff_until:
            wait = _rate_limit_backoff_until - now
            print(f"[rate-limit] global backoff {wait:.1f}s", flush=True)
            time.sleep(wait)

        # Make the call
        try:
            return fn()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limited" in error_str.lower():
                # Increase delay and extend backoff window
                _rate_limit_delay = min(60, max(1, _rate_limit_delay * 1.5 or 2))
                _rate_limit_backoff_until = time.time() + _rate_limit_delay
                _last_rate_limit_time = time.time()
                print(f"[rate-limit] hit 429, backing off all calls for {_rate_limit_delay:.1f}s", flush=True)
                # Retry this specific call after the global backoff
                time.sleep(_rate_limit_delay)
                return fn()
            else:
                raise

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
                response = self._call_with_backoff(
                    lambda: self._client.chat.complete(
                        model=self._model,
                        messages=messages,
                        temperature=LLM_TEMPERATURE,
                        response_format={
                            "type": "json_object",
                            "schema": json_schema,
                        },
                    )
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
