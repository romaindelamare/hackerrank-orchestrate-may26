"""Generic rate-limit backoff for all LLM clients.

Per-provider module-level state means two clients using the same provider share
a single backoff budget, while different providers are isolated from each other.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

# {provider: [backoff_until, current_delay, last_hit_time]}
_state: dict[str, list[float]] = {}


def _get(provider: str) -> list[float]:
    if provider not in _state:
        _state[provider] = [0.0, 0.0, 0.0]
    return _state[provider]


def call_with_backoff(fn: Callable[[], T], *, provider: str) -> T:
    """Call fn, retrying once after an exponential backoff on 429 / rate-limit errors.

    Backoff decays by 50% after 30 s of clean requests, grows ×1.5 on each hit,
    caps at 60 s.
    """
    s = _get(provider)
    # s[0] = backoff_until, s[1] = current_delay, s[2] = last_hit_time

    now = time.time()

    # Decay: if it's been >30 s since the last rate-limit hit, halve the delay
    if s[2] > 0 and (now - s[2]) > 30:
        s[1] = max(0.0, s[1] * 0.5)
        s[2] = now
        if s[1] < 0.5:
            s[1] = 0.0

    # Wait out any active global backoff window
    if now < s[0]:
        wait = s[0] - now
        print(f"[rate-limit:{provider}] backing off {wait:.1f}s", flush=True)
        time.sleep(wait)

    try:
        return fn()
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limited" in err.lower():
            s[1] = min(60.0, max(1.0, (s[1] * 1.5) if s[1] else 2.0))
            s[0] = time.time() + s[1]
            s[2] = time.time()
            print(f"[rate-limit:{provider}] hit 429, backing off {s[1]:.1f}s", flush=True)
            time.sleep(s[1])
            return fn()
        raise
