"""Project-wide configuration constants.

Loaded once at startup. All paths are resolved relative to the repo root
(the parent of this `code/` directory) so the agent runs the same way
from any working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CODE_DIR: Path = REPO_ROOT / "code"
DATA_DIR: Path = REPO_ROOT / "data"
TICKETS_DIR: Path = REPO_ROOT / "support_tickets"
INPUT_CSV: Path = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV: Path = TICKETS_DIR / "output.csv"
CHROMA_DIR: Path = CODE_DIR / "chroma_db"

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME: str = "support_docs"


@dataclass
class LLMConfig:
    """Provider + model selector for a single node or agent.

    Supported providers: "mistral", "anthropic", "openai"
    """
    provider: str
    model: str


# ---------------------------------------------------------------------------
# Per-node LLM configuration
# ---------------------------------------------------------------------------
# Classify: lightweight structured extraction — small/fast model is fine.
CLASSIFY_LLM = LLMConfig(provider="mistral", model="mistral-small-latest")

# Reply: grounded answer generation — benefits from a capable model.
REPLY_LLM = LLMConfig(provider="mistral", model="mistral-small-latest")

# Escalate: empathetic message composition — small model sufficient.
ESCALATE_LLM = LLMConfig(provider="mistral", model="mistral-small-latest")

# Reviewer: meta-reasoning quality gate — use a stronger model for better catches.
REVIEWER_LLM = LLMConfig(provider="mistral", model="mistral-large-latest")

# Optimal configuration
# CLASSIFY_LLM  = LLMConfig(provider="mistral",   model="mistral-small-latest")
# REPLY_LLM     = LLMConfig(provider="anthropic", model="claude-sonnet-4-6")
# ESCALATE_LLM  = LLMConfig(provider="mistral",   model="mistral-small-latest")
# REVIEWER_LLM  = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")


RETRIEVAL_TOP_K: int = 8
LLM_TEMPERATURE: float = 0.0
LLM_MAX_RETRIES: int = 2

SEMANTIC_CACHE_COLLECTION: str = "llm_response_cache"
SEMANTIC_CACHE_THRESHOLD: float = 0.93

VALID_COMPANIES: tuple[str, ...] = ("hackerrank", "claude", "visa")

# Risk flags that force the escalate branch regardless of retrieval quality.
ESCALATE_TRIGGERS: frozenset[str] = frozenset({
    "billing_dispute",
    "fraud",
    "identity_theft",
    "account_access_non_owner",
    "security_vulnerability",
    "legal_or_regulatory",
    "score_manipulation",
    "sitewide_outage",
    "prompt_injection",
    "pii_request",
})

LOG_DIR: Path = Path(os.path.expanduser("~")) / "hackerrank_orchestrate"
LOG_FILE: Path = LOG_DIR / "log.txt"
