"""Project-wide configuration constants.

Loaded once at startup. All paths are resolved relative to the repo root
(the parent of this `code/` directory) so the agent runs the same way
from any working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CODE_DIR: Path = REPO_ROOT / "code"
DATA_DIR: Path = REPO_ROOT / "data"
TICKETS_DIR: Path = REPO_ROOT / "support_tickets"
INPUT_CSV: Path = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV: Path = TICKETS_DIR / "output.csv"
CHROMA_DIR: Path = CODE_DIR / "chroma_db"

MISTRAL_MODEL: str = "mistral-small-latest"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME: str = "support_docs"

RETRIEVAL_TOP_K: int = 8
LLM_TEMPERATURE: float = 0.0
LLM_MAX_RETRIES: int = 2

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
