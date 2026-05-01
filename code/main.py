"""Entry point for the support-triage agent.

Usage:
    python -m code.main                    # process support_tickets.csv
    python -m code.main --reindex          # rebuild Chroma index first
    python -m code.main --limit 5          # process only first N rows (debug)

The entry point's job is composition only:
  1. parse args
  2. build the index if needed
  3. wire dependencies (LLM client + retriever) into the graph
  4. iterate input CSV, invoke graph per row, write output CSV
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from code.cache.semantic_cache import SemanticCache
from code.config import (
    CODE_DIR,
    INPUT_CSV,
    OUTPUT_CSV,
    REPO_ROOT,
    CLASSIFY_LLM,
    REPLY_LLM,
    ESCALATE_LLM,
    REVIEWER_LLM,
)
from code.agents.triage import build_graph
from code.agents.reviewer import build_reviewer_graph
from code.llm.factory import create_llm_client
from code.retrieval.indexer import build_index, collection_size
from code.retrieval.retriever import ChromaRetriever
from code.utils.agent_logger import AgentLogger
from code.processing import run_batch, print_recap

logging.getLogger("chromadb").setLevel(logging.ERROR)


@contextlib.contextmanager
def _suppress_chroma_stderr():
    """Suppress ChromaDB's telemetry stderr noise."""
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = old_stderr

OUTPUT_COLUMNS = [
    "Issue",
    "Subject",
    "Company",
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HackerRank Orchestrate support triage agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python -m code.main                         # process all tickets\n"
               "  python -m code.main --reindex --limit 5     # rebuild index, process 5 tickets\n"
               "  python -m code.main --reset-cache           # clear semantic cache, then process\n"
    )

    # Index management
    index_group = p.add_argument_group("Index Management")
    index_group.add_argument("--reindex", action="store_true", help="Rebuild the vector index from scratch")

    # Cache management
    cache_group = p.add_argument_group("Cache Management")
    cache_group.add_argument("--reset-cache", action="store_true", help="Clear semantic cache before processing")

    # Processing control
    control_group = p.add_argument_group("Processing Control")
    control_group.add_argument("--limit", type=int, default=None, help="Process only the first N rows (for debugging)")

    # I/O paths
    io_group = p.add_argument_group("Input/Output Paths")
    io_group.add_argument("--input", type=Path, default=INPUT_CSV, help="Input CSV path")
    io_group.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Output CSV path")

    return p.parse_args()


def _ensure_index(reindex: bool) -> None:
    if reindex or collection_size() == 0:
        print("[indexer] building vector index from data/ ...", flush=True)
        with _suppress_chroma_stderr():
            n = build_index(reset=reindex)
        print(f"[indexer] indexed {n} chunks", flush=True)
    else:
        with _suppress_chroma_stderr():
            size = collection_size()
        print(f"[indexer] reusing existing index ({size} chunks)", flush=True)


def _time_remaining() -> str:
    end = datetime.fromisoformat("2026-05-02T11:00:00+05:30")
    now = datetime.now().astimezone()
    delta = end - now
    if delta.total_seconds() <= 0:
        return "challenge ended"
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes, _ = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m"


def main() -> int:
    start_time = time.time()
    load_dotenv(dotenv_path=CODE_DIR / ".env")
    args = _parse_args()

    logger = AgentLogger()
    logger.session_start(
        agent="hackerrank-orchestrate-agent",
        repo_root=str(REPO_ROOT),
        language="py",
        time_remaining=_time_remaining(),
    )

    if not args.input.exists():
        print(f"[error] input CSV not found: {args.input}", file=sys.stderr)
        return 1

    # Suppress ChromaDB telemetry noise during entire setup and processing
    with _suppress_chroma_stderr():
        _ensure_index(args.reindex)

        print("[graph] wiring dependencies ...", flush=True)
        print(
            f"[llm] classify={CLASSIFY_LLM.provider}/{CLASSIFY_LLM.model} "
            f"reply={REPLY_LLM.provider}/{REPLY_LLM.model} "
            f"escalate={ESCALATE_LLM.provider}/{ESCALATE_LLM.model} "
            f"reviewer={REVIEWER_LLM.provider}/{REVIEWER_LLM.model}",
            flush=True,
        )
        retriever = ChromaRetriever()
        graph = build_graph(
            retriever,
            classify_llm=create_llm_client(CLASSIFY_LLM),
            reply_llm=create_llm_client(REPLY_LLM),
            escalate_llm=create_llm_client(ESCALATE_LLM),
        )
        reviewer = build_reviewer_graph(create_llm_client(REVIEWER_LLM))
        cache = SemanticCache()

        if args.reset_cache:
            cache.reset()
            print("[cache] semantic cache cleared", flush=True)

        df = pd.read_csv(args.input)
        if args.limit:
            df = df.head(args.limit)

        result = run_batch(df, graph, reviewer, cache)

        if result.cache_hits:
            total = len(df)
            print(f"[cache] {result.cache_hits}/{total} ticket(s) served from semantic cache", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(result.rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(args.output, index=False)

    elapsed = time.time() - start_time
    print_recap(
        len(result.rows),
        dict(result.status_counter),
        dict(result.request_type_counter),
        dict(result.company_counter),
        dict(result.reviewer_counter),
        args.output,
        elapsed,
    )

    logger.event(
        title="Batch run complete",
        summary=f"Processed {len(result.rows)} tickets, wrote {args.output}.",
        actions=[f"read {args.input}", f"wrote {args.output}"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
