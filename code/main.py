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
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

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

from code.config import (
    CODE_DIR,
    INPUT_CSV,
    OUTPUT_CSV,
    REPO_ROOT,
)
from code.graph import build_graph
from code.llm.mistral_client import MistralClient
from code.retrieval.indexer import build_index, collection_size
from code.retrieval.retriever import ChromaRetriever
from code.schemas.ticket import TicketInput
from code.utils.agent_logger import AgentLogger

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
    p = argparse.ArgumentParser(description="HackerRank Orchestrate support triage agent")
    p.add_argument("--reindex", action="store_true", help="Rebuild the vector index")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows")
    p.add_argument("--input", type=Path, default=INPUT_CSV, help="Input CSV path")
    p.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Output CSV path")
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


def _row_to_state(row: pd.Series) -> dict:
    issue = str(row.get("Issue", "") or "")
    subject = str(row.get("Subject", "") or "")
    company = str(row.get("Company", "None") or "None")
    inp = TicketInput(issue=issue, subject=subject, company=company)
    return {"issue": inp.issue, "subject": inp.subject, "company": inp.company}


def main() -> int:
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

    _ensure_index(args.reindex)

    print("[graph] wiring dependencies ...", flush=True)
    llm = MistralClient()
    with _suppress_chroma_stderr():
        retriever = ChromaRetriever()
        graph = build_graph(llm, retriever)

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    rows: list[dict] = []
    total = len(df)
    print(f"[run] processing {total} ticket(s) ...", flush=True)

    for i, row in df.iterrows():
        idx = int(i) + 1 if isinstance(i, int) else len(rows) + 1
        state_in = _row_to_state(row)
        try:
            with _suppress_chroma_stderr():
                state_out = graph.invoke(state_in)
        except Exception as err:
            state_out = {
                "status": "escalated",
                "product_area": "general",
                "response": "Unable to process ticket automatically; human follow-up required.",
                "justification": f"Graph failure: {err}",
                "request_type": "invalid",
            }
        rows.append({
            "Issue": state_in["issue"],
            "Subject": state_in["subject"],
            "Company": state_in["company"],
            "status": state_out.get("status", "escalated"),
            "product_area": state_out.get("product_area", "general"),
            "response": state_out.get("response", ""),
            "justification": state_out.get("justification", ""),
            "request_type": state_out.get("request_type", "invalid"),
        })
        print(f"  [{idx}/{total}] {state_out.get('status')} / {state_out.get('product_area')}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(args.output, index=False)
    print(f"[done] wrote {len(rows)} rows to {args.output}", flush=True)

    logger.event(
        title="Batch run complete",
        summary=f"Processed {len(rows)} tickets, wrote {args.output}.",
        actions=[f"read {args.input}", f"wrote {args.output}"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
