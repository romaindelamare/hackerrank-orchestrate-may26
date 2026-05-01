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
from collections import Counter
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

from code.cache.semantic_cache import SemanticCache
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


def _print_recap(
    total: int,
    status_counts: dict[str, int],
    request_type_counts: dict[str, int],
    company_counts: dict[str, int],
    output_path: Path,
    elapsed_seconds: float,
) -> None:
    """Print a formatted recap box with execution summary."""
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    width = 70
    border = "═" * width

    # Color mapping
    status_colors = {
        "resolved": (GREEN, "✓"),
        "replied": (GREEN, "✉"),
        "escalated": (YELLOW, "⚠"),
        "pending": (BLUE, "⏳"),
        "closed": (MAGENTA, "🔒"),
    }

    def build_breakdown(counts: dict[str, int], total_val: int, sort_by_count: bool = False) -> list[str]:
        """Build visual breakdown lines with bars and percentages."""
        lines = []
        total_val = total_val if total_val > 0 else 1
        if sort_by_count:
            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_items = sorted(counts.items(), key=lambda x: x[0])
        for key, count in sorted_items:
            pct = (count / total_val) * 100
            bar_fill = int((count / total_val) * 15)
            bar = "█" * bar_fill + "░" * (15 - bar_fill)
            lines.append(f"    {key:<18} {bar} {count:3} ({pct:5.1f}%)")
        return lines

    # Build status breakdown
    status_lines = []
    for status in sorted(status_counts.keys()):
        count = status_counts[status]
        pct = (count / total) * 100
        color, emoji = status_colors.get(status, (WHITE, "•"))
        bar_fill = int((count / total) * 15)
        bar = "█" * bar_fill + "░" * (15 - bar_fill)
        status_lines.append(
            f"  {emoji} {status.capitalize():<18} {color}{bar}{RESET} {count:3} ({pct:5.1f}%)"
        )

    # Print recap
    print()
    print(f"{CYAN}{border}{RESET}")
    print(f"{GREEN}{BOLD}✓ PROCESSING COMPLETE{RESET}")
    print(f"{CYAN}{border}{RESET}")
    print(f"{BOLD}Tickets Processed:{RESET} {WHITE}{total}{RESET}")
    print()
    print(f"{BOLD}Status Breakdown:{RESET}")
    for line in status_lines:
        print(line)
    print()
    print(f"{BOLD}Request Type:{RESET}")
    for line in build_breakdown(request_type_counts, total, sort_by_count=True):
        print(line)
    print()
    print(f"{BOLD}Companies:{RESET}")
    for company, count in sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = (count / total) * 100
        bar_fill = int((count / total) * 15)
        bar = "█" * bar_fill + "░" * (15 - bar_fill)
        print(f"    {company[:18]:<18} {bar} {count:3} ({pct:5.1f}%)")
    print()
    print(f"{BOLD}Output:{RESET} {CYAN}{output_path.name}{RESET}")
    print(f"{BOLD}Time Taken:{RESET} {WHITE}{elapsed_seconds:.1f}s{RESET}")
    print(f"{CYAN}{border}{RESET}")
    print()


def _row_to_state(row: pd.Series) -> dict:
    issue = str(row.get("Issue", "") or "")
    subject = str(row.get("Subject", "") or "")
    company = str(row.get("Company", "None") or "None")
    inp = TicketInput(issue=issue, subject=subject, company=company)
    return {"issue": inp.issue, "subject": inp.subject, "company": inp.company}


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
        llm = MistralClient()
        retriever = ChromaRetriever()
        graph = build_graph(llm, retriever)
        cache = SemanticCache()

        df = pd.read_csv(args.input)
        if args.limit:
            df = df.head(args.limit)

        rows: list[dict] = []
        total = len(df)
        cache_hits = 0
        status_counter: Counter[str] = Counter()
        request_type_counter: Counter[str] = Counter()
        company_counter: Counter[str] = Counter()
        print(f"[run] processing {total} ticket(s) ...", flush=True)

        for i, row in df.iterrows():
            idx = int(i) + 1 if isinstance(i, int) else len(rows) + 1
            state_in = _row_to_state(row)
            ticket_text = f"{state_in['issue']} | {state_in['subject']} | {state_in['company']}"

            cached = cache.get(ticket_text)
            if cached is not None:
                state_out = cached
                cache_hits += 1
                hit_label = "[CACHE HIT] "
            else:
                try:
                    state_out = graph.invoke(state_in)
                except Exception as err:
                    state_out = {
                        "status": "escalated",
                        "product_area": "general",
                        "response": "Unable to process ticket automatically; human follow-up required.",
                        "justification": f"Graph failure: {err}",
                        "request_type": "invalid",
                    }
                cache.set(ticket_text, {
                    "status": state_out.get("status", "escalated"),
                    "product_area": state_out.get("product_area", "general"),
                    "response": state_out.get("response", ""),
                    "justification": state_out.get("justification", ""),
                    "request_type": state_out.get("request_type", "invalid"),
                })
                hit_label = ""

            status = state_out.get("status", "escalated")
            product_area = state_out.get("product_area", "general")
            request_type = state_out.get("request_type", "invalid")
            company = state_in["company"]

            status_counter[status] += 1
            request_type_counter[request_type] += 1
            company_counter[company] += 1

            rows.append({
                "Issue": state_in["issue"],
                "Subject": state_in["subject"],
                "Company": company,
                "status": status,
                "product_area": product_area,
                "response": state_out.get("response", ""),
                "justification": state_out.get("justification", ""),
                "request_type": request_type,
            })
            # Format: [index/total] status | product_area | company | request_type
            print(f"  [{idx:2}/{total}] {hit_label}{status:12} | {product_area:35} | {company:18} | {request_type}", flush=True)

        if cache_hits:
            print(f"[cache] {cache_hits}/{total} ticket(s) served from semantic cache", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(args.output, index=False)

    elapsed = time.time() - start_time
    _print_recap(
        len(rows),
        dict(status_counter),
        dict(request_type_counter),
        dict(company_counter),
        args.output,
        elapsed,
    )

    logger.event(
        title="Batch run complete",
        summary=f"Processed {len(rows)} tickets, wrote {args.output}.",
        actions=[f"read {args.input}", f"wrote {args.output}"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
