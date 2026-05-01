"""Batch orchestration for ticket processing."""

import pandas as pd

from code.cache.semantic_cache import SemanticCache
from code.schemas.ticket import TicketInput
from code.processing.result import BatchResult
from code.processing.ticket import process_ticket, build_output_row


def _row_to_state(row: pd.Series) -> dict:
    """Convert a CSV row to agent state dict."""
    issue = str(row.get("Issue", "") or "")
    subject = str(row.get("Subject", "") or "")
    company = str(row.get("Company", "None") or "None")
    inp = TicketInput(issue=issue, subject=subject, company=company)
    return {"issue": inp.issue, "subject": inp.subject, "company": inp.company}


def run_batch(
    df: pd.DataFrame,
    graph,
    reviewer,
    cache: SemanticCache,
) -> BatchResult:
    """Process all tickets in the dataframe. Returns counts and output rows."""
    result = BatchResult()
    total = len(df)
    print(f"[run] processing {total} ticket(s) ...", flush=True)

    for i, row in df.iterrows():
        idx = int(i) + 1 if isinstance(i, int) else len(result.rows) + 1
        state_in = _row_to_state(row)
        ticket_text = f"{state_in['issue']} | {state_in['subject']} | {state_in['company']}"

        state_out, hit_label, was_cache_hit = process_ticket(cache, graph, reviewer, ticket_text, state_in)
        if was_cache_hit:
            result.cache_hits += 1

        status = state_out.get("status", "escalated")
        product_area = state_out.get("product_area", "general")
        request_type = state_out.get("request_type", "invalid")
        company = state_in["company"]
        reviewer_action = state_out.get("reviewer_action", "approved")

        result.status_counter[status] += 1
        result.request_type_counter[request_type] += 1
        result.company_counter[company] += 1
        result.reviewer_counter[reviewer_action] += 1

        result.rows.append(build_output_row(state_in, state_out))
        review_tag = "" if reviewer_action == "approved" else f" [{reviewer_action.upper()}]"
        print(f"  [{idx:2}/{total}] {hit_label}{status:12} | {product_area:35} | {company:18} | {request_type}{review_tag}", flush=True)

    return result
