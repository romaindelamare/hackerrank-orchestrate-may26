"""Single-ticket processing logic."""

from code.cache.semantic_cache import SemanticCache


def process_ticket(
    cache: SemanticCache,
    graph,
    reviewer,
    ticket_text: str,
    state_in: dict,
) -> tuple[dict, str, bool]:
    """Process a single ticket: check cache, invoke graph+reviewer, handle errors.

    Returns:
        (state_out, hit_label, was_cache_hit)
    """
    cached = cache.get(ticket_text)
    if cached is not None:
        return cached, "[CACHE HIT] ", True

    try:
        state_out = graph.invoke(state_in)
        state_out = reviewer.invoke(state_out)
    except Exception as err:
        state_out = {
            "status": "escalated",
            "product_area": "general",
            "response": "Unable to process ticket automatically; human follow-up required.",
            "justification": f"Graph failure: {err}",
            "request_type": "invalid",
            "reviewer_action": "approved",
            "reviewer_notes": "Reviewer skipped due to upstream error.",
        }
    cache.set(ticket_text, {
        "status": state_out.get("status", "escalated"),
        "product_area": state_out.get("product_area", "general"),
        "response": state_out.get("response", ""),
        "justification": state_out.get("justification", ""),
        "request_type": state_out.get("request_type", "invalid"),
        "reviewer_action": state_out.get("reviewer_action", "approved"),
        "reviewer_notes": state_out.get("reviewer_notes", ""),
    })
    return state_out, "", False


def build_output_row(state_in: dict, state_out: dict) -> dict:
    """Build a single CSV output row from input and output state."""
    return {
        "Issue": state_in["issue"],
        "Subject": state_in["subject"],
        "Company": state_in["company"],
        "status": state_out.get("status", "escalated"),
        "product_area": state_out.get("product_area", "general"),
        "response": state_out.get("response", ""),
        "justification": state_out.get("justification", ""),
        "request_type": state_out.get("request_type", "invalid"),
    }
