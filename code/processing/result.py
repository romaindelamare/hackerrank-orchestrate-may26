"""Result aggregation for batch processing."""

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class BatchResult:
    """Aggregates rows and metrics from batch ticket processing."""

    rows: list[dict] = field(default_factory=list)
    cache_hits: int = 0
    status_counter: Counter = field(default_factory=Counter)
    request_type_counter: Counter = field(default_factory=Counter)
    company_counter: Counter = field(default_factory=Counter)
    reviewer_counter: Counter = field(default_factory=Counter)
