"""Output formatting for batch processing results."""

from pathlib import Path


def print_recap(
    total: int,
    status_counts: dict[str, int],
    request_type_counts: dict[str, int],
    company_counts: dict[str, int],
    reviewer_counts: dict[str, int],
    output_path: Path,
    elapsed_seconds: float,
) -> None:
    """Print a formatted recap box with execution summary."""
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

    status_colors = {
        "resolved": GREEN,
        "replied": GREEN,
        "escalated": YELLOW,
        "pending": BLUE,
        "closed": MAGENTA,
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

    status_lines = []
    for status in sorted(status_counts.keys()):
        count = status_counts[status]
        pct = (count / total) * 100
        color = status_colors.get(status, WHITE)
        bar_fill = int((count / total) * 15)
        bar = "█" * bar_fill + "░" * (15 - bar_fill)
        status_lines.append(
            f"    {status.capitalize():<18} {color}{bar}{RESET} {count:3} ({pct:5.1f}%)"
        )

    print()
    print(f"{CYAN}{border}{RESET}")
    print(f"{GREEN}{BOLD}PROCESSING COMPLETE{RESET}")
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
    print(f"{BOLD}Reviewer Quality Gate:{RESET}")
    reviewer_colors = {
        "approved": GREEN,
        "refined": BLUE,
        "escalated": YELLOW,
    }
    for action in ["approved", "refined", "escalated"]:
        count = reviewer_counts.get(action, 0)
        pct = (count / total) * 100 if total > 0 else 0
        bar_fill = int((count / total) * 15) if total > 0 else 0
        bar = "█" * bar_fill + "░" * (15 - bar_fill)
        color = reviewer_colors.get(action, WHITE)
        print(f"    {action.capitalize():<18} {color}{bar}{RESET} {count:3} ({pct:5.1f}%)")
    print()
    print(f"{BOLD}Output:{RESET} {CYAN}{output_path.name}{RESET}")
    print(f"{BOLD}Time Taken:{RESET} {WHITE}{elapsed_seconds:.1f}s{RESET}")
    print(f"{CYAN}{border}{RESET}")
    print()
