"""Append-only agent log writer (AGENTS.md §5 format).

Writes to ~/hackerrank_orchestrate/log.txt with UTF-8 LF line endings.
Never rewrites or reorders prior entries. Cross-platform path resolution.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from code.config import LOG_DIR, LOG_FILE


class AgentLogger:
    """Tiny structured-log helper used by main.py at session start."""

    def __init__(self, path: Path = LOG_FILE) -> None:
        self._path = path
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def session_start(
        self,
        *,
        agent: str,
        repo_root: str,
        branch: str = "unknown",
        worktree: str = "main",
        parent_agent: str = "none",
        language: str = "py",
        time_remaining: str = "",
    ) -> None:
        ts = datetime.now().astimezone().isoformat()
        entry = (
            f"## [{ts}] SESSION START\n\n"
            f"Agent: {agent}\n"
            f"Repo Root: {repo_root}\n"
            f"Branch: {branch}\n"
            f"Worktree: {worktree}\n"
            f"Parent Agent: {parent_agent}\n"
            f"Language: {language}\n"
            f"Time Remaining: {time_remaining}\n\n"
        )
        self._append(entry)

    def event(self, title: str, summary: str, *, actions: list[str] | None = None) -> None:
        ts = datetime.now().astimezone().isoformat()
        actions = actions or []
        actions_str = "\n".join(f"* {a}" for a in actions) or "* (none)"
        entry = (
            f"## [{ts}] {title[:80]}\n\n"
            f"Agent Response Summary:\n{summary}\n\n"
            f"Actions:\n{actions_str}\n\n"
        )
        self._append(entry)

    def _append(self, content: str) -> None:
        with open(self._path, "a", encoding="utf-8", newline="\n") as f:
            f.write(content)
