"""Markdown chunker.

Splits each `.md` file at H2 headings (`## ...`). Each chunk inherits the
file's YAML front matter (title, source_url, breadcrumbs) and derives
company + product area from its directory path under `data/`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import yaml

from code.config import DATA_DIR
from code.schemas.ticket import Chunk

_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_H2_SPLIT_RE = re.compile(r"\n(?=##\s)")
_MIN_CHUNK_CHARS = 60


class MarkdownChunker:
    """Concrete IChunker for the support corpus markdown files."""

    def chunk(self, path: Path) -> Iterable[Chunk]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        front_matter, body = self._split_front_matter(raw)
        meta = self._parse_front_matter(front_matter)
        company, product_area = self._derive_taxonomy(path)
        source_file = str(path.relative_to(DATA_DIR.parent)).replace("\\", "/")
        source_url = meta.get("source_url", "") or ""

        for section in self._split_sections(body):
            text = section.strip()
            if len(text) < _MIN_CHUNK_CHARS:
                continue
            yield Chunk(
                text=text,
                company=company,
                product_area=product_area,
                source_file=source_file,
                source_url=source_url,
            )

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[str, str]:
        m = _FRONT_MATTER_RE.match(raw)
        if not m:
            return "", raw
        return m.group(1), raw[m.end():]

    @staticmethod
    def _parse_front_matter(front_matter: str) -> dict:
        if not front_matter:
            return {}
        try:
            data = yaml.safe_load(front_matter)
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError:
            return {}

    @staticmethod
    def _split_sections(body: str) -> list[str]:
        # Keep H1 + intro as the first chunk; subsequent H2s become their own.
        parts = _H2_SPLIT_RE.split(body)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _derive_taxonomy(path: Path) -> tuple[str, str]:
        try:
            rel = path.relative_to(DATA_DIR)
        except ValueError:
            return "unknown", "unknown"
        parts = rel.parts
        company = parts[0].lower() if parts else "unknown"
        product_area = parts[1].lower() if len(parts) > 1 else "general"
        return company, product_area
