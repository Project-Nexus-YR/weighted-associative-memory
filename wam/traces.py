"""Streaming plain-text trace ingestion and cache-line normalization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_addresses(path: str | Path) -> Iterator[int]:
    """Yield integer or hexadecimal addresses without loading a file in memory.

    Blank lines and lines beginning with ``#`` are ignored. Inline comments
    are accepted, which makes hand-authored traces convenient.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            token = raw_line.split("#", 1)[0].strip().split()
            if not token:
                continue
            yield int(token[0], 0)


def load_trace(path: str | Path) -> list[int]:
    """Materialize a trace when an experiment needs random access/splitting."""
    return list(iter_addresses(path))


def normalize_addresses(addresses: Iterable[int], cache_line_size: int = 64) -> Iterator[int]:
    if cache_line_size < 1:
        raise ValueError("cache_line_size must be positive")
    for address in addresses:
        yield address // cache_line_size
