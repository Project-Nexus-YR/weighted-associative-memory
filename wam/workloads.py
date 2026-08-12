"""Deterministic synthetic traces for evaluating memory predictors."""

from __future__ import annotations

import random


def sequential(length: int = 512, start: int = 0) -> list[int]:
    return list(range(start, start + length))


def repeating(pattern: tuple[int, ...] = (0, 1, 2, 3), repeats: int = 128) -> list[int]:
    return list(pattern) * repeats


def branching(length: int = 512, branch_probability: float = 0.8, seed: int = 7) -> list[int]:
    """Choose A-B-C or A-D-E cycles with a deterministic branch mix."""
    rng = random.Random(seed)
    trace: list[int] = []
    while len(trace) < length:
        trace.extend((0, 1, 2) if rng.random() < branch_probability else (0, 3, 4))
    return trace[:length]


def contextual(repeats: int = 128) -> list[int]:
    """A-B-X and C-B-Y: B alone is ambiguous, (A,B)/(C,B) is not."""
    return [address for _ in range(repeats) for address in (0, 1, 10, 2, 1, 11)]


def random_access(length: int = 512, address_space: int = 1024, seed: int = 11) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(address_space) for _ in range(length)]


def all_workloads(length: int = 512) -> dict[str, list[int]]:
    return {
        "Sequential": sequential(length),
        "Repeating": repeating(repeats=max(1, length // 4)),
        "Branching": branching(length),
        "Contextual": contextual(repeats=max(1, length // 6)),
        "Random": random_access(length),
    }
