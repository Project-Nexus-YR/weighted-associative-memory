"""Deterministic synthetic line-address workloads."""

from __future__ import annotations

import random
from collections.abc import Iterable


def sequential(length: int = 512, start: int = 0) -> list[int]:
    return list(range(start, start + length))


def constant_stride(length: int = 512, stride: int = 4, start: int = 0) -> list[int]:
    return [start + index * stride for index in range(length)]


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


def longer_dependency(repeats: int = 100) -> list[int]:
    """A-P-Q-R-X and B-P-Q-R-Y require more than the last two accesses."""
    return [address for _ in range(repeats) for address in (0, 1, 2, 3, 10, 4, 1, 2, 3, 11)]


def higher_order_ambiguous(context_count: int = 100, repeats: int = 32) -> list[int]:
    """Many ``prefix,B,C,target`` families; B/C alone are ambiguous."""
    trace: list[int] = []
    for _ in range(repeats):
        for family in range(context_count):
            prefix = family * 10 + 10
            trace.extend((prefix, 1, 2, family * 10 + 3))
    return trace


def higher_order_depth4(context_count: int = 100, repeats: int = 32) -> list[int]:
    """Many ``prefix,P,Q,R,target`` families; depth 4 identifies the branch."""
    trace: list[int] = []
    for _ in range(repeats):
        for family in range(context_count):
            prefix = family * 10 + 10
            trace.extend((prefix, 1, 2, 3, family * 10 + 4))
    return trace


def probabilistic_branching(length: int = 600, probability: float = 0.75, seed: int = 0) -> list[int]:
    """A-B-C with ``probability`` and A-B-D otherwise."""
    rng = random.Random(seed)
    trace: list[int] = []
    while len(trace) < length:
        trace.extend((0, 1, 2) if rng.random() < probability else (0, 1, 3))
    return trace[:length]


def phase_changing(length: int = 600, switch_fraction: float = 0.5) -> list[int]:
    """A-B-C in the first phase, then A-D-E."""
    first = int(length * switch_fraction)
    return ([0, 1, 2] * ((first + 2) // 3))[:first] + ([0, 3, 4] * ((length - first + 2) // 3))[: length - first]


def random_access(length: int = 512, address_space: int = 1024, seed: int = 11) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(address_space) for _ in range(length)]


def to_byte_addresses(line_trace: Iterable[int], cache_line_size: int = 64) -> list[int]:
    """Convert generated line IDs to raw byte addresses for the simulator."""
    return [line * cache_line_size for line in line_trace]


def all_workloads(length: int = 512, seed: int = 11) -> dict[str, list[int]]:
    return {
        "Sequential": sequential(length),
        "Stride": constant_stride(length, stride=4),
        "Repeating": repeating(repeats=max(1, length // 4)),
        "Contextual": contextual(repeats=max(1, length // 6)),
        "LongerDependency": longer_dependency(repeats=max(1, length // 10)),
        "Probabilistic": probabilistic_branching(length, seed=seed),
        "PhaseChanging": phase_changing(length),
        "Random": random_access(length, seed=seed),
    }
