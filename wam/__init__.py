"""Weighted Associative Memory research prototype."""

from .predictor import LastTransitionPredictor, NextLinePredictor, StridePredictor, WeightedTriePredictor
from .simulator import SimulatorConfig, SimulationResult, simulate
from .trie import WeightedTrie, WeightedTrieNode
from .traces import iter_addresses, load_trace

__all__ = [
    "LastTransitionPredictor",
    "NextLinePredictor",
    "StridePredictor",
    "SimulationResult",
    "SimulatorConfig",
    "WeightedTrie",
    "WeightedTrieNode",
    "WeightedTriePredictor",
    "simulate",
    "iter_addresses",
    "load_trace",
]
