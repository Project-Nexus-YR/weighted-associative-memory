from pathlib import Path

from scripts.convert_trace import convert
from wam.real_predictors import GMCStylePredictor, SPPStylePredictor, DeltaContextPredictor
from wam.real_trace_evaluation import discover_traces, workload_class


def test_external_trace_converter_keeps_explicit_data_addresses(tmp_path: Path) -> None:
    source = tmp_path / "raw.txt"
    destination = tmp_path / "data.trace"
    source.write_text(" L 0x1000,4\nI 0x2000\nS 0x1040,8\nnoise\n", encoding="utf-8")
    kept, skipped = convert(source, destination)
    assert kept == 2
    assert skipped == 2
    assert destination.read_text(encoding="utf-8").splitlines() == ["0x1000", "0x1040"]


def test_real_trace_discovery_and_categories(tmp_path: Path) -> None:
    (tmp_path / "graph_bfs.trace").write_text("0x0\n0x40\n", encoding="utf-8")
    assert discover_traces(tmp_path)[0][0] == "graph_bfs"
    assert workload_class("graph_bfs") == "graph"
    assert workload_class("matrix_scan") == "dynamic_programming"


def test_strong_temporal_baseline_families_share_the_predictor_interface() -> None:
    trace = [0, 8, 16, 0, 8, 16, 0, 8, 16] * 8
    for predictor in (DeltaContextPredictor("VLDP", 4, 4, 2048, longest_match=True), SPPStylePredictor(4, 4, 2048), GMCStylePredictor(4, 2048)):
        predictor.fit(trace)
        assert predictor.predict_horizon(trace[-5:], 1)
        assert predictor.storage_stats()["estimated_bytes"] <= 2048
