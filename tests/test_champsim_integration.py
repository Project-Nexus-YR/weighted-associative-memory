from pathlib import Path

from scripts.generate_champsim_smoke_trace import RECORD, write_trace


ROOT = Path(__file__).resolve().parents[1]


def test_native_champsim_record_is_64_bytes(tmp_path):
    output = tmp_path / "smoke.champsimtrace"
    write_trace(output, 64)
    assert RECORD.size == 64
    assert output.stat().st_size == 64 * RECORD.size


def test_wam_h16_source_has_fixed_budget_and_delayed_training_contract():
    source = (ROOT / "champsim/prefetcher/wam_h16/wam_h16.h").read_text(encoding="utf-8")
    implementation = (ROOT / "champsim/prefetcher/wam_h16/wam_h16.cc").read_text(encoding="utf-8")
    assert "TABLE_ENTRIES = 256" in source
    assert "static_assert(sizeof(entry) == 32" in source
    assert "HORIZON = 16" in source
    assert "pending_.size() >= HORIZON" in implementation
    assert "type == access_type::PREFETCH" in implementation


def test_set_associative_wam_preserves_budget_and_uses_four_ways():
    source = (ROOT / "champsim/prefetcher/set_associative_wam/set_associative_wam.h").read_text(encoding="utf-8")
    implementation = (ROOT / "champsim/prefetcher/set_associative_wam/set_associative_wam.cc").read_text(encoding="utf-8")
    assert "SETS = 64" in source
    assert "WAYS = 4" in source
    assert "TABLE_ENTRIES = 256" in source
    assert "ENTRY_STORAGE_BYTES = TABLE_ENTRIES * 32" in source
    assert "HORIZON = 16" in source
    assert "pending_.size() >= HORIZON" in implementation
    assert "CONFIDENCE_THRESHOLD = 8" in source
    assert "0xbf58476d1ce4e5b9ULL" in implementation
    assert "direct_shadow_train" in implementation


def test_champsim_smoke_results_are_explicitly_non_paper_quality():
    metadata = ROOT / "results/champsim_validation/smoke/run_metadata.json"
    assert metadata.exists()
    assert '"paper_quality_trace": false' in metadata.read_text(encoding="utf-8")
