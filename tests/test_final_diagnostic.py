from array import array

from scripts.analyze_champsim_final_diagnostic import EVENT, PREDICTION, chronological_oracle, diagnostic_alignment


def test_diagnostic_binary_records_have_stable_sizes():
    assert EVENT.size == 40
    assert PREDICTION.size == 32


def test_chronological_oracle_uses_only_pre_split_targets():
    sequence = array("Q", (100 + index % 5 for index in range(100)))
    result = chronological_oracle(sequence, depth=4, horizon=16)
    assert result["train_examples"] == int(100 * 0.7) - 16 - (4 - 1)
    assert result["evaluation_examples"] == 30 - 16
    assert result["coverage"] > 0.0
    assert result["oracle_top1_accuracy"] == 1.0


def test_alignment_validation_covers_required_horizons():
    rows = diagnostic_alignment()
    assert [row["test"] for row in rows] == ["H1_alignment", "H8_alignment", "H16_alignment"]
    assert all(row["status"] == "pass" for row in rows)
