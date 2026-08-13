from wam.hybrid_analysis import sampled_starts, selector_choice


def test_phase_sampling_covers_early_middle_and_late_windows():
    assert sampled_starts(3_000, 100) == [0, 1_500, 2_900]


def test_confidence_selector_uses_only_start_confidence():
    row = {"gmc_start_confidence": 0.2, "wam_start_confidence": 0.8}
    assert selector_choice(row, "ConfidenceSelector", []) == "WAM"


def test_window_oracle_selects_lower_cycle_predictor():
    row = {"gmc_cycles": 20, "wam_cycles": 10}
    assert selector_choice(row, "WindowOracle", []) == "WAM"
