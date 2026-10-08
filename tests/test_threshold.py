from numerics import find_threshold_2d, linear_growth_rate
from storage import load_config


def test_short_linear_probe_does_not_guess_threshold():
    config = load_config("configs/validation.yaml")
    config["threshold"]["linear_fit_window_ps"] = 2.0
    config["threshold"]["initial_P0_scan"] = [0.5, 1.5, 4.0]
    point = linear_growth_rate(config, duration_ps=6.0)
    assert point["status"] in ("STABLE", "UNSTABLE", "UNRESOLVED")
    assert point["sigma_late_per_ps"] == point["sigma_late_per_ps"]
    result = find_threshold_2d(config, save=False, duration_ps=6.0,
                               max_bisections=2)
    assert result["status"] in ("NOT_BRACKETED", "THRESHOLD_UNRESOLVED", "PRELIMINARY")
    assert len(result["points"]) >= 3


def test_threshold_resume_reuses_saved_points(tmp_path):
    config = load_config("configs/validation.yaml")
    config["threshold"]["linear_fit_window_ps"] = 2.0
    config["threshold"]["initial_P0_scan"] = [0.5, 1.5, 4.0]
    first = find_threshold_2d(config, duration_ps=6.0, max_bisections=0,
                              result_root=tmp_path)
    second = find_threshold_2d(config, duration_ps=6.0, max_bisections=0,
                               resume_dir=first["run_dir"])
    assert len(second["points"]) == len(first["points"])
    assert "THRESHOLD_RESUMED" in (
        tmp_path/next(tmp_path.iterdir()).name/"run_events.jsonl").read_text()
