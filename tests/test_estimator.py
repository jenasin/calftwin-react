"""State estimation: accuracy, calibration and fault tolerance."""
import numpy as np
import pytest

from calftwin.runner import run_episode


@pytest.mark.parametrize("scenario", ["normal", "heat", "water_block"])
def test_filter_tracks_core_temperature(scenario):
    r = run_episode(scenario=scenario, seed=11, agent="shadow_only")
    assert r["metrics"]["rmse_core_c"] < 0.30, "core-temperature RMSE too large"


def test_uncertainty_is_broadly_calibrated():
    """The 95 % interval should cover the truth roughly 95 % of the time."""
    cov = [run_episode(scenario=s, seed=s_i, agent="shadow_only")["metrics"]["coverage95_core"]
           for s_i, s in enumerate(["normal", "heat", "water_block", "sensor_fault"])]
    assert np.mean(cov) > 0.80, f"mean 95 % coverage only {np.mean(cov):.2f}"


def test_fever_is_detected_without_environmental_heat():
    r = run_episode(scenario="fever", seed=4, agent="shadow_only")
    d = r["trace"]
    assert d["p_fever_driven"].max() > 0.5, "febrile episode was never inferred"
    assert d.loc[d["true_pyrogen"] < 0.01, "p_fever_driven"].head(20).max() < 0.5


def test_blocked_trough_is_inferred_from_the_water_balance_residual():
    r = run_episode(scenario="water_block", seed=4, agent="shadow_only")
    assert r["trace"]["p_trough_ok"].min() < 0.3, "blockage never inferred"


def test_no_false_trough_alarm_on_a_normal_day():
    r = run_episode(scenario="normal", seed=4, agent="shadow_only")
    assert r["trace"]["p_trough_ok"].min() > 0.5, "false blockage alarm"


def test_forecast_predicts_benefit_of_cooling_on_a_hot_day():
    r = run_episode(scenario="heat", seed=6, agent="shadow_only")
    twin = r["twin"]
    fc = twin.forecast_options(3.0, ["observe", "cooling_on"])
    assert (fc["options"]["cooling_on"]["mean_peak_core_c"]
            <= fc["options"]["observe"]["mean_peak_core_c"])
