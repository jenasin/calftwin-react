"""Ground-truth simulator: physiological plausibility and regulation."""
import numpy as np
import pytest

from calftwin.calf import CalfSimulator, Environment, thi
from calftwin.config import STEPS_PER_DAY


def run_day(base, amp, seed=0, days=2, **env_kw):
    sim = CalfSimulator(env=Environment(t_air_base_c=base, t_air_amp_c=amp, **env_kw),
                        rng=np.random.default_rng(seed))
    rec = []
    for i in range(STEPS_PER_DAY * days):
        sim.step()
        if i >= STEPS_PER_DAY * (days - 1):
            rec.append(sim.truth_record())
    return rec


def test_thermoneutral_day_stays_in_normal_range():
    rec = run_day(17.0, 5.0)
    tc = np.array([r["t_core_c"] for r in rec])
    assert 38.2 <= tc.min(), f"core fell to {tc.min():.2f} C"
    assert tc.max() <= 39.35, f"core rose to {tc.max():.2f} C without heat load"


def test_cold_day_is_defended_by_thermogenesis():
    """A cold pen must not drag core temperature down like a passive body."""
    rec = run_day(0.0, 4.0)
    tc = np.array([r["t_core_c"] for r in rec])
    assert tc.mean() > 38.0, f"cold-day mean core {tc.mean():.2f} C is not defended"


def test_heat_load_raises_core_and_respiration():
    rec = run_day(29.0, 6.0)
    tc = np.array([r["t_core_c"] for r in rec])
    rr = np.array([r["resp_bpm"] for r in rec])
    assert tc.max() > 39.4, "heat load did not produce hyperthermia"
    assert rr.max() > 90.0, "heat load did not produce panting"


def test_blocked_trough_produces_dehydration():
    rec = run_day(22.0, 5.0, water_blocked_from_h=8.0, water_blocked_to_h=20.0)
    wd = np.array([r["water_deficit"] for r in rec])
    assert wd.max() > 0.02, f"peak deficit only {wd.max():.3f}"


def test_fever_raises_core_without_environmental_heat():
    sim = CalfSimulator(env=Environment(t_air_base_c=16.0, t_air_amp_c=4.0),
                        rng=np.random.default_rng(1))
    sim.pyrogen_schedule = [(10.0, 0.8)]
    peak = 0.0
    for _ in range(STEPS_PER_DAY):
        sim.step()
        peak = max(peak, sim.state.t_core_c)
    assert peak > 39.5, f"fever peak only {peak:.2f} C in a cool pen"


def test_cooling_reduces_thermal_load():
    hot = dict(t_air_base_c=29.0, t_air_amp_c=6.0)
    a = CalfSimulator(env=Environment(**hot), rng=np.random.default_rng(5))
    b = CalfSimulator(env=Environment(**hot), rng=np.random.default_rng(5))
    b.effects.cooling_on = True
    for _ in range(STEPS_PER_DAY):
        a.step()
        b.step()
    assert b.state.t_core_c < a.state.t_core_c, "cooling had no thermal effect"


def test_thi_matches_published_form():
    # NRC (1971): THI = (1.8*Tdb + 32) - (0.55 - 0.0055*RH)*(1.8*Tdb - 26)
    # 25 C / 50 % RH -> 77.0 - 0.275*19.0 = 71.775
    assert thi(25.0, 50.0) == pytest.approx(71.775, abs=0.05)
    assert thi(30.0, 60.0) > 78.0          # heat-stress range for cattle
