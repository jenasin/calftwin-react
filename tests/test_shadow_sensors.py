"""Sensor faults and the shadow's data-quality checks."""
import numpy as np

from calftwin.calf import CalfSimulator, Environment
from calftwin.sensors import SensorArray, SensorFault
from calftwin.shadow import DigitalShadow


def build(faults=None, seed=0):
    sim = CalfSimulator(env=Environment(t_air_base_c=24.0), rng=np.random.default_rng(seed))
    arr = SensorArray(sim=sim, rng=np.random.default_rng(seed + 1), faults=faults or [])
    sh = DigitalShadow(calf_id=sim.params.calf_id)
    return sim, arr, sh


def test_core_temperature_is_never_exposed_by_the_sensor_array():
    _, arr, _ = build()
    assert "t_core_c" not in arr.sample()
    assert "core_temp_c" not in arr.sample()


def test_frozen_channel_is_flagged():
    sim, arr, sh = build(faults=[SensorFault("ear_temp_c", "stuck", 0.0, 24.0)])
    flagged = False
    for _ in range(40):
        sim.step()
        f = sh.ingest(sim.step_idx, sim.hour, arr.sample(), sim.params.calf_id)
        flagged |= bool(f and f["ear_temp_c"].frozen)
    assert flagged, "a stuck ear tag was never flagged as frozen"


def test_dropout_is_recorded_as_missing():
    sim, arr, sh = build(faults=[SensorFault("resp_bpm", "dropout", 0.0, 24.0)])
    sim.step()
    f = sh.ingest(sim.step_idx, sim.hour, arr.sample(), sim.params.calf_id)
    assert f["resp_bpm"].missing


def test_duplicate_and_out_of_order_records_are_rejected():
    sim, arr, sh = build()
    sim.step()
    obs = arr.sample()
    sh.ingest(sim.step_idx, sim.hour, obs, sim.params.calf_id)
    assert sh.ingest(sim.step_idx, sim.hour, obs, sim.params.calf_id) == {}
    assert sh.n_rejected_duplicate == 1


def test_identity_mismatch_raises():
    sim, arr, sh = build()
    sim.step()
    try:
        sh.ingest(sim.step_idx, sim.hour, arr.sample(), "CALF-999")
    except ValueError:
        return
    raise AssertionError("shadow accepted data for a different animal")


def test_gaps_are_counted():
    sim, arr, sh = build()
    sim.step()
    sh.ingest(1, 0.25, arr.sample(), sim.params.calf_id)
    sh.ingest(5, 1.25, arr.sample(), sim.params.calf_id)
    assert sh.n_gaps == 1
