"""Day scenarios: the stress patterns the testbed can generate.

Each scenario fixes a pen micro-climate, an optional water-trough blockage, an
optional sensor fault and an optional pyrogen (fever) event. Seed-dependent
jitter makes every simulated day different while keeping the scenario's
qualitative structure intact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .calf import Environment
from .sensors import SensorFault

SCENARIOS = ("normal", "heat", "water_block", "sensor_fault",
             "fever", "fever_hot", "combined")


@dataclass
class ScenarioSpec:
    name: str
    env: Environment
    faults: list[SensorFault] = field(default_factory=list)
    pyrogen_schedule: list[tuple[float, float]] = field(default_factory=list)
    description: str = ""


def build(name: str, rng: np.random.Generator) -> ScenarioSpec:
    """Instantiate one scenario day with seed-dependent jitter."""
    j = lambda s: float(rng.normal(0.0, s))                      # noqa: E731

    if name == "normal":
        return ScenarioSpec(
            name, Environment(t_air_base_c=17.0 + j(1.5), t_air_amp_c=5.0 + j(0.6),
                              rh_base_pct=62 + j(5)),
            description="Thermoneutral day, water available, no sensor fault.")

    if name == "heat":
        return ScenarioSpec(
            name, Environment(t_air_base_c=27.0 + j(1.0), t_air_amp_c=6.0 + j(0.7),
                              rh_base_pct=58 + j(6)),
            description="Hot day with radiant load; peak air temperature ~33 C.")

    if name == "water_block":
        start = 9.0 + j(0.8)
        return ScenarioSpec(
            name, Environment(t_air_base_c=22.0 + j(1.2), t_air_amp_c=5.5 + j(0.6),
                              water_blocked_from_h=start,
                              water_blocked_to_h=start + 9.0 + j(1.0)),
            description="Warm day; drinking water unavailable for ~9 h from mid-morning.")

    if name == "sensor_fault":
        s = 10.0 + j(1.0)
        return ScenarioSpec(
            name, Environment(t_air_base_c=25.0 + j(1.2), t_air_amp_c=5.5 + j(0.6)),
            faults=[SensorFault("ear_temp_c", "stuck", s, s + 6.0 + j(0.8)),
                    SensorFault("resp_bpm", "dropout", s + 1.0, s + 2.5)],
            description="Warm day; ear-tag temperature frozen for ~6 h plus a "
                        "respiration-rate dropout burst.")

    if name == "fever":
        return ScenarioSpec(
            name, Environment(t_air_base_c=17.0 + j(1.5), t_air_amp_c=5.0 + j(0.6)),
            pyrogen_schedule=[(11.0 + j(0.5), 0.80)],
            description="Thermoneutral day with a febrile episode starting late morning.")

    if name == "fever_hot":
        return ScenarioSpec(
            name, Environment(t_air_base_c=26.0 + j(1.0), t_air_amp_c=6.0 + j(0.7)),
            pyrogen_schedule=[(11.5 + j(0.5), 0.75)],
            description="Hot day with a concurrent febrile episode: hyperthermia is "
                        "ambiguous between environmental load and fever.")

    if name == "combined":
        s = 10.5 + j(1.0)
        wb = 9.5 + j(0.8)
        return ScenarioSpec(
            name, Environment(t_air_base_c=27.0 + j(1.0), t_air_amp_c=6.0 + j(0.7),
                              water_blocked_from_h=wb, water_blocked_to_h=wb + 8.0 + j(1.0)),
            faults=[SensorFault("ear_temp_c", "stuck", s, s + 5.0 + j(0.8)),
                    SensorFault("heart_bpm", "drift", 14.0, 22.0, magnitude=2.5)],
            description="Hot day, blocked trough, frozen ear-tag and a drifting "
                        "heart-rate channel.")

    raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
