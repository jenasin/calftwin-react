"""Simulated sensor layer.

Sensors are the only channel between the animal and the twin. Two properties of
this layer drive the whole experiment:

1. **Core temperature is not observed.** The ear-tag returns a skin temperature
   that is confounded by air temperature, so recovering thermal state requires a
   model, not a threshold.
2. **Faults are indistinguishable from physiology at the sample level.** A stuck
   ear-tag looks like a stable calf; a zero trough reading means either a blocked
   trough or a calf that is simply not thirsty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calf import CalfSimulator
from .config import DEFAULT_SENSORS, VERIFY_CORE_SIGMA, SensorSpec


@dataclass
class SensorFault:
    """A scheduled fault on one channel."""
    channel: str
    kind: str                # "stuck" | "drift" | "dropout" | "bias"
    start_h: float
    end_h: float
    magnitude: float = 0.0   # drift K/h, or bias K
    _stuck_value: float | None = field(default=None, repr=False)

    def active(self, hour: float) -> bool:
        return self.start_h <= hour < self.end_h


@dataclass
class SensorArray:
    sim: CalfSimulator
    rng: np.random.Generator
    specs: dict[str, SensorSpec] = field(default_factory=lambda: dict(DEFAULT_SENSORS))
    noise_mult: float = 1.0
    faults: list[SensorFault] = field(default_factory=list)
    verify_count: int = 0

    # ---- clean (noise-free) transducer outputs ---------------------------
    def _clean(self) -> dict[str, float]:
        s = self.sim.state
        t_air = self.sim.effective_air_temp()
        rh = self.sim.env.rh(self.sim.hour)
        # ear skin temperature: core signal heavily confounded by air temperature
        ear = s.t_core_c - 1.2 - 0.16 * (24.0 - t_air)
        return {
            "ear_temp_c": ear,
            "resp_bpm": s.resp_bpm,
            "heart_bpm": s.heart_bpm,
            "activity": s.activity,
            "air_temp_c": t_air,
            "rh_pct": rh,
            "water_flow_l": getattr(self.sim, "_last_trough_intake_l", 0.0),
        }

    # ---- one sampling instant --------------------------------------------
    def sample(self) -> dict[str, float | None]:
        clean = self._clean()
        hour = self.sim.hour
        out: dict[str, float | None] = {}
        for name, spec in self.specs.items():
            val = clean[name]
            sigma = spec.sigma * self.noise_mult

            fault = next((f for f in self.faults
                          if f.channel == name and f.active(hour)), None)
            if fault is not None:
                if fault.kind == "dropout":
                    out[name] = None
                    continue
                if fault.kind == "stuck":
                    if fault._stuck_value is None:
                        fault._stuck_value = val + self.rng.normal(0, sigma)
                    out[name] = float(fault._stuck_value)
                    continue
                if fault.kind == "drift":
                    val = val + fault.magnitude * (hour - fault.start_h)
                elif fault.kind == "bias":
                    val = val + fault.magnitude

            if self.rng.random() < spec.dropout_p:
                out[name] = None
                continue
            noise = self.rng.normal(0.0, sigma)
            if self.rng.random() < spec.spike_p:
                noise += self.rng.normal(0.0, sigma * spec.spike_sigma_mult)
            out[name] = float(val + noise)
        return out

    # ---- costly confirmatory measurement ---------------------------------
    def verify(self) -> dict[str, float]:
        """Hands-on / bolus check: accurate core temperature + trough function test.

        This is the *information action* whose value the experiment measures.
        """
        self.verify_count += 1
        return {
            "core_temp_c": float(self.sim.state.t_core_c
                                 + self.rng.normal(0.0, VERIFY_CORE_SIGMA)),
            "trough_functional": float(self.sim.water_available()),
            "hydration_score": float(np.clip(self.sim.state.water_deficit
                                             + self.rng.normal(0, 0.004), 0.0, 0.2)),
        }
