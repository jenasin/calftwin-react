"""Digital shadow: the one-way data path from animal to digital representation.

The shadow does no inference about physiological state. It owns identity,
timestamps, ordering and *data quality*. Keeping it separate from the state
estimator means an experiment can attribute a failure either to the data path or
to the inference layer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DEFAULT_SENSORS, SensorSpec


@dataclass
class QualityFlags:
    missing: bool = False
    out_of_range: bool = False
    frozen: bool = False       # identical value repeated -> suspected stuck sensor
    jump: bool = False         # implausible rate of change
    stale_h: float = 0.0

    @property
    def suspect(self) -> bool:
        return self.out_of_range or self.frozen or self.jump

    def to_dict(self) -> dict[str, Any]:
        return {"missing": self.missing, "out_of_range": self.out_of_range,
                "frozen": self.frozen, "jump": self.jump,
                "stale_h": round(self.stale_h, 2), "suspect": self.suspect}


# maximum plausible change per 15-min step, per channel
MAX_RATE = {"ear_temp_c": 1.2, "resp_bpm": 45.0, "heart_bpm": 45.0,
            "activity": 0.9, "air_temp_c": 2.5, "rh_pct": 18.0, "water_flow_l": 1.5}
FROZEN_N = 6   # >=6 identical samples (90 min) is not physiology


@dataclass
class DigitalShadow:
    """Append-only observation store for one identified animal."""
    calf_id: str
    specs: dict[str, SensorSpec] = field(default_factory=lambda: dict(DEFAULT_SENSORS))
    maxlen: int = 4000
    records: deque = field(init=False)
    _last_seen: dict[str, tuple[int, float]] = field(default_factory=dict)
    _repeat: dict[str, int] = field(default_factory=dict)
    n_ingested: int = 0
    n_rejected_duplicate: int = 0
    n_gaps: int = 0

    def __post_init__(self) -> None:
        self.records = deque(maxlen=self.maxlen)

    # ---- write path -------------------------------------------------------
    def ingest(self, step: int, hour: float, obs: dict[str, float | None],
               calf_id: str | None = None) -> dict[str, QualityFlags]:
        if calf_id is not None and calf_id != self.calf_id:
            raise ValueError(f"identity mismatch: {calf_id} != {self.calf_id}")
        if self.records and step <= self.records[-1]["step"]:
            self.n_rejected_duplicate += 1
            return {}
        if self.records and step > self.records[-1]["step"] + 1:
            self.n_gaps += 1

        flags: dict[str, QualityFlags] = {}
        for name, spec in self.specs.items():
            v = obs.get(name)
            f = QualityFlags()
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                f.missing = True
                prev = self._last_seen.get(name)
                f.stale_h = (step - prev[0]) * 0.25 if prev else 0.0
                self._repeat[name] = 0
            else:
                if not (spec.lo <= v <= spec.hi):
                    f.out_of_range = True
                prev = self._last_seen.get(name)
                if prev is not None:
                    dstep = max(1, step - prev[0])
                    if abs(v - prev[1]) > MAX_RATE.get(name, 1e9) * dstep:
                        f.jump = True
                    if abs(v - prev[1]) < 1e-9:
                        self._repeat[name] = self._repeat.get(name, 0) + 1
                    else:
                        self._repeat[name] = 0
                if self._repeat.get(name, 0) >= FROZEN_N:
                    f.frozen = True
                self._last_seen[name] = (step, float(v))
            flags[name] = f

        self.records.append({"step": step, "hour": hour, "calf_id": self.calf_id,
                             "obs": dict(obs),
                             "flags": {k: v.to_dict() for k, v in flags.items()}})
        self.n_ingested += 1
        return flags

    # ---- read path --------------------------------------------------------
    def latest(self) -> dict[str, Any] | None:
        return self.records[-1] if self.records else None

    def window(self, n: int) -> list[dict[str, Any]]:
        return list(self.records)[-n:]

    def series(self, channel: str, n: int = 96) -> tuple[list[float], list[float]]:
        hs, vs = [], []
        for r in self.window(n):
            v = r["obs"].get(channel)
            if v is not None and np.isfinite(v):
                hs.append(r["hour"])
                vs.append(float(v))
        return hs, vs

    def summary(self, n: int = 8) -> dict[str, Any]:
        """Compact, agent-facing description of recent data quality."""
        w = self.window(n)
        if not w:
            return {"calf_id": self.calf_id, "n": 0}
        out: dict[str, Any] = {"calf_id": self.calf_id, "n_records": len(self.records),
                               "window_steps": len(w),
                               "duplicates_rejected": self.n_rejected_duplicate,
                               "gaps": self.n_gaps, "channels": {}}
        for ch in self.specs:
            vals = [r["obs"][ch] for r in w if r["obs"].get(ch) is not None]
            fl = [r["flags"][ch] for r in w if ch in r["flags"]]
            out["channels"][ch] = {
                "last": round(vals[-1], 3) if vals else None,
                "mean": round(float(np.mean(vals)), 3) if vals else None,
                "missing_frac": round(sum(f["missing"] for f in fl) / max(1, len(fl)), 3),
                "suspect": any(f["suspect"] for f in fl),
                "frozen": any(f["frozen"] for f in fl),
            }
        return out
