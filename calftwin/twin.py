"""The digital twin: shadow + estimator + forecast + command path.

This class defines the *entire* interface available to an agent. An agent can
only read what ``get_state``, ``get_history`` and ``forecast_options`` return, can
only buy information through ``verify_vitals``, and can only act through
``propose_action``. It has no handle on :class:`~calftwin.calf.CalfSimulator`, so
it cannot see the hidden state or the future weather.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .actuators import ACTIONS, ActuatorBus, Command
from .calf import CalfSimulator, thi
from .config import DT_H, CostWeights
from .estimator import ParticleFilter, TwinModelParams
from .sensors import SensorArray
from .shadow import DigitalShadow


class DigitalTwin:
    def __init__(self, sim: CalfSimulator, sensors: SensorArray, bus: ActuatorBus,
                 costs: CostWeights, seed: int = 0, n_particles: int = 400,
                 forecast_noise_c: float = 1.0) -> None:
        self.sim = sim
        self.sensors = sensors
        self.bus = bus
        self.costs = costs
        self.shadow = DigitalShadow(calf_id=sim.params.calf_id)
        self.pf = ParticleFilter(n=n_particles, seed=seed + 17,
                                 params=TwinModelParams())
        self.rng = np.random.default_rng(seed + 99)
        self.forecast_noise_c = forecast_noise_c
        self.step_idx = 0
        self.hour = 0.0
        self.n_verify = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []

    # ================= data path (animal -> twin) =========================
    def observe(self) -> None:
        """Pull one sensor frame, store it in the shadow, update the estimate."""
        self.step_idx = self.sim.step_idx
        self.hour = self.sim.hour
        obs = self.sensors.sample()
        flags = self.shadow.ingest(self.step_idx, self.hour, obs,
                                   self.sim.params.calf_id)
        self.pf.update(obs, {k: v.to_dict() for k, v in flags.items()},
                       self.hour, self.sim.effects.cooling_on,
                       self.sim.effects.treatment_active)

    # ================= weather forecast (imperfect) ========================
    def _air_forecast(self, horizon_steps: int) -> list[float]:
        path = []
        for k in range(1, horizon_steps + 1):
            h = (self.hour + k * DT_H) % 24.0
            path.append(self.sim.env.air_temp(h)
                        + float(self.rng.normal(0.0, self.forecast_noise_c)))
        return path

    # ================= tools exposed to the agent ==========================
    def _record(self, tool: str, args: dict[str, Any], cost: float = 0.0) -> None:
        self.tool_calls.append({"step": self.step_idx, "hour": round(self.hour, 2),
                                "tool": tool, "args": args, "cost": cost})

    def get_state(self) -> dict[str, Any]:
        """Current twin estimate of latent state, with uncertainty and data quality."""
        self._record("get_state", {})
        est = self.pf.estimate().to_dict()
        latest = self.shadow.latest() or {"obs": {}, "flags": {}}
        obs = latest["obs"]
        t_air = obs.get("air_temp_c")
        rh = obs.get("rh_pct")
        summary = self.shadow.summary(8)
        suspect = [ch for ch, v in summary.get("channels", {}).items() if v["suspect"]]
        missing = [ch for ch, v in summary.get("channels", {}).items()
                   if v["missing_frac"] > 0.4]
        return {
            "calf_id": self.shadow.calf_id,
            "hour_of_day": round(self.hour, 2),
            "estimate": est,
            "measured": {k: (round(v, 3) if isinstance(v, (int, float)) else v)
                         for k, v in obs.items()},
            "thi": round(thi(t_air, rh), 1) if (t_air is not None and rh is not None) else None,
            "data_quality": {"suspect_channels": suspect,
                             "mostly_missing_channels": missing,
                             "gaps": summary.get("gaps", 0),
                             "duplicates_rejected": summary.get("duplicates_rejected", 0)},
            "pen": {"cooling_on": bool(self.sim.effects.cooling_on),
                    "clinical_response_active": bool(self.sim.effects.treatment_active)},
            "recent_commands": self.bus.recent(4),
            "verifications_used": self.n_verify,
        }

    def get_history(self, channel: str, hours: float = 6.0) -> dict[str, Any]:
        """Recent measured series for one channel (downsampled to ~12 points)."""
        self._record("get_history", {"channel": channel, "hours": hours})
        if channel not in self.shadow.specs:
            return {"error": f"unknown channel {channel!r}",
                    "available": list(self.shadow.specs)}
        n = max(2, int(hours / DT_H))
        hs, vs = self.shadow.series(channel, n)
        if not vs:
            return {"channel": channel, "n": 0, "note": "no valid samples in window"}
        stride = max(1, len(vs) // 12)
        return {"channel": channel, "hours": hours, "n_samples": len(vs),
                "hours_axis": [round(h, 2) for h in hs[::stride]],
                "values": [round(v, 3) for v in vs[::stride]],
                "trend_per_h": round((vs[-1] - vs[0]) / max(1e-6, (len(vs) - 1) * DT_H), 3),
                "min": round(min(vs), 3), "max": round(max(vs), 3)}

    def verify_vitals(self) -> dict[str, Any]:
        """Buy a confirmatory hands-on measurement. Costs information budget."""
        self.n_verify += 1
        self._record("verify_vitals", {}, cost=self.costs.c_verify)
        v = self.sensors.verify()
        before = self.pf.estimate()
        self.pf.assimilate_verification(v)
        after = self.pf.estimate()
        return {"confirmed_core_temp_c": round(v["core_temp_c"], 2),
                "trough_functional": bool(v["trough_functional"] > 0.5),
                "hydration_score": round(v["hydration_score"], 4),
                "estimate_shift_c": round(after.t_core_c - before.t_core_c, 3),
                "uncertainty_before_sd": round(before.t_core_sd, 3),
                "uncertainty_after_sd": round(after.t_core_sd, 3),
                "cost": self.costs.c_verify}

    def forecast_options(self, horizon_h: float = 3.0,
                         options: list[str] | None = None) -> dict[str, Any]:
        """Predicted thermal outcome over the horizon under candidate actions."""
        options = options or ["observe", "cooling_on", "restore_water"]
        self._record("forecast_options", {"horizon_h": horizon_h, "options": options})
        steps = max(1, int(horizon_h / DT_H))
        air = self._air_forecast(steps)
        act = 0.35
        out: dict[str, Any] = {"horizon_h": horizon_h,
                               "air_temp_forecast_c": [round(a, 1) for a in air[::max(1, steps // 6)]],
                               "options": {}}
        for opt in options:
            if opt not in ACTIONS:
                out["options"][opt] = {"error": "not an allowed action"}
                continue
            cooling = (opt == "cooling_on") or (self.sim.effects.cooling_on and opt != "cooling_off")
            water_ok = 0.95 if opt == "restore_water" else None
            treat = self.sim.effects.treatment_active or (opt == "flag_human_check")
            f = self.pf.forecast(steps, self.hour, air, act, cooling, treat, water_ok)
            f.pop("core_trajectory", None)
            out["options"][opt] = {k: round(v, 4) for k, v in f.items()}
        return out

    def propose_action(self, action: str, rationale: str,
                       evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        """Submit a command to the approval gate. Execution is delayed and fallible."""
        self._record("propose_action", {"action": action})
        if action not in ACTIONS:
            return {"error": f"action {action!r} not permitted",
                    "allowed_actions": list(ACTIONS)}
        cmd: Command = self.bus.issue(action, self.step_idx, rationale, evidence)
        return {"cmd_id": cmd.cmd_id, "action": cmd.action, "status": cmd.status,
                "scheduled_execute_step": cmd.execute_step,
                "note": ("no physical change" if action == "observe"
                         else "execution is delayed and may fail; re-check state later")}

    # ================= per-step logging ====================================
    def log_step(self, truth: dict[str, float], decision: dict[str, Any] | None) -> None:
        est = self.pf.estimate()
        rec = {**{f"true_{k}": v for k, v in truth.items()}, **est.to_dict(),
               "n_verify": self.n_verify}
        if decision:
            rec["action"] = decision.get("action")
            rec["agent_reasoning"] = decision.get("reasoning", "")[:500]
            rec["n_tool_calls"] = decision.get("n_tool_calls", 0)
        self.trace.append(rec)
