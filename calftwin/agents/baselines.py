"""Non-agentic baselines.

* ``ShadowOnlyAgent``  -- a digital shadow: it estimates and reports, never acts.
* ``ThresholdAgent``   -- the incumbent in commercial precision-livestock systems:
  fixed alarm thresholds on raw sensor values, no state estimation, no
  verification, no forecast.
* ``FullInfoAgent``    -- the same rule structure as the bounded controller but
  evaluated on the hidden state. It is a *perfect-observation reference*, not an
  optimal controller: it shows how much of the gap to ideal behaviour is caused
  by imperfect sensing rather than by the policy. It is not deployable.
"""
from __future__ import annotations

from typing import Any

from ..twin import DigitalTwin
from .base import decision


class ShadowOnlyAgent:
    name = "shadow_only"

    def decide(self, twin: DigitalTwin) -> dict[str, Any]:
        twin.get_state()
        return decision("observe", "Monitoring only; this configuration has no "
                                   "command path to the animal.", 1)


class ThresholdAgent:
    """Fixed thresholds on raw measurements, as in a conventional alarm system."""
    name = "threshold"

    def __init__(self, ear_on_c: float = 38.6, ear_off_c: float = 37.9,
                 resp_on_bpm: float = 75.0, dry_steps: int = 8) -> None:
        self.ear_on_c = ear_on_c
        self.ear_off_c = ear_off_c
        self.resp_on_bpm = resp_on_bpm
        self.dry_steps = dry_steps
        self._dry = 0
        self._water_fixed_sent = False

    def decide(self, twin: DigitalTwin) -> dict[str, Any]:
        st = twin.get_state()
        m = st["measured"]
        ear = m.get("ear_temp_c")
        resp = m.get("resp_bpm")
        flow = m.get("water_flow_l")
        cooling = st["pen"]["cooling_on"]

        self._dry = self._dry + 1 if (flow is not None and flow <= 0.05) else 0

        if self._dry >= self.dry_steps and not self._water_fixed_sent:
            self._water_fixed_sent = True
            twin.propose_action("restore_water",
                                f"No trough flow for {self._dry} consecutive samples.")
            return decision("restore_water", "Dry-trough alarm threshold reached.", 2)

        hot = ((ear is not None and ear > self.ear_on_c)
               or (resp is not None and resp > self.resp_on_bpm))
        if hot and not cooling:
            twin.propose_action("cooling_on",
                                f"Alarm: ear temp {ear}, respiration {resp}.")
            return decision("cooling_on", "Raw-threshold heat alarm.", 2)
        if cooling and ear is not None and ear < self.ear_off_c:
            twin.propose_action("cooling_off", f"Ear temp back to {ear}.")
            return decision("cooling_off", "Raw-threshold release.", 2)
        return decision("observe", "No threshold exceeded.", 1)


class FullInfoAgent:
    """Perfect-observation reference policy. Reads hidden state; not deployable."""
    name = "full_info"

    def __init__(self, min_dwell_steps: int = 4) -> None:
        self.min_dwell_steps = min_dwell_steps
        self._last_switch = -999
        self._last_check = -999
        self._last_fix = -999

    def decide(self, twin: DigitalTwin) -> dict[str, Any]:
        s = twin.sim.state
        step = twin.step_idx
        cooling = twin.sim.effects.cooling_on
        water_ok = twin.sim.water_available()
        treated = twin.sim.effects.treatment_active
        dwell = (step - self._last_switch) >= self.min_dwell_steps

        if not water_ok and s.water_deficit > 0.010 and (step - self._last_fix) > 16:
            self._last_fix = step
            twin.propose_action("restore_water", "FULL-INFO: trough is blocked.")
            return decision("restore_water", "full information", 1)
        if s.pyrogen > 0.25 and not treated and (step - self._last_check) > 24:
            self._last_check = step
            twin.propose_action("flag_human_check", "FULL-INFO: pyrogen-driven fever.")
            return decision("flag_human_check", "full information", 1)
        if s.t_core_c > 38.95 and not cooling and dwell:
            self._last_switch = step
            twin.propose_action("cooling_on", "FULL-INFO: rising thermal load.")
            return decision("cooling_on", "full information", 1)
        if s.t_core_c < 38.70 and cooling and dwell:
            self._last_switch = step
            twin.propose_action("cooling_off", "FULL-INFO: thermal load resolved.")
            return decision("cooling_off", "full information", 1)
        return decision("observe", "full information: within bounds", 1)
