"""Ground-truth calf and pen environment.

This module is the *simulated animal*. Its state is never exposed to the agent;
the agent only ever sees the output of :mod:`calftwin.sensors` filtered through
:mod:`calftwin.shadow`. Keeping the two apart is what makes the testbed a
falsifiable rig rather than a self-confirming narrative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import DT_H, BODY_WATER_FRAC, CalfParams, ActuatorParams


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
@dataclass
class Environment:
    """Pen micro-climate and resource availability."""
    t_air_base_c: float = 17.0
    t_air_amp_c: float = 5.0        # diurnal half-range
    peak_hour: float = 15.0
    rh_base_pct: float = 62.0
    solar_on: bool = True
    water_blocked_from_h: float | None = None
    water_blocked_to_h: float | None = None

    def air_temp(self, hour: float) -> float:
        return self.t_air_base_c + self.t_air_amp_c * math.sin(
            2 * math.pi * (hour - self.peak_hour + 6.0) / 24.0
        )

    def rh(self, hour: float) -> float:
        # relative humidity runs anti-phase with temperature
        return float(np.clip(self.rh_base_pct - 0.9 * (self.air_temp(hour) - self.t_air_base_c) * 2.2,
                             25.0, 98.0))

    def solar(self, hour: float) -> float:
        if not self.solar_on:
            return 0.0
        x = math.sin(math.pi * (hour - 6.0) / 12.0)
        return max(0.0, x) ** 1.5

    def water_blocked(self, hour: float) -> bool:
        if self.water_blocked_from_h is None:
            return False
        return self.water_blocked_from_h <= hour < (self.water_blocked_to_h or 24.0)


def thi(t_air_c: float, rh_pct: float) -> float:
    """Temperature-humidity index (NRC 1971 form), reported for interpretability."""
    return (1.8 * t_air_c + 32.0) - (0.55 - 0.0055 * rh_pct) * (1.8 * t_air_c - 26.0)


# ---------------------------------------------------------------------------
# hidden state
# ---------------------------------------------------------------------------
@dataclass
class CalfState:
    t_core_c: float = 38.65
    water_deficit: float = 0.004     # fraction of body mass
    fatigue: float = 0.05
    pyrogen: float = 0.0             # 0..1 endogenous fever driver
    resp_bpm: float = 30.0
    heart_bpm: float = 96.0
    activity: float = 0.35
    lying: bool = False

    def as_array(self) -> np.ndarray:
        return np.array([self.t_core_c, self.water_deficit, self.fatigue, self.pyrogen])


@dataclass
class PenEffects:
    """Effects currently applied to the pen by executed commands."""
    cooling_on: bool = False
    water_fixed: bool = False        # a maintenance fix has cleared the blockage
    treatment_active: bool = False   # vet / antipyretic response in progress


@dataclass
class CalfSimulator:
    """Discrete-time ground-truth simulator, 15-minute steps."""
    params: CalfParams = field(default_factory=CalfParams)
    env: Environment = field(default_factory=Environment)
    actuator: ActuatorParams = field(default_factory=ActuatorParams)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    state: CalfState = field(default_factory=CalfState)
    effects: PenEffects = field(default_factory=PenEffects)
    pyrogen_schedule: list[tuple[float, float]] = field(default_factory=list)
    step_idx: int = 0

    # -- derived, observable-through-sensors quantities ---------------------
    @property
    def hour(self) -> float:
        return (self.step_idx * DT_H) % 24.0

    def effective_air_temp(self) -> float:
        t = self.env.air_temp(self.hour)
        if self.effects.cooling_on:
            t -= self.actuator.cooling_delta_air_c
        return t

    def water_available(self) -> bool:
        if not self.env.water_blocked(self.hour):
            return True
        return self.effects.water_fixed

    def set_point(self) -> float:
        return self.params.t_set_c + self.params.pyrogen_set_gain_k * self.state.pyrogen

    # -- one step -----------------------------------------------------------
    def step(self) -> None:
        p, s = self.params, self.state
        hour = self.hour
        t_air = self.effective_air_temp()
        rh = self.env.rh(hour)

        # scheduled pyrogen (fever) injection
        for h_on, amount in self.pyrogen_schedule:
            if abs(hour - h_on) < DT_H / 2:
                s.pyrogen = float(np.clip(s.pyrogen + amount, 0.0, 1.0))

        # ---- behaviour: calves lie ~17 h/day, more at night ---------------
        p_lying = 0.78 if (hour < 6 or hour >= 21) else 0.55
        p_lying = float(np.clip(p_lying + 0.35 * s.fatigue, 0.0, 0.95))
        s.lying = bool(self.rng.random() < p_lying)
        s.activity = float(np.clip((0.08 if s.lying else 0.55)
                                   + self.rng.normal(0, 0.06)
                                   - 0.15 * s.fatigue, 0.0, 1.0))

        # ---- thermoregulatory targets -------------------------------------
        set_pt = self.set_point()
        rr_target = (p.rr_rest_bpm
                     + p.rr_gain_air * max(0.0, t_air - 22.0)
                     + p.rr_gain_core * max(0.0, s.t_core_c - set_pt)
                     + 6.0 * s.activity)
        rr_target = float(np.clip(rr_target, 12.0, p.rr_max_bpm))
        s.resp_bpm += (rr_target - s.resp_bpm) * (DT_H / p.rr_tau_h)

        hr_target = (p.hr_rest_bpm
                     + p.hr_gain_core * (s.t_core_c - p.t_set_c)
                     + p.hr_gain_dehyd * s.water_deficit
                     + p.hr_gain_act * s.activity
                     + 14.0 * s.pyrogen)
        s.heart_bpm += (hr_target - s.heart_bpm) * (DT_H / p.hr_tau_h)

        # ---- autonomic thermoregulation -----------------------------------
        # err > 0 means the calf is below its defended temperature.
        err = set_pt - s.t_core_c
        # peripheral vasomotion: constrict when cold, dilate when warm
        vaso = float(np.clip(1.0 - p.vaso_gain * err, p.vaso_min, p.vaso_max))
        # shivering / non-shivering thermogenesis when cold
        h_thermo = float(np.clip(p.thermo_gain_w_per_k * err, 0.0, p.thermo_max_w))

        # ---- heat balance --------------------------------------------------
        h_prod = (p.met_heat_w + p.act_heat_w * s.activity
                  + p.pyrogen_heat_w * s.pyrogen + h_thermo)
        h_solar = p.solar_gain_w * self.env.solar(hour) * (0.0 if self.effects.cooling_on else 1.0)
        k_sens = p.k_sensible_w_per_k * vaso * (self.actuator.cooling_conductance_mult
                                               if self.effects.cooling_on else 1.0)
        t_skin = s.t_core_c - p.skin_core_offset_k * vaso
        h_sens = k_sens * (t_skin - t_air)
        vpd = max(0.15, 1.0 - rh / 100.0)
        hydration_factor = float(np.clip(1.0 - 6.0 * s.water_deficit, 0.45, 1.0))
        # respiratory (panting) + cutaneous evaporative heat loss
        h_evap = (p.k_evap_w_per_bpm * s.resp_bpm
                  + p.k_skin_evap_w_per_k * max(0.0, t_air - 24.0)) * vpd * hydration_factor

        dq_w = h_prod + h_solar - h_sens - h_evap
        dt_core = (dq_w * DT_H * 3.6) / p.heat_capacity_kj_per_k   # W*h -> kJ
        s.t_core_c = float(np.clip(s.t_core_c + dt_core + self.rng.normal(0, 0.010),
                                   35.0, 43.0))

        # ---- water balance -------------------------------------------------
        tbw_l = p.bw_kg * BODY_WATER_FRAC
        loss_l = (p.loss_base_l_per_day
                  + p.loss_resp_l_per_bpm_day * s.resp_bpm
                  + p.loss_skin_l_per_k_day * max(0.0, t_air - 25.0)) * DT_H / 24.0
        intake_l = 0.0
        for mh in p.meal_hours:
            if abs(hour - mh) < DT_H / 2:
                intake_l += p.milk_l_per_meal
        if self.water_available() and s.water_deficit > p.thirst_threshold:
            thirst = float(np.clip(s.water_deficit / 0.03, 0.0, 1.6))
            drink = p.drink_gain_l_per_h * DT_H * thirst * (0.3 if s.lying else 1.0)
            intake_l += max(0.0, drink + self.rng.normal(0, 0.03))
        self._last_trough_intake_l = 0.0 if not self.water_available() else max(
            0.0, intake_l - sum(p.milk_l_per_meal for mh in p.meal_hours
                                if abs(hour - mh) < DT_H / 2))
        s.water_deficit = float(np.clip(s.water_deficit + (loss_l - intake_l) / tbw_l,
                                        0.0, 0.14))

        # ---- fatigue -------------------------------------------------------
        heat_strain = max(0.0, s.t_core_c - 39.2)
        dehyd_strain = max(0.0, s.water_deficit - 0.02)
        d_fat = (p.fatigue_gain_heat * heat_strain
                 + p.fatigue_gain_dehyd * dehyd_strain) * DT_H
        if heat_strain <= 0.0 and dehyd_strain <= 0.0 and s.lying:
            d_fat -= p.fatigue_recovery_per_h * DT_H
        s.fatigue = float(np.clip(s.fatigue + d_fat, 0.0, 1.0))

        # ---- pyrogen dynamics ---------------------------------------------
        decay = p.pyrogen_decay_per_h + (p.treatment_clearance_per_h
                                         if self.effects.treatment_active else 0.0)
        s.pyrogen = float(np.clip(s.pyrogen * math.exp(-decay * DT_H), 0.0, 1.0))

        self.step_idx += 1

    # -- introspection for logging/metrics only -----------------------------
    def truth_record(self) -> dict[str, float]:
        hour = self.hour
        t_air = self.effective_air_temp()
        rh = self.env.rh(hour)
        return {
            "step": self.step_idx,
            "hour": hour,
            "t_core_c": self.state.t_core_c,
            "water_deficit": self.state.water_deficit,
            "fatigue": self.state.fatigue,
            "pyrogen": self.state.pyrogen,
            "resp_bpm": self.state.resp_bpm,
            "heart_bpm": self.state.heart_bpm,
            "activity": self.state.activity,
            "lying": float(self.state.lying),
            "t_air_c": t_air,
            "rh_pct": rh,
            "thi": thi(t_air, rh),
            "water_available": float(self.water_available()),
            "cooling_on": float(self.effects.cooling_on),
            "treatment_active": float(self.effects.treatment_active),
            "trough_intake_l": getattr(self, "_last_trough_intake_l", 0.0),
        }
