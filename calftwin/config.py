"""Central configuration: physical constants, sensor specs, costs, weights.

All numeric values are *illustrative but physiologically plausible* parameters for a
~6-week-old dairy calf. They are not calibrated against a real herd; see README
"Validation status".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

DT_H = 0.25                 # simulation step, hours (15 min)
STEPS_PER_DAY = 96
BODY_WATER_FRAC = 0.70      # fraction of BW that is water


@dataclass
class CalfParams:
    """Ground-truth biophysical parameters (hidden from the agent)."""
    calf_id: str = "CALF-001"
    age_days: int = 42
    bw_kg: float = 65.0

    # --- heat balance -------------------------------------------------------
    heat_capacity_kj_per_k: float = 3.5 * 65.0   # C = c_p * BW
    met_heat_w: float = 106.0                    # resting metabolic heat production
    act_heat_w: float = 55.0                     # extra heat when active/standing
    k_sensible_w_per_k: float = 6.4              # convective+radiative conductance
    skin_core_offset_k: float = 1.5              # T_skin = T_core - offset
    k_evap_w_per_bpm: float = 1.05               # respiratory evaporative cooling
    k_skin_evap_w_per_k: float = 7.0             # cutaneous evaporation above 24 C
    solar_gain_w: float = 45.0                   # midday radiant gain (outdoor pen)
    t_set_c: float = 38.75                       # hypothalamic set point

    # --- autonomic thermoregulation ----------------------------------------
    vaso_gain: float = 2.2                       # vasomotor response per K of error
    vaso_min: float = 0.50                       # full vasoconstriction (cold)
    vaso_max: float = 1.55                       # full vasodilation (warm)
    thermo_gain_w_per_k: float = 260.0           # shivering thermogenesis
    thermo_max_w: float = 130.0
    pyrogen_heat_w: float = 22.0

    # --- respiration / heart rate ------------------------------------------
    rr_rest_bpm: float = 30.0
    rr_gain_air: float = 3.2                     # per K above 22 C
    rr_gain_core: float = 70.0                   # per K above set point
    rr_max_bpm: float = 140.0
    rr_tau_h: float = 0.33
    hr_rest_bpm: float = 95.0
    hr_gain_core: float = 26.0
    hr_gain_dehyd: float = 520.0                 # tachycardia per unit water deficit
    hr_gain_act: float = 22.0
    hr_tau_h: float = 0.17

    # --- water balance ------------------------------------------------------
    loss_base_l_per_day: float = 3.5             # urine + faecal water
    loss_resp_l_per_bpm_day: float = 0.022       # respiratory water loss
    loss_skin_l_per_k_day: float = 0.075         # cutaneous loss per K above 25 C
    milk_l_per_meal: float = 3.0
    meal_hours: tuple[float, ...] = (7.0, 16.0)
    drink_gain_l_per_h: float = 2.4              # max voluntary water intake rate
    thirst_threshold: float = 0.004              # deficit at which drinking starts

    # --- fatigue ------------------------------------------------------------
    fatigue_gain_heat: float = 0.55              # per (K above 39.2) per hour
    fatigue_gain_dehyd: float = 3.0              # per unit deficit per hour
    fatigue_recovery_per_h: float = 0.085

    # --- pyrogen (fever) ----------------------------------------------------
    pyrogen_set_gain_k: float = 1.35             # set-point shift at pyrogen = 1
    pyrogen_decay_per_h: float = 0.045
    treatment_clearance_per_h: float = 0.45      # antipyretic / vet response

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SensorSpec:
    name: str
    sigma: float                 # Gaussian noise sd in engineering units
    dropout_p: float = 0.01      # probability of a missing sample
    spike_p: float = 0.004       # probability of a gross outlier
    spike_sigma_mult: float = 12.0
    lo: float = -1e9             # plausible range for the shadow's range check
    hi: float = 1e9


DEFAULT_SENSORS: dict[str, SensorSpec] = {
    "ear_temp_c":  SensorSpec("ear_temp_c",  sigma=0.25, dropout_p=0.015, lo=15.0, hi=42.0),
    "resp_bpm":    SensorSpec("resp_bpm",    sigma=4.0,  dropout_p=0.02,  lo=5.0,  hi=200.0),
    "heart_bpm":   SensorSpec("heart_bpm",   sigma=6.0,  dropout_p=0.02,  lo=40.0, hi=220.0),
    "activity":    SensorSpec("activity",    sigma=0.05, dropout_p=0.01,  lo=0.0,  hi=1.0),
    "air_temp_c":  SensorSpec("air_temp_c",  sigma=0.30, dropout_p=0.005, lo=-30.0, hi=55.0),
    "rh_pct":      SensorSpec("rh_pct",      sigma=2.0,  dropout_p=0.005, lo=0.0,  hi=100.0),
    "water_flow_l": SensorSpec("water_flow_l", sigma=0.04, dropout_p=0.01, lo=0.0, hi=5.0),
}

# Confirmatory ("hands-on") measurement obtained only via the verify tool.
VERIFY_CORE_SIGMA = 0.08


@dataclass
class ActuatorParams:
    approval_p: float = 0.97       # simulated stockperson / safety-gate approval
    latency_steps: int = 2         # 30 min from decision to effect
    exec_success_p: float = 0.95
    cooling_delta_air_c: float = 4.0
    cooling_conductance_mult: float = 1.40
    water_fix_success_p: float = 0.90
    human_check_latency_steps: int = 4


@dataclass
class CostWeights:
    """Illustrative weights. Not monetary units and not a validated welfare score."""
    w_heat: float = 1.0        # per degree-hour above the heat threshold
    w_dehyd: float = 1.0       # per (100 x deficit)-hour above threshold
    w_fatigue: float = 0.6     # per fatigue-hour above threshold
    heat_threshold_c: float = 39.2
    dehyd_threshold: float = 0.02
    fatigue_threshold: float = 0.30

    c_cooling_per_h: float = 0.06
    c_water_fix: float = 0.10
    c_human_check: float = 0.55
    c_verify: float = 0.12
    c_llm_call: float = 0.0    # accounted separately in token terms


@dataclass
class RunConfig:
    scenario: str = "normal"
    seed: int = 0
    days: int = 1
    agent: str = "bounded"
    calf: CalfParams = field(default_factory=CalfParams)
    actuator: ActuatorParams = field(default_factory=ActuatorParams)
    costs: CostWeights = field(default_factory=CostWeights)
    sensor_noise_mult: float = 1.0
    decide_every: int = 2           # agent decision cadence in steps (30 min)
