"""Twin-side state estimator.

A bootstrap particle filter over the latent vector

    x = [t_core_c, water_deficit, fatigue, pyrogen, resp_bpm]

plus a scalar Bayesian belief that the water trough is functional.

The internal dynamics deliberately differ from :mod:`calftwin.calf` (different
conductance, metabolic and gain constants, no solar term). The twin therefore
suffers structural model mismatch, as any deployed twin does. Observations that
the shadow flagged as suspect are down-weighted rather than trusted, and the
likelihood is a Gaussian/broad mixture so that spikes cannot capture the filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DT_H, BODY_WATER_FRAC, VERIFY_CORE_SIGMA

# latent indices
TC, WD, FT, PY, RR = 0, 1, 2, 3, 4
N_STATE = 5

# inflated measurement sigmas: sensor noise + allowance for model error
SIG_EAR = 0.40
SIG_RESP = 6.0
SIG_HEART = 9.0
SIG_FLOW = 0.10
OUTLIER_W = 0.03          # weight of the broad mixture component
OUTLIER_MULT = 10.0
FLOW_WINDOW_STEPS = 8     # 2 h rolling window for the water-balance residual
# A flow meter cannot read negative, so Gaussian noise on a true zero rectifies to
# a positive mean and the twin would credit the calf with water it never drank.
# Soft-thresholding at k sigma removes that bias at the price of ignoring very
# small genuine draughts.
FLOW_SHRINK_K = float(__import__("os").environ.get("CALFTWIN_FLOW_SHRINK_K", "0.5"))
FLOW_SIGMA = 0.04


@dataclass
class TwinModelParams:
    """Twin's *belief* about the animal -- intentionally not the truth."""
    t_set_c: float = 38.75
    pyrogen_set_gain_k: float = 1.30
    met_heat_w: float = 100.0
    act_heat_w: float = 50.0
    pyrogen_heat_w: float = 20.0
    k_sensible_w_per_k: float = 6.0
    skin_core_offset_k: float = 1.5
    k_evap_w_per_bpm: float = 1.00
    k_skin_evap_w_per_k: float = 6.5
    vaso_gain: float = 2.0
    vaso_min: float = 0.55
    vaso_max: float = 1.50
    thermo_gain_w_per_k: float = 240.0
    thermo_max_w: float = 130.0
    heat_capacity_kj_per_k: float = 225.0
    rr_rest_bpm: float = 30.0
    rr_gain_air: float = 3.0
    rr_gain_core: float = 65.0
    rr_tau_h: float = 0.35
    hr_rest_bpm: float = 95.0
    hr_gain_core: float = 26.0
    hr_gain_dehyd: float = 500.0
    hr_gain_act: float = 22.0
    hr_gain_pyr: float = 14.0
    loss_base_l_per_day: float = 3.6
    loss_resp_l_per_bpm_day: float = 0.021
    milk_l_per_meal: float = 3.0
    meal_hours: tuple[float, ...] = (7.0, 16.0)
    drink_gain_l_per_h: float = 2.4
    thirst_threshold: float = 0.004
    fatigue_gain_heat: float = 0.55
    fatigue_gain_dehyd: float = 3.0
    fatigue_recovery_per_h: float = 0.085
    pyrogen_decay_per_h: float = 0.05
    treatment_clearance_per_h: float = 0.45
    cooling_delta_air_c: float = 4.0
    cooling_conductance_mult: float = 1.35
    bw_kg: float = 65.0


# process noise (sd per step)
Q = np.array([0.120, 0.0018, 0.010, 0.035, 3.0])

# Fever onset is a regime change, not a random walk: a small fraction of
# particles receives a large pyrogen perturbation each step so that the filter
# can follow a step change instead of crawling towards it.
JUMP_P = 0.04
JUMP_PYROGEN_SD = 0.45


@dataclass
class StateEstimate:
    t_core_c: float
    t_core_sd: float
    water_deficit: float
    water_deficit_sd: float
    fatigue: float
    fatigue_sd: float
    pyrogen: float
    pyrogen_sd: float
    resp_bpm: float
    p_trough_ok: float
    ess: float                     # effective sample size (filter health)
    p_hyperthermia: float          # P(t_core > 39.2)
    p_dehydrated: float            # P(water_deficit > 0.04)
    p_fever_driven: float          # P(pyrogen > 0.25)

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class ParticleFilter:
    def __init__(self, n: int = 400, seed: int = 0,
                 params: TwinModelParams | None = None) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.p = params or TwinModelParams()
        self.x = np.zeros((n, N_STATE))
        self.x[:, TC] = 38.7 + self.rng.normal(0, 0.20, n)
        self.x[:, WD] = np.clip(0.005 + self.rng.normal(0, 0.004, n), 0.0, 0.14)
        self.x[:, FT] = np.clip(0.05 + self.rng.normal(0, 0.03, n), 0.0, 1.0)
        self.x[:, PY] = np.clip(np.abs(self.rng.normal(0, 0.05, n)), 0.0, 1.0)
        self.x[:, RR] = 30.0 + self.rng.normal(0, 4.0, n)
        self.w = np.full(n, 1.0 / n)
        self.p_trough_ok = 0.97
        self._no_flow_steps = 0
        self._flow_window: list[float] = []
        self._loss_window: list[float] = []

    # ---- dynamics ---------------------------------------------------------
    def _propagate(self, x: np.ndarray, hour: float, t_air_obs: float,
                   activity_obs: float, cooling: bool, treatment: bool,
                   water_ok_p: float, add_noise: bool = True,
                   measured_intake_l: float | None = None,
                   rh_obs: float = 60.0) -> np.ndarray:
        p = self.p
        x = x.copy()
        tc, wd, ft, py, rr = x[:, TC], x[:, WD], x[:, FT], x[:, PY], x[:, RR]

        set_pt = p.t_set_c + p.pyrogen_set_gain_k * py
        err = set_pt - tc
        vaso = np.clip(1.0 - p.vaso_gain * err, p.vaso_min, p.vaso_max)
        h_thermo = np.clip(p.thermo_gain_w_per_k * err, 0.0, p.thermo_max_w)

        t_air_eff = t_air_obs - (p.cooling_delta_air_c if cooling else 0.0)
        k = p.k_sensible_w_per_k * vaso * (p.cooling_conductance_mult if cooling else 1.0)
        h_sens = k * ((tc - p.skin_core_offset_k * vaso) - t_air_eff)
        # Humidity is measured, so use it rather than a constant proxy: a fixed
        # vapour-pressure deficit biases evaporative heat loss and the bias leaks
        # into the water-deficit estimate through the shared heart-rate channel.
        vpd = max(0.15, 1.0 - rh_obs / 100.0)
        hyd = np.clip(1.0 - 6.0 * wd, 0.45, 1.0)
        h_evap = (p.k_evap_w_per_bpm * rr
                  + p.k_skin_evap_w_per_k * np.maximum(0.0, t_air_eff - 24.0)) * vpd * hyd
        h_prod = p.met_heat_w + p.act_heat_w * activity_obs + p.pyrogen_heat_w * py + h_thermo

        dq = h_prod - h_sens - h_evap
        tc_new = tc + dq * DT_H * 3.6 / p.heat_capacity_kj_per_k

        rr_target = np.clip(p.rr_rest_bpm + p.rr_gain_air * np.maximum(0.0, t_air_eff - 22.0)
                            + p.rr_gain_core * np.maximum(0.0, tc - set_pt)
                            + 6.0 * activity_obs, 12.0, 145.0)
        rr_new = rr + (rr_target - rr) * (DT_H / p.rr_tau_h)

        tbw = p.bw_kg * BODY_WATER_FRAC
        loss = (p.loss_base_l_per_day + p.loss_resp_l_per_bpm_day * rr) * DT_H / 24.0
        intake = np.zeros_like(wd)
        for mh in p.meal_hours:
            if abs(hour - mh) < DT_H / 2:
                intake = intake + p.milk_l_per_meal
        if measured_intake_l is not None:
            # Trough flow is measured, so use it directly rather than predicting
            # how much the calf chose to drink. p_trough_ok then serves its proper
            # role: explaining *why* flow is absent, not standing in for it.
            intake = intake + measured_intake_l
        else:
            thirsty = np.clip(wd / 0.03, 0.0, 1.6) * (wd > p.thirst_threshold)
            intake = intake + water_ok_p * p.drink_gain_l_per_h * DT_H * thirsty * 0.75
        wd_new = np.clip(wd + (loss - intake) / tbw, 0.0, 0.14)

        heat_strain = np.maximum(0.0, tc - 39.2)
        dehyd_strain = np.maximum(0.0, wd - 0.02)
        d_ft = (p.fatigue_gain_heat * heat_strain + p.fatigue_gain_dehyd * dehyd_strain) * DT_H
        d_ft = d_ft - p.fatigue_recovery_per_h * DT_H * ((heat_strain <= 0) & (dehyd_strain <= 0))
        ft_new = np.clip(ft + d_ft, 0.0, 1.0)

        decay = p.pyrogen_decay_per_h + (p.treatment_clearance_per_h if treatment else 0.0)
        py_new = np.clip(py * np.exp(-decay * DT_H), 0.0, 1.0)

        out = np.stack([tc_new, wd_new, ft_new, py_new, rr_new], axis=1)
        if add_noise:
            out += self.rng.normal(0.0, 1.0, out.shape) * Q
            jump = self.rng.random(out.shape[0]) < JUMP_P
            if jump.any():
                out[jump, PY] += self.rng.normal(0.0, JUMP_PYROGEN_SD, int(jump.sum()))
            out[:, WD] = np.clip(out[:, WD], 0.0, 0.14)
            out[:, FT] = np.clip(out[:, FT], 0.0, 1.0)
            out[:, PY] = np.clip(out[:, PY], 0.0, 1.0)
            out[:, RR] = np.clip(out[:, RR], 10.0, 150.0)
        return out

    # ---- likelihood -------------------------------------------------------
    @staticmethod
    def _robust_logpdf(resid: np.ndarray, sigma: float) -> np.ndarray:
        core = np.exp(-0.5 * (resid / sigma) ** 2) / sigma
        broad = np.exp(-0.5 * (resid / (sigma * OUTLIER_MULT)) ** 2) / (sigma * OUTLIER_MULT)
        return np.log((1 - OUTLIER_W) * core + OUTLIER_W * broad + 1e-300)

    def update(self, obs: dict[str, float | None], flags: dict[str, Any],
               hour: float, cooling: bool, treatment: bool) -> None:
        t_air = obs.get("air_temp_c")
        t_air = float(t_air) if t_air is not None else 18.0
        act = obs.get("activity")
        act = float(np.clip(act, 0.0, 1.0)) if act is not None else 0.3
        rh = obs.get("rh_pct")
        rh = float(np.clip(rh, 5.0, 100.0)) if rh is not None else 60.0
        self._last_rh = rh

        flow_obs = obs.get("water_flow_l")
        flow_flag = flags.get("water_flow_l", {})
        flow_flag = flow_flag if isinstance(flow_flag, dict) else flow_flag.to_dict()
        measured_intake = None
        if (flow_obs is not None and np.isfinite(flow_obs)
                and not flow_flag.get("suspect", False)):
            measured_intake = max(0.0, float(flow_obs) - FLOW_SHRINK_K * FLOW_SIGMA)
        self.x = self._propagate(self.x, hour, t_air, act, cooling, treatment,
                                 self.p_trough_ok, measured_intake_l=measured_intake,
                                 rh_obs=rh)

        logw = np.log(self.w + 1e-300)

        def usable(ch: str) -> bool:
            v = obs.get(ch)
            f = flags.get(ch, {})
            f = f if isinstance(f, dict) else f.to_dict()
            return v is not None and np.isfinite(v) and not f.get("suspect", False)

        if usable("ear_temp_c"):
            pred = self.x[:, TC] - 1.2 - 0.16 * (24.0 - t_air)
            logw += self._robust_logpdf(float(obs["ear_temp_c"]) - pred, SIG_EAR)
        if usable("resp_bpm"):
            logw += self._robust_logpdf(float(obs["resp_bpm"]) - self.x[:, RR], SIG_RESP)
        if usable("heart_bpm"):
            p = self.p
            pred = (p.hr_rest_bpm + p.hr_gain_core * (self.x[:, TC] - p.t_set_c)
                    + p.hr_gain_dehyd * self.x[:, WD] + p.hr_gain_act * act
                    + p.hr_gain_pyr * self.x[:, PY])
            logw += self._robust_logpdf(float(obs["heart_bpm"]) - pred, SIG_HEART)

        # --- trough-function belief -----------------------------------------
        # The test compares cumulative measured intake with cumulative obligatory
        # water loss over a rolling window. Loss is driven by the well-observed
        # respiration rate, so the test does not depend on the drinking belief it
        # is meant to inform -- which would otherwise be circular.
        flow = obs.get("water_flow_l")
        if flow is not None and np.isfinite(flow):
            rr_mean = float(self.x[:, RR] @ self.w)
            step_loss = (self.p.loss_base_l_per_day
                         + self.p.loss_resp_l_per_bpm_day * rr_mean) * DT_H / 24.0
            self._flow_window.append(max(0.0, float(flow)))
            self._loss_window.append(step_loss)
            if len(self._flow_window) > FLOW_WINDOW_STEPS:
                self._flow_window.pop(0)
                self._loss_window.pop(0)
            if len(self._flow_window) >= FLOW_WINDOW_STEPS:
                obs_flow = sum(self._flow_window)
                exp_flow = sum(self._loss_window)
                ratio = obs_flow / max(1e-6, exp_flow)
                # likelihood ratio for "trough functional" vs "trough blocked"
                if ratio < 0.25:
                    lr = 0.33
                elif ratio < 0.6:
                    lr = 0.80
                else:
                    lr = 4.0
                prior = self.p_trough_ok
                self.p_trough_ok = float(np.clip((lr * prior) / (lr * prior + (1 - prior)),
                                                 0.02, 0.999))
            # No likelihood term on flow here. Flow already enters the dynamics as
            # measured intake; scoring particles a second time against a *predicted*
            # drinking rate would assume the calf drinks whenever it is thirsty and
            # so penalise exactly the dehydrated particles a blockage produces.

        self._normalise_and_resample(logw)

    # ---- costly confirmatory measurement ---------------------------------
    def assimilate_verification(self, v: dict[str, float]) -> None:
        logw = np.log(self.w + 1e-300)
        logw += self._robust_logpdf(v["core_temp_c"] - self.x[:, TC], VERIFY_CORE_SIGMA * 1.5)
        if "hydration_score" in v:
            logw += self._robust_logpdf(v["hydration_score"] - self.x[:, WD], 0.006)
        self.p_trough_ok = 0.99 if v.get("trough_functional", 1.0) > 0.5 else 0.01
        self._normalise_and_resample(logw)

    # ---- bookkeeping ------------------------------------------------------
    def _normalise_and_resample(self, logw: np.ndarray) -> None:
        logw -= logw.max()
        w = np.exp(logw)
        s = w.sum()
        self.w = np.full(self.n, 1.0 / self.n) if s <= 0 else w / s
        ess = 1.0 / np.sum(self.w ** 2)
        if ess < 0.5 * self.n:
            idx = self._systematic_resample()
            self.x = self.x[idx]
            self.w = np.full(self.n, 1.0 / self.n)

    def _systematic_resample(self) -> np.ndarray:
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        return np.searchsorted(np.cumsum(self.w), pos).clip(0, self.n - 1)

    # ---- outputs ----------------------------------------------------------
    def estimate(self) -> StateEstimate:
        m = self.x.T @ self.w
        var = ((self.x - m) ** 2).T @ self.w
        sd = np.sqrt(np.maximum(var, 0.0))
        return StateEstimate(
            t_core_c=float(m[TC]), t_core_sd=float(sd[TC]),
            water_deficit=float(m[WD]), water_deficit_sd=float(sd[WD]),
            fatigue=float(m[FT]), fatigue_sd=float(sd[FT]),
            pyrogen=float(m[PY]), pyrogen_sd=float(sd[PY]),
            resp_bpm=float(m[RR]),
            p_trough_ok=float(self.p_trough_ok),
            ess=float(1.0 / np.sum(self.w ** 2)),
            p_hyperthermia=float(self.w[self.x[:, TC] > 39.2].sum()),
            p_dehydrated=float(self.w[self.x[:, WD] > 0.04].sum()),
            p_fever_driven=float(self.w[self.x[:, PY] > 0.25].sum()),
        )

    # ---- forecasting ------------------------------------------------------
    def forecast(self, horizon_steps: int, hour: float, t_air_path: list[float],
                 activity: float, cooling: bool, treatment: bool,
                 water_ok_p: float | None = None) -> dict[str, Any]:
        """Propagate the particle cloud forward under a fixed action assumption."""
        x = self.x.copy()
        w = self.w
        wop = self.p_trough_ok if water_ok_p is None else water_ok_p
        peak_tc = np.zeros(self.n)
        deg_hours = np.zeros(self.n)
        traj = []
        h = hour
        for k in range(horizon_steps):
            ta = t_air_path[min(k, len(t_air_path) - 1)]
            x = self._propagate(x, h, ta, activity, cooling, treatment, wop,
                                rh_obs=getattr(self, "_last_rh", 60.0))
            peak_tc = np.maximum(peak_tc, x[:, TC])
            deg_hours += np.maximum(0.0, x[:, TC] - 39.2) * DT_H
            traj.append(float(x[:, TC] @ w))
            h = (h + DT_H) % 24.0
        return {
            "mean_peak_core_c": float(peak_tc @ w),
            "p95_peak_core_c": float(np.quantile(peak_tc, 0.95)),
            "expected_degree_hours": float(deg_hours @ w),
            "expected_final_deficit": float(x[:, WD] @ w),
            "expected_final_fatigue": float(x[:, FT] @ w),
            "core_trajectory": [round(t, 3) for t in traj],
        }
