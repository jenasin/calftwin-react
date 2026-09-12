"""Bounded ReAct reference controller (offline, no language model).

This is a deterministic reasoning-and-acting loop: it reads the twin's estimate,
interrogates history when the data path looks unreliable, *buys* a confirmatory
measurement only when uncertainty would change its decision, forecasts the
candidate actions, and then commits one command.

It is the reference against which the language-model agent is measured. Calling
it "ReAct-style" refers to the interleaving of reasoning steps with tool calls,
not to the use of a language model.
"""
from __future__ import annotations

from typing import Any

from ..twin import DigitalTwin
from .base import decision


class BoundedReActAgent:
    name = "bounded_react"

    def __init__(self, allow_verify: bool = True, allow_forecast: bool = True,
                 verify_sd_c: float = 0.16, max_verify_per_day: int = 6,
                 verify_cooldown_steps: int = 8, cooling_gain_deg_h: float | None = None,
                 fever_p: float = 0.40, trough_p: float = 0.35,
                 min_dwell_steps: int = 4) -> None:
        self.allow_verify = allow_verify
        self.allow_forecast = allow_forecast
        self.verify_sd_c = verify_sd_c
        self.max_verify_per_day = max_verify_per_day
        self.verify_cooldown_steps = verify_cooldown_steps
        self.cooling_gain_deg_h = cooling_gain_deg_h
        self.fever_p = fever_p
        self.trough_p = trough_p
        self._last_verify_step = -999
        self._last_human_check_step = -999
        self._last_water_fix_step = -999
        self._last_switch_step = -999
        self.min_dwell_steps = min_dwell_steps
        if not allow_verify and not allow_forecast:
            self.name = "bounded_react_reactive"
        elif not allow_verify:
            self.name = "bounded_react_noverify"
        elif not allow_forecast:
            self.name = "bounded_react_noforecast"

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _tentative_action(t_core: float, pyrogen: float, p_trough: float,
                          cooling: bool, treated: bool) -> str:
        """Action the policy would take if the point estimate were exact."""
        if p_trough < 0.35:
            return "restore_water"
        if pyrogen > 0.25 and not treated:
            return "flag_human_check"
        if t_core > 39.15 and not cooling:
            return "cooling_on"
        if t_core < 38.75 and cooling:
            return "cooling_off"
        return "observe"

    def _uncertainty_matters(self, est: dict[str, Any], cooling: bool,
                             treated: bool) -> bool:
        """Would a plausible excursion of the estimate change the action?"""
        lo = self._tentative_action(est["t_core_c"] - 1.64 * est["t_core_sd"],
                                    max(0.0, est["pyrogen"] - 1.64 * est["pyrogen_sd"]),
                                    min(0.999, est["p_trough_ok"] + 0.2), cooling, treated)
        hi = self._tentative_action(est["t_core_c"] + 1.64 * est["t_core_sd"],
                                    min(1.0, est["pyrogen"] + 1.64 * est["pyrogen_sd"]),
                                    max(0.0, est["p_trough_ok"] - 0.2), cooling, treated)
        return lo != hi

    # ---- the loop ---------------------------------------------------------
    def decide(self, twin: DigitalTwin) -> dict[str, Any]:
        calls = 0
        thoughts: list[str] = []

        # --- 1. read the twin ---------------------------------------------
        st = twin.get_state()
        calls += 1
        est = st["estimate"]
        cooling = st["pen"]["cooling_on"]
        treated = st["pen"]["clinical_response_active"]
        suspect = st["data_quality"]["suspect_channels"]
        missing = st["data_quality"]["mostly_missing_channels"]
        step = twin.step_idx
        thoughts.append(
            f"core {est['t_core_c']:.2f}+-{est['t_core_sd']:.2f} C, "
            f"deficit {est['water_deficit']:.3f}, pyrogen {est['pyrogen']:.2f}, "
            f"P(trough ok) {est['p_trough_ok']:.2f}")

        # --- 2. data path integrity ---------------------------------------
        if suspect or missing:
            ch = (suspect + missing)[0]
            h = twin.get_history(ch, hours=4.0)
            calls += 1
            thoughts.append(f"channel {ch} flagged; trend {h.get('trend_per_h')} /h")

        # --- 3. value of information --------------------------------------
        uncertainty_matters = self._uncertainty_matters(est, cooling, treated)
        budget_left = twin.n_verify < self.max_verify_per_day
        cooled_down = (step - self._last_verify_step) >= self.verify_cooldown_steps
        if (self.allow_verify and uncertainty_matters
                and est["t_core_sd"] > self.verify_sd_c and budget_left and cooled_down):
            v = twin.verify_vitals()
            calls += 1
            self._last_verify_step = step
            thoughts.append(
                f"verified: core {v['confirmed_core_temp_c']} C "
                f"(shift {v['estimate_shift_c']:+.2f}, sd {v['uncertainty_before_sd']}"
                f"->{v['uncertainty_after_sd']}), trough_ok={v['trough_functional']}")
            st = twin.get_state()
            calls += 1
            est = st["estimate"]
        elif uncertainty_matters and not self.allow_verify:
            thoughts.append("decision is uncertainty-sensitive but verification "
                            "is disabled in this configuration")

        # --- 4. water availability ----------------------------------------
        if (est["p_trough_ok"] < self.trough_p
                and (step - self._last_water_fix_step) > 16):
            self._last_water_fix_step = step
            twin.propose_action(
                "restore_water",
                f"P(trough functional) = {est['p_trough_ok']:.2f} while estimated "
                f"deficit is {est['water_deficit']:.3f}: flow evidence is "
                f"inconsistent with expected drinking.",
                evidence={"p_trough_ok": est["p_trough_ok"],
                          "water_deficit": est["water_deficit"]})
            calls += 1
            return decision("restore_water", " | ".join(thoughts), calls)

        # --- 5. fever vs environmental load ------------------------------
        if (est["p_fever_driven"] > self.fever_p and not treated
                and (step - self._last_human_check_step) > 24):
            self._last_human_check_step = step
            twin.propose_action(
                "flag_human_check",
                f"P(pyrogen-driven fever) = {est['p_fever_driven']:.2f}: core "
                f"{est['t_core_c']:.2f} C is not explained by the measured "
                f"microclimate (THI {st.get('thi')}).",
                evidence={"p_fever_driven": est["p_fever_driven"],
                          "thi": st.get("thi"), "t_core_c": est["t_core_c"]})
            calls += 1
            act = "flag_human_check"
            if est["t_core_c"] > 39.4 and not cooling:
                twin.propose_action("cooling_on", "Concurrent environmental load.")
                calls += 1
            return decision(act, " | ".join(thoughts), calls)

        # --- 6. cost-aware thermal control --------------------------------
        # The comparison is always "keep doing what the pen is doing" against the
        # switch, so the option set depends on the current actuator state. Doing
        # otherwise makes "observe" mean two different things and the controller
        # chatters on and off every epoch.
        if self.allow_forecast:
            horizon_h = 3.0
            threshold = (self.cooling_gain_deg_h
                         if self.cooling_gain_deg_h is not None
                         else twin.costs.c_cooling_per_h * horizon_h)
            dwell_ok = (step - self._last_switch_step) >= self.min_dwell_steps
            if not cooling:
                fc = twin.forecast_options(horizon_h, ["observe", "cooling_on"])
                calls += 1
                base = fc["options"]["observe"]["expected_degree_hours"]
                cool = fc["options"]["cooling_on"]["expected_degree_hours"]
                gain = base - cool
                thoughts.append(f"3 h degree-hours: stay {base:.3f} vs cool {cool:.3f} "
                                f"(gain {gain:.3f} vs cost {threshold:.3f})")
                if gain > threshold and dwell_ok:
                    self._last_switch_step = step
                    twin.propose_action(
                        "cooling_on",
                        f"Forecast heat load {base:.2f} degree-hours without cooling "
                        f"versus {cool:.2f} with it; expected gain {gain:.2f} exceeds "
                        f"the {threshold:.2f} running cost over {horizon_h:g} h.",
                        evidence={"forecast": fc["options"]})
                    calls += 1
                    return decision("cooling_on", " | ".join(thoughts), calls)
            else:
                fc = twin.forecast_options(horizon_h, ["observe", "cooling_off"])
                calls += 1
                keep = fc["options"]["observe"]["expected_degree_hours"]
                off = fc["options"]["cooling_off"]["expected_degree_hours"]
                penalty = off - keep
                thoughts.append(f"3 h degree-hours: keep cooling {keep:.3f} vs stop "
                                f"{off:.3f} (penalty {penalty:.3f} vs saving {threshold:.3f})")
                if penalty < threshold and dwell_ok:
                    self._last_switch_step = step
                    twin.propose_action(
                        "cooling_off",
                        f"Stopping cooling costs only {penalty:.2f} degree-hours over "
                        f"{horizon_h:g} h, below the {threshold:.2f} running cost saved.",
                        evidence={"forecast": fc["options"]})
                    calls += 1
                    return decision("cooling_off", " | ".join(thoughts), calls)
        else:
            if est["p_hyperthermia"] > 0.5 and not cooling:
                twin.propose_action("cooling_on", "P(hyperthermia) > 0.5.")
                calls += 1
                return decision("cooling_on", " | ".join(thoughts), calls)

        twin.propose_action("observe", "State within bounds; no command warranted.")
        calls += 1
        return decision("observe", " | ".join(thoughts), calls)
