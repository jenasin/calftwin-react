"""Outcome metrics.

``burden`` aggregates modelled physiological strain of the *simulated* animal
(ground truth, used for evaluation only -- never visible to the agent).
``total_score`` adds intervention and information cost so that a controller
cannot win by spending without limit. Weights are illustrative: the score is a
model-internal quantity, not money and not a validated welfare index.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DT_H, CostWeights


def burden_components(df: pd.DataFrame, w: CostWeights) -> dict[str, float]:
    heat = float(np.sum(np.maximum(0.0, df["true_t_core_c"] - w.heat_threshold_c)) * DT_H)
    dehyd = float(np.sum(np.maximum(0.0, df["true_water_deficit"] - w.dehyd_threshold))
                  * 100.0 * DT_H)
    fat = float(np.sum(np.maximum(0.0, df["true_fatigue"] - w.fatigue_threshold)) * DT_H)
    return {"heat_degree_hours": heat, "dehydration_pct_hours": dehyd,
            "fatigue_hours": fat}


def evaluate(df: pd.DataFrame, twin, w: CostWeights,
             llm_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    comp = burden_components(df, w)
    burden = (w.w_heat * comp["heat_degree_hours"]
              + w.w_dehyd * comp["dehydration_pct_hours"]
              + w.w_fatigue * comp["fatigue_hours"])

    counts = twin.bus.counts()
    intervention_cost = (w.c_cooling_per_h * counts["cooling_hours"]
                         + w.c_water_fix * counts["water_fix"]
                         + w.c_human_check * counts["human_check"])
    info_cost = w.c_verify * twin.n_verify
    total = burden + intervention_cost + info_cost

    # estimation accuracy of the twin (evaluation-only comparison with truth)
    err_core = df["t_core_c"] - df["true_t_core_c"]
    err_wd = df["water_deficit"] - df["true_water_deficit"]
    inside = np.mean(np.abs(err_core) <= 1.96 * np.maximum(df["t_core_sd"], 1e-6))

    # detection latency: first hyperthermia episode in truth vs first alarm
    lat = _detection_latency(df, w)

    out = {
        **comp,
        "burden": burden,
        "intervention_cost": intervention_cost,
        "info_cost": info_cost,
        "total_score": total,
        "peak_true_core_c": float(df["true_t_core_c"].max()),
        "peak_true_deficit": float(df["true_water_deficit"].max()),
        "final_fatigue": float(df["true_fatigue"].iloc[-1]),
        "rmse_core_c": float(np.sqrt(np.mean(err_core ** 2))),
        "bias_core_c": float(np.mean(err_core)),
        "rmse_deficit": float(np.sqrt(np.mean(err_wd ** 2))),
        "coverage95_core": float(inside),
        "mean_core_sd": float(df["t_core_sd"].mean()),
        "n_verify": int(twin.n_verify),
        "cooling_hours": float(counts["cooling_hours"]),
        "n_human_check": int(counts["human_check"]),
        "n_water_fix": int(counts["water_fix"]),
        "n_commands_issued": int(counts["issued"]),
        "n_rejected": int(counts["rejected"]),
        "n_failed": int(counts["failed"]),
        "n_tool_calls": int(len(twin.tool_calls)),
        "detection_latency_h": lat,
        "false_cooling_hours": _false_cooling(df, w),
    }
    if llm_stats:
        out.update({f"llm_{k}": v for k, v in llm_stats.items()})
    return out


def _detection_latency(df: pd.DataFrame, w: CostWeights) -> float:
    """Hours from true threshold crossing to the first command that follows it."""
    hot = df.index[df["true_t_core_c"] > w.heat_threshold_c]
    if len(hot) == 0:
        return float("nan")
    onset = int(hot[0])
    acted = df.index[(df.index >= onset) & df["action"].isin(
        ["cooling_on", "flag_human_check", "restore_water"])]
    if len(acted) == 0:
        return float("nan")
    return float((int(acted[0]) - onset) * DT_H)


def _false_cooling(df: pd.DataFrame, w: CostWeights) -> float:
    """Cooling hours accrued while the animal was not thermally challenged."""
    if "true_cooling_on" not in df:
        return 0.0
    mask = (df["true_cooling_on"] > 0.5) & (df["true_t_core_c"] < w.heat_threshold_c - 0.3)
    return float(mask.sum() * DT_H)
