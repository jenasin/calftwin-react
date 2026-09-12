"""Figures for the manuscript and the repository.

Colour follows the validated categorical order (blue, orange, aqua, yellow,
magenta, green). Because conference proceedings are often read in print, every
series also carries a distinct marker or line style, so identity never depends on
colour alone. Each panel has a single y-axis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- validated categorical palette (light surface) -------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e2de"
SURFACE = "#fcfcfb"
STATUS_BAD = "#e34948"

AGENT_ORDER = ["shadow_only", "threshold", "bounded_react_noforecast",
               "bounded_react_noverify", "bounded_react", "full_info"]
AGENT_LABEL = {
    "shadow_only": "Digital shadow (no action)",
    "threshold": "Threshold controller",
    "bounded_react_noforecast": "Agent, no forecast",
    "bounded_react_noverify": "Agent, no verification",
    "bounded_react": "Bounded ReAct agent",
    "full_info": "Full-information reference",
}
AGENT_STYLE = {a: (SERIES[i], m) for i, (a, m) in enumerate(
    zip(AGENT_ORDER, ["o", "s", "^", "D", "v", "P"]))}
SCEN_LABEL = {"normal": "Normal", "heat": "Heat", "water_block": "Water block",
              "sensor_fault": "Sensor fault", "fever": "Fever",
              "fever_hot": "Fever + heat", "combined": "Combined"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "regular", "axes.titlelocation": "left",
        "lines.linewidth": 1.6, "figure.dpi": 150, "savefig.dpi": 300,
        "legend.frameon": False, "axes.axisbelow": True,
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Figure 1 -- architecture
# ===========================================================================
def fig_architecture(path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.2, 7.9)
    ax.axis("off")

    def box(x0, y0, x1, y1, text, fc, ec, fs=8.3):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.1, edgecolor=ec, facecolor=fc))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, linespacing=1.45)

    def arrow(p0, p1, color, ls="-"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.3, color=color, linestyle=ls,
                                     shrinkA=1, shrinkB=1))

    def lab(x, y, t, color=INK2, ha="center", va="center", fs=7.3):
        ax.text(x, y, t, fontsize=fs, color=color, ha=ha, va=va, linespacing=1.35)

    # ---- physical layer (left column) ---------------------------------
    box(0.15, 6.10, 2.75, 7.15, "Pen environment\nair temperature, humidity,\nwater availability",
        "#f4f3f0", MUTED)
    box(0.15, 4.15, 2.75, 5.85, "Simulated calf  CALF-001\n"
        r"$x_t=[\,T_{core},\ w,\ f,\ \pi\,]$" + "\nhidden from the agent",
        "#eaf2fd", SERIES[0])
    box(0.15, 1.00, 2.75, 2.20, "Actuators\ncooling, trough maintenance,\nstockperson",
        "#fdeee7", SERIES[1])

    # ---- twin (middle column) -----------------------------------------
    box(3.45, 6.10, 6.20, 7.15, "Sensor layer\nnoise, dropout, spikes,\nstuck and drifting channels",
        "#f4f3f0", MUTED)
    box(3.45, 4.50, 6.20, 5.70, "Digital shadow\nidentity, ordering, duplicates,\ngaps, quality flags",
        "#eaf2fd", SERIES[0])
    box(3.45, 2.70, 6.20, 4.10, "State estimator\nparticle filter  "
        + r"$p(x_t \mid z_{1:t})$" + "\nestimate and uncertainty", "#eaf2fd", SERIES[0])

    # ---- decision layer (right column) --------------------------------
    box(7.10, 3.30, 9.85, 6.55, "ReAct agent\n\nget_state\nget_history\n"
        "forecast_options\nverify_vitals  (costs)\npropose_action", "#e7f7f1", SERIES[2])
    box(7.10, 1.00, 9.85, 2.20, "Command path\napproval gate, 30 min latency,\nfailure probability",
        "#fdeee7", SERIES[1])

    # ---- data path (blue) ---------------------------------------------
    arrow((1.45, 6.10), (1.45, 5.85), MUTED)
    arrow((2.75, 5.35), (3.45, 6.25), SERIES[0]); lab(2.88, 5.72, "measure", ha="left")
    arrow((4.82, 6.10), (4.82, 5.70), SERIES[0]); lab(4.95, 5.90, "ingest", ha="left")
    arrow((4.82, 4.50), (4.82, 4.10), SERIES[0]); lab(4.95, 4.30, "assimilate", ha="left")
    arrow((6.20, 3.72), (7.10, 3.72), SERIES[0])
    lab(6.65, 4.34, "state and\nuncertainty", fs=7.0)

    # ---- information purchase (green, dashed) -------------------------
    arrow((7.10, 3.12), (6.20, 3.12), SERIES[2], ls=(0, (4, 2)))
    lab(6.65, 2.84, "verify\n(costly)", color=SERIES[2])

    # ---- command path (orange) ----------------------------------------
    arrow((8.47, 3.30), (8.47, 2.20), SERIES[1]); lab(8.60, 2.75, "command", ha="left")
    arrow((7.10, 1.60), (2.75, 1.60), SERIES[1])
    lab(4.92, 1.80, "approved and executed intervention")
    arrow((1.45, 2.20), (1.45, 4.15), SERIES[1])

    lab(0.15, 7.62, "Physical layer: ground truth, never visible to the agent",
        ha="left", fs=8.2)
    lab(7.10, 6.85, "Decision layer", ha="left", fs=8.2)
    lab(0.15, 0.42, "Blue: data path (animal to twin).    Orange: command path "
        "(twin to animal).    Green dashed: purchased information.", ha="left", fs=7.5)
    return _save(fig, path)


# ===========================================================================
# Figure 2 -- one closed-loop day
# ===========================================================================
def fig_day_trace(res: dict[str, Any], path: Path, title: str = "") -> Path:
    apply_style()
    d = res["trace"].copy()
    d = d[d["true_step"] > 8].reset_index(drop=True)
    h = np.arange(len(d)) * 0.25
    fig, axes = plt.subplots(4, 1, figsize=(6.9, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1.5, 1.5, 1.3]})

    # cooling spans
    cool = d["true_cooling_on"].values > 0.5
    def shade(ax):
        start = None
        for i, c in enumerate(cool):
            if c and start is None:
                start = i
            if (not c or i == len(cool) - 1) and start is not None:
                ax.axvspan(h[start], h[i], color=SERIES[0], alpha=0.07, lw=0)
                start = None

    # -- panel 1: core temperature
    ax = axes[0]
    shade(ax)
    ax.fill_between(h, d["t_core_c"] - 1.96 * d["t_core_sd"],
                    d["t_core_c"] + 1.96 * d["t_core_sd"],
                    color=SERIES[0], alpha=0.17, lw=0, label="Twin estimate, 95 % interval")
    ax.plot(h, d["t_core_c"], color=SERIES[0], label="Twin estimate")
    ax.plot(h, d["true_t_core_c"], color=INK, lw=1.4, ls=(0, (5, 2)),
            label="True core temperature")
    ax.axhline(39.2, color=STATUS_BAD, lw=1.0, ls=":")
    ax.text(0.15, 39.24, "strain threshold 39.2 °C", fontsize=7.2, color=STATUS_BAD)
    vh = h[d["n_verify"].diff().fillna(0) > 0]
    if len(vh):
        ax.plot(vh, np.full(len(vh), ax.get_ylim()[0] + 0.06), marker="^", ls="none",
                ms=6, color=SERIES[2], clip_on=False, label="Verification purchased")
    ax.set_ylabel("Core temperature (°C)")
    ax.legend(loc="upper left", ncol=2, handlelength=1.8)
    ax.set_title(title or "Closed-loop day", color=INK)

    # -- panel 2: water deficit
    ax = axes[1]
    shade(ax)
    ax.plot(h, d["true_water_deficit"] * 100, color=INK, lw=1.4, ls=(0, (5, 2)),
            label="True deficit")
    ax.plot(h, d["water_deficit"] * 100, color=SERIES[1], label="Twin estimate")
    ax.axhline(2.0, color=STATUS_BAD, lw=1.0, ls=":")
    ax.set_ylabel("Water deficit\n(% of body mass)")
    ax.legend(loc="upper left", ncol=2, handlelength=1.8)

    # -- panel 3: fever evidence
    ax = axes[2]
    shade(ax)
    ax.plot(h, d["true_pyrogen"], color=INK, lw=1.4, ls=(0, (5, 2)), label="True pyrogen")
    ax.plot(h, d["p_fever_driven"], color=SERIES[4], label="P(fever-driven)")
    ax.plot(h, d["p_trough_ok"], color=SERIES[2], ls=(0, (1, 1.6)), label="P(trough functional)")
    ax.set_ylabel("Probability /\npyrogen level")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper left", ncol=3, handlelength=1.8)

    # -- panel 4: environment and commands
    ax = axes[3]
    shade(ax)
    ax.plot(h, d["true_t_air_c"], color=SERIES[3], label="Pen air temperature")
    ax.set_ylabel("Air temp (°C)")
    ax.set_xlabel("Hour of day")
    cmds = [c for c in res["twin"].bus.log
            if c.action != "observe" and c.status == "executed"]
    lo, hi = ax.get_ylim()
    for c in cmds:
        x = (c.executed_step - 9) * 0.25
        ax.axvline(x, color=SERIES[1], lw=1.0, alpha=0.75)
        ax.annotate(c.action.replace("_", " "), (x, hi), rotation=90, fontsize=6.8,
                    color=SERIES[1], ha="right", va="top")
    ax.legend(loc="upper left", handlelength=1.8)
    ax.set_xlim(0, h[-1])
    ax.set_xticks(np.arange(0, 25, 3))
    fig.align_ylabels(axes)
    return _save(fig, path)


# ===========================================================================
# Figure 3 -- main comparison
# ===========================================================================
def fig_main_comparison(e1: pd.DataFrame, path: Path) -> Path:
    """Two questions, two scales.

    The upper panel answers "is it worth closing the loop at all?", where the
    no-action shadow dominates the range. The lower panel drops the shadow so that
    the differences *between acting controllers* -- the actual research question --
    are legible rather than compressed into the bottom 5 % of the axis.
    """
    apply_style()
    scen = [s for s in SCEN_LABEL if s in set(e1["scenario"])]
    agents = [a for a in AGENT_ORDER if a in set(e1["agent"])]
    acting = [a for a in agents if a != "shadow_only"]
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 5.8))
    x = np.arange(len(scen))

    def panel(ax, agent_list, metric, ylab):
        width = 0.82 / len(agent_list)
        g = e1.groupby(["scenario", "agent"])[metric].agg(["mean", "sem"])
        for i, a in enumerate(agent_list):
            mu = [g.loc[(s, a), "mean"] for s in scen]
            se = [g.loc[(s, a), "sem"] for s in scen]
            ax.bar(x + i * width - 0.41 + width / 2, mu, width * 0.86, yerr=se,
                   capsize=1.5, color=AGENT_STYLE[a][0], label=AGENT_LABEL[a],
                   edgecolor=SURFACE, linewidth=0.8,
                   error_kw={"lw": 0.8, "ecolor": INK2})
        ax.set_ylabel(ylab)
        ax.set_xticks(x, [SCEN_LABEL[s] for s in scen])

    panel(axes[0], agents, "burden", "Welfare burden (model units)")
    axes[0].legend(loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.2,
                   fontsize=7.6)
    axes[0].set_title("(a) All configurations: the cost of not acting", color=INK2,
                      fontsize=8.6)
    axes[0].set_ylim(0, e1.groupby(["scenario", "agent"])["burden"].mean().max() * 1.38)

    panel(axes[1], acting, "total_score", "Total score = burden + cost")
    axes[1].set_title("(b) Acting controllers only: the cost of how you act",
                      color=INK2, fontsize=8.6)
    axes[1].set_xlabel("Scenario")
    fig.align_ylabels(axes)
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# Figure 4 -- price of information
# ===========================================================================
def fig_info_price(e3: pd.DataFrame, path: Path) -> Path:
    apply_style()
    scen = sorted(set(e3["scenario"]), key=lambda s: list(SCEN_LABEL).index(s))
    fig, axes = plt.subplots(1, len(scen), figsize=(6.9, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, scen):
        sub = e3[e3["scenario"] == s]
        for a in ["bounded_react", "bounded_react_noverify"]:
            g = sub[sub["agent"] == a].groupby("verify_price")["total_score"].agg(["mean", "sem"])
            col, mk = AGENT_STYLE[a]
            ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=col, marker=mk, ms=4.5,
                        capsize=2, lw=1.6, label=AGENT_LABEL[a],
                        ls="-" if a == "bounded_react" else (0, (4, 2)))
        ax.set_title(SCEN_LABEL[s], color=INK2, fontsize=8.5)
        ax.set_xlabel("Price per verification")
    axes[0].set_ylabel("Total score")
    axes[0].legend(loc="upper left", handlelength=1.8)
    return _save(fig, path)


# ===========================================================================
# Figure 5 -- sensor-noise sensitivity
# ===========================================================================
def fig_noise_sensitivity(e2: pd.DataFrame, path: Path) -> Path:
    apply_style()
    scen = sorted(set(e2["scenario"]), key=lambda s: list(SCEN_LABEL).index(s))
    fig, axes = plt.subplots(1, len(scen), figsize=(6.9, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, scen):
        sub = e2[e2["scenario"] == s]
        for a in ["threshold", "bounded_react_noverify", "bounded_react"]:
            g = sub[sub["agent"] == a].groupby("noise_mult")["burden"].agg(["mean", "sem"])
            col, mk = AGENT_STYLE[a]
            ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=col, marker=mk, ms=4.5,
                        capsize=2, lw=1.6, label=AGENT_LABEL[a],
                        ls={"threshold": (0, (4, 2)), "bounded_react": "-",
                            "bounded_react_noverify": (0, (1, 1.6))}[a])
        ax.set_title(SCEN_LABEL[s], color=INK2, fontsize=8.5)
        ax.set_xlabel("Sensor-noise multiplier")
    axes[0].set_ylabel("Welfare burden")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, handlelength=2.2,
               bbox_to_anchor=(0.5, -0.10), columnspacing=1.6)
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# Figure 6 -- language-model agents
# ===========================================================================
def fig_llm(llm: pd.DataFrame, ref: pd.DataFrame, path: Path) -> Path:
    apply_style()
    scen = sorted(set(llm["scenario"]), key=lambda s: list(SCEN_LABEL).index(s))
    models = sorted(set(llm["agent"]))
    baselines = [a for a in ["threshold", "bounded_react"] if a in set(ref["agent"])]
    labels = [AGENT_LABEL[a] for a in baselines] + [m.split("[")[-1].rstrip("]") for m in models]
    colors = [AGENT_STYLE[a][0] for a in baselines] + SERIES[3:3 + len(models)]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    x = np.arange(len(scen))
    width = 0.8 / len(labels)
    for j, (src, key) in enumerate([(None, "burden"), (None, "total_score")]):
        ax = axes[j]
        for i, (lab, col) in enumerate(zip(labels, colors)):
            if i < len(baselines):
                sub = ref[(ref["agent"] == baselines[i]) & (ref["scenario"].isin(scen))]
            else:
                sub = llm[llm["agent"] == models[i - len(baselines)]]
            g = sub.groupby("scenario")[key].agg(["mean", "sem"])
            mu = [g.loc[s, "mean"] if s in g.index else np.nan for s in scen]
            se = [g.loc[s, "sem"] if s in g.index else np.nan for s in scen]
            ax.bar(x + i * width - 0.4 + width / 2, mu, width * 0.88, yerr=se,
                   color=col, label=lab, edgecolor=SURFACE, linewidth=0.8,
                   capsize=1.5, error_kw={"lw": 0.8, "ecolor": INK2})
        ax.set_xticks(x, [SCEN_LABEL[s] for s in scen], rotation=42, ha="right",
                      fontsize=7.4)
        ax.set_ylabel("Welfare burden" if j == 0 else "Total score")
    axes[0].legend(loc="upper left", ncol=2, fontsize=7.0, handlelength=1.2,
                   columnspacing=0.9)
    fig.tight_layout()
    return _save(fig, path)
