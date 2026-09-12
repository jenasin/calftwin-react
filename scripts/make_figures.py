#!/usr/bin/env python3
"""Render every figure and results table from the experiment outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from calftwin import plots
from calftwin.runner import run_episode

AGENT_ORDER = plots.AGENT_ORDER
SCEN_ORDER = list(plots.SCEN_LABEL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="outputs/experiments")
    ap.add_argument("--llm", default="outputs/llm")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tables", default="docs")
    args = ap.parse_args()
    exp, out, tab = Path(args.exp), Path(args.out), Path(args.tables)
    out.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    # Figure 1 -- architecture
    plots.fig_architecture(out / "fig1_architecture.png")
    print("fig1 architecture")

    # Figure 2 -- one closed-loop day
    res = run_episode(scenario="combined", seed=7, agent="bounded_react")
    plots.fig_day_trace(res, out / "fig2_day_trace.png",
                        title="Combined scenario: heat, blocked trough, frozen ear tag")
    print("fig2 day trace")

    e1 = pd.read_csv(exp / "e1_main.csv")
    plots.fig_main_comparison(e1, out / "fig3_main_comparison.png")
    print("fig3 main comparison")

    if (exp / "e3_info_price.csv").exists():
        e3 = pd.read_csv(exp / "e3_info_price.csv")
        plots.fig_info_price(e3, out / "fig4_info_price.png")
        print("fig4 price of information")

    if (exp / "e2_noise.csv").exists():
        e2 = pd.read_csv(exp / "e2_noise.csv")
        plots.fig_noise_sensitivity(e2, out / "fig5_noise.png")
        print("fig5 sensor-noise sensitivity")

    llm_csv = Path(args.llm) / "llm_results.csv"
    if llm_csv.exists():
        llm = pd.read_csv(llm_csv)
        plots.fig_llm(llm, e1, out / "fig6_llm.png")
        print("fig6 language-model agents")

    # ---- tables --------------------------------------------------------
    def piv(metric):
        p = e1.pivot_table(index="scenario", columns="agent", values=metric,
                           aggfunc="mean")
        return p.reindex(index=[s for s in SCEN_ORDER if s in p.index],
                         columns=[a for a in AGENT_ORDER if a in p.columns]).round(3)

    lines = ["# Computed results\n",
             f"Generated from `{exp}`. All values are means over "
             f"{e1.groupby(['scenario', 'agent']).size().max()} seeds.\n"]
    for metric, title in [("burden", "Welfare burden (model units, lower is better)"),
                          ("total_score", "Total score = burden + intervention + information cost"),
                          ("n_verify", "Confirmatory measurements purchased per day"),
                          ("cooling_hours", "Cooling run-time (h per day)"),
                          ("detection_latency_h", "Detection latency (h)")]:
        lines += [f"\n## {title}\n", piv(metric).to_markdown(), ""]

    est = (e1[e1["agent"] == "shadow_only"]
           .groupby("scenario")[["rmse_core_c", "bias_core_c", "coverage95_core",
                                 "mean_core_sd", "rmse_deficit"]].mean().round(4))
    est = est.reindex([s for s in SCEN_ORDER if s in est.index])
    lines += ["\n## State-estimation quality (open loop)\n", est.to_markdown(), ""]

    if llm_csv.exists():
        g = (pd.read_csv(llm_csv)
             .groupby(["model", "scenario"])[["burden", "total_score", "n_verify",
                                              "llm_tool_calls", "llm_in_tokens",
                                              "llm_out_tokens"]].mean().round(3))
        lines += ["\n## Language-model agents\n", g.to_markdown(), ""]

    (tab / "RESULTS.md").write_text("\n".join(lines))
    print(f"tables -> {tab / 'RESULTS.md'}")

    meta = json.loads((exp / "meta.json").read_text()) if (exp / "meta.json").exists() else {}
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
