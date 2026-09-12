"""Command-line interface.

    calftwin demo     --scenario combined --seed 7 --agent bounded_react
    calftwin compare  --scenario fever_hot --seeds 20
    calftwin scenarios
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .runner import AGENTS, run_episode, save_episode
from .scenarios import SCENARIOS, build as build_scenario


def _demo(a: argparse.Namespace) -> None:
    kwargs = {}
    if a.agent == "llm_react":
        kwargs = {"model": a.model}
    res = run_episode(scenario=a.scenario, seed=a.seed, agent=a.agent, days=a.days,
                      decide_every=a.decide_every, agent_kwargs=kwargs)
    out = save_episode(res, a.out)
    from .plots import fig_day_trace
    fig = fig_day_trace(res, Path(out) / "day_trace.png",
                        title=f"{a.scenario} / {res['metrics']['agent']} / seed {a.seed}")
    m = res["metrics"]
    print(f"\nscenario   {a.scenario}: {res['scenario_spec'].description}")
    print(f"agent      {m['agent']}")
    print(f"burden     {m['burden']:.3f}   (heat {m['heat_degree_hours']:.2f} deg-h, "
          f"dehydration {m['dehydration_pct_hours']:.2f} %-h, fatigue {m['fatigue_hours']:.2f} h)")
    print(f"cost       intervention {m['intervention_cost']:.3f} + information "
          f"{m['info_cost']:.3f}")
    print(f"total      {m['total_score']:.3f}")
    print(f"estimation RMSE {m['rmse_core_c']:.3f} C, 95 % coverage "
          f"{m['coverage95_core']:.2f}")
    print(f"actions    cooling {m['cooling_hours']:.2f} h, human checks "
          f"{m['n_human_check']}, trough fixes {m['n_water_fix']}, "
          f"verifications {m['n_verify']}")
    print(f"commands   {m['n_commands_issued']} issued, {m['n_rejected']} rejected, "
          f"{m['n_failed']} failed")
    print(f"\nwritten to {out}/  (trace.csv, metrics.json, commands.json, {fig.name})")


def _compare(a: argparse.Namespace) -> None:
    rows = []
    for ag in a.agents:
        for sd in range(a.seeds):
            rows.append(run_episode(scenario=a.scenario, seed=sd, agent=ag)["metrics"])
    df = pd.DataFrame(rows)
    g = df.groupby("agent")[["burden", "intervention_cost", "info_cost", "total_score",
                             "n_verify", "cooling_hours"]].mean().round(3)
    print(f"\nscenario {a.scenario}, {a.seeds} seeds\n")
    print(g.to_string())


def _scenarios(a: argparse.Namespace) -> None:
    import numpy as np
    for s in SCENARIOS:
        print(f"{s:14s} {build_scenario(s, np.random.default_rng(0)).description}")


def main() -> None:
    p = argparse.ArgumentParser(prog="calftwin",
                                description="CalfTwin-ReAct digital-twin testbed")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run one episode and write artefacts")
    d.add_argument("--scenario", default="combined", choices=SCENARIOS)
    d.add_argument("--agent", default="bounded_react", choices=AGENTS)
    d.add_argument("--model", default="gpt-4.1-mini", help="model id for llm_react")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--days", type=int, default=1)
    d.add_argument("--decide-every", dest="decide_every", type=int, default=4)
    d.add_argument("--out", default="outputs/demo")
    d.set_defaults(func=_demo)

    c = sub.add_parser("compare", help="compare agents on one scenario")
    c.add_argument("--scenario", default="combined", choices=SCENARIOS)
    c.add_argument("--seeds", type=int, default=20)
    c.add_argument("--agents", nargs="+", default=["shadow_only", "threshold",
                                                   "bounded_react", "full_info"])
    c.set_defaults(func=_compare)

    s = sub.add_parser("scenarios", help="list scenarios")
    s.set_defaults(func=_scenarios)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
