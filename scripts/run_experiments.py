#!/usr/bin/env python3
"""Offline experiments: main comparison, ablations and sensitivity analysis.

Usage:  python scripts/run_experiments.py [--seeds 50] [--out outputs/experiments]

Every result in the manuscript that does not involve a language model is
produced by this script. It uses no API key and is deterministic given --seeds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from calftwin.config import ActuatorParams, CostWeights
from calftwin.runner import run_episode, run_grid
from calftwin.scenarios import SCENARIOS

AGENTS = ["shadow_only", "threshold", "bounded_react", "bounded_react_noverify",
          "bounded_react_noforecast", "full_info"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default="outputs/experiments")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    t0 = time.time()

    # ---- E1: main comparison -------------------------------------------
    print(f"E1  main grid: {len(SCENARIOS)} scenarios x {len(AGENTS)} agents "
          f"x {len(seeds)} seeds")
    e1 = run_grid(list(SCENARIOS), seeds, AGENTS)
    e1.to_csv(out / "e1_main.csv", index=False)

    # ---- E2: sensor-noise sensitivity ----------------------------------
    print("E2  sensor-noise sweep")
    rows = []
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for sc in ["heat", "water_block", "fever_hot", "combined"]:
            for ag in ["threshold", "bounded_react", "bounded_react_noverify"]:
                for sd in seeds[:25]:
                    m = run_episode(scenario=sc, seed=sd, agent=ag,
                                    sensor_noise_mult=mult)["metrics"]
                    m["noise_mult"] = mult
                    rows.append(m)
    pd.DataFrame(rows).to_csv(out / "e2_noise.csv", index=False)

    # ---- E3: price of information --------------------------------------
    print("E3  verification-price sweep")
    rows = []
    for price in [0.0, 0.06, 0.12, 0.25, 0.50, 1.00]:
        w = CostWeights(c_verify=price)
        for sc in ["fever_hot", "combined", "sensor_fault"]:
            for ag in ["bounded_react", "bounded_react_noverify"]:
                for sd in seeds[:25]:
                    m = run_episode(scenario=sc, seed=sd, agent=ag, costs=w)["metrics"]
                    m["verify_price"] = price
                    rows.append(m)
    pd.DataFrame(rows).to_csv(out / "e3_info_price.csv", index=False)

    # ---- E4: actuator latency and reliability --------------------------
    print("E4  command-path degradation sweep")
    rows = []
    for lat in [0, 2, 4, 8]:
        for ok in [1.0, 0.8, 0.5]:
            a = ActuatorParams(latency_steps=lat, exec_success_p=ok)
            for sc in ["heat", "combined"]:
                for ag in ["threshold", "bounded_react"]:
                    for sd in seeds[:20]:
                        m = run_episode(scenario=sc, seed=sd, agent=ag,
                                        actuator=a)["metrics"]
                        m["latency_steps"] = lat
                        m["exec_success_p"] = ok
                        rows.append(m)
    pd.DataFrame(rows).to_csv(out / "e4_actuator.csv", index=False)

    # ---- E5: particle count --------------------------------------------
    print("E5  particle-count sweep")
    rows = []
    for n in [50, 100, 200, 400, 800, 1600]:
        for sc in ["fever_hot", "combined"]:
            for sd in seeds[:20]:
                m = run_episode(scenario=sc, seed=sd, agent="shadow_only",
                                n_particles=n)["metrics"]
                m["n_particles"] = n
                rows.append(m)
    pd.DataFrame(rows).to_csv(out / "e5_particles.csv", index=False)

    meta = {"seeds": args.seeds, "scenarios": list(SCENARIOS), "agents": AGENTS,
            "episodes": int(len(e1)), "runtime_s": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone in {meta['runtime_s']} s -> {out}")

    piv = e1.pivot_table(index="scenario", columns="agent",
                         values="burden", aggfunc="mean").round(3)
    print("\nmean burden by scenario x agent\n")
    print(piv.to_string())


if __name__ == "__main__":
    main()
