#!/usr/bin/env python3
"""Language-model experiments.

Runs the ReAct agent with real models against the same scenarios and seeds as the
offline experiments. Spend is capped by a process-wide budget guard; if the cap is
reached the run stops and whatever completed is still written out.

    python scripts/run_llm_experiments.py --pilot              # measure cost/epoch
    python scripts/run_llm_experiments.py --budget 5.0         # full grid
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from calftwin.agents.llm_react import PRICES, BudgetExceeded, BudgetTracker
from calftwin.runner import run_episode

_print_lock = threading.Lock()


def one(scenario: str, seed: int, model: str, budget: BudgetTracker,
        reasoning: str | None, decide_every: int, transcripts: Path | None):
    t0 = time.time()
    res = run_episode(scenario=scenario, seed=seed, agent="llm_react",
                      decide_every=decide_every,
                      agent_kwargs={"model": model, "budget": budget,
                                    "reasoning_effort": reasoning})
    m = res["metrics"]
    m["model"] = model
    m["wall_s"] = round(time.time() - t0, 1)
    if transcripts is not None:
        transcripts.mkdir(parents=True, exist_ok=True)
        (transcripts / f"{model}_{scenario}_s{seed}.json").write_text(
            json.dumps(res["agent"].transcript, indent=1, default=str))
    with _print_lock:
        print(f"  {model:14s} {scenario:13s} seed {seed:<3d} "
              f"burden {m['burden']:7.3f}  total {m['total_score']:7.3f}  "
              f"verify {m['n_verify']}  spend ${budget.spent_usd:.3f}", flush=True)
    return m


def run_block(jobs, budget, out_rows, workers: int):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        try:
            for f in as_completed(futs):
                out_rows.append(f.result())
        except BudgetExceeded as exc:
            print(f"\n  BUDGET STOP: {exc}")
            for f in futs:
                f.cancel()
        except Exception as exc:                       # noqa: BLE001
            print(f"\n  ERROR: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--out", default="outputs/llm")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--decide-every", dest="decide_every", type=int, default=4)
    ap.add_argument("--pilot", action="store_true",
                    help="one episode per model to measure cost per decision epoch")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = BudgetTracker(limit_usd=args.budget)
    rows: list[dict] = []

    if args.pilot:
        print("PILOT: one fever_hot day per model\n")
        for model, eff in [("gpt-4.1-mini", None), ("gpt-5-mini", "low"), ("gpt-5", "low")]:
            before = budget.spent_usd
            run_block([("fever_hot", 0, model, budget, eff, args.decide_every, None)],
                      budget, rows, 1)
            spent = budget.spent_usd - before
            n_ep = 96 // args.decide_every
            print(f"    -> ${spent:.4f} per day, ${spent / n_ep:.5f} per decision epoch\n")
        pd.DataFrame(rows).to_csv(out / "pilot.csv", index=False)
        print(json.dumps(budget.snapshot(), indent=1))
        return

    # ---- full grid, cheapest and broadest first ------------------------
    scen_all = ["normal", "heat", "water_block", "sensor_fault", "fever",
                "fever_hot", "combined"]
    scen_key = ["heat", "fever_hot", "combined"]
    # Grid sized from the pilot's measured cost per simulated day:
    # gpt-4.1-mini ~$0.044, gpt-5-mini ~$0.035, gpt-5 ~$0.33.
    plan = [
        ("gpt-4.1-mini", None, scen_all, [0, 1, 2]),
        ("gpt-5-mini", "low", scen_all, [0, 1, 2]),
        ("gpt-5", "low", scen_key, [0, 1]),
    ]
    tdir = out / "transcripts"
    for model, eff, scens, seeds in plan:
        print(f"\n{model} ({len(scens)} scenarios x {len(seeds)} seeds), "
              f"spend so far ${budget.spent_usd:.3f}")
        jobs = [(sc, sd, model, budget, eff, args.decide_every, tdir)
                for sc in scens for sd in seeds]
        run_block(jobs, budget, rows, args.workers)
        pd.DataFrame(rows).to_csv(out / "llm_results.csv", index=False)
        if budget.spent_usd > budget.limit_usd:
            break

    df = pd.DataFrame(rows)
    df.to_csv(out / "llm_results.csv", index=False)
    (out / "budget.json").write_text(json.dumps(
        {**budget.snapshot(), "prices_usd_per_mtok": PRICES,
         "decide_every": args.decide_every}, indent=2))
    print("\n--- mean by model ---")
    print(df.groupby("model")[["burden", "total_score", "n_verify", "cooling_hours",
                               "n_human_check", "llm_tool_calls",
                               "llm_no_action_epochs"]].mean().round(3).to_string())
    print("\n" + json.dumps(budget.snapshot(), indent=1))


if __name__ == "__main__":
    main()
