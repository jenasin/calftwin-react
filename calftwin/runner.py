"""Simulation loop and experiment orchestration.

Step ordering per 15-minute interval:

1. ``bus.tick``      -- due commands take physical effect (or fail)
2. ``sim.step``      -- the animal evolves under those conditions
3. ``twin.observe``  -- sensors are sampled, shadow ingests, filter updates
4. ``agent.decide``  -- at decision epochs only
5. ``twin.log_step`` -- truth and estimate recorded side by side

Truth is recorded only for evaluation; nothing in the agent path reads it
(``OracleAgent`` excepted, and it is reported as an upper bound).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .actuators import ActuatorBus
from .agents.baselines import FullInfoAgent, ShadowOnlyAgent, ThresholdAgent
from .agents.bounded import BoundedReActAgent
from .calf import CalfSimulator, CalfState
from .config import STEPS_PER_DAY, ActuatorParams, CalfParams, CostWeights, RunConfig
from .metrics import evaluate
from .scenarios import build as build_scenario
from .sensors import SensorArray
from .twin import DigitalTwin

AGENTS = ("shadow_only", "threshold", "bounded_react", "bounded_react_noverify",
          "bounded_react_noforecast", "full_info", "llm_react")


def make_agent(name: str, **kw: Any):
    if name == "shadow_only":
        return ShadowOnlyAgent()
    if name == "threshold":
        return ThresholdAgent()
    if name == "bounded_react":
        return BoundedReActAgent()
    if name == "bounded_react_noverify":
        return BoundedReActAgent(allow_verify=False)
    if name == "bounded_react_noforecast":
        return BoundedReActAgent(allow_forecast=False)
    if name == "full_info":
        return FullInfoAgent()
    if name == "llm_react":
        from .agents.llm_react import LLMReActAgent
        return LLMReActAgent(**kw)
    raise ValueError(f"unknown agent {name!r}; choose from {AGENTS}")


def run_episode(scenario: str = "normal", seed: int = 0, agent: str = "bounded_react",
                days: int = 1, decide_every: int = 4, n_particles: int = 400,
                sensor_noise_mult: float = 1.0, costs: CostWeights | None = None,
                actuator: ActuatorParams | None = None,
                agent_kwargs: dict[str, Any] | None = None,
                allow_actions: bool = True) -> dict[str, Any]:
    """Run one episode and return trace, metrics and artefacts."""
    costs = costs or CostWeights()
    actuator = actuator or ActuatorParams()
    spec = build_scenario(scenario, np.random.default_rng(10_000 + seed))

    sim = CalfSimulator(params=CalfParams(), env=spec.env, actuator=actuator,
                        rng=np.random.default_rng(seed), state=CalfState())
    sim.pyrogen_schedule = list(spec.pyrogen_schedule)
    sensors = SensorArray(sim=sim, rng=np.random.default_rng(seed + 1),
                          noise_mult=sensor_noise_mult, faults=list(spec.faults))
    bus = ActuatorBus(sim, actuator, np.random.default_rng(seed + 2))
    twin = DigitalTwin(sim, sensors, bus, costs, seed=seed, n_particles=n_particles)
    ag = make_agent(agent, **(agent_kwargs or {}))

    n_steps = days * STEPS_PER_DAY
    # burn-in: let the filter settle before the agent is allowed to act
    for _ in range(8):
        bus.tick(sim.step_idx)
        sim.step()
        twin.observe()
        twin.log_step(sim.truth_record(), None)

    for k in range(n_steps):
        bus.tick(sim.step_idx)
        sim.step()
        twin.observe()
        dec = None
        if allow_actions and (k % decide_every == 0):
            dec = ag.decide(twin)
        twin.log_step(sim.truth_record(), dec)

    df = pd.DataFrame(twin.trace)
    df["action"] = df.get("action", pd.Series([None] * len(df))).ffill().fillna("observe")
    llm_stats = getattr(ag, "stats", None)
    met = evaluate(df, twin, costs, llm_stats)
    met.update({"scenario": scenario, "seed": seed, "agent": getattr(ag, "name", agent),
                "days": days, "decide_every": decide_every,
                "sensor_noise_mult": sensor_noise_mult})
    return {"trace": df, "metrics": met, "twin": twin, "sim": sim,
            "scenario_spec": spec, "agent": ag}


def run_grid(scenarios: list[str], seeds: list[int], agents: list[str],
             progress: bool = True, **kw: Any) -> pd.DataFrame:
    rows = []
    total = len(scenarios) * len(seeds) * len(agents)
    i = 0
    for sc in scenarios:
        for ag in agents:
            for sd in seeds:
                i += 1
                res = run_episode(scenario=sc, seed=sd, agent=ag, **kw)
                rows.append(res["metrics"])
                if progress and i % 25 == 0:
                    print(f"  [{i}/{total}] {sc}/{ag}/seed{sd}", flush=True)
    return pd.DataFrame(rows)


def save_episode(res: dict[str, Any], outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    res["trace"].to_csv(out / "trace.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(res["metrics"], indent=2, default=str))
    (out / "commands.json").write_text(json.dumps(
        [c.to_dict() for c in res["twin"].bus.log], indent=2))
    (out / "tool_calls.json").write_text(json.dumps(res["twin"].tool_calls, indent=2))
    (out / "scenario.json").write_text(json.dumps(
        {"name": res["scenario_spec"].name,
         "description": res["scenario_spec"].description,
         "env": asdict(res["scenario_spec"].env),
         "faults": [{k: v for k, v in f.__dict__.items() if not k.startswith("_")}
                    for f in res["scenario_spec"].faults],
         "pyrogen_schedule": res["scenario_spec"].pyrogen_schedule}, indent=2))
    return out
