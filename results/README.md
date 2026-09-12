# Committed experiment outputs

These are the exact files that `scripts/run_experiments.py --seeds 50` produced
and that `scripts/make_figures.py` and the manuscript read, so every reported
number can be checked without re-running anything.

| File | Contents |
|---|---|
| `e1_main.csv` | Main grid: 7 scenarios x 6 configurations x 50 seeds. |
| `e2_noise.csv` | Sensor-noise multiplier sweep. |
| `e3_info_price.csv` | Price-of-verification sweep. |
| `e4_actuator.csv` | Command-path latency and reliability sweep. |
| `e5_particles.csv` | Particle-count sweep. |
| `meta.json` | Run configuration and wall-clock time. |
| `llm_results.csv` | Language-model episodes. |
| `pilot.csv` | One fever-with-heat day per model, used to size the grid by measured cost. |
| `budget.json` | Measured token usage and estimated spend. |

One row is one simulated day. Re-running the scripts reproduces these files
exactly for the offline experiments; language-model rows are not deterministic.

**The language-model grid is incomplete.** The run stopped when the API account's
credit was exhausted, so `gpt-5-mini` never reached the `combined` and
`fever_hot` scenarios and `gpt-5` ran only the single pilot day in `pilot.csv`
(fever + heat, burden 2.68 against the bounded agent's 2.98 on that scenario --
one episode, far too few to compare). The manuscript therefore reports the
language-model comparison only on the five scenarios both completed models ran.
Re-running `scripts/run_llm_experiments.py` with credit available completes the
grid.
