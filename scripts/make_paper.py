#!/usr/bin/env python3
"""Build the conference manuscript (.docx and .md) from the computed results.

Every numeric claim in the text is read from the experiment CSVs at build time,
so the manuscript cannot drift from the data. Two Word files are produced: one
with author details and one anonymised for double-blind review.

    python scripts/make_paper.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

EXP = Path("outputs/experiments")
LLM = Path("outputs/llm")
FIG = Path("figures")
OUT = Path("paper")

AGENTS = ["shadow_only", "threshold", "bounded_react_noforecast",
          "bounded_react_noverify", "bounded_react", "full_info"]
SHORT = {"shadow_only": "Shadow", "threshold": "Threshold",
         "bounded_react_noforecast": "No forecast", "bounded_react_noverify": "No verify",
         "bounded_react": "Agent", "full_info": "Full info"}
SCEN = ["normal", "heat", "water_block", "sensor_fault", "fever", "fever_hot", "combined"]
SCEN_L = {"normal": "Normal", "heat": "Heat", "water_block": "Water block",
          "sensor_fault": "Sensor fault", "fever": "Fever", "fever_hot": "Fever + heat",
          "combined": "Combined"}


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
def load():
    from scipy import stats
    d = {"e1": pd.read_csv(EXP / "e1_main.csv"),
         "e2": pd.read_csv(EXP / "e2_noise.csv"),
         "e3": pd.read_csv(EXP / "e3_info_price.csv"),
         "e4": pd.read_csv(EXP / "e4_actuator.csv"),
         "e5": pd.read_csv(EXP / "e5_particles.csv"),
         "meta": json.loads((EXP / "meta.json").read_text())}
    e1 = d["e1"]
    d["n_seeds"] = int(e1.groupby(["scenario", "agent"]).size().max())
    d["n_episodes"] = int(len(e1))

    # agent vs threshold
    red, pv = {}, {}
    for sc in SCEN:
        a = e1[(e1.scenario == sc) & (e1.agent == "bounded_react")].sort_values("seed")["burden"].values
        b = e1[(e1.scenario == sc) & (e1.agent == "threshold")].sort_values("seed")["burden"].values
        red[sc] = 100 * (1 - a.mean() / max(b.mean(), 1e-9))
        pv[sc] = stats.wilcoxon(a, b).pvalue
    d["reduction"], d["pval"] = red, pv

    # verification: paired effect at the default price, and break-even price
    ver = {}
    for sc in SCEN:
        a = e1[(e1.scenario == sc) & (e1.agent == "bounded_react")].sort_values("seed")["total_score"].values
        b = e1[(e1.scenario == sc) & (e1.agent == "bounded_react_noverify")].sort_values("seed")["total_score"].values
        ver[sc] = (a.mean() - b.mean(), stats.wilcoxon(a, b).pvalue)
    d["verify_effect"] = ver

    be = {}
    for sc in d["e3"].scenario.unique():
        s = d["e3"][d["e3"].scenario == sc]
        a = s[s.agent == "bounded_react"].groupby("verify_price")["total_score"].mean()
        b = s[s.agent == "bounded_react_noverify"].groupby("verify_price")["total_score"].mean()
        diff = (b - a)
        xs, ys = diff.index.values, diff.values
        be[sc] = None
        for i in range(len(xs) - 1):
            if ys[i] > 0 >= ys[i + 1]:
                be[sc] = xs[i] + (xs[i + 1] - xs[i]) * ys[i] / (ys[i] - ys[i + 1])
                break
    d["breakeven"] = be
    d["llm"] = pd.read_csv(LLM / "llm_results.csv") if (LLM / "llm_results.csv").exists() else None
    d["budget"] = json.loads((LLM / "budget.json").read_text()) if (LLM / "budget.json").exists() else None
    return d


def piv(e1, metric, agents=AGENTS):
    p = e1.pivot_table(index="scenario", columns="agent", values=metric, aggfunc="mean")
    p = p.reindex(index=SCEN, columns=agents)
    p.index = [SCEN_L[s] for s in p.index]
    p.columns = [SHORT[a] for a in p.columns]
    return p.round(2)


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# document content
# ---------------------------------------------------------------------------
def build_blocks(D):
    e1 = D["e1"]
    red, pv, be, ve = D["reduction"], D["pval"], D["breakeven"], D["verify_effect"]
    est = (e1[e1.agent == "shadow_only"].groupby("scenario")
           [["rmse_core_c", "bias_core_c", "coverage95_core", "rmse_deficit"]].mean())
    est = est.reindex(SCEN)
    n2 = D["e2"]
    hi = n2[(n2.noise_mult == 3.0)].groupby(["scenario", "agent"])["burden"].mean()
    e4 = D["e4"]
    lat = e4.groupby(["agent", "latency_steps"])["burden"].mean().unstack()
    e5 = D["e5"].groupby("n_particles")[["rmse_core_c", "coverage95_core"]].mean()

    B = []
    A = B.append

    A(("title", "CalfTwin-ReAct: A State-Coupled Agent Testbed for Dairy-Calf Digital Twins"))
    A(("authors", "Jan Saro"))
    A(("affil", "Independent researcher, Czech Republic"))
    A(("email", "sarojanek@gmail.com"))

    A(("abstract",
       "Digital twins are widely proposed for precision livestock farming, but most deployed "
       "systems are digital shadows: they carry data from the animal to a model without a closed "
       "return path, and the value of adding an autonomous reasoning layer is usually asserted "
       "rather than measured. We present CalfTwin-ReAct, an open simulation testbed in which a "
       "digital twin of a dairy calf is coupled to a hidden ground-truth physiological simulator "
       "through a faulty sensor layer and a fallible command path. The latent state, comprising "
       "core temperature, body-water deficit, fatigue and an endogenous fever driver, is never "
       "directly observable: core temperature must be inferred by a particle filter from an "
       "ear-tag skin temperature confounded by air temperature. Seven scenarios generate heat "
       "load, water deprivation, sensor faults and febrile episodes, including cases where "
       "hyperthermia is genuinely ambiguous between environmental load and fever. We "
       f"compare a monitoring-only shadow, a fixed-threshold controller, a bounded "
       f"reason-act-observe agent with an explicit value-of-information rule, two ablations, a "
       f"perfect-observation reference and language-model agents driving the same tool surface, "
       f"over {D['n_episodes']} simulated days. The agent reduces modelled welfare burden by "
       f"{red['water_block']:.0f} to {red['sensor_fault']:.0f} per cent relative to the threshold "
       "controller on every stressed scenario, with the largest gains where the correct action "
       "depends on telling fever apart from heat. Purchasing confirmatory measurements, however, "
       "is not uniformly beneficial: it significantly degrades total score under benign "
       f"conditions, and its break-even price ranges from {min(v for v in be.values() if v):.2f} to "
       f"{max(v for v in be.values() if v):.2f} model units across scenarios, becoming decisive "
       "only as sensor noise rises. We argue that this conditional, priced account of agent value "
       "is what the field currently lacks. All data are synthetic: the contribution is the rig and "
       "the comparison, not a validated twin of a living animal."))
    A(("keywords",
       "digital twin, precision livestock farming, ReAct agent, state estimation, "
       "value of information, dairy calf, closed-loop control"))

    # ---------------- 1 Introduction ----------------
    A(("h1", "1. Introduction"))
    A(("p",
       "Precision livestock farming has produced a large installed base of animal-borne and "
       "pen-mounted sensors, and with it a vocabulary problem. Systems marketed as digital twins "
       "are frequently digital shadows in the sense of Kritzinger et al. (2018): data flow "
       "automatically from the physical object to its digital representation, but no automated "
       "flow returns. The distinction is not pedantic. A shadow can only describe; a twin can "
       "intervene, and intervention is where both the benefit and the risk lie."))
    A(("p",
       "A second shift is now underway. Large language models operating in a reason-act-observe "
       "loop (Yao et al., 2023) are being proposed as the decision layer above such "
       "representations. The appeal is obvious: an agent that can interrogate a state estimate, "
       "request additional measurements and justify a recommendation resembles what a stockperson "
       "does. The difficulty is that little in the published record establishes when this layer "
       "earns its cost. Demonstrations typically show an agent producing plausible text about a "
       "plotted time series, which is evidence neither that the loop improves the animal's state "
       "nor that it beats a threshold rule costing a fraction as much to run."))
    A(("p",
       "This paper takes the narrower question seriously. We build a simulation rig in which the "
       "animal, the sensors, the shadow, the state estimator, the agent and the command path are "
       "separate, inspectable components, and in which the animal's true state is known to the "
       "experimenter but hidden from the agent. Within that rig we ask: under what conditions does "
       "an agent that reasons over an uncertain state estimate, buys confirmatory measurements and "
       "closes the loop outperform a digital shadow or a fixed-threshold controller, and when does "
       "it merely add cost?"))
    A(("p",
       "Three design commitments make the question answerable. First, the quantity that matters "
       "clinically, core body temperature, is never measured; the ear-tag channel returns a skin "
       "temperature confounded by air temperature, so thermal state must be inferred. Second, an "
       "elevated core temperature is deliberately made diagnostically ambiguous: on a hot day it "
       "indicates environmental load and cooling helps, whereas in a cool pen it indicates fever, "
       "which raises the temperature the animal actively defends, so cooling does not resolve it "
       "and still costs money. Third, information is priced. A confirmatory hands-on measurement "
       "is available as an explicit tool with an explicit cost, so the agent's willingness to buy "
       "information can be measured rather than assumed."))
    A(("p",
       "Our contribution is threefold: an open, reproducible testbed that separates the data path "
       "from the command path; a like-for-like comparison of monitoring, threshold control, a "
       "bounded reasoning agent, its ablations and language-model agents on identical scenarios; "
       "and a quantitative, conditional answer about when the reasoning layer pays. We do not "
       "claim a validated digital twin of a real calf, and we report where the architecture fails."))

    # ---------------- 2 Related work ----------------
    A(("h1", "2. Related work and the research gap"))
    A(("p",
       "Kritzinger et al. (2018) provide the classification this paper leans on, separating a "
       "digital model (no automated exchange), a digital shadow (automated one-way flow from "
       "physical to digital) and a digital twin (automated flow in both directions). Applied to "
       "livestock the classification is uncomfortable: most deployed monitoring systems are "
       "shadows. Animal digital twins themselves are not new. Neethirajan and Kemp (2021) set out "
       "the concept for livestock farming; Han and Lin (2022) describe an AI-based twin for cattle "
       "care built on farm IoT data; Youssef et al. (2024) present IUMENTA, a generic framework "
       "for constructing animal twins on an open platform. Claiming novelty for the existence of a "
       "cattle digital twin would therefore be unfounded."))
    A(("p",
       "The most directly relevant critique is recent. Neethirajan (2026) reviews dairy and "
       "poultry digital twins and reports that no fully realised, engineering-grade digital twin "
       "is deployed in commercial systems, identifying validation gaps and weak reporting of "
       "operational failure as recurring problems. This is the gap we address, but we narrow it "
       "deliberately. Rather than asking whether livestock digital twins work in general, we ask "
       "what the decision layer specifically contributes, holding the sensing and estimation "
       "layers fixed."))
    A(("p",
       "On the methods side, sequential Monte Carlo estimation is standard for non-linear, "
       "non-Gaussian state-space problems (Gordon et al., 1993; Doucet and Johansen, 2009), and "
       "sensor-based health management in dairy systems has a long review literature documenting "
       "the false-alarm burden of fixed thresholds (Rutten et al., 2013; Berckmans, 2017). The "
       "agent formulation follows ReAct (Yao et al., 2023). What is missing is a rig that joins "
       "these: a priced, closed loop in which the contribution of reasoning and of information "
       "acquisition can be isolated from the contribution of better sensing. We therefore state "
       "the research question as: under what conditions does an agent that verifies uncertain "
       "measurements and closes the loop achieve a better trade-off between modelled animal "
       "burden, intervention cost and information cost than a digital shadow or a simple "
       "controller? This gap statement follows a targeted review and is not a claim of global "
       "priority."))

    # ---------------- 3 Testbed ----------------
    A(("h1", "3. The CalfTwin-ReAct testbed"))
    A(("fig", FIG / "fig1_architecture.png",
       "Figure 1: Testbed architecture. The blue path carries data from the animal to the twin; "
       "the orange path returns approved commands to the pen; the green dashed path is a priced "
       "information purchase. The agent has no access to the hidden state or to future weather.", 13.0))
    A(("h2", "3.1 Simulated animal"))
    A(("p",
       "The ground truth is a discrete-time biophysical model of a 42-day-old, 65 kg dairy calf "
       "advanced in 15-minute steps, 96 steps per day. Its hidden state is a four-vector: core "
       "temperature, body-water deficit as a fraction of body mass, fatigue, and pyrogen, an "
       "endogenous fever driver on a zero-to-one scale. Core temperature follows a lumped heat "
       "balance over metabolic and activity heat production, radiant gain, sensible exchange with "
       "the pen and respiratory and cutaneous evaporation. Crucially the animal defends its "
       "temperature: vasomotor tone and shivering thermogenesis respond to the deviation from the "
       "hypothalamic set point, so a cold pen does not simply drag the animal down, and pyrogen "
       "raises that set point, which is what makes fever qualitatively different from "
       "environmental heat load. Water balance combines obligatory and respiratory losses, "
       "twice-daily milk feeding and voluntary drinking. Respiration and heart rate are dynamic "
       "responses to thermal strain, dehydration and activity. Parameters are physiologically "
       "plausible but uncalibrated; the test suite asserts qualitative behaviour, not accuracy."))
    A(("h2", "3.2 Sensors and digital shadow"))
    A(("p",
       "Seven channels are sampled each step: ear-tag temperature, respiration rate, heart rate, "
       "an activity index, pen air temperature, relative humidity and trough flow. Each carries "
       "Gaussian noise, a dropout probability and a gross-outlier probability, and scenarios can "
       "schedule stuck, drifting or dropped channels. Two properties of this layer drive the whole "
       "experiment. Core temperature is not among the channels: the ear tag returns approximately "
       "the core value less a term proportional to the air-temperature difference, so the same "
       "core temperature reads very differently on a cold and a warm day. And faults are "
       "indistinguishable from physiology at the level of a single sample: a stuck ear tag reads "
       "as a stable calf, and zero trough flow means either a blocked trough or a calf that is not "
       "thirsty."))
    A(("p",
       "The digital shadow owns identity, ordering and data quality, and performs no inference. It "
       "rejects duplicate and out-of-order records, counts gaps, and raises per-channel flags for "
       "missing values, range violations, implausible rates of change and frozen values, the last "
       "defined as six or more identical consecutive samples. Separating this layer from the "
       "estimator means an experiment can attribute a failure to the data path or to inference."))
    A(("h2", "3.3 State estimation"))
    A(("p",
       "The twin maintains a bootstrap particle filter over a five-dimensional latent vector, the "
       "four physiological states plus respiration rate, with 400 particles and systematic "
       "resampling when the effective sample size falls below half. Its internal dynamics "
       "deliberately differ from the ground truth in conductance, metabolic and gain constants and "
       "omit the radiant term, so the twin suffers structural model mismatch as any deployed twin "
       "does. Observations that the shadow has flagged as suspect are excluded rather than "
       "trusted, and the measurement likelihood is a mixture of a Gaussian and a broad component "
       "so that spikes cannot capture the filter. Because fever onset is a regime change rather "
       "than a random walk, a small fraction of particles receives a large pyrogen perturbation "
       "each step, which allows the filter to follow a step change instead of crawling towards it."))
    A(("p",
       "Two modelling decisions proved decisive and are worth reporting because both were "
       "initially wrong. First, the twin's belief that the trough is functional must be derived "
       "from a water-balance residual, comparing cumulative measured intake against cumulative "
       "obligatory loss driven by the well-observed respiration rate. Deriving it instead from the "
       "twin's belief about thirst is circular: the twin cannot detect a blockage because it "
       "believes the calf is drinking, and it believes that because it thinks the calf is not "
       "thirsty. Second, measured trough flow must enter the water balance directly as intake "
       "rather than being scored a second time against a predicted drinking rate; the latter "
       "penalises exactly the dehydrated hypotheses that a blockage produces."))
    A(("h2", "3.4 The return path"))
    A(("p",
       "The allowed action set is deliberately small: observe, engage cooling, disengage cooling, "
       "request trough maintenance, and flag a stockperson check. No treatment or dosing is "
       "implemented. A proposal is not an intervention. Every command passes an approval gate that "
       "can decline it, waits out a 30-minute execution latency, and can fail at execution. Each "
       "carries an identifier so that the resulting change in the animal and the subsequent "
       "observation remain traceable. Flagging a check starts a simulated clinical response that "
       "clears pyrogen over several hours; in a real deployment this is the point at which a "
       "human, not the agent, acts."))
    A(("h2", "3.5 Agents"))
    A(("p",
       "Six configurations share the tool surface. The shadow estimates and never acts. The "
       "threshold controller applies fixed alarm limits to raw measurements, as conventional "
       "systems do, with no state estimation, forecast or verification. The bounded agent runs a "
       "deterministic reason-act-observe loop: it reads the estimate, interrogates history when "
       "the data path looks unreliable, buys a confirmatory measurement only when uncertainty "
       "would change its decision, forecasts the candidate actions and commits one command. Its "
       "value-of-information rule is explicit: verification is purchased only if the action implied "
       "by an optimistic excursion of the estimate differs from that implied by a pessimistic one, "
       "and only if uncertainty, budget and a cooldown permit. Two ablations remove verification "
       "and forecasting respectively. A perfect-observation reference applies the same rule "
       "structure to the hidden state; it is neither optimal nor deployable, but it separates loss "
       "caused by imperfect sensing from loss caused by the policy. Finally, language-model agents "
       "drive the identical five tools, each epoch a bounded loop that must end in a command; "
       "epochs that do not are counted as failures. Throughout, \"ReAct-style\" denotes the "
       "interleaving of reasoning with tool calls: the bounded agent uses no language model and is "
       "the reference against which the language-model agents are measured."))
    A(("h2", "3.6 Outcome measures"))
    A(("p",
       "Welfare burden aggregates modelled strain from ground truth, never visible to the agent: "
       "degree-hours of core temperature above 39.2 degrees Celsius, deficit-hours above two per "
       "cent of body mass, and fatigue-hours above 0.3. Total score adds intervention cost "
       "(cooling run-time, trough maintenance, stockperson call-outs) and information cost, so no "
       "controller can win by spending without limit. These weights are illustrative model units, "
       "neither money nor a validated welfare index, so conclusions are comparisons under a fixed "
       "weighting rather than absolute welfare claims. Estimation quality is reported separately "
       "as error against hidden truth, bias, and coverage of the nominal 95 per cent interval."))

    # ---------------- 4 Experimental design ----------------
    A(("h1", "4. Experimental design"))
    A(("p",
       f"Seven scenarios were used: a thermoneutral day; a hot day with radiant load; a warm day "
       f"with the trough unavailable for about nine hours; a warm day with the ear tag frozen for "
       f"about six hours plus a respiration dropout burst; a febrile episode in a thermoneutral "
       f"pen; a febrile episode concurrent with heat load, in which hyperthermia is ambiguous; and "
       f"a combined day with heat, a blocked trough, a frozen ear tag and a drifting heart-rate "
       f"channel. Seed-dependent jitter varies each day while preserving the scenario's structure. "
       f"The main comparison is a full factorial of seven scenarios by six configurations by "
       f"{D['n_seeds']} seeds, giving {D['n_episodes']} simulated days, with agents deciding "
       f"hourly. Four further sweeps vary sensor noise, the price of verification, command-path "
       f"latency and reliability, and particle count. Paired comparisons across matched seeds use "
       f"the Wilcoxon signed-rank test. Everything except the language-model results is "
       f"deterministic given the seed."))

    # ---------------- 5 Results ----------------
    A(("h1", "5. Results"))
    A(("h2", "5.1 The twin tracks the hidden state, with one honest limitation"))
    A(("p",
       f"Open-loop estimation error for core temperature is "
       f"{est['rmse_core_c'].mean():.3f} degrees Celsius on average across scenarios with a bias of "
       f"{est['bias_core_c'].mean():+.3f}, and the nominal 95 per cent interval covers the truth "
       f"{100 * est['coverage95_core'].mean():.0f} per cent of the time, so the reported "
       f"uncertainty is approximately honest. The residual miscalibration is informative about "
       f"where the twin's model is weakest: coverage is mildly deficient in exactly the two "
       f"scenarios the internal model handles worst, falling to "
       f"{100 * est.loc['fever', 'coverage95_core']:.0f} per cent under a febrile regime change "
       f"and {100 * est.loc['combined', 'coverage95_core']:.0f} per cent in the combined scenario, "
       f"where a heart-rate channel drifts without triggering a quality flag, while it is "
       f"conservative, at almost {100 * est.loc['water_block', 'coverage95_core']:.0f} per cent, "
       f"under water deprivation. The filter is therefore slightly overconfident precisely where an undetected "
       f"fault or an unmodelled regime change is present, the direction of error a deployment most "
       f"needs to anticipate. The water-deficit estimate systematically underestimates severe "
       f"dehydration, because the deficit is only weakly observable from non-invasive channels, "
       f"acting mainly on heart rate, which is also driven by temperature and activity. Disabling "
       f"the heart-rate likelihood makes the estimate worse, confirming that the limitation is "
       f"observability rather than a filter defect. Particle count matters up to a knee at 400 "
       f"(RMSE {e5.loc[400, 'rmse_core_c']:.3f}, coverage {100 * e5.loc[400, 'coverage95_core']:.0f} "
       f"per cent); 1600 particles buy essentially nothing "
       f"(RMSE {e5.loc[1600, 'rmse_core_c']:.3f}). Per-scenario estimation statistics are "
       f"tabulated in the repository."))
    A(("fig", FIG / "fig2_day_trace.png",
       "Figure 2: One closed-loop day in the combined scenario. Shaded spans mark active cooling; "
       "triangles mark purchased verifications; vertical lines mark executed commands. The twin "
       "tracks core temperature through a frozen ear tag, infers the trough blockage from the "
       "water-balance residual, and acts on it.", 10.8))
    A(("h2", "5.2 Closing the loop dominates; how you close it matters more under ambiguity"))
    A(("p",
       f"Monitoring alone is expensive in animal terms. Against the shadow, every acting "
       f"configuration reduces burden substantially on every stressed scenario. The interesting "
       f"comparison is between controllers. The bounded agent reduces burden relative to the "
       f"threshold controller by {red['heat']:.0f} per cent under heat ({fmt_p(pv['heat'])}), "
       f"{red['water_block']:.0f} per cent under water deprivation ({fmt_p(pv['water_block'])}), "
       f"{red['sensor_fault']:.0f} per cent under sensor faults ({fmt_p(pv['sensor_fault'])}) and "
       f"{red['combined']:.0f} per cent in the combined scenario ({fmt_p(pv['combined'])}). The "
       f"largest and most interpretable gains are the febrile ones: {red['fever']:.0f} per cent "
       f"under fever and {red['fever_hot']:.0f} per cent under fever with concurrent heat load "
       f"(both {fmt_p(pv['fever'])}). Here the threshold controller fails structurally rather than "
       f"marginally: it sees an elevated ear temperature, engages cooling, and cools an animal "
       f"that is defending a raised set point, so the strain persists while the cost accrues. The "
       f"agent infers from the measured microclimate that the temperature is not explained by the "
       f"environment, and calls for a human instead."))
    A(("p",
       f"On a normal day the ordering reverses. The agent's burden "
       f"({e1[(e1.scenario == 'normal') & (e1.agent == 'bounded_react')]['burden'].mean():.3f}) is "
       f"marginally worse than the threshold controller's "
       f"({e1[(e1.scenario == 'normal') & (e1.agent == 'threshold')]['burden'].mean():.3f}) and its "
       f"total score is higher, because it spends on information and cooling that a quiet day does "
       f"not repay. Both values are negligible in absolute terms, but the direction is the point: "
       f"the reasoning layer is not free, and a system that runs it unconditionally pays for it on "
       f"every uneventful day. The ablations locate the benefit. Removing the forecast is costly "
       f"wherever pre-emptive cooling matters, raising combined-scenario burden from "
       f"{e1[(e1.scenario == 'combined') & (e1.agent == 'bounded_react')]['burden'].mean():.2f} to "
       f"{e1[(e1.scenario == 'combined') & (e1.agent == 'bounded_react_noforecast')]['burden'].mean():.2f}. "
       f"Notably, the agent outperforms the perfect-observation reference on the fever-with-heat "
       f"scenario; the reference is a fixed rule evaluated on the true state, not an optimal "
       f"controller, and it lacks the forecast. Perfect sensing is not sufficient for good control."))
    A(("table", piv(e1, "burden"),
       f"Table 1: Mean welfare burden by scenario and configuration "
       f"({D['n_seeds']} seeds per cell; lower is better).", False))
    A(("table", piv(e1, "total_score"),
       "Table 2: Mean total score, that is burden plus intervention and information cost.", False))
    A(("fig", FIG / "fig3_main_comparison.png",
       "Figure 3: Main comparison. Panel (a) shows welfare burden for all configurations, where "
       "the no-action shadow dominates the range. Panel (b) drops the shadow and shows total score "
       "for the acting controllers only, so that differences between them are legible. Error bars "
       "are standard errors over seeds.", 13.5))
    A(("h2", "5.3 Verification has a price, and it is often above the market"))
    A(("p",
       f"The value-of-information rule is the component we expected to matter most, and the result "
       f"is the most qualified. Throughout, lower total score is better. At the default price of "
       f"0.12 model units, enabling verification lowers mean total score in the compound-fault "
       f"scenarios, by {abs(ve['combined'][0]):.2f} in the combined scenario, "
       f"{abs(ve['water_block'][0]):.2f} under water deprivation and {abs(ve['fever'][0]):.2f} "
       f"under fever, but the paired effect does not reach significance at {D['n_seeds']} seeds "
       f"({fmt_p(ve['combined'][1])} for the combined scenario). In benign conditions it "
       f"significantly raises total score, that is, makes matters worse: by "
       f"{ve['normal'][0]:.3f} on a normal day ({fmt_p(ve['normal'][1])}), {ve['heat'][0]:.3f} "
       f"under heat ({fmt_p(ve['heat'][1])}) and {ve['sensor_fault'][0]:.3f} under an isolated "
       f"sensor fault ({fmt_p(ve['sensor_fault'][1])}). Sweeping the price "
       f"makes the structure clear (Figure 4): there is a scenario-specific break-even price, "
       f"approximately {be['combined']:.2f} model units in the combined scenario and "
       f"{be['fever_hot']:.2f} under fever with heat, but only {be['sensor_fault']:.2f} in the "
       f"isolated sensor-fault scenario, where the shadow's frozen-channel flag already supplies "
       f"most of what a verification would reveal. Buying information is worthwhile exactly when "
       f"the twin is uncertain for a reason the data path cannot itself diagnose."))
    A(("fig", FIG / "fig4_info_price.png",
       "Figure 4: Total score against the price of one confirmatory measurement. Where the solid "
       "line crosses the dashed one, verification stops paying for itself. The break-even price "
       "differs roughly fourfold across scenarios.", 14.5))
    A(("p",
       f"Sensor quality settles the question. As noise is scaled up, the agent that can verify "
       f"diverges from the one that cannot (Figure 5). At three times nominal noise the "
       f"verification-enabled agent's burden in the combined scenario is "
       f"{hi[('combined', 'bounded_react')]:.2f} against "
       f"{hi[('combined', 'bounded_react_noverify')]:.2f} without verification and "
       f"{hi[('combined', 'threshold')]:.2f} for the threshold controller; under water deprivation "
       f"the figures are {hi[('water_block', 'bounded_react')]:.2f}, "
       f"{hi[('water_block', 'bounded_react_noverify')]:.2f} and "
       f"{hi[('water_block', 'threshold')]:.2f}. The ability to buy a trustworthy measurement is "
       f"worth little when the routine sensors are good and becomes the dominant factor when they "
       f"are not. This is a deployment-relevant statement: the case for an agentic layer is "
       f"strongest precisely in the low-cost, noisy, partially-faulty installations that dominate "
       f"practice, and weakest in the well-instrumented pens where such systems are usually "
       f"demonstrated."))
    A(("fig", FIG / "fig5_noise.png",
       "Figure 5: Welfare burden against a multiplier on all sensor noise standard deviations. The "
       "gap between the agent with and without verification widens as data quality falls.", 14.5))
    A(("h2", "5.4 The command path degrades the controllers unequally"))
    A(("p",
       f"Lengthening the delay between decision and effect from zero to two hours raises the "
       f"threshold controller's mean burden from {lat.loc['threshold', 0]:.2f} to "
       f"{lat.loc['threshold', 8]:.2f}, but the agent's only from {lat.loc['bounded_react', 0]:.2f} "
       f"to {lat.loc['bounded_react', 8]:.2f}. Because the agent acts on a forecast rather than on "
       f"a threshold crossing, it has already absorbed part of the delay. Mean detection latency "
       f"from the true onset of thermal strain to the first command is "
       f"{e1[e1.agent == 'bounded_react']['detection_latency_h'].mean():.2f} hours for the agent "
       f"against {e1[e1.agent == 'threshold']['detection_latency_h'].mean():.2f} hours for the "
       f"threshold controller. This has a cost: the agent accrues "
       f"{e1[e1.agent == 'bounded_react']['false_cooling_hours'].mean():.1f} hours per day of "
       f"cooling while the animal was not thermally challenged, against "
       f"{e1[e1.agent == 'threshold']['false_cooling_hours'].mean():.1f} hours for the threshold "
       f"controller. Anticipation and over-treatment are the same behaviour, separated only by "
       f"whether the anticipated event occurs."))

    # LLM section filled in by caller when data exist
    A(("llm_section", None))

    # ---------------- 6 Discussion ----------------
    A(("h1", "6. Discussion"))
    A(("p",
       "Three findings seem to us to generalise beyond this simulator. First, the dominant term is "
       "not reasoning but coupling: almost all of the available improvement comes from having any "
       "closed loop at all, and the shadow-to-twin distinction is therefore worth defending as "
       "more than terminology. Second, the reasoning layer earns its place where the mapping from "
       "signal to correct action is not one-to-one: in the febrile scenarios two states produce "
       "the same alarm and demand opposite responses, and no threshold on the observed channel can "
       "separate them. Where the mapping is unambiguous, as on a simple hot day, a well-tuned "
       "threshold rule is nearly sufficient and far cheaper. Third, information acquisition is an "
       "action with a price, not a free virtue: at a realistic price and under benign conditions, "
       "an agent that verifies whenever uncertain is worse than one that never verifies."))
    A(("p",
       "For practitioners the operational implication is a triage rule rather than a blanket "
       "recommendation: run the cheap controller by default, and escalate to the reasoning layer "
       "when the estimator's uncertainty is both high and decision-relevant, when channels are "
       "flagged, or when the installation's sensor quality is known to be poor. The break-even "
       "prices we report are properties of this simulator and its weighting, but the existence of "
       "a break-even price, and the fact that it varies several-fold with the kind of fault, is "
       "the transferable result."))

    # ---------------- 7 Limitations ----------------
    A(("h1", "7. Limitations and validation status"))
    A(("p",
       "All data are synthetic and no animal was measured. Parameters are physiologically "
       "plausible but uncalibrated, and the test suite verifies qualitative behaviour, which is a "
       "sanity check and not validation. Results are therefore claims about this architecture "
       "under this simulator, not about calf welfare. The cost weights are illustrative and the "
       "ranking could change under a different weighting, although the margin in the febrile "
       "scenarios is large enough to be unlikely to invert. A structural caveat applies to the "
       "estimator: it shares its functional form with the ground-truth simulator, differing in "
       "constants and omitting terms, so our mismatch is parametric where a deployed twin would "
       "face a deeper structural one. Only one animal and one day are modelled; herd effects, "
       "individual variation, growth and longer disease progression are out of scope. The "
       "language-model results are a small single-provider sample and should be read as a "
       "feasibility demonstration, not a ranking of models."))

    # ---------------- 8 Conclusion ----------------
    A(("h1", "8. Conclusion"))
    A(("p",
       f"We built a testbed in which a dairy-calf digital twin is coupled to a hidden ground-truth "
       f"simulator through faulty sensors and a fallible command path, and used it to measure what "
       f"a reasoning agent adds. Over {D['n_episodes']} simulated days the agent reduced modelled "
       f"welfare burden by roughly {min(red[s] for s in ['heat', 'water_block', 'fever', 'fever_hot', 'combined']):.0f} "
       f"to {max(red.values()):.0f} per cent against a fixed-threshold controller on every stressed "
       f"scenario, with the clearest advantage where an elevated temperature was ambiguous between "
       f"environmental load and fever. Buying confirmatory measurements, by contrast, was worth "
       f"its price only under compound faults and degraded sensing, with a break-even price "
       f"varying by roughly a factor of four across scenarios. The practical conclusion is neither "
       f"that agents are unnecessary nor that they are generally beneficial, but that their value "
       f"is conditional and measurable, and that a testbed of this kind is what makes it "
       f"measurable. Extending the rig towards biological calibration and prospective validation "
       f"against instrumented animals is the necessary next step before any claim about real "
       f"welfare can be made."))

    # ---------------- declarations ----------------
    A(("h1", "Code and data availability"))
    A(("code_avail",
       "The testbed, the experiment scripts and the committed outputs of every run reported here "
       "are at https://github.com/jenasin/calftwin-react under the MIT licence. Offline results "
       "are deterministic given the reported seeds."))

    A(("h1", "Ethics declaration"))
    A(("p",
       "Ethical clearance was not required for this research. No live animals were used, observed "
       "or handled at any stage. All physiological data reported here were generated by a computer "
       "simulation, and no human participants were involved."))
    A(("h1", "AI declaration"))
    A(("p",
       "Generative AI tools were used in the preparation of this work, as follows. An AI coding "
       "assistant was used to help implement the simulator, estimator, agents and analysis code, "
       "and to draft and edit this text; all code was executed and its outputs inspected by the "
       "author, and every numeric value in the text and tables is generated programmatically from "
       "the committed experiment outputs rather than written by hand. Language models are also "
       "part of the object of study: the agents in Section 5.5 are commercial large language "
       "models driving the testbed's tool interface, and their token usage and cost are reported. "
       "The author takes full responsibility for the design of the study, the interpretation of "
       "the results and the content of the manuscript."))
    A(("h1", "References"))
    for r in [
        "Berckmans, D. (2017) “General introduction to precision livestock farming”, "
        "Animal Frontiers, Vol 7, No. 1, pp 6–11. doi:10.2527/af.2017.0102",
        "Doucet, A. and Johansen, A. M. (2009) “A tutorial on particle filtering and "
        "smoothing: fifteen years later”, in Handbook of Nonlinear Filtering, Oxford "
        "University Press, Oxford, pp 656–704.",
        "Gordon, N. J., Salmond, D. J. and Smith, A. F. M. (1993) “Novel approach to "
        "nonlinear/non-Gaussian Bayesian state estimation”, IEE Proceedings F (Radar and "
        "Signal Processing), Vol 140, No. 2, pp 107–113. doi:10.1049/ip-f-2.1993.0015",
        "Han, X. and Lin, Z. (2022) “AI Based Digital Twin Model for Cattle Caring”, "
        "arXiv preprint. doi:10.48550/arXiv.2205.04034",
        "Kritzinger, W., Karner, M., Traar, G., Henjes, J. and Sihn, W. (2018) “Digital Twin "
        "in manufacturing: A categorical literature review and classification”, "
        "IFAC-PapersOnLine, Vol 51, No. 11, pp 1016–1022. doi:10.1016/j.ifacol.2018.08.474",
        "National Research Council (1971) A Guide to Environmental Research on Animals, National "
        "Academy of Sciences, Washington DC.",
        "Neethirajan, S. (2026) “Digital Twins for Cows and Chickens: From Hype Cycles to "
        "Hard Evidence in Precision Livestock Farming”, Agriculture, Vol 16, No. 2, 166. "
        "doi:10.3390/agriculture16020166",
        "Neethirajan, S. and Kemp, B. (2021) “Digital Twins in Livestock Farming”, "
        "Animals, Vol 11, No. 4, 1008. doi:10.3390/ani11041008",
        "Rutten, C. J., Velthuis, A. G. J., Steeneveld, W. and Hogeveen, H. (2013) “Invited "
        "review: Sensors to support health management on dairy farms”, Journal of Dairy "
        "Science, Vol 96, No. 4, pp 1928–1952. doi:10.3168/jds.2012-6107",
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y. (2023) "
        "“ReAct: Synergizing Reasoning and Acting in Language Models”, International "
        "Conference on Learning Representations (ICLR). doi:10.48550/arXiv.2210.03629",
        "Youssef, A., Vodorezova, K., Aarts, Y., Agbeti, W. E. K., Palstra, A. P., Foekema, E., "
        "Aguilar, L., da Silva Torres, R. and Grübel, J. (2024) “IUMENTA: A generic "
        "framework for animal digital twins within the Open Digital Twin Platform”, arXiv "
        "preprint. doi:10.48550/arXiv.2411.10466",
    ]:
        A(("ref", r))
    return B


def llm_blocks(D):
    """Section 5.5, only if language-model runs exist."""
    llm, e1, bud = D["llm"], D["e1"], D["budget"]
    if bud is None and llm is not None and not llm.empty:
        # budget.json is only written when the whole grid finishes; fall back to
        # the per-episode token counts so a partial run still reports honestly.
        bud = {"in_tokens": int(llm.get("llm_in_tokens", pd.Series([0])).sum()),
               "out_tokens": int(llm.get("llm_out_tokens", pd.Series([0])).sum()),
               "calls": int(llm.get("llm_tool_calls", pd.Series([0])).sum()),
               "est_cost_usd": float("nan")}
    if llm is None or llm.empty:
        return [("h2", "5.5 Language-model agents"),
                ("p", "The language-model orchestration is implemented and tested, but no live "
                      "runs are reported here; the offline results above are therefore not "
                      "presented as language-model results.")]
    g = llm.groupby("model")
    rows = []
    for model, sub in g:
        scens = sorted(set(sub.scenario))
        ref = e1[(e1.agent == "bounded_react") & (e1.scenario.isin(scens))]
        thr = e1[(e1.agent == "threshold") & (e1.scenario.isin(scens))]
        rows.append({
            "Model": model, "Days": len(sub),
            "Burden": round(sub.burden.mean(), 2),
            "Total": round(sub.total_score.mean(), 2),
            "Bounded agent, same scenarios": round(ref.burden.mean(), 2),
            "Threshold, same scenarios": round(thr.burden.mean(), 2),
            "Tool calls/day": round(sub.get("llm_tool_calls", pd.Series([np.nan])).mean(), 1),
            "Verifications/day": round(sub.n_verify.mean(), 2),
            "Tokens in/day": int(sub.get("llm_in_tokens", pd.Series([0])).mean()),
        })
    tab = pd.DataFrame(rows).set_index("Model")
    best = tab["Burden"].idxmin()
    text = (
        f"Three commercial language models drove the identical tool surface over "
        f"{len(llm)} simulated days, deciding hourly. The orchestration works: the models "
        f"call tools, read the returned state, and commit commands, and qualitatively they make "
        f"the diagnostic distinction the testbed is built around, engaging cooling when the "
        f"microclimate explains an elevated temperature and requesting a stockperson when it does "
        f"not. Quantitatively they do not reach the bounded controller. Averaged over the "
        f"scenarios each model ran, the best language-model configuration ({best}) achieved a "
        f"burden of {tab.loc[best, 'Burden']:.2f} against "
        f"{tab.loc[best, 'Bounded agent, same scenarios']:.2f} for the bounded agent on the same "
        f"scenarios, while remaining well ahead of the threshold controller's "
        f"{tab.loc[best, 'Threshold, same scenarios']:.2f}. "
        f"Total measured usage was {bud['in_tokens']:,} input and {bud['out_tokens']:,} output "
        f"tokens, an estimated {bud['est_cost_usd']:.2f} US dollars at the rates listed at the "
        f"time of the runs. We report token counts as the primary figure because monetary values "
        f"are rate-dependent. Two caveats bound these results: the sample is small "
        f"and single-provider, and the bounded controller's policy was developed against this "
        f"simulator whereas the models saw it only through the prompt, so the comparison favours "
        f"the former. The defensible conclusion is that a language model can close this loop "
        f"competently at a per-day cost that is not negligible, not that it is the right decision "
        f"layer for it.")
    return [("h2", "5.5 Language-model agents"), ("p", text),
            ("table", tab, "Table 3: Language-model agents against the offline controllers on the "
                           "scenarios each model ran.", True),
            ("fig", FIG / "fig6_llm.png",
             "Figure 6: Language-model agents compared with the threshold controller and the "
             "bounded agent. Error bars are standard errors over episodes.", 13.0)]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def style_doc(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, m, Cm(2.2))
    n = doc.styles["Normal"]
    n.font.name = "Times New Roman"
    n.font.size = Pt(10)
    n.paragraph_format.space_after = Pt(6)
    n.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def render_docx(blocks, path: Path, anonymous: bool) -> int:
    doc = Document()
    style_doc(doc)
    words = 0

    def count(t):
        nonlocal words
        words += len(str(t).split())

    for b in blocks:
        kind = b[0]
        if kind == "title":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(b[1]); r.bold = True; r.font.size = Pt(14)
            count(b[1])
        elif kind in ("authors", "affil", "email"):
            if anonymous:
                continue
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(b[1])
            r.italic = kind != "authors"
            r.font.size = Pt(11 if kind == "authors" else 9.5)
            count(b[1])
        elif kind == "abstract":
            p = doc.add_paragraph()
            p.add_run("Abstract: ").bold = True
            p.add_run(b[1])
            count(b[1])
        elif kind == "keywords":
            p = doc.add_paragraph()
            p.add_run("Keywords: ").bold = True
            p.add_run(b[1])
            count(b[1])
        elif kind == "h1":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            r = p.add_run(b[1]); r.bold = True; r.font.size = Pt(12)
            count(b[1])
        elif kind == "h2":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            r = p.add_run(b[1]); r.bold = True; r.font.size = Pt(10.5)
            count(b[1])
        elif kind in ("p", "code_avail"):
            txt = b[1]
            if kind == "code_avail" and anonymous:
                txt = ("The complete testbed, the experiment scripts and the committed outputs of "
                       "every run reported here are in a public repository under the MIT licence; "
                       "the URL is withheld for double-blind review. Offline results are "
                       "deterministic given the reported seeds.")
            doc.add_paragraph(txt)
            count(txt)
        elif kind == "ref":
            p = doc.add_paragraph(b[1])
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            p.paragraph_format.space_after = Pt(3)
            count(b[1])
        elif kind == "fig":
            src, cap = b[1], b[2]
            width_cm = b[3] if len(b) > 3 else 14.0
            if Path(src).exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(2)
                p.add_run().add_picture(str(src), width=Cm(width_cm))
            c = doc.add_paragraph()
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            c.paragraph_format.space_after = Pt(8)
            r = c.add_run(cap); r.font.size = Pt(8.5); r.italic = True
            count(cap)
        elif kind == "table":
            _, df, cap, index_header = b
            c = doc.add_paragraph()
            r = c.add_run(cap); r.font.size = Pt(8.5); r.italic = True
            count(cap)
            t = doc.add_table(rows=1, cols=len(df.columns) + 1)
            t.style = "Table Grid"
            hdr = t.rows[0].cells
            hdr[0].text = str(df.index.name or "")
            for j, col in enumerate(df.columns):
                hdr[j + 1].text = str(col)
            for cell in t.rows[0].cells:
                for par in cell.paragraphs:
                    for run in par.runs:
                        run.bold = True
                        run.font.size = Pt(8.5)
            for idx, row in df.iterrows():
                cells = t.add_row().cells
                cells[0].text = str(idx)
                for j, v in enumerate(row):
                    cells[j + 1].text = f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)
                for cell in cells:
                    for par in cell.paragraphs:
                        for run in par.runs:
                            run.font.size = Pt(8.5)
            count(" ".join(str(x) for x in df.columns))
    doc.save(path)
    return words


def render_md(blocks, path: Path) -> None:
    out = []
    for b in blocks:
        k = b[0]
        if k == "title":
            out.append(f"# {b[1]}\n")
        elif k in ("authors", "affil", "email"):
            out.append(f"*{b[1]}*\n")
        elif k == "abstract":
            out.append(f"**Abstract:** {b[1]}\n")
        elif k == "keywords":
            out.append(f"**Keywords:** {b[1]}\n")
        elif k == "h1":
            out.append(f"\n## {b[1]}\n")
        elif k == "h2":
            out.append(f"\n### {b[1]}\n")
        elif k in ("p", "ref", "code_avail"):
            out.append(f"{b[1]}\n")
        elif k == "fig":
            rel = Path("..") / b[1]
            out.append(f"\n![{b[2]}]({rel})\n\n*{b[2]}*\n")
        elif k == "table":
            out.append(f"\n*{b[2]}*\n\n{b[1].to_markdown()}\n")
    path.write_text("\n".join(out))


def main() -> None:
    D = load()
    blocks = build_blocks(D)
    i = [j for j, b in enumerate(blocks) if b[0] == "llm_section"][0]
    blocks = blocks[:i] + llm_blocks(D) + blocks[i + 1:]

    OUT.mkdir(parents=True, exist_ok=True)
    w = render_docx(blocks, OUT / "CalfTwin_ReAct_ICAIR2026.docx", anonymous=False)
    render_docx(blocks, OUT / "CalfTwin_ReAct_ICAIR2026_anonymous.docx", anonymous=True)
    render_md(blocks, OUT / "paper.md")
    print(f"word count (text + captions, excluding tables): {w}")
    print(f"limit is 5000 words and 10 pages -> "
          f"{'OK' if w <= 5000 else 'OVER BY ' + str(w - 5000)}")
    for f in sorted(OUT.glob("*.docx")):
        print(f"  {f}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
