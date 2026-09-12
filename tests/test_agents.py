"""Agent contract, command path and closed-loop behaviour."""
import numpy as np
import pytest

from calftwin.actuators import ACTIONS
from calftwin.config import ActuatorParams
from calftwin.runner import run_episode

OFFLINE = ["shadow_only", "threshold", "bounded_react",
           "bounded_react_noverify", "full_info"]


@pytest.mark.parametrize("agent", OFFLINE)
def test_agents_only_emit_allowed_actions(agent):
    r = run_episode(scenario="combined", seed=2, agent=agent)
    assert set(r["trace"]["action"].unique()) <= set(ACTIONS)


def test_agent_cannot_read_hidden_state():
    """The tool surface must not leak ground truth to the agent."""
    r = run_episode(scenario="fever_hot", seed=2, agent="bounded_react")
    twin = r["twin"]
    leaked = {"t_core_c", "water_deficit", "fatigue", "pyrogen"}
    state = twin.get_state()
    assert not (leaked & set(state["measured"])), "hidden variables exposed as measurements"
    assert state["estimate"]["t_core_c"] != pytest.approx(twin.sim.state.t_core_c, abs=1e-9)


def test_shadow_only_never_changes_the_pen():
    r = run_episode(scenario="heat", seed=2, agent="shadow_only")
    c = r["twin"].bus.counts()["executed_by_action"]
    assert all(v == 0 for k, v in c.items() if k != "observe")


def test_commands_are_delayed_and_traceable():
    r = run_episode(scenario="heat", seed=2, agent="bounded_react")
    log = [c for c in r["twin"].bus.log if c.action != "observe" and c.status == "executed"]
    assert log, "no command was ever executed"
    for c in log:
        assert c.executed_step > c.issued_step, "command executed without latency"
        assert c.cmd_id.startswith("CMD-")


def test_rejected_commands_have_no_effect():
    """With the approval gate closed, nothing may reach the animal."""
    r = run_episode(scenario="heat", seed=2, agent="bounded_react",
                    actuator=ActuatorParams(approval_p=0.0))
    assert r["twin"].bus.counts()["cooling_hours"] == 0.0
    assert r["twin"].bus.n_rejected > 0


def test_closed_loop_beats_monitoring_only_under_heat_load():
    passive = run_episode(scenario="heat", seed=2, agent="shadow_only")["metrics"]
    closed = run_episode(scenario="heat", seed=2, agent="bounded_react")["metrics"]
    assert closed["burden"] < passive["burden"], "closing the loop did not help"


def test_verification_is_used_only_when_uncertainty_matters():
    calm = run_episode(scenario="normal", seed=2, agent="bounded_react")["metrics"]
    hard = run_episode(scenario="combined", seed=2, agent="bounded_react")["metrics"]
    assert calm["n_verify"] <= hard["n_verify"]


def test_ablation_without_verification_spends_nothing_on_information():
    r = run_episode(scenario="combined", seed=2, agent="bounded_react_noverify")
    assert r["metrics"]["n_verify"] == 0
    assert r["metrics"]["info_cost"] == 0.0


def test_cooling_does_not_chatter():
    """Switching the pen on and off every epoch is a control defect."""
    r = run_episode(scenario="heat", seed=2, agent="bounded_react")
    acts = r["trace"]["action"].tolist()
    switches = sum(1 for a, b in zip(acts, acts[1:])
                   if a != b and {a, b} <= {"cooling_on", "cooling_off"})
    assert switches <= 6, f"{switches} direct on/off reversals"


def test_reproducible_given_a_seed():
    a = run_episode(scenario="combined", seed=9, agent="bounded_react")["metrics"]
    b = run_episode(scenario="combined", seed=9, agent="bounded_react")["metrics"]
    assert a["burden"] == pytest.approx(b["burden"])
    assert a["total_score"] == pytest.approx(b["total_score"])


def test_every_agent_key_reports_a_distinct_name():
    """Two configurations sharing a reported name silently merge experiment rows."""
    from calftwin.runner import AGENTS, make_agent
    names = {}
    for key in AGENTS:
        if key == "llm_react":
            continue
        n = make_agent(key).name
        assert n not in names, f"{key!r} and {names[n]!r} both report name {n!r}"
        names[n] = key
