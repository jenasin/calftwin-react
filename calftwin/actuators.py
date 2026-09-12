"""Command path: the return leg that turns a digital decision into a physical act.

A proposal is not an intervention. Every command passes an approval gate, waits
out an execution latency, and can fail. Each command carries an id so that the
resulting change in the animal and the subsequent observation remain traceable.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calf import CalfSimulator
from .config import ActuatorParams

ACTIONS = ("observe", "cooling_on", "cooling_off", "restore_water", "flag_human_check")
_ids = itertools.count(1)


@dataclass
class Command:
    cmd_id: str
    action: str
    issued_step: int
    rationale: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"      # proposed|approved|rejected|executed|failed
    execute_step: int | None = None
    executed_step: int | None = None
    effect: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"cmd_id": self.cmd_id, "action": self.action,
                "issued_step": self.issued_step, "status": self.status,
                "execute_step": self.execute_step, "executed_step": self.executed_step,
                "rationale": self.rationale[:400], "effect": self.effect}


class ActuatorBus:
    """Holds pending commands and applies them to the simulated pen."""

    def __init__(self, sim: CalfSimulator, params: ActuatorParams,
                 rng: np.random.Generator) -> None:
        self.sim = sim
        self.p = params
        self.rng = rng
        self.pending: list[Command] = []
        self.log: list[Command] = []
        self.cooling_hours = 0.0
        self.n_water_fix = 0
        self.n_human_check = 0
        self.n_rejected = 0
        self.n_failed = 0
        self._treatment_until_step: int | None = None

    # ---- issue ------------------------------------------------------------
    def issue(self, action: str, step: int, rationale: str,
              evidence: dict[str, Any] | None = None) -> Command:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; allowed: {ACTIONS}")
        cmd = Command(cmd_id=f"CMD-{next(_ids):05d}", action=action, issued_step=step,
                      rationale=rationale, evidence=evidence or {})
        if action == "observe":
            cmd.status = "executed"
            cmd.executed_step = step
            cmd.effect = "no physical change"
            self.log.append(cmd)
            return cmd
        if self.rng.random() > self.p.approval_p:
            cmd.status = "rejected"
            cmd.effect = "approval gate declined"
            self.n_rejected += 1
            self.log.append(cmd)
            return cmd
        cmd.status = "approved"
        lat = (self.p.human_check_latency_steps if action == "flag_human_check"
               else self.p.latency_steps)
        cmd.execute_step = step + lat
        self.pending.append(cmd)
        self.log.append(cmd)
        return cmd

    # ---- apply ------------------------------------------------------------
    def tick(self, step: int) -> list[Command]:
        """Execute any due commands. Call once per simulation step."""
        done: list[Command] = []
        for cmd in list(self.pending):
            if cmd.execute_step is not None and step >= cmd.execute_step:
                self.pending.remove(cmd)
                if self.rng.random() > self.p.exec_success_p:
                    cmd.status = "failed"
                    cmd.effect = "actuator reported failure"
                    self.n_failed += 1
                    done.append(cmd)
                    continue
                self._apply(cmd, step)
                cmd.status = "executed"
                cmd.executed_step = step
                done.append(cmd)

        # cooling run-time accounting and treatment expiry
        if self.sim.effects.cooling_on:
            self.cooling_hours += 0.25
        if (self._treatment_until_step is not None
                and step >= self._treatment_until_step):
            self.sim.effects.treatment_active = False
            self._treatment_until_step = None
        return done

    def _apply(self, cmd: Command, step: int) -> None:
        eff = self.sim.effects
        if cmd.action == "cooling_on":
            eff.cooling_on = True
            cmd.effect = "pen cooling and forced airflow engaged"
        elif cmd.action == "cooling_off":
            eff.cooling_on = False
            cmd.effect = "pen cooling disengaged"
        elif cmd.action == "restore_water":
            self.n_water_fix += 1
            if self.rng.random() < self.p.water_fix_success_p:
                eff.water_fixed = True
                cmd.effect = "trough blockage cleared"
            else:
                cmd.effect = "maintenance attempt did not clear the blockage"
        elif cmd.action == "flag_human_check":
            self.n_human_check += 1
            eff.treatment_active = True
            self._treatment_until_step = step + 24      # 6 h of clinical response
            cmd.effect = "stockperson examination logged; supportive care started"

    # ---- reporting --------------------------------------------------------
    def recent(self, n: int = 6) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.log[-n:]]

    def counts(self) -> dict[str, int | float]:
        by_action = {a: sum(1 for c in self.log if c.action == a and c.status == "executed")
                     for a in ACTIONS}
        return {"executed_by_action": by_action,
                "cooling_hours": round(self.cooling_hours, 2),
                "water_fix": self.n_water_fix, "human_check": self.n_human_check,
                "rejected": self.n_rejected, "failed": self.n_failed,
                "issued": len(self.log)}
