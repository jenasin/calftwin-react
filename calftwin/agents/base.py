"""Agent interface.

An agent receives the twin (its tool surface) and returns one decision per
decision epoch. It must not touch ``twin.sim``; the only exception is
:class:`~calftwin.agents.baselines.OracleAgent`, which is an evaluation-only
upper bound and is labelled as such everywhere it is reported.
"""
from __future__ import annotations

from typing import Any, Protocol

from ..twin import DigitalTwin


class Agent(Protocol):
    name: str

    def decide(self, twin: DigitalTwin) -> dict[str, Any]:
        """Return {"action": str, "reasoning": str, "n_tool_calls": int}."""
        ...


def decision(action: str, reasoning: str, n_tool_calls: int = 0,
             **extra: Any) -> dict[str, Any]:
    return {"action": action, "reasoning": reasoning,
            "n_tool_calls": n_tool_calls, **extra}
