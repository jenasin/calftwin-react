"""Language-model ReAct agent.

The model is given the same tool surface as the bounded controller and nothing
else: it cannot read the hidden state, the future weather, or the evaluation
metric. Each decision epoch is one bounded reason-act-observe loop that must end
in ``propose_action``; if it does not, the epoch defaults to ``observe`` and the
failure is counted. Token usage is recorded per episode so that the cost of the
agent is reported alongside its effect on the animal.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..actuators import ACTIONS
from ..twin import DigitalTwin
from .base import decision

# USD per 1M tokens, as listed at the time of the reported runs. Token counts are
# the primary figure reported; monetary values are derived and rate-dependent.
PRICES = {
    "gpt-4.1":      (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-5":        (1.25, 10.00),
    "gpt-5-mini":   (0.25, 2.00),
}

SYSTEM_PROMPT = """You are the decision layer of a digital twin of one dairy calf (CALF-001, 42 days old, 65 kg) housed in an instrumented pen.

You cannot see the calf. You can only call tools. The twin maintains a particle-filter estimate of four latent variables you never observe directly:
- t_core_c: core body temperature. Normal 38.5-39.2 C. Above 39.2 C is thermal strain.
- water_deficit: body-water deficit as a fraction of body mass. Above 0.02 is a concern, above 0.05 is serious.
- pyrogen: 0-1 endogenous fever driver. Fever raises the temperature the calf actively defends, so cooling does NOT resolve it; clinical attention does.
- fatigue: 0-1 accumulated strain.

Every estimate comes with a standard deviation and with probabilities (p_hyperthermia, p_dehydrated, p_fever_driven, p_trough_ok). Sensors fail: an ear tag can freeze, a channel can drop out, a heart-rate channel can drift. The twin flags suspect channels but cannot catch every fault.

The central judgement you must make is DIAGNOSTIC, not just reactive:
- A high core temperature on a hot day (high THI) is environmental load -> cooling_on helps.
- A high core temperature when the pen is NOT hot points to fever -> flag_human_check; cooling is the wrong answer and still costs money.
- Absent trough flow may mean a blocked trough OR a calf that is simply not thirsty. p_trough_ok is the twin's belief that the trough works.

Tools:
- get_state(): current estimate, latest measurements, THI, data-quality flags, pen status. Free.
- get_history(channel, hours): recent measured series and trend for one channel. Free.
- forecast_options(horizon_h, options): predicted degree-hours of heat load under candidate actions. Free.
- verify_vitals(): a hands-on confirmatory core-temperature and trough check. COSTS 0.12 score units. Use it only when your action would actually change depending on the answer.
- propose_action(action, rationale): submit exactly one command. Required to end the epoch.

Allowed actions: observe, cooling_on, cooling_off, restore_water, flag_human_check.

Costs charged against you: cooling 0.06 per hour of running time, restore_water 0.10, flag_human_check 0.55, verify_vitals 0.12. Animal strain is charged at 1.0 per degree-hour above 39.2 C, 1.0 per (100 x deficit)-hour above 0.02, and 0.6 per fatigue-hour above 0.3. Your objective is to minimise strain PLUS cost, so act when the expected strain avoided exceeds the price of acting, and not otherwise.

Commands are not instantaneous: approval can be declined, execution is delayed about 30 minutes, and it can fail. Do not re-issue a command that is already pending. If cooling is already on, leaving it on requires no new command - propose observe.

Work in at most 4 tool calls before proposing. Be brief.
"""


class BudgetExceeded(RuntimeError):
    pass


class BudgetTracker:
    """Process-wide spend guard so an experiment cannot run away with the bill."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.in_tokens = 0
        self.out_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    def charge(self, model: str, usage: Any) -> None:
        pin, pout = PRICES.get(model, (2.0, 8.0))
        it = int(getattr(usage, "prompt_tokens", 0) or 0)
        ot = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._lock:
            self.in_tokens += it
            self.out_tokens += ot
            self.calls += 1
            self.spent_usd += it / 1e6 * pin + ot / 1e6 * pout
            spent = self.spent_usd
        if spent > self.limit_usd:
            raise BudgetExceeded(
                f"spend {spent:.3f} USD exceeded limit {self.limit_usd:.2f}")

    def snapshot(self) -> dict[str, Any]:
        return {"calls": self.calls, "in_tokens": self.in_tokens,
                "out_tokens": self.out_tokens, "est_cost_usd": round(self.spent_usd, 4)}


GLOBAL_BUDGET = BudgetTracker(limit_usd=float(os.environ.get("CALFTWIN_BUDGET_USD", "5.0")))


def load_api_key(dotenv: str | Path = ".env") -> str | None:
    for var in ("OPENAI_API_KEY", "OPEN_AI_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    p = Path(dotenv)
    if p.exists():
        m = dict(re.findall(r"^(\w+)\s*=\s*(.*)$", p.read_text(), re.M))
        for var in ("OPENAI_API_KEY", "OPEN_AI_KEY"):
            if m.get(var):
                return m[var].strip().strip('"').strip("'")
    return None


TOOL_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_state", "description": "Current twin state estimate with uncertainty, latest measurements, THI, data-quality flags and pen status.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_history", "description": "Recent measured series and trend for one sensor channel.",
        "parameters": {"type": "object", "properties": {
            "channel": {"type": "string", "enum": ["ear_temp_c", "resp_bpm", "heart_bpm",
                                                   "activity", "air_temp_c", "rh_pct",
                                                   "water_flow_l"]},
            "hours": {"type": "number", "description": "Look-back window in hours (1-12)."}},
            "required": ["channel"]}}},
    {"type": "function", "function": {
        "name": "forecast_options", "description": "Predicted heat load (degree-hours above 39.2 C) over the horizon under candidate actions.",
        "parameters": {"type": "object", "properties": {
            "horizon_h": {"type": "number"},
            "options": {"type": "array", "items": {"type": "string", "enum": list(ACTIONS)}}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "verify_vitals", "description": "Buy a confirmatory hands-on core-temperature and trough check. Costs 0.12 score units.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "propose_action", "description": "Submit exactly one command and end the decision epoch.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "rationale": {"type": "string", "description": "One or two sentences citing the measurements or estimates that justify the command."}},
            "required": ["action", "rationale"]}}},
]


class LLMReActAgent:
    name = "llm_react"

    def __init__(self, model: str = "gpt-4.1-mini", max_iters: int = 5,
                 temperature: float = 0.2, reasoning_effort: str | None = None,
                 budget: BudgetTracker | None = None, memory_len: int = 3,
                 max_retries: int = 3, verbose: bool = False) -> None:
        from openai import OpenAI  # imported lazily so offline runs need no openai
        key = load_api_key()
        if not key:
            raise RuntimeError("no OpenAI API key found in environment or .env "
                               "(OPENAI_API_KEY or OPEN_AI_KEY)")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.max_iters = max_iters
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.budget = budget or GLOBAL_BUDGET
        self.memory: list[str] = []
        self.memory_len = memory_len
        self.max_retries = max_retries
        self.verbose = verbose
        self.name = f"llm_react[{model}]"
        self.stats = {"episodes": 0, "tool_calls": 0, "no_action_epochs": 0,
                      "invalid_action_epochs": 0, "api_errors": 0,
                      "in_tokens": 0, "out_tokens": 0, "est_cost_usd": 0.0}
        self.transcript: list[dict[str, Any]] = []

    # ---- API call with retry ---------------------------------------------
    def _chat(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages,
                                  "tools": TOOL_SCHEMA, "tool_choice": "auto"}
        if self.model.startswith("gpt-5") or self.model.startswith("o"):
            kwargs["max_completion_tokens"] = 2000
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = 700
            kwargs["temperature"] = self.temperature
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                if resp.usage is not None:
                    self.budget.charge(self.model, resp.usage)
                    self.stats["in_tokens"] += int(resp.usage.prompt_tokens or 0)
                    self.stats["out_tokens"] += int(resp.usage.completion_tokens or 0)
                return resp
            except BudgetExceeded:
                raise
            except Exception as exc:                       # transient API failure
                self.stats["api_errors"] += 1
                if attempt == self.max_retries - 1:
                    raise
                if self.verbose:
                    print(f"    API error ({type(exc).__name__}), retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    # ---- the decision epoch ----------------------------------------------
    def decide(self, twin: DigitalTwin) -> dict[str, Any]:
        self.stats["episodes"] += 1
        mem = ("\n".join(f"- {m}" for m in self.memory[-self.memory_len:])
               or "- (no previous decisions this episode)")
        user = (f"Decision epoch at hour {twin.hour:.2f} of the day for CALF-001.\n"
                f"Your recent decisions:\n{mem}\n\n"
                f"Assess the calf and submit exactly one command with propose_action.")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]

        n_calls = 0
        thoughts: list[str] = []
        chosen: str | None = None
        rationale = ""

        for _ in range(self.max_iters):
            resp = self._chat(messages)
            msg = resp.choices[0].message
            if msg.content:
                thoughts.append(msg.content.strip()[:300])
            tcs = msg.tool_calls or []
            if not tcs:
                break
            messages.append({"role": "assistant", "content": msg.content,
                             "tool_calls": [{"id": t.id, "type": "function",
                                             "function": {"name": t.function.name,
                                                          "arguments": t.function.arguments}}
                                            for t in tcs]})
            done = False
            for tc in tcs:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                n_calls += 1
                self.stats["tool_calls"] += 1
                result, ended = self._dispatch(twin, name, args)
                if ended:
                    chosen = result.get("action")
                    rationale = args.get("rationale", "")
                    done = True
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, default=str)[:4000]})
            if done:
                break

        if chosen is None:
            self.stats["no_action_epochs"] += 1
            twin.propose_action("observe", "LLM epoch ended without a command; "
                                           "defaulting to observation.")
            chosen = "observe"
            rationale = "no command proposed within the iteration budget"

        self.memory.append(f"h{twin.hour:.1f}: {chosen} ({rationale[:90]})")
        self.stats["est_cost_usd"] = round(self.budget.spent_usd, 4)
        self.transcript.append({"hour": round(twin.hour, 2), "action": chosen,
                                "rationale": rationale,
                                "thoughts": thoughts, "n_tool_calls": n_calls})
        return decision(chosen, " || ".join(thoughts) or rationale, n_calls,
                        rationale=rationale)

    # ---- tool dispatch ----------------------------------------------------
    def _dispatch(self, twin: DigitalTwin, name: str,
                  args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if name == "get_state":
            return twin.get_state(), False
        if name == "get_history":
            return twin.get_history(str(args.get("channel", "ear_temp_c")),
                                    float(args.get("hours", 6.0))), False
        if name == "forecast_options":
            return twin.forecast_options(float(args.get("horizon_h", 3.0)),
                                         args.get("options")), False
        if name == "verify_vitals":
            return twin.verify_vitals(), False
        if name == "propose_action":
            action = str(args.get("action", "")).strip()
            if action not in ACTIONS:
                self.stats["invalid_action_epochs"] += 1
                return {"error": f"{action!r} is not an allowed action",
                        "allowed_actions": list(ACTIONS)}, False
            res = twin.propose_action(action, str(args.get("rationale", "")))
            res["action"] = action
            return res, True
        return {"error": f"unknown tool {name!r}"}, False
