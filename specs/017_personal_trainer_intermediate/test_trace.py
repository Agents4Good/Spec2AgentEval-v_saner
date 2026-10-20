"""Trace-based evaluation of the Personal Trainer agent (spec 017).

The agent entrypoint returns ``output, trace``. All behavioral checks are made
on the trace (tool calls, ordering, B1 -> B2 | B3 -> B4 | B5 -> B6 | B7);
the output is only used for the schema checks the spec attaches to the
observable result (Error: prefix, seven sections, Tool Used).

The trace format is not fixed by the spec, so the trace is normalized
defensively into a flat list of events. If the trace carries no usable
evidence for a requirement, the test FAILS (nothing is inferred).

Functional runs require real OpenAI/Notion credentials in the environment.
"""
import re

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from agent import agent  # agent(user_profile) -> output, trace

EQUIP = "retrieve_available_equipment"
RESTR = "retrieve_restrictions"
TOOLS = (EQUIP, RESTR)

SECTIONS = [
    "Assessment:",
    "Weekly Plan:",
    "Intensity & Progression:",
    "Nutrition Guidance:",
    "Recovery Protocol:",
    "Important Notes:",
    "Tool Used:",
]

RESULT_HINTS = ("result", "end", "output", "response", "return", "observation")
LLM_HINTS = ("llm", "chat", "model", "validat", "ai_message", "aimessage")


# --------------------------------------------------------------------------
# Trace normalization
# --------------------------------------------------------------------------
def _to_dict(ev):
    if isinstance(ev, dict):
        return ev
    if hasattr(ev, "model_dump"):
        try:
            return ev.model_dump()
        except Exception:
            pass
    if hasattr(ev, "__dict__"):
        return dict(vars(ev))
    return {"value": str(ev)}


def _events(trace):
    """Flatten a trace (list / dict with events / nested) into ordered dicts."""
    if trace is None:
        return []
    if isinstance(trace, dict):
        for key in ("events", "steps", "trace", "spans", "messages", "trajectory"):
            if key in trace and isinstance(trace[key], (list, tuple)):
                return _events(trace[key])
        return [trace]
    if isinstance(trace, (list, tuple)):
        out = []
        for ev in trace:
            if isinstance(ev, (list, tuple)):
                out.extend(_events(ev))
            else:
                out.append(_to_dict(ev))
        return out
    return [{"value": str(trace)}]


def _kind(ev):
    parts = []
    for k in ("type", "kind", "event", "event_type", "phase", "node", "role", "status"):
        v = ev.get(k)
        if isinstance(v, str):
            parts.append(v.lower())
    return " ".join(parts)


def _tool_invocations(ev):
    """Tool names invoked by this event (one entry per invocation)."""
    names = []
    calls = ev.get("tool_calls")
    if isinstance(calls, (list, tuple)):
        for c in calls:
            c = _to_dict(c)
            fn = c.get("function") if isinstance(c.get("function"), dict) else {}
            n = c.get("name") or c.get("tool") or fn.get("name")
            if n in TOOLS:
                names.append(n)
        if names:
            return names
    kind = _kind(ev)
    if any(h in kind for h in RESULT_HINTS) and "call" not in kind and "start" not in kind:
        return []
    if "tool" in kind or "call" in kind or "action" in kind or not kind:
        for k in ("tool", "tool_name", "name", "action", "function"):
            v = ev.get(k)
            if v in TOOLS:
                return [v]
    return []


def tool_calls(trace):
    out = []
    for i, ev in enumerate(_events(trace)):
        for n in _tool_invocations(ev):
            out.append((i, n))
    return out


def has_llm_evidence_before(trace, index):
    """Evidence that an LLM step (B1 validation) happened before ``index``."""
    for ev in _events(trace)[:index]:
        text = (_kind(ev) + " " + " ".join(
            str(ev.get(k, "")).lower() for k in ("name", "node", "step", "component"))).strip()
        if any(h in text for h in LLM_HINTS) and not _tool_invocations(ev):
            return True
    return False


def has_any_llm_evidence(trace):
    return has_llm_evidence_before(trace, len(_events(trace)))


# --------------------------------------------------------------------------
# DeepEval metric wrapping a trace check (deterministic, no judge LLM)
# --------------------------------------------------------------------------
class TraceCheckMetric(BaseMetric):
    def __init__(self, name, check):
        self._name = name
        self.check = check
        self.threshold = 1.0
        self.score = 0.0
        self.success = False
        self.reason = ""
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True
        self.error = None
        self.evaluation_model = None
        self.verbose_mode = False

    @property
    def __name__(self):
        return self._name

    def measure(self, test_case, *args, **kwargs):
        trace = (test_case.additional_metadata or {}).get("trace")
        try:
            problem = self.check(trace, test_case.actual_output)
        except AssertionError as e:
            problem = str(e) or "assertion failed"
        self.success = problem is None
        self.score = 1.0 if self.success else 0.0
        self.reason = problem or "trace satisfies the behavioral graph"
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return self.success


def evaluate(profile, output, trace, name, check):
    tc = LLMTestCase(
        input=profile,
        actual_output=output if isinstance(output, str) else str(output),
        additional_metadata={"trace": trace},
    )
    assert_test(tc, [TraceCheckMetric(name, check)])


# --------------------------------------------------------------------------
# Trace checks (behavioral graph oracle)
# --------------------------------------------------------------------------
def _require_trace(trace):
    if not _events(trace):
        return "trace is empty: no evidence of execution"
    return None


def check_invalid(trace, output):
    """B1 -> B2 -> ERROR: zero tool calls, Error: string, no plan."""
    p = _require_trace(trace)
    if p:
        return p
    calls = tool_calls(trace)
    if calls:
        return f"invalid input must cause zero tool calls, got {calls}"
    if not has_any_llm_evidence(trace):
        return "no trace evidence of LLM validation (B1)"
    if not output.startswith("Error:"):
        return "invalid input must return a string beginning with 'Error:'"
    if any(s in output for s in SECTIONS[:2]):
        return "error output must not contain a plan"
    return None


def _check_valid(expected, forbidden):
    def check(trace, output):
        p = _require_trace(trace)
        if p:
            return p
        calls = tool_calls(trace)
        if not calls:
            return "no tool call observable in the trace for a valid input"
        names = {n for _, n in calls}
        if forbidden in names:
            return f"{forbidden} must not be called on this route"
        if names != {expected}:
            return f"expected only {expected}, got {sorted(names)}"
        if len(calls) > 3:
            return f"at most three total attempts allowed, got {len(calls)}"
        # B1 validation by LLM precedes the tool decision
        if not has_llm_evidence_before(trace, calls[0][0]):
            return "no LLM validation/decision evidence before the first tool call"
        # no parallel / combined retrieval inside a single event
        for ev in _events(trace):
            if len(_tool_invocations(ev)) > 1:
                return "parallel tool calls observed in a single event"
        # B6 only after usable retrieval: an LLM synthesis step after the last call
        last = calls[-1][0]
        evs = _events(trace)
        if output.startswith("Error:"):
            if any(s in output for s in SECTIONS[:2]):
                return "Error output must not contain a plan"
            return None  # B7 path: still consistent with the graph
        if not has_llm_evidence_before(trace, len(evs)) or last >= len(evs):
            return "no evidence of B6 synthesis"
        if not any(
            any(h in (_kind(e) + " " + str(e.get("name", "")).lower()) for h in LLM_HINTS)
            and not _tool_invocations(e)
            for e in evs[last + 1:]
        ):
            return "no LLM synthesis event after the retrieval (B6)"
        return check_success_output(output, expected)
    return check


def check_success_output(output, expected):
    pos = -1
    for s in SECTIONS:
        i = output.find(s)
        if i == -1:
            return f"missing section title '{s}'"
        if i < pos:
            return f"section '{s}' out of order"
        pos = i
    if re.search(r"```|\*\*|^\s*#{1,6}\s|^\s*[-*•]\s|^\s*\d+\.\s", output, re.M):
        return "output contains Markdown / list markup"
    tool_line = output[output.rfind("Tool Used:"):]
    if expected not in tool_line:
        return f"Tool Used must name {expected}"
    other = RESTR if expected == EQUIP else EQUIP
    if other in tool_line:
        return "Tool Used names the wrong tool"
    return None


def check_retries(trace, output):
    """Retries (B7): only the selected tool, never more than 3 attempts."""
    p = _require_trace(trace)
    if p:
        return p
    calls = tool_calls(trace)
    if not calls:
        return "no tool call observable in the trace"
    if len({n for _, n in calls}) != 1:
        return "retries must never switch tools"
    if len(calls) > 3:
        return f"more than three attempts: {len(calls)}"
    return None


# --------------------------------------------------------------------------
# Controlled inputs
# --------------------------------------------------------------------------
INVALID = [
    "",
    "   \n\t  ",
    "@@@ ### $$$ !!!",
    "asdf qwer zxcv",
    "I want to lose weight.",  # goal without fitness level or age
    "I have knee pain and a bad back.",  # invalid yet mentions pain: still zero tools
    "I am 30 years old.",  # age without goal
]

RESTRICTIONS = [
    "I am 35 years old, intermediate level, I want to build muscle. I have a bad knee that hurts when I squat.",
    "Beginner, goal is weight loss. I have lower back pain and hypertension.",
    "I'm 52, want to improve general fitness, I recently had shoulder surgery.",
]

EQUIPMENT = [
    "I am 28 years old, beginner, I want to build strength. I train 4 days a week for 45 minutes.",
    "Intermediate level, goal is to run a 10k faster. I am vegetarian.",
    "I'm 40, advanced, I want to gain muscle mass in 12 weeks.",
]


def run(profile):
    result = agent(profile)
    assert isinstance(result, tuple) and len(result) == 2, "agent must return (output, trace)"
    return result


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
@pytest.mark.parametrize("profile", INVALID)
def test_invalid_input_b1_b2_no_tools(profile):
    output, trace = run(profile)
    evaluate(profile, output, trace, "InvalidInputTrace", check_invalid)


@pytest.mark.parametrize("profile", RESTRICTIONS)
def test_restrictions_route_b3_b4_b6(profile):
    output, trace = run(profile)
    evaluate(profile, output, trace, "RestrictionsRouteTrace", _check_valid(RESTR, EQUIP))


@pytest.mark.parametrize("profile", EQUIPMENT)
def test_equipment_route_b3_b5_b6(profile):
    output, trace = run(profile)
    evaluate(profile, output, trace, "EquipmentRouteTrace", _check_valid(EQUIP, RESTR))


@pytest.mark.parametrize("profile", [RESTRICTIONS[0], EQUIPMENT[0]])
def test_retry_bound_and_no_tool_switch(profile):
    output, trace = run(profile)
    evaluate(profile, output, trace, "RetryBoundTrace", check_retries)


def test_restrictions_take_priority_over_equipment():
    profile = (
        "I'm 30, intermediate, I want to build muscle. I have a home gym with dumbbells "
        "and a barbell, but I have chronic wrist pain."
    )
    output, trace = run(profile)
    evaluate(profile, output, trace, "PriorityTrace", _check_valid(RESTR, EQUIP))


def test_empty_trace_is_insufficient_evidence():
    assert _require_trace([]) is not None
    assert _require_trace(None) is not None
