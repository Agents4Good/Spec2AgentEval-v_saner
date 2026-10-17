import argparse
import subprocess
from pathlib import Path


TRACE_EVAL_PROMPT = """
You are a test designer specialized in trace-based evaluation of AI agents.

Read ONLY the following specification:

{spec_path}

Generate trace-based evaluation tests from this specification.

You MUST use the DeepEval skill when applicable.

The specification is the ONLY source of requirements.

The evaluation MUST be based on the execution trace captured by the
benchmark instrumentation.

TRACE REQUIREMENT:
- The agent entrypoint declared in the specification returns only the
  output: `agent(...) -> output`. Import it with `from agent import agent`.
- The tests MUST invoke the agent ONLY through the benchmark wrapper, which
  executes it under instrumentation and returns the output and the trace:

      from trace_capture import traced
      from agent import agent

      output, trace = traced(agent)(<arguments of the entrypoint>)

  `trace_capture` is provided by the benchmark at test time; do not
  implement, mock or modify it. Never call `agent(...)` directly.
- Each scenario (fixture) invokes the agent once with its own inputs.
- The tests MUST NOT assume that behavioral execution can be determined
  from the final output alone.
- The tests MUST NOT reconstruct or infer the trace from the output.
- If the trace does not provide sufficient evidence to evaluate a
  trace-based requirement, the corresponding test MUST fail rather than
  infer that the behavior occurred.

Do not inspect the agent implementation.
Do not inspect source code.
Do not inspect existing tests.
Do not infer implementation details.
Do not invent requirements.
Do not make any semantic validation of the agent output.

Use the behavioral graph defined in the specification as the oracle
for evaluating the agent trajectory.

The trace evaluation may verify:

1. Whether required tools specified by the specification were called,
   when such calls are observable in the trace.
2. Whether the observed behavioral trajectory is compatible with the
   behavioral graph.
3. Whether required behavioral transitions occurred in the expected order.
4. Whether trace events provide sufficient evidence that the observed
   behaviors actually occurred.

The observed trajectory does not need to contain exactly the same
events as the behavioral graph. Intermediate trace events are allowed,
provided that the relevant behavioral transitions and ordering constraints
are preserved.

Generate tests only for trace-based requirements explicitly supported
by the specification.

IMPORTANT:
- The installed DeepEval version is 3.3.9.
- Generate code compatible with DeepEval 3.3.9.
- Follow the current DeepEval API described by the DeepEval skill.
- Do not assume APIs from older DeepEval versions.
- Do not invent DeepEval APIs or metric interfaces.

TRACE FORMAT (returned by `traced(agent)(...)`; this is the only trace
format, the agent does not produce its own trace):

    {{
      "run_id": str,
      "query": list | dict,          # entrypoint arguments (secrets as "***")
      "status": "ok" | "error",      # "error" if the agent raised
      "error": str | null,           # "<ExceptionType>: <message>"
      "runs_collected": int,
      "trajectory": [str, ...],      # LangGraph nodes and tools, by start time
      "tool_calls": [
        {{
          "name": str,               # tool name
          "input": dict,             # tool arguments
          "output": any,             # tool result (absent when "error")
          "error": str,              # present only if the tool failed
          "start": str,              # ISO-8601
          "end": str                 # ISO-8601
        }}
      ]
    }}

Example:

    {{
      "run_id": "ee66dd81f59642bd9b3fd2ebc0b35f9e",
      "query": ["octocat/Hello-World", "Liste as issues abertas.", "***"],
      "status": "ok",
      "error": null,
      "runs_collected": 5,
      "trajectory": ["fetch_issues", "list_issues", "filter_issues", "respond"],
      "tool_calls": [
        {{
          "name": "list_issues",
          "input": {{"owner": "octocat", "repo": "Hello-World", "state": "OPEN"}},
          "output": [{{"number": 1, "title": "Found a bug", "state": "open"}}],
          "start": "2026-10-15T12:00:01.000000+00:00",
          "end": "2026-10-15T12:00:02.000000+00:00"
        }}
      ]
    }}

If the agent raises, `output` is None and `trace["status"] == "error"`.
Node names in `trajectory` are chosen by each agent implementation and are
NOT defined by the specification: do not rely on specific node names.

Output: file named test_trace.py with a pytest test suite that evaluates
the agent execution trace against the specification.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec_path")
    args = parser.parse_args()

    spec_path = Path(args.spec_path).resolve()

    if not spec_path.exists():
        raise FileNotFoundError(spec_path)

    subprocess.run(
        [
            "claude",
            "--dangerously-skip-permissions",
            "--model", "claude-opus-5-5",
            "-p",
            TRACE_EVAL_PROMPT.format(
                spec_path=spec_path
            ),
            "--effort", "low",
        ],
        cwd=spec_path.parent,
        check=True,
    )


if __name__ == "__main__":
    main()