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

The evaluation is TRACE-BASED:

- Evaluate the execution trace returned by the agent.
- Do not inspect the agent implementation.
- Do not inspect source code.
- Do not inspect existing tests.
- Do not infer implementation details.
- Do not invent requirements.

Use the behavioral graph defined in the specification as the oracle
for evaluating the agent trajectory.

The trace evaluation may verify:

1. Whether required tools specified by the specification were called.
2. Whether the observed behavioral trajectory is compatible with the
   behavioral graph.
3. Whether required behavioral transitions occurred in the expected order.

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
            "-p",
            TRACE_EVAL_PROMPT.format(
                spec_path=spec_path
            ),
        ],
        cwd=spec_path.parent,
        check=True,
    )


if __name__ == "__main__":
    main()