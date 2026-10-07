import argparse
import subprocess
from pathlib import Path


BLACKBOX_EVAL_PROMPT = """
You are a test designer specialized in behavioral testing of AI agents.

Read ONLY the following specification:

{spec_path}

Generate black-box behavioral evaluation tests from this specification.

You MUST use the DeepEval skill.

The specification is the ONLY source of requirements.

Do NOT read, inspect, search for, or access any other files.

The evaluation is BLACK-BOX:

- Evaluate only observable agent behavior.
- Do not depend on implementation details.
- Do not infer requirements from source code.
- Do not inspect existing tests.
- Do not inspect the agent implementation.
- Do not inspect the project structure.

Generate tests only for behavioral requirements explicitly defined in
the specification.

Do not invent requirements.

Use the DeepEval skill to implement the evaluation suite.

IMPORTANT:
- The installed DeepEval version is 3.3.9.
- Generate code compatible with DeepEval 3.3.9.
- Follow the current DeepEval API described by the DeepEval skill.
- Do not assume APIs from older DeepEval versions.
- Do not invent DeepEval APIs or metric interfaces.

Output: file named test_deepeval.py with a pytest test suite that evaluates
the agent's observable behavior against the specification.
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
            BLACKBOX_EVAL_PROMPT.format(
                spec_path=spec_path
            ),
        ],
        cwd=spec_path.parent,
        check=True,
    )


if __name__ == "__main__":
    main()