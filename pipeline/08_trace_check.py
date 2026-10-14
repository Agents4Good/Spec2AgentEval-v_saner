"""Stage 08 — trace-based evaluation runner.

For each generated agent, executes the agent once per scenario declared in the
spec's ``test_trace.py`` under LangChain run collection (``trace_capture``),
then runs ``test_trace.py`` over the captured traces (trajectory, tool calls
and output), passed in memory. The test file never invokes the agent itself.

The traces are embedded in the stage report alongside pytest's structured JSON
report, so every verdict can be audited against the exact evidence it used.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from utils.pytest_runner import run_pytest_for_agents


load_dotenv(override=True)

GENERATED_AGENTS_DIR = Path(os.getenv("GENERATED_DIR") or "")
BENCH_DIR = Path(os.getenv("BENCH_DIR") or "")
EXPERIMENTS_RESULTS = Path(os.getenv("EXPERIMENTS_RESULTS") or "")

TRACE_RUNNER = Path(__file__).parent / "utils" / "trace_runner.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs trace-based evaluation for generated agents."
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="Filters execution by LLM (e.g., copilot, gemini, claude)",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Filters execution for a specific agent/directory",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run index (1..N). Reads/writes under run_<N>/ subpath.",
    )
    parser.add_argument(
        "--experiment",
        default="by_tool",
        help="Experiment slug (e.g., by_tool, by_claude_version).",
    )
    args = parser.parse_args()

    run_pytest_for_agents(
        llm=args.llm,
        agent_filter=args.agent,
        test_filename="test_trace.py",
        generated_dir=GENERATED_AGENTS_DIR,
        bench_dir=BENCH_DIR,
        stage_label="trace_evaluation",
        stage_prefix="08",
        report_name_stem="trace_eval",
        experiments_results=EXPERIMENTS_RESULTS,
        run=args.run,
        experiment=args.experiment,
        runner_script=TRACE_RUNNER,
    )


if __name__ == "__main__":
    main()
