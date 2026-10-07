"""Stage 08 — trace-based evaluation runner.

Executes ``test_trace.py`` from each spec against the matching generated
agent, embedding pytest's structured JSON report so no test is lost.
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
    )


if __name__ == "__main__":
    main()
