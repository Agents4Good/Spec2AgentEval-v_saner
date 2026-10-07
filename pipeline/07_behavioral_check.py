"""Stage 07 — behavioral evaluation (DeepEval) runner.

Executes ``test_deepeval.py`` from each spec against the matching generated
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
        description="Runs behavioral evaluation for generated agents."
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="LLM used to generate the agents (e.g., claude, gemini, copilot)",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Specific agent directory (e.g., 001_list_issues_agent)",
    )
    parser.add_argument(
        "--test",
        default="test_deepeval.py",
        help="Test file to execute (defaults to test_deepeval.py)",
    )
    args = parser.parse_args()

    run_pytest_for_agents(
        llm=args.llm,
        agent_filter=args.agent,
        test_filename=args.test,
        generated_dir=GENERATED_AGENTS_DIR,
        bench_dir=BENCH_DIR,
        stage_label="evaluation",
        stage_prefix="07",
        report_name_stem=Path(args.test).stem,
        experiments_results=EXPERIMENTS_RESULTS,
    )


if __name__ == "__main__":
    main()
