"""Stage 08 — trace-based evaluation runner.

Executes ``test_trace.py`` from each spec against the matching generated
agent. The test invokes the agent through ``traced(agent)(...)``
(``utils/trace_capture.py``, put on ``PYTHONPATH``), which runs it under
LangChain run collection and returns ``(output, trace)``.

Every execution is also exported to
``<EXPERIMENTS_RESULTS>/<experiment>/run_<N>/08_traces/<llm>/<agent>/``,
so each verdict can be audited against the exact evidence it used.
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

UTILS_DIR = Path(__file__).resolve().parent / "utils"


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

    traces_root = (EXPERIMENTS_RESULTS / args.experiment / f"run_{args.run}"
                   / "08_traces" / args.llm).resolve()

    def trace_env(agent_folder: Path) -> dict[str, str]:
        # `from trace_capture import traced` no teste + exportação dos traces
        pythonpath = os.pathsep.join(
            p for p in (str(UTILS_DIR), os.environ.get("PYTHONPATH")) if p)
        return {
            "PYTHONPATH": pythonpath,
            "TRACE_DIR": str(traces_root / agent_folder.name),
        }

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
        extra_env=trace_env,
    )


if __name__ == "__main__":
    main()
