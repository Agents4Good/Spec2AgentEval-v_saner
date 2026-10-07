import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(override=True)

GENERATED_AGENTS_DIR = Path(os.getenv("GENERATED_DIR") or "")
BENCH_DIR = Path(os.getenv("BENCH_DIR") or "")
EXPERIMENTS_RESULTS = Path(os.getenv("EXPERIMENTS_RESULTS") or "")


def save_report(report_path: Path, report: dict):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Runs trace-based evaluation for generated agents."
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="Filters execution by LLM (e.g., copilot, gemini, claude)"
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Filters execution for a specific agent/directory"
    )
    args = parser.parse_args()

    if not GENERATED_AGENTS_DIR.exists():
        raise FileNotFoundError(
            f"Generated dir not found: {GENERATED_AGENTS_DIR}"
        )

    if not BENCH_DIR.exists():
        raise FileNotFoundError(
            f"Bench dir not found: {BENCH_DIR}"
        )

    model_folder = GENERATED_AGENTS_DIR / args.llm

    if not model_folder.exists():
        raise FileNotFoundError(
            f"LLM directory not found: {model_folder}"
        )

    results = []

    for agent_folder in model_folder.iterdir():

        if not agent_folder.is_dir():
            continue

        if args.agent and agent_folder.name != args.agent:
            continue

        entry = {
            "model": args.llm,
            "agent_folder": agent_folder.name,
            "agent_file": "agent.py",
            "test_file": "test_trace.py",
            "passed": None,
            "returncode": None,
            "stdout": None,
            "stderr": None,
            "error": None
        }

        agent_file = agent_folder / "agent.py"
        source_test_file = BENCH_DIR / agent_folder.name / "test_trace.py"
        target_test_file = agent_folder / "test_trace.py"

        if not agent_file.exists():
            entry["error"] = "agent.py not found"
            results.append(entry)
            continue

        if not source_test_file.exists():
            entry["error"] = "test_trace.py not found"
            results.append(entry)
            continue

        try:
            shutil.copy2(source_test_file, target_test_file)

            print(
                f"Running trace evaluation for {agent_folder.name}..."
            )

            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "test_trace.py",
                    "-q"
                ],
                cwd=agent_folder,
                capture_output=True,
                text=True
            )

            entry["passed"] = result.returncode == 0
            entry["returncode"] = result.returncode
            entry["stdout"] = result.stdout
            entry["stderr"] = result.stderr

        except Exception as e:
            entry["error"] = f"Error running trace eval: {e}"

        finally:
            if target_test_file.exists():
                target_test_file.unlink()

        results.append(entry)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_report = {
        "run_id": run_id,
        "stage": "trace_evaluation",
        "model": args.llm,
        "agent_filter": args.agent,
        "bench_dir": str(BENCH_DIR),
        "generated_dir": str(GENERATED_AGENTS_DIR),
        "total_agents": len(results),
        "results": results
    }

    report_path = (
        EXPERIMENTS_RESULTS
        / f"08_trace_eval_{args.llm}_{run_id}.json"
    )

    save_report(report_path, final_report)

    print(
        f"Trace evaluation completed. "
        f"Report saved to {report_path}"
    )


if __name__ == "__main__":
    main()