import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import os

if os.getenv("SPEC2AGENT_SKIP_DOTENV") != "1":
    load_dotenv(override=True)

GENERATED_AGENTS_DIR = os.getenv("GENERATED_DIR")
EXPERIMENTS_RESULTS = os.getenv("EXPERIMENTS_RESULTS")


def save_report(report_path: Path, report: dict):
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Checks existence of test files for generated agents.")
    parser.add_argument("--llm", required=True, help="Filters execution by LLM (e.g., copilot, gemini, claude)")
    parser.add_argument("--agent", default=None, help="Filters execution for a specific agent/directory")
    parser.add_argument("--run", type=int, default=1, help="Run index (1..N). Reads/writes under run_<N>/ subpath.")
    parser.add_argument("--experiment", default="by_tool", help="Experiment slug (e.g., by_tool, by_claude_version).")
    args = parser.parse_args()

    generated_agents = Path(GENERATED_AGENTS_DIR) / args.experiment / f"run_{args.run}"
    experiment_dir = Path(EXPERIMENTS_RESULTS) / args.experiment / f"run_{args.run}"

    model_folder = generated_agents / args.llm
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    tests_results = []

    selection_error = None
    if not generated_agents.is_dir():
        selection_error = "missing_generated_root"
    elif not model_folder.is_dir():
        selection_error = "missing_model_root"
    folders = ([model_folder / args.agent] if args.agent else
               sorted(p for p in model_folder.iterdir() if p.is_dir()) if model_folder.is_dir() else [])
    if not folders and selection_error is None:
        selection_error = "empty_selection"

    for agent_folder in folders:

        agent_file = agent_folder / "agent.py"

        test_files = [
            str(p.relative_to(agent_folder))
            for p in agent_folder.rglob("*.py")
            if p.name.startswith("test") or p.name.endswith("test.py")
        ]

        report_entry = {
            "model": args.llm,
            "agent_folder": agent_folder.name,
            "agent_file": str(agent_file.name),
            "status": "ok" if agent_file.is_file() else "error",
            "error": None if agent_file.is_file() else "agent.py not found",
            "tests": {
                "exists": len(test_files) > 0,
                "count": len(test_files),
                "files": test_files
            }
        }

        tests_results.append(report_entry)
        if not agent_file.is_file():
            report_entry["reason_code"] = "missing_agent"

    final_report = {
        "run_id": run_id,
        "run": args.run,
        "experiment": args.experiment,
        "model": args.llm,
        "agent_filter": args.agent,
        "generated_dir": str(generated_agents),
        "total_agents": len(tests_results),
        "results": tests_results
    }

    final_report["status"] = "error" if selection_error or any(
        entry["status"] == "error" for entry in tests_results) else "ok"
    if final_report["status"] == "error":
        final_report["reason_code"] = selection_error or "target_error"

    report_path = experiment_dir / f"04_tests_check_{args.llm}_{run_id}.json"
    save_report(report_path, final_report)

    print(f"Test existence check completed. Report saved to {report_path}")

    return 1 if final_report["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
