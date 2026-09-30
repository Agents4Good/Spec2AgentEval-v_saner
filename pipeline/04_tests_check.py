import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import os

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
    args = parser.parse_args()

    generated_agents = Path(GENERATED_AGENTS_DIR)
    experiment_dir = Path(EXPERIMENTS_RESULTS)

    if not generated_agents.exists():
        raise FileNotFoundError(f"Agents directory not found: {generated_agents}")

    model_folder = generated_agents / args.llm
    if not model_folder.exists():
        raise FileNotFoundError(f"LLM directory not found: {model_folder}")

    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    tests_results = []

    for agent_folder in model_folder.iterdir():
        if not agent_folder.is_dir():
            continue
        if args.agent and agent_folder.name != args.agent:
            continue

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
            "tests": {
                "exists": len(test_files) > 0,
                "count": len(test_files),
                "files": test_files
            }
        }

        tests_results.append(report_entry)

    final_report = {
        "run_id": run_id,
        "model": args.llm,
        "agent_filter": args.agent,
        "generated_dir": str(generated_agents),
        "total_agents": len(tests_results),
        "results": tests_results
    }

    report_path = experiment_dir / f"04_tests_check_{args.llm}_{run_id}.json"
    save_report(report_path, final_report)

    print(f"Test existence check completed. Report saved to {report_path}")


if __name__ == "__main__":
    main()