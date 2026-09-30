import argparse
import importlib.util
from pathlib import Path
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

GENERATED_AGENTS_DIR = Path(os.getenv("GENERATED_DIR"))
BENCH_DIR = Path(os.getenv("BENCH_DIR"))
EXPERIMENTS_RESULTS = Path(os.getenv("EXPERIMENTS_RESULTS"))


def save_report(report_path: Path, report: dict):
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Runs static spec LLM judge for generated agents.")
    parser.add_argument("--llm", required=True, help="Filters execution by LLM (e.g., copilot, gemini, claude)")
    parser.add_argument("--agent", default=None, help="Filters execution for a specific agent/directory")
    args = parser.parse_args()

    if not GENERATED_AGENTS_DIR.exists():
        raise FileNotFoundError(f"Generated agents dir not found: {GENERATED_AGENTS_DIR}")

    if not BENCH_DIR.exists():
        raise FileNotFoundError(f"Bench dir not found: {BENCH_DIR}")

    model_folder = GENERATED_AGENTS_DIR / args.llm
    if not model_folder.exists():
        raise FileNotFoundError(f"LLM directory not found: {model_folder}")

    EXPERIMENTS_RESULTS.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []

    for agent_folder in model_folder.iterdir():
        if not agent_folder.is_dir():
            continue
        if args.agent and agent_folder.name != args.agent:
            continue

        agent_file_path = agent_folder / "agent.py"
        spec_path = BENCH_DIR / agent_folder.name / "spec.md"
        eval_module_path = BENCH_DIR / agent_folder.name / "eval_1_static_spec_llm_judge.py"

        if not agent_file_path.exists() or not spec_path.exists() or not eval_module_path.exists():
            continue

        source_code = agent_file_path.read_text(encoding="utf-8")
        spec_text = load_text_file(spec_path)

        static_eval_module = load_module(eval_module_path)

        judge_result = static_eval_module.run_static_eval(
            source_code=source_code,
            spec_text=spec_text,
        )

        results.append({
            "model": args.llm,
            "agent_folder": agent_folder.name,
            "agent_file": "agent.py",
            "judge": judge_result
        })

    final_report = {
        "run_id": run_id,
        "model": args.llm,
        "agent_filter": args.agent,
        "generated_dir": str(GENERATED_AGENTS_DIR),
        "total_agents": len(results),
        "results": results
    }

    report_path = EXPERIMENTS_RESULTS / f"05_spec_judge_{args.llm}_{run_id}.json"
    save_report(report_path, final_report)

    print(f"Spec judge evaluation completed. Report saved to {report_path}")


if __name__ == "__main__":
    main()