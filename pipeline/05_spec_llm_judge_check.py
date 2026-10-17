import sys
import argparse
import importlib.util
from pathlib import Path
import os
import json
from datetime import datetime
from dotenv import load_dotenv

if os.getenv("SPEC2AGENT_SKIP_DOTENV") != "1":
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
    parser.add_argument("--run", type=int, default=1, help="Run index (1..N). Reads/writes under run_<N>/ subpath.")
    parser.add_argument("--experiment", default="by_tool", help="Experiment slug (e.g., by_tool, by_claude_version).")
    args = parser.parse_args()

    generated_root = GENERATED_AGENTS_DIR / args.experiment / f"run_{args.run}"
    experiments_root = EXPERIMENTS_RESULTS / args.experiment / f"run_{args.run}"

    model_folder = generated_root / args.llm
    experiments_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []

    selection_error = None
    if not generated_root.is_dir():
        selection_error = "missing_generated_root"
    elif not model_folder.is_dir():
        selection_error = "missing_model_root"
    folders = ([model_folder / args.agent] if args.agent else
               sorted(p for p in model_folder.iterdir() if p.is_dir()) if model_folder.is_dir() else [])
    if not folders and selection_error is None:
        selection_error = "empty_selection"

    for agent_folder in folders:

        agent_file_path = agent_folder / "agent.py"
        spec_path = BENCH_DIR / agent_folder.name / "spec.md"
        eval_module_path = BENCH_DIR / agent_folder.name / "spec_adherence.py"

        entry = {
            "model": args.llm,
            "agent_folder": agent_folder.name,
            "agent_file": "agent.py",
            "judge": None,
            "error": None,
            "status": "ok",
        }
        results.append(entry)
        missing = next(((path, reason) for path, reason in (
            (agent_file_path, "missing_agent"), (spec_path, "missing_spec"),
            (eval_module_path, "missing_evaluator")) if not path.is_file()), None)
        if missing:
            entry.update(status="error", reason_code=missing[1], error=f"{missing[0].name} not found")
            continue
        try:
            source_code = agent_file_path.read_text(encoding="utf-8")
            spec_text = load_text_file(spec_path)
            static_eval_module = load_module(eval_module_path)
            entry["judge"] = static_eval_module.run_static_eval(
                source_code=source_code, spec_text=spec_text)
        except Exception as exc:
            entry.update(status="error", reason_code="evaluation_error",
                         error=f"Spec evaluation failed ({type(exc).__name__})")

    final_report = {
        "run_id": run_id,
        "run": args.run,
        "experiment": args.experiment,
        "model": args.llm,
        "agent_filter": args.agent,
        "generated_dir": str(generated_root),
        "total_agents": len(results),
        "results": results
    }

    final_report["status"] = "error" if selection_error or any(
        entry["status"] == "error" for entry in results) else "ok"
    if final_report["status"] == "error":
        final_report["reason_code"] = selection_error or "target_error"

    report_path = experiments_root / f"05_spec_judge_{args.llm}_{run_id}.json"
    save_report(report_path, final_report)

    print(f"Spec judge evaluation completed. Report saved to {report_path}")

    return 1 if final_report["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
