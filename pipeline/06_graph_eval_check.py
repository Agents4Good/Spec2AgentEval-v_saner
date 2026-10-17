import sys
import argparse
from pathlib import Path
import importlib.util
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import inspect

if os.getenv("SPEC2AGENT_SKIP_DOTENV") != "1":
    load_dotenv(override=True)

GENERATED_AGENTS_DIR = Path(os.getenv("GENERATED_DIR") or "")
BENCH_DIR = Path(os.getenv("BENCH_DIR") or "")
EXPERIMENTS_RESULTS = Path(os.getenv("EXPERIMENTS_RESULTS") or "")

def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def save_report(report_path: Path, report: dict):
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Runs graph architecture evaluation for generated agents.")
    parser.add_argument("--llm", required=True, help="Filters execution by LLM (e.g., copilot, gemini, claude)")
    parser.add_argument("--agent", default=None, help="Filters execution for a specific agent/directory")
    parser.add_argument("--run", type=int, default=1, help="Run index (1..N). Reads/writes under run_<N>/ subpath.")
    parser.add_argument("--experiment", default="by_tool", help="Experiment slug (e.g., by_tool, by_claude_version).")
    args = parser.parse_args()

    generated_root = GENERATED_AGENTS_DIR / args.experiment / f"run_{args.run}"
    experiments_root = EXPERIMENTS_RESULTS / args.experiment / f"run_{args.run}"

    model_folder = generated_root / args.llm
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

        entry = {
            "model": args.llm,
            "agent_folder": agent_folder.name,
            "agent_file": "agent.py",
            "judge": None,
            "status": "ok",
            "error": None
        }

        agent_file_path = agent_folder / "agent.py"
        eval_module_path = BENCH_DIR / agent_folder.name / "eval_2_graph.py"

        if not agent_file_path.is_file():
            entry.update(status="error", reason_code="missing_agent")
            entry["error"] = "agent.py not found"
            results.append(entry)
            continue

        if not eval_module_path.is_file():
            entry.update(status="error", reason_code="missing_evaluator")
            entry["error"] = "eval_2_graph.py not found"
            results.append(entry)
            continue

        try:
            agent_module = load_module_from_path(agent_file_path)
        except Exception as e:
            entry.update(status="error", reason_code="agent_import_error")
            entry["error"] = f"Error loading agent module: {e}"
            results.append(entry)
            continue

        if not hasattr(agent_module, "build_graph"):
            entry.update(status="error", reason_code="missing_build_graph")
            entry["error"] = "build_graph not implemented"
            results.append(entry)
            continue

        try:
            build_graph = agent_module.build_graph
            sig = inspect.signature(build_graph)

            if len(sig.parameters) == 0:
                graph = build_graph()
            else:
                graph = build_graph([])
        except Exception as e:
            entry.update(status="error", reason_code="graph_build_error")
            entry["error"] = f"build_graph failed: {e}"
            results.append(entry)
            continue

        try:
            arch_spec_check_module = load_module_from_path(eval_module_path)
            entry["judge"] = arch_spec_check_module.validate_architecture(graph)
        except Exception as exc:
            entry.update(status="error", reason_code="evaluation_error",
                         error=f"Graph evaluation failed ({type(exc).__name__})")
        results.append(entry)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    experiments_root.mkdir(parents=True, exist_ok=True)
    final_report["status"] = "error" if selection_error or any(
        entry["status"] == "error" for entry in results) else "ok"
    if final_report["status"] == "error":
        final_report["reason_code"] = selection_error or "target_error"

    report_path = experiments_root / f"06_graph_eval_{args.llm}_{run_id}.json"
    save_report(report_path, final_report)

    print(f"Graph evaluation completed. Report saved to {report_path}")
    return 1 if final_report["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
