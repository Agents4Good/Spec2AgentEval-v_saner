import argparse
from pathlib import Path
import importlib.util
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import inspect

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
    args = parser.parse_args()

    if not GENERATED_AGENTS_DIR.exists():
        raise FileNotFoundError(f"Generated dir not found: {GENERATED_AGENTS_DIR}")

    if not BENCH_DIR.exists():
        raise FileNotFoundError(f"Bench dir not found: {BENCH_DIR}")

    model_folder = GENERATED_AGENTS_DIR / args.llm
    if not model_folder.exists():
        raise FileNotFoundError(f"LLM directory not found: {model_folder}")

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
            "judge": None,
            "error": None
        }

        agent_file_path = agent_folder / "agent.py"
        eval_module_path = BENCH_DIR / agent_folder.name / "eval_2_graph.py"

        if not agent_file_path.exists():
            entry["error"] = "agent.py not found"
            results.append(entry)
            continue

        if not eval_module_path.exists():
            entry["error"] = "eval_2_graph.py not found"
            results.append(entry)
            continue

        try:
            agent_module = load_module_from_path(agent_file_path)
        except Exception as e:
            entry["error"] = f"Error loading agent module: {e}"
            results.append(entry)
            continue

        if not hasattr(agent_module, "build_graph"):
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
            entry["error"] = f"build_graph failed: {e}"
            results.append(entry)
            continue

        arch_spec_check_module = load_module_from_path(eval_module_path)

        judge_result = arch_spec_check_module.validate_architecture(
            graph
        )

        entry["judge"] = judge_result
        results.append(entry)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_report = {
        "run_id": run_id,
        "model": args.llm,
        "agent_filter": args.agent,
        "generated_dir": str(GENERATED_AGENTS_DIR),
        "total_agents": len(results),
        "results": results
    }

    report_path = EXPERIMENTS_RESULTS / f"06_graph_eval_{args.llm}_{run_id}.json"
    save_report(report_path, final_report)

    print(f"Graph evaluation completed. Report saved to {report_path}")

if __name__ == "__main__":
    main()