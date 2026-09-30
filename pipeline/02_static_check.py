import argparse
import json
import py_compile
import importlib.util
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv(override=True)
GENERATED_DIR = os.getenv("GENERATED_DIR")
EXPERIMENTS_RESULTS = os.getenv("EXPERIMENTS_RESULTS")

def load_module_from_path(path: Path):
    module_name = f"agent_{path.stem}_{hash(path)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_report(report_path: Path, report: dict):
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def validate_agent_folder(model_name: str, agent_folder: Path):
    agent_file = agent_folder / "agent.py"

    result = {
        "model": model_name,
        "agent_folder": agent_folder.name,
        "checks": {
            "file_exists": {"passed": False},
            "syntax_valid": {"passed": False},
            "dependency_error": {"passed": False},
            "has_entrypoint": {"passed": False},
            "entrypoint_callable": {"passed": False}
        },
        "final_status": "FAILED"
    }

    success = True
    agent_module = None

    # 1. File existence
    if agent_file.exists():
        result["checks"]["file_exists"]["passed"] = True
    else:
        result["checks"]["file_exists"]["error"] = "agent.py not found"
        success = False

    # 2. Syntax
    if result["checks"]["file_exists"]["passed"]:
        try:
            py_compile.compile(str(agent_file), doraise=True)
            result["checks"]["syntax_valid"]["passed"] = True
        except py_compile.PyCompileError as e:
            result["checks"]["syntax_valid"]["error"] = str(e)
            success = False
        except Exception as e:
            result["checks"]["syntax_valid"]["error"] = f"unexpected_error: {e}"
            success = False
    else:
        result["checks"]["syntax_valid"]["error"] = "skipped_missing_file"

    # 3. Import
    if result["checks"]["syntax_valid"]["passed"]:
        try:
            agent_module = load_module_from_path(agent_file)
            result["checks"]["dependency_error"]["passed"] = True
        except ModuleNotFoundError as e:
            result["checks"]["dependency_error"]["error"] = f"missing_dependency: {e}"
            success = False
        except Exception as e:
            result["checks"]["dependency_error"]["error"] = f"import_runtime_error: {e}"
            success = False
    else:
        result["checks"]["dependency_error"]["error"] = "skipped_invalid_syntax"

    # 4. Has entrypoint
    if agent_module:
        if hasattr(agent_module, "agent"):
            result["checks"]["has_entrypoint"]["passed"] = True
        else:
            result["checks"]["has_entrypoint"]["error"] = "agent function not found"
            success = False
    else:
        result["checks"]["has_entrypoint"]["error"] = "skipped_not_importable"

    # 5. Callable
    if agent_module and hasattr(agent_module, "agent"):
        if callable(agent_module.agent):
            result["checks"]["entrypoint_callable"]["passed"] = True
        else:
            result["checks"]["entrypoint_callable"]["error"] = "agent is not callable"
            success = False
    else:
        result["checks"]["entrypoint_callable"]["error"] = "skipped_no_entrypoint"

    result["final_status"] = "SUCCESS" if success else "FAILED"
    return result


def main():
    parser = argparse.ArgumentParser(description="Static validation of generated agents.")
    parser.add_argument("--llm", required=True, help="Filters execution by LLM (e.g., copilot, gemini, claude)")
    parser.add_argument("--agent", default=None, help="Filters execution for a specific agent/directory")
    args = parser.parse_args()

    generated_agents = Path(GENERATED_DIR)
    experiment_dir = Path(EXPERIMENTS_RESULTS)

    if not generated_agents.exists():
        raise FileNotFoundError(f"Agents directory not found: {generated_agents}")

    model_folder = generated_agents / args.llm
    if not model_folder.exists():
        raise FileNotFoundError(f"LLM directory not found: {model_folder}")

    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    execution_results = []

    for agent_folder in model_folder.iterdir():
        if not agent_folder.is_dir():
            continue
        if args.agent and agent_folder.name != args.agent:
            continue
        execution_results.append(
            validate_agent_folder(args.llm, agent_folder)
        )

    report = {
        "run_id": run_id,
        "model": args.llm,
        "agent_filter": args.agent,
        "generated_dir": str(generated_agents),
        "total_agents": len(execution_results),
        "results": execution_results
    }

    report_path = experiment_dir / f"02_validation_{args.llm}_{run_id}.json"

    save_report(report_path, report)

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()