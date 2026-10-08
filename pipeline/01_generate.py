# pipeline/steps/01_generate.py
import argparse
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

from adapters.copilot import CopilotAdapter
from adapters.gemini import GeminiAdapter
from adapters.claude import ClaudeCodeAdapter
from adapters.codex import CodexAdapter
from adapters.open_code import OpenCodeAdapter

ADAPTERS: dict[str, tuple[type, dict]] = {
    "copilot":   (CopilotAdapter,     {}),
    "gemini":    (GeminiAdapter,      {}),
    "claude":    (ClaudeCodeAdapter,  {}),
    "codex":     (CodexAdapter,       {}),
    "opencode":  (OpenCodeAdapter,    {}),

    # Claude version comparison experiment — all at effort "low"
    "claude-opus-4-6-low": (ClaudeCodeAdapter, {"model": "claude-opus-4-6", "effort": "low"}),
    "claude-opus-4-8-low": (ClaudeCodeAdapter, {"model": "claude-opus-4-8", "effort": "low"}),
    "claude-opus-5-5-low": (ClaudeCodeAdapter, {"model": "claude-opus-5-5", "effort": "low"}),
}

load_dotenv(override=True)
BENCH_DIR = Path(os.getenv("BENCH_DIR"))
GENERATED_DIR = Path(os.getenv("GENERATED_DIR"))
EXPERIMENTS_RESULTS = Path(os.getenv("EXPERIMENTS_RESULTS"))

def get_all_specs() -> list[Path]:
    return sorted([p for p in BENCH_DIR.iterdir() if p.is_dir() and (p / "spec.md").exists()])


def generate(llm: str, spec_folder: Path, generated_root: Path) -> dict:
    spec_file = spec_folder / "spec.md"
    output_dir = generated_root / llm / spec_folder.name
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_class, adapter_kwargs = ADAPTERS[llm]
    adapter = adapter_class(spec_file, output_dir, **adapter_kwargs)
    generated_file_path = adapter.run()

    return {
        "agent": spec_folder.name,
        "llm": llm,
        "spec_file": str(spec_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "generated_file": str(Path(generated_file_path).resolve()),
        "status": "ok",
        "usage": getattr(adapter, "usage", {})
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True, choices=list(ADAPTERS.keys()))
    parser.add_argument("--agent", default=None, help="Filter a specific agent (folder name in bench/)")
    parser.add_argument("--run", type=int, default=1, help="Run index (1..N). Writes under run_<N>/ subpath.")
    parser.add_argument("--experiment", default="by_tool", help="Experiment slug (e.g., by_tool, by_claude_version).")
    args = parser.parse_args()

    if not BENCH_DIR or not GENERATED_DIR or not EXPERIMENTS_RESULTS:
        raise ValueError("Required environment variables not configured.")

    generated_root = Path(GENERATED_DIR) / args.experiment / f"run_{args.run}"
    experiments_root = Path(EXPERIMENTS_RESULTS) / args.experiment / f"run_{args.run}"

    specs = get_all_specs()

    if args.agent:
        specs = [s for s in specs if s.name == args.agent]
        if not specs:
            raise ValueError(f"Agent '{args.agent}' not found in {BENCH_DIR}")

    results = []

    for spec_folder in specs:
        print(f"[generate] experiment={args.experiment} run={args.run} {args.llm} ← {spec_folder.name}")

        try:
            result = generate(args.llm, spec_folder, generated_root)
        except Exception as e:
            result = {
                "model": args.llm,
                "agent_folder": spec_folder.name,
                "status": "error",
                "error": str(e),
            }

        results.append(result)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    final_report = {
        "run_id": run_id,
        "run": args.run,
        "experiment": args.experiment,
        "stage": "generation",
        "model": args.llm,
        "bench_dir": str(Path(BENCH_DIR).resolve()),
        "generated_dir": str(generated_root.resolve()),
        "total_agents": len(results),
        "results": results,
    }

    experiments_root.mkdir(parents=True, exist_ok=True)

    report_path = experiments_root / f"01_generation_{args.llm}_{run_id}.json"

    report_path.write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[generate] Report saved → {report_path}")


if __name__ == "__main__":
    main()
