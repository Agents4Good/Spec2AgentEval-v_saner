import os
import sys
import json
import argparse
import subprocess
from sonar_scripts import sonar_pipeline
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir / "sonar_scripts"))

load_dotenv(find_dotenv(), override=True)

GENERATED_AGENTS_DIR = Path(os.getenv("GENERATED_DIR"))
EXPERIMENTS_RESULTS = Path(os.getenv("EXPERIMENTS_RESULTS"))


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def main():
    parser = argparse.ArgumentParser(description="Executes lints and SonarQube analysis in batch.")
    parser.add_argument("--llm", required=True, help="Filters execution by LLM (e.g., copilot, gemini)")
    parser.add_argument("--agent", default=None, help="Filters execution for a specific agent/directory")
    args = parser.parse_args()

    if not GENERATED_AGENTS_DIR or not EXPERIMENTS_RESULTS:
        raise ValueError("Required environment variables not configured.")

    llm_dir = GENERATED_AGENTS_DIR / args.llm
    if not llm_dir.exists():
        print(f"Base directory not found: {llm_dir}")
        sys.exit(1)

    agents_to_analyze = []
    for agent_dir in llm_dir.iterdir():
        if agent_dir.is_dir():
            if args.agent and agent_dir.name != args.agent:
                continue
            agents_to_analyze.append(agent_dir)

    if not agents_to_analyze:
        print(f"No agents found for LLM '{args.llm}'.")
        sys.exit(0)

    print("Starting SonarQube server in the background...")
    sonar_process = sonar_pipeline.start_sonarqube_server()

    try:
        print("Waiting for SonarQube boot...")
        sonar_pipeline.wait_for_sonarqube_boot()
        print("SonarQube UP!")

        session = sonar_pipeline.requests.Session()
        print("Logging into SonarQube...")
        xsrf = sonar_pipeline.sonar_login(session)

        all_results = []

        for target_dir in agents_to_analyze:
            print(f"\n[lint] Analyzing: {args.llm} -> {target_dir.name}")
            
            tools = {
                "ruff": f"ruff check {target_dir} --output-format=json",
                "pylint": f"pylint {target_dir} --output-format=json",
                "radon_cc": f"radon cc {target_dir} -a -j",
                "radon_mi": f"radon mi {target_dir} -j"
            }

            agent_results = {}
            passed = True

            for name, cmd in tools.items():
                res = run_command(cmd)
                
                parsed_stdout = res["stdout"]
                if parsed_stdout.strip():
                    try:
                        parsed_stdout = json.loads(parsed_stdout)
                    except json.JSONDecodeError:
                        pass
                    
                agent_results[name] = {
                    "command": res["command"],
                    "exit_code": res["exit_code"],
                    "stdout": parsed_stdout,
                    "stderr": res["stderr"]
                }
                
                if res["exit_code"] != 0:
                    passed = False

            print(f"Executing SonarQube for {target_dir.name}...")
            try:
                project_key = f"{args.llm}_{target_dir.name}_{sonar_pipeline.uuid.uuid4().hex[:4]}"
                project_name = f"{args.llm} - {target_dir.name}"
                
                sonar_pipeline.create_project(session, xsrf, project_key, project_name)
                token = sonar_pipeline.generate_project_token(session, xsrf, project_key)
                
                sonar_pipeline.run_pysonar(token, str(target_dir), project_key)
                sonar_pipeline.wait_for_analysis(session, project_key)
                metrics = sonar_pipeline.extract_metrics(session, project_key)
                
                agent_results["sonarqube"] = {
                    "exit_code": 0,
                    "metrics": metrics
                }
            except Exception as e:
                print(f"SonarQube error for {target_dir.name}: {e}")
                agent_results["sonarqube"] = {
                    "exit_code": 1,
                    "error": str(e)
                }
                passed = False

            all_results.append({
                "agent": target_dir.name,
                "target_dir": str(target_dir.resolve()),
                "passed": passed,
                "tools": agent_results
            })

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        final_report = {
            "run_id": run_id,
            "stage": "static_analysis",
            "model": args.llm,
            "timestamp": datetime.utcnow().isoformat(),
            "total_agents": len(all_results),
            "results": all_results
        }

        EXPERIMENTS_RESULTS.mkdir(parents=True, exist_ok=True)
        report_path = EXPERIMENTS_RESULTS / f"03_lint_{args.llm}_{run_id}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        print(f"\n[lint] Batch analysis completed. Report saved at → {report_path}")

    finally:
        print("\nShutting down SonarQube server...")
        sonar_process.terminate()

if __name__ == "__main__":
    main()