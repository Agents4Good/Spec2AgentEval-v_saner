import os
import sys
import json
import shlex
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit
from dotenv import load_dotenv, find_dotenv

if os.getenv('SPEC2AGENT_SKIP_DOTENV') != '1':
    load_dotenv(find_dotenv(), override=True)

GENERATED_AGENTS_DIR = Path(os.getenv('GENERATED_DIR', 'generated_agents'))
EXPERIMENTS_RESULTS = Path(os.getenv('EXPERIMENTS_RESULTS', 'experiments_results'))


def run_command(command, tool=None):
    argv = shlex.split(command) if isinstance(command, str) else command
    tool = tool or Path(argv[0]).name
    result = dict(command=command, exit_code=None, stdout='', stderr='', passed=False,
                  status='error', error=None, reason_code=None)
    try:
        proc = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=300)
        rc = proc.returncode
        normal = (rc in (0, 1) if tool == 'ruff' else
                  0 <= rc <= 63 and not rc & 33 if tool == 'pylint' else rc == 0)
        result.update(exit_code=rc, stdout=proc.stdout, stderr=proc.stderr,
                      passed=rc == 0, status='ok' if normal else 'error')
        if not normal:
            result.update(error='Tool execution failed', reason_code='signal' if rc < 0 else 'tool_error')
    except (OSError, subprocess.TimeoutExpired) as exc:
        def text(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        result.update(stdout=text(getattr(exc, 'stdout', '')), stderr=text(getattr(exc, 'stderr', '')),
                      error='Tool execution failed', reason_code='timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'spawn_error')
    return result


def run_sonar(results):
    mode = os.getenv('SONAR_MODE', 'local')
    required = os.getenv('SONAR_REQUIRED', '1')
    outcome = dict(status='error', exit_code=1, passed=False, error=None, reason_code=None)
    try:
        host = urlsplit(os.getenv('SONAR_HOST_URL', ''))
        unsafe_host = bool(host.username is not None or host.password is not None or host.query or host.fragment)
    except ValueError:
        unsafe_host = True
    keys = ['SONAR_HOST_URL', 'SONAR_LOGIN', 'SONAR_PASSWORD', 'USER_TOKEN']
    if mode == 'local':
        keys.append('SONAR_COMMAND')
    if mode == 'disabled':
        outcome.update(status='skipped', exit_code=None, passed=None, reason_code='disabled')
    elif mode not in ('local', 'external') or required not in ('0', '1'):
        outcome.update(error='Invalid Sonar configuration', reason_code='invalid_configuration')
    elif not all(os.getenv(key) for key in keys):
        outcome.update(status='skipped' if required == '0' else 'error',
                       exit_code=None if required == '0' else 1, passed=None if required == '0' else False,
                       error=None if required == '0' else 'Missing Sonar configuration',
                       reason_code='optional_unconfigured' if required == '0' else 'missing_configuration')
    elif unsafe_host:
        outcome.update(error='Sonar URL must not contain credentials or query data', reason_code='invalid_configuration')
    else:
        process = session = None
        try:
            from sonar_scripts import sonar_pipeline as sonar
            if mode == 'local':
                process = sonar.start_sonarqube_server()
                sonar.wait_for_sonarqube_boot()
            session = sonar.requests.Session()
            xsrf = sonar.sonar_login(session)
            for entry in results:
                if entry['status'] == 'error' and not entry['tools']:
                    continue
                try:
                    key = f"{entry['model']}_{entry['agent']}_{sonar.uuid.uuid4().hex[:4]}"
                    sonar.create_project(session, xsrf, key, f"{entry['model']} - {entry['agent']}")
                    token = sonar.generate_project_token(session, xsrf, key)
                    sonar.run_pysonar(token, entry['target_dir'], key)
                    sonar.wait_for_analysis(session, key)
                    entry['tools']['sonarqube'] = dict(status='ok', exit_code=0, passed=True,
                        metrics=sonar.extract_metrics(session, key), error=None, reason_code=None)
                except Exception as exc:
                    entry['tools']['sonarqube'] = dict(outcome, error=f'Sonar analysis failed ({type(exc).__name__})', reason_code='sonar_error')
            return
        except Exception as exc:
            outcome.update(error=f'Sonar setup failed ({type(exc).__name__})', reason_code='sonar_error')
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            if process is not None:
                try:
                    try:
                        process.terminate()
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    outcome.update(error='Sonar cleanup failed', reason_code='cleanup_error')
                    for entry in results:
                        entry['tools']['sonarqube'] = dict(outcome)
    for entry in results:
        entry['tools']['sonarqube'] = dict(outcome)


def main():
    parser = argparse.ArgumentParser(description='Executes lints and SonarQube analysis in batch.')
    parser.add_argument('--llm', required=True)
    parser.add_argument('--agent', default=None)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--experiment', default='by_tool')
    args = parser.parse_args()
    llm_dir = GENERATED_AGENTS_DIR / args.experiment / f'run_{args.run}' / args.llm
    root = EXPERIMENTS_RESULTS / args.experiment / f'run_{args.run}'
    targets = [llm_dir / args.agent] if args.agent else sorted(p for p in llm_dir.iterdir() if p.is_dir()) if llm_dir.is_dir() else []
    results = []
    for target in targets:
        entry = dict(agent=target.name, model=args.llm, target_dir=str(target.resolve()),
                     passed=False, tools={}, status='ok', error=None, reason_code=None)
        results.append(entry)
        if not (target / 'agent.py').is_file():
            entry.update(status='error', error='Agent file missing', reason_code='missing_agent')
            continue
        tools = {'ruff': ['ruff', 'check', str(target), '--output-format=json'],
                 'pylint': ['pylint', str(target), '--output-format=json'],
                 'radon_cc': ['radon', 'cc', str(target), '-a', '-j'],
                 'radon_mi': ['radon', 'mi', str(target), '-j']}
        for name, argv in tools.items():
            res = run_command(argv, name)
            res['command'] = shlex.join(argv)
            try:
                res['stdout'] = json.loads(res['stdout'])
            except (ValueError, TypeError):
                pass
            entry['tools'][name] = res
    if results:
        run_sonar(results)
    for entry in results:
        if any(tool['status'] == 'error' for tool in entry['tools'].values()):
            entry.update(status='error', error=entry['error'] or 'Component execution failed',
                         reason_code=entry['reason_code'] or 'component_error')
        evaluated = [tool for tool in entry['tools'].values() if tool['status'] != 'skipped']
        entry['passed'] = bool(evaluated) and entry['status'] != 'error' and all(t['passed'] is True for t in evaluated)
    status = 'error' if not results or any(e['status'] == 'error' for e in results) else 'ok'
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report = dict(run_id=run_id, run=args.run, experiment=args.experiment, stage='static_analysis',
                  model=args.llm, timestamp=datetime.utcnow().isoformat(), total_agents=len(results),
                  results=results, status=status, error='No agents found' if not results else 'Component execution failed' if status == 'error' else None,
                  reason_code='no_agents' if not results else 'component_error' if status == 'error' else None)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'03_lint_{args.llm}_{run_id}.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[lint] Report saved at → {path}')
    return 1 if status == 'error' else 0


if __name__ == '__main__':
    sys.exit(main())
