"""Private allowlisted Docker evaluation staging."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).absolute().parent.parent
STEPS = {
    '02': ('02_static_check.py', 'validation'),
    '03': ('03_lints_check.py', 'lint'),
    '04': ('04_tests_check.py', 'tests_check'),
    '05': ('05_spec_llm_judge_check.py', 'spec_judge'),
    '06': ('06_graph_eval_check.py', 'graph_eval'),
    '07': ('07_behavioral_check.py', 'test_deepeval'),
    '08': ('08_trace_check.py', 'trace_eval'),
}
BENCH = {'05': ('spec.md', 'spec_adherence.py'),
         '06': ('eval_2_graph.py',), '07': ('test_deepeval.py',),
         '08': ('test_trace.py',)}


def safe(path):
    """Reject symlinks, including dangling links and existing ancestors."""
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        if part.is_symlink():
            raise ValueError(f'symlink forbidden: {part}')
    return path


def copy_file(source, target, required=False):
    source = safe(source)
    if not source.exists() and not required:
        return
    if not source.is_file():
        raise ValueError(f'not a regular input file: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def component(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise argparse.ArgumentTypeError('expected safe alphanumeric component')
    return value


def publish_file(source, destination):
    source, destination = safe(source), safe(destination)
    if not stat.S_ISREG(source.stat().st_mode):
        raise ValueError(f'not a regular product: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe(destination.parent)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as out, source.open('rb') as inp:
            shutil.copyfileobj(inp, out)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def publish_trace_reports(sources, source_root, destination_root, llm, agent):
    """Revalidate only manifest-selected job traces; publish reports last."""
    traces, reports, seen = [], [], set()
    def read_regular(path):
        path = safe(path)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('unsafe trace/report product')
        with path.open('rb') as stream:
            return stream.read()
    def strict_json(raw):
        return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('invalid JSON constant')))
    try:
        for source in sources:
            raw = read_regular(source)
            reports.append((destination_root / source.name, raw))
            try:
                report = strict_json(raw)
            except (ValueError, UnicodeError):
                # Preserve legacy report publication; no manifest means no traces.
                continue
            for entry in report.get('results', []):
                artifacts = entry.get('trace_artifacts', [])
                if not isinstance(artifacts, list):
                    raise ValueError('invalid trace manifest')
                for record in artifacts:
                    if (report.get('model') != llm or entry.get('agent_folder') != agent
                            or entry.get('model') != llm or not isinstance(record, dict)):
                        raise ValueError('foreign trace job')
                    path = record.get('path')
                    parts = path.split('/') if isinstance(path, str) else []
                    if (len(parts) != 5 or parts[:3] != ['08_traces', llm, agent]
                            or not re.fullmatch(r'[0-9a-f]{32}', parts[3])
                            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.json', parts[4])):
                        raise ValueError('invalid trace path')
                    relative = Path(*parts)
                    payload = read_regular(source_root / relative)
                    trace = strict_json(payload)
                    if (not isinstance(trace, dict) or type(trace.get('schema_version')) is not int
                            or trace['schema_version'] != 1
                            or not isinstance(trace.get('run_id'), str)
                            or not re.fullmatch(r'[0-9a-f]{32}', trace['run_id'])
                            or trace['run_id'] != record.get('run_id')
                            or trace.get('status') not in ('ok', 'error')
                            or trace['status'] != record.get('capture_status')
                            or not isinstance(trace.get('test_nodeid'), str)
                            or trace['test_nodeid'] != record.get('test_nodeid')
                            or not {'query', 'error', 'trajectory', 'tool_calls', 'runs_collected'} <= trace.keys()
                            or trace['error'] is not None and not isinstance(trace['error'], str)
                            or type(trace['runs_collected']) is not int or trace['runs_collected'] < 0
                            or not isinstance(trace['trajectory'], list) or not isinstance(trace['tool_calls'], list)
                            or not all(isinstance(name, str) for name in trace['trajectory'])
                            or not all(isinstance(call, dict) and {'name', 'input', 'start', 'end'} <= call.keys()
                                and isinstance(call['name'], str)
                                and all(call[key] is None or isinstance(call[key], str) for key in ('start', 'end'))
                                and (('output' in call) != ('error' in call))
                                and ('error' not in call or isinstance(call['error'], str))
                                for call in trace['tool_calls'])
                            or path in seen):
                        raise ValueError('invalid trace identity/schema')
                    seen.add(path)
                    traces.append((destination_root / relative, payload))
        products = traces + reports
        for destination, _ in products:
            if safe(destination).exists():
                raise ValueError('trace/report destination exists')
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        raise ValueError('trace/report validation failed') from None
    created = []
    try:
        for destination, payload in products:
            for directory in (*reversed(destination.parent.parents), destination.parent):
                safe(directory).mkdir(mode=0o700, exist_ok=True)
            fd = os.open(safe(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            created.append(destination)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(payload)
    except BaseException:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        raise ValueError('trace/report publication failed') from None


class ClientStillAlive(RuntimeError):
    """A bounded kill/reap attempt could not establish client termination."""


def terminate_client(process):
    """Bounded process-group cancellation, including setup subprocesses."""
    if process.poll() is not None:
        return
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            continue
    raise ClientStillAlive('Docker client could not be reaped')


def invoke(command, cancelled, **kwargs):
    """Track every client; cancellation is observed before the next operation."""
    if cancelled():
        raise InterruptedError('evaluation cancelled')
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        while True:
            if cancelled():
                raise InterruptedError('evaluation cancelled')
            try:
                return process.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                continue
    finally:
        terminate_client(process)


def teardown(name):
    """An error from inspect is not proof of absence: require a successful list."""
    try:
        removal = subprocess.run(['docker', 'rm', '-f', name], timeout=3,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        listed = subprocess.run(['docker', 'container', 'ls', '-a', '--filter',
                                 f'name=^/{name}$', '--format', '{{.Names}}'],
                                capture_output=True, text=True, timeout=3)
        absent = listed.returncode == 0 and not listed.stdout.strip()
        if removal.returncode and not absent:
            print('[docker-eval] container removal refused or unavailable', file=sys.stderr)
        return absent
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'[docker-eval] teardown: {exc}', file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--step', choices=STEPS, required=True)
    parser.add_argument('--llm', type=component, required=True)
    parser.add_argument('--agent', type=component, required=True)
    parser.add_argument('--experiment', type=component, default='by_tool')
    parser.add_argument('--run', default='1')
    args = parser.parse_args()
    if not re.fullmatch(r'[1-9][0-9]{0,8}', args.run):
        parser.error('run must be a positive canonical integer')
    selinux = os.environ.get('EVAL_SELINUX', 'Z')
    if selinux not in ('Z', 'none', 'disable'):
        parser.error('EVAL_SELINUX must be Z, none or disable')
    provider = os.environ.get('EVAL_PROVIDER', 'openai')
    if provider not in ('openai', 'gemini', 'none'):
        parser.error('EVAL_PROVIDER must be openai, gemini or none')
    network = os.environ.get('EVAL_NETWORK', 'bridge')
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', network)
            or network.lower() == 'host'):
        parser.error('unsafe EVAL_NETWORK')
    if args.step in ('02', '04', '06'):
        network = 'none'
    bench = safe(os.environ.get('BENCH_DIR', ROOT / 'specs'))
    generated = safe(os.environ.get('GENERATED_DIR', ROOT / 'generated_agents'))
    results = safe(os.environ.get('EXPERIMENTS_RESULTS', ROOT / 'experiments_results'))
    relative = Path(args.experiment) / f'run_{args.run}'
    job = relative / args.llm / args.agent
    filename, stem = STEPS[args.step]
    image = os.environ.get('EVAL_IMAGE', 'spec2agent-eval:infra')
    name = 'spec2agent-eval-' + uuid.uuid4().hex
    interrupted = None
    private = None
    preserve = False

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        private = Path(tempfile.mkdtemp(prefix='spec2agent-eval-'))
        try:
            for directory in ('repo', 'bench', 'data', 'results'):
                (private / directory).mkdir(mode=0o700)
            copy_file(ROOT / 'pipeline' / filename,
                      private / 'repo/pipeline' / filename, True)
            extra = {'03': 'sonar_scripts/sonar_pipeline.py',
                     '07': 'utils/pytest_runner.py', '08': 'utils/pytest_runner.py'}
            if args.step in extra:
                copy_file(ROOT / 'pipeline' / extra[args.step],
                          private / 'repo/pipeline' / extra[args.step], True)
            if args.step == '08':
                copy_file(ROOT / 'pipeline/utils/trace_capture.py',
                          private / 'repo/pipeline/utils/trace_capture.py', True)
            (private / 'data' / job).mkdir(parents=True)
            copy_file(generated / job / 'agent.py', private / 'data' / job / 'agent.py')
            # Stage04 inventories generated tests: direct Python test files only.
            if args.step == '04' and safe(generated / job).is_dir():
                for source in safe(generated / job).iterdir():
                    if source.suffix == '.py' and (source.name.startswith('test')
                                                   or source.name.endswith('test.py')):
                        copy_file(source, private / 'data' / job / source.name, True)
            (private / 'bench' / args.agent).mkdir()
            for item in BENCH.get(args.step, ()):
                copy_file(bench / args.agent / item, private / 'bench' / args.agent / item)
            cancelled = lambda: interrupted is not None
            inspect = invoke(['docker', 'image', 'inspect', image], cancelled,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if inspect:
                code = invoke(['docker', 'build', '-f', str(ROOT / 'docker/Dockerfile.eval'),
                               '-t', image, str(ROOT)], cancelled)
                if code:
                    raise subprocess.CalledProcessError(code, 'docker build')
            command = ['docker', 'run', '--rm', '--name', name,
                       '--user', f'{os.getuid()}:{os.getgid()}', '--network', network]
            if selinux == 'disable':
                print('WARNING: SELinux labeling explicitly disabled', file=sys.stderr)
                command += ['--security-opt', 'label=disable']
            for directory, mode in [('repo', 'ro'), ('bench', 'ro'),
                                    ('data', 'rw'), ('results', 'rw')]:
                suffix = ',Z' if selinux == 'Z' else ''
                command += ['-v', f'{private / directory}:/{directory}:{mode}{suffix}']
            environment = {'GENERATED_DIR': '/data', 'EXPERIMENTS_RESULTS': '/results',
                           'BENCH_DIR': '/bench', 'SPEC2AGENT_SKIP_DOTENV': '1',
                           'PYTHON_DOTENV_DISABLED': '1'}
            config = {'03': ('SONAR_MODE', 'SONAR_REQUIRED'),
                      '07': ('PYTEST_TIMEOUT_SECONDS',), '08': ('PYTEST_TIMEOUT_SECONDS',)}
            for key in config.get(args.step, ()):
                if key in os.environ:
                    environment[key] = os.environ[key]
            allowed = ()
            if args.step == '03':
                allowed = ('SONAR_HOST_URL', 'SONAR_LOGIN', 'SONAR_PASSWORD',
                           'USER_TOKEN', 'SONAR_COMMAND')
            elif args.step in ('05', '07', '08') and provider != 'none':
                allowed = ('OPENAI_API_KEY' if provider == 'openai' else 'GEMINI_API_KEY',)
            for key in allowed:
                if os.environ.get(key):
                    command += ['-e', key]
            for key, value in environment.items():
                command += ['-e', f'{key}={value}']
            command += [image, 'python3', f'/repo/pipeline/{filename}',
                        '--llm', args.llm, '--agent', args.agent, '--run', args.run,
                        '--experiment', args.experiment]
            if interrupted:
                return 128 + interrupted
            try:
                try:
                    code = invoke(command, cancelled)
                except InterruptedError:
                    if not interrupted:
                        raise
                    code = 128 + interrupted
            finally:
                # Always stabilize the named workload, even on client disconnect
                # or wait exceptions. Never publish from a potentially live run.
                preceding = sys.exc_info()[1]
                preserve = True
                absent = teardown(name)
                if not absent or isinstance(preceding, ClientStillAlive):
                    detail = f'; preceding error: {preceding}' if preceding else ''
                    raise RuntimeError(f'container absence unconfirmed: {name}{detail}')
                preserve = False
            # stdout/stderr remain inherited; only real stage reports are products.
            report_dir = safe(private / 'results' / relative)
            pattern = re.compile(rf'{args.step}_{stem}_{re.escape(args.llm)}_[0-9]{{8}}_[0-9]{{6}}\.json')
            if report_dir.exists():
                sources = [source for source in report_dir.iterdir() if pattern.fullmatch(source.name)]
                if args.step == '08':
                    publish_trace_reports(sources, report_dir, results / relative, args.llm, args.agent)
                else:
                    for source in sources:
                        publish_file(source, results / relative / source.name)
            # Known per-job diagnostics only, never arbitrary container files.
            # Pytest embeds stdout/stderr and its JSON report in stage reports.
            if args.step in ('07', '08'):
                for item in ('trace.json', 'trace.jsonl', 'agent.log', '.pytest_report.json'):
                    source = safe(private / 'data' / job / item)
                    if source.exists():
                        destination = (results / relative / 'eval_artifacts' /
                                       f'{args.step}_{args.llm}_{args.agent}_{name}' / item)
                        publish_file(source, destination)
            return 128 + interrupted if interrupted else (code if code >= 0 else 128 - code)
        except InterruptedError:
            if interrupted:
                return 128 + interrupted
            raise
        finally:
            if preserve:
                print(f'[docker-eval] PRIVATE STAGING PRESERVED: {private}; '
                      f'recover after confirming container {name} absent', file=sys.stderr)
            else:
                shutil.rmtree(private)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'[docker-eval] {exc}', file=sys.stderr)
        sys.exit(1)
