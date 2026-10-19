"""Run only synthetic tracing/staging contracts; never build or run Docker."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
def main():
    root = Path(__file__).absolute().parent.parent
    python = Path(sys.executable).absolute()
    uv = shutil.which('uv')
    if not uv or not (root / '.venv/bin/python').is_file():
        print('Need uv on PATH and the prepared project .venv; no installation attempted.', file=sys.stderr)
        return 1
    if python != root / '.venv/bin/python':
        print('Run with the absolute project .venv/bin/python interpreter.', file=sys.stderr)
        return 1
    tools = {name: shutil.which(name) for name in ('bash', 'dirname', 'cut', 'seq', 'sleep', 'mkdir', 'id')}
    if not all(tools.values()):
        print('Missing required shell helper; no installation attempted.', file=sys.stderr)
        return 1
    private = Path(tempfile.mkdtemp(prefix='spec2agent-smoke-',
                                  dir='/tmp/opencode' if Path('/tmp/opencode').is_dir() else None))
    process, cleaned, tearing_down = None, True, False
    def cancelled(signum, frame):
        if not tearing_down:
            raise KeyboardInterrupt
    old_handlers = {s: signal.signal(s, cancelled) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name in ('home', 'cache', 'tmp', 'bin'):
            (private / name).mkdir(mode=0o700)
        for name, target in tools.items():
            (private / 'bin' / name).symlink_to(target)
        config = private / 'pytest.ini'
        config.write_text('[pytest]\n')
        env = dict(HOME=str(private / 'home'), TMPDIR=str(private / 'tmp'),
                   XDG_CACHE_HOME=str(private / 'cache'), UV_CACHE_DIR=str(private / 'cache/uv'),
                   PATH=str(private / 'bin'), SPEC2AGENT_SKIP_DOTENV='1', PYTHON_DOTENV_DISABLED='1',
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', DEEPEVAL_TELEMETRY_OPT_OUT='1',
                   LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false')
        tests = ['test_trace_capture.py', 'test_trace_runner.py', 'test_docker_eval_isolation.py', 'test_mvp_contract.py']
        args = [uv, '--no-config', '--offline', 'run', '--no-project', '--no-env-file',
                '--python', str(python), str(python), '-I', '-m', 'pytest', '--noconftest',
                '--confcutdir', str(private), '-c', str(config), '-q',
                *(str(root / 'tests' / name) for name in tests)]
        cleaned = False  # A spawn/cancellation failure may leave a live client.
        process = subprocess.Popen(args, cwd=private, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=240)
            cleaned = True
            return code
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            tearing_down = True
            for signum in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    cleaned = True
                    return 1
                except subprocess.TimeoutExpired:
                    continue
            return 1
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
        return 1
    finally:
        tearing_down = True
        try:
            if cleaned:
                cleaned = False
                shutil.rmtree(private)
                cleaned = True
        except (OSError, KeyboardInterrupt):
            cleaned = False
            return 1
        finally:
            if not cleaned:
                print(f'Smoke cleanup/termination incomplete; private files preserved: {private}', file=sys.stderr)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
if __name__ == '__main__':
    sys.exit(main())
