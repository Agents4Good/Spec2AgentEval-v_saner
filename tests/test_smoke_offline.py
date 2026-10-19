"""Mock-only lifecycle checks: no uv, Docker or providers are executed."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('smoke_lifecycle', ROOT / 'scripts/smoke_offline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(os, 'environ', {'HOME': str(tmp_path), 'PATH': '', 'SPEC2AGENT_SKIP_DOTENV': '1',
        'PYTHON_DOTENV_DISABLED': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'DEEPEVAL_TELEMETRY_OPT_OUT': '1',
        'LANGSMITH_TRACING': 'false', 'LANGCHAIN_TRACING_V2': 'false'})
    monkeypatch.setattr(module.sys, 'executable', str(ROOT / '.venv/bin/python'))
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/synthetic/bin/' + name)
    handlers = {signal.SIGINT: object(), signal.SIGTERM: object()}
    originals = handlers.copy()
    def install(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        return previous
    monkeypatch.setattr(module.signal, 'signal', install)
    private = tmp_path / 'private'
    allocations, removed, events = [], [], []
    def allocate(**kwargs):
        allocations.append(kwargs)
        private.mkdir()
        return str(private)
    monkeypatch.setattr(module.tempfile, 'mkdtemp', allocate)
    process = type('Process', (), {'pid': 123, 'reaped': False})()
    def remove(path):
        assert process.reaped, 'removed before confirmed reap'
        removed.append(path)
    monkeypatch.setattr(module.shutil, 'rmtree', remove)
    def start(args, **kwargs):
        assert kwargs['start_new_session'] is True
        assert kwargs['env']['PATH'] == str(private / 'bin')
        assert kwargs['env']['LANGSMITH_TRACING'] == 'false'
        return process
    monkeypatch.setattr(module.subprocess, 'Popen', start)
    monkeypatch.setattr(module.os, 'killpg', lambda pid, signum: events.append(signum))
    return module, process, handlers, originals, private, allocations, removed, events


@pytest.mark.parametrize('first', [signal.SIGINT, signal.SIGTERM])
def test_repeated_signals_escalate_before_cleanup(smoke, first):
    module, process, handlers, originals, private, _, removed, events = smoke
    def wait(timeout):
        if timeout == 240:
            handlers[first](first, None)
        # Further real signals during teardown must not raise.
        for signum in (signal.SIGINT, signal.SIGTERM, first):
            handlers[signum](signum, None)
        if events == [signal.SIGTERM]:
            raise subprocess.TimeoutExpired('synthetic', timeout)
        process.reaped = True
        return -9
    process.wait = wait
    assert module.main() == 1
    assert events[:2] == [signal.SIGTERM, signal.SIGKILL]
    assert removed == [private] and handlers == originals


@pytest.mark.parametrize('failure', ['unreaped', 'wait', 'kill', 'interrupt_cleanup'])
def test_failure_preserves_private_files(smoke, monkeypatch, capsys, failure):
    module, process, handlers, originals, private, _, removed, events = smoke
    def wait(timeout):
        if timeout == 240:
            if failure == 'wait':
                raise OSError('synthetic failure')
            raise subprocess.TimeoutExpired('synthetic', timeout)
        if failure == 'interrupt_cleanup':
            raise KeyboardInterrupt
        raise subprocess.TimeoutExpired('synthetic', timeout)
    process.wait = wait
    if failure == 'kill':
        monkeypatch.setattr(module.os, 'killpg', lambda *args: (_ for _ in ()).throw(OSError('synthetic')))
    assert module.main() == 1
    assert not removed and private.is_dir()
    assert 'private files preserved:' in capsys.readouterr().err
    assert handlers == originals


@pytest.mark.parametrize('preferred', [False, True])
def test_normal_exit_and_portable_temp(smoke, monkeypatch, preferred):
    module, process, handlers, originals, private, allocations, removed, _ = smoke
    actual = Path.is_dir
    monkeypatch.setattr(Path, 'is_dir', lambda path: preferred if path == Path('/tmp/opencode') else actual(path))
    def wait(timeout):
        assert timeout == 240
        process.reaped = True
        return 0
    process.wait = wait
    assert module.main() == 0
    assert allocations[0]['dir'] == ('/tmp/opencode' if preferred else None)
    assert removed == [private] and handlers == originals


def test_cleanup_filesystem_failure_reports_preservation(smoke, monkeypatch, capsys):
    module, process, _, originals, private, _, _, _ = smoke
    def wait(timeout):
        process.reaped = True
        return 0
    process.wait = wait
    monkeypatch.setattr(module.shutil, 'rmtree', lambda path: (_ for _ in ()).throw(OSError('synthetic')))
    assert module.main() == 1
    assert private.is_dir()
    assert 'private files preserved:' in capsys.readouterr().err
