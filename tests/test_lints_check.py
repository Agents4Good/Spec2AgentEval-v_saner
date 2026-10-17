import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def lint(monkeypatch, tmp_path):
    monkeypatch.setattr('os.environ', {
        'HOME': str(tmp_path), 'GENERATED_DIR': str(tmp_path / 'generated'),
        'EXPERIMENTS_RESULTS': str(tmp_path / 'results'), 'SONAR_MODE': 'disabled',
        'SPEC2AGENT_SKIP_DOTENV': '1', 'PYTHON_DOTENV_DISABLED': '1',
    })
    spec = importlib.util.spec_from_file_location('lint_check', Path(__file__).resolve().parents[1] / 'pipeline/03_lints_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('tool,rc,status', [
    *[('ruff', r, s) for r, s in [(0, 'ok'), (1, 'ok'), (2, 'error'), (3, 'error'), (-9, 'error')]],
    *[('pylint', r, 'error' if r & 33 or r > 63 else 'ok') for r in [0, 1, 2, 4, 8, 16, 6, 30, 31, 32, 33, 34, 63, 64]],
    ('radon_cc', 0, 'ok'), ('radon_cc', 1, 'error'), ('radon_mi', 1, 'error'),
])
def test_codes(lint, monkeypatch, tool, rc, status):
    def run(argv, **kwargs):
        assert isinstance(argv, list) and kwargs['shell'] is False and kwargs['timeout'] == 300
        return subprocess.CompletedProcess(argv, rc, '[]', '')
    monkeypatch.setattr(lint.subprocess, 'run', run)
    result = lint.run_command(['tool', 'path with spaces'], tool)
    assert result['status'] == status
    assert result['passed'] is (rc == 0)


@pytest.mark.parametrize('exc,reason', [(OSError('spawn'), 'spawn_error'), (subprocess.TimeoutExpired('tool', 300, output=b'partial'), 'timeout')])
def test_execution_errors(lint, monkeypatch, exc, reason):
    def run(*a, **kw):
        raise exc
    monkeypatch.setattr(lint.subprocess, 'run', run)
    assert lint.run_command(['tool'], 'ruff')['reason_code'] == reason


@pytest.mark.parametrize('ruff_rc', [0, 1])
@pytest.mark.parametrize('mode,required,configured,broken,expected', [
    ('disabled', '1', False, False, 'skipped'),
    ('external', '0', False, False, 'skipped'),
    ('external', '1', False, False, 'error'),
    ('external', '0', True, True, 'error'),
    ('external', '1', True, False, 'ok'),
    ('local', '1', True, True, 'error'),
])
def test_batch(lint, monkeypatch, mode, required, configured, broken, expected, ruff_rc):
    target = lint.GENERATED_AGENTS_DIR / 'by_tool/run_1/model/a'
    target.mkdir(parents=True)
    (target / 'agent.py').write_text('synthetic')
    events = []
    monkeypatch.setenv('SONAR_MODE', mode)
    monkeypatch.setenv('SONAR_REQUIRED', required)
    if configured:
        for key in ['SONAR_HOST_URL', 'SONAR_LOGIN', 'SONAR_PASSWORD', 'USER_TOKEN', 'SONAR_COMMAND']:
            monkeypatch.setenv(key, 'synthetic')
    def run(*a, **kw):
        events.append('lint')
        return subprocess.CompletedProcess(a[0], ruff_rc if a[0][0] == 'ruff' else 0, '[]', '')
    monkeypatch.setattr(lint.subprocess, 'run', run)
    def wait(timeout):
        assert timeout == 10
        if mode == 'local' and broken and 'kill' not in events:
            raise subprocess.TimeoutExpired('synthetic-server', timeout)
    process = types.SimpleNamespace(terminate=lambda: events.append('terminate'),
        wait=wait, kill=lambda: events.append('kill'))
    def login(session):
        assert events[:4] == ['lint'] * 4
        if broken:
            raise RuntimeError('secret must not leak')
        return 'xsrf'
    fake = types.SimpleNamespace(
        start_sonarqube_server=lambda: (events.append('start') or process),
        wait_for_sonarqube_boot=lambda: None, requests=types.SimpleNamespace(Session=lambda: types.SimpleNamespace(close=lambda: None)),
        sonar_login=login, uuid=types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex='1234')),
        create_project=lambda *a: None, generate_project_token=lambda *a: 'token', run_pysonar=lambda *a: None,
        wait_for_analysis=lambda *a: None, extract_metrics=lambda *a: {},
    )
    monkeypatch.setitem(sys.modules, 'sonar_scripts', types.SimpleNamespace(sonar_pipeline=fake))
    monkeypatch.setattr(sys, 'argv', ['lint', '--llm', 'model'])
    code = lint.main()
    report = json.loads(next(lint.EXPERIMENTS_RESULTS.rglob('03_*.json')).read_text())
    assert report['results'][0]['tools']['sonarqube']['status'] == expected
    assert report['results'][0]['passed'] is (ruff_rc == 0 and expected != 'error')
    assert report['status'] == ('error' if expected == 'error' else 'ok')
    assert report['results'][0]['status'] == report['status']
    if expected == 'skipped':
        sonar = report['results'][0]['tools']['sonarqube']
        assert sonar['passed'] is None
        assert sonar['reason_code'] == ('disabled' if mode == 'disabled' else 'optional_unconfigured')
    assert code == (1 if expected == 'error' else 0)
    assert 'secret must not leak' not in json.dumps(report)
    assert ('start' in events) is (mode == 'local' and configured)
    assert ('terminate' in events) is (mode == 'local' and configured)
    assert ('kill' in events) is (mode == 'local' and configured and broken)


def test_missing_agent_saved(lint, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['lint', '--llm', 'model', '--agent', 'missing'])
    assert lint.main() == 1
    report = json.loads(next(lint.EXPERIMENTS_RESULTS.rglob('03_*.json')).read_text())
    assert report['results'][0]['reason_code'] == 'missing_agent'


@pytest.fixture
def sonar(lint):
    spec = importlib.util.spec_from_file_location('sonar_under_test', Path(__file__).resolve().parents[1] / 'pipeline/sonar_scripts/sonar_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scanner_secret_only_in_environment(sonar, monkeypatch):
    calls = []
    monkeypatch.setattr(sonar.subprocess, 'run', lambda *a, **kw: calls.append((a, kw)))
    sonar.run_pysonar('secret-token', '/synthetic', 'project')
    args, kwargs = calls[0]
    assert 'secret-token' not in repr(args)
    assert kwargs['env']['SONAR_TOKEN'] == 'secret-token'
    assert kwargs['timeout'] == 300 and kwargs['shell'] is False


def test_http_bounded_and_no_logging(sonar, capsys):
    calls = []
    response = types.SimpleNamespace(status_code=200, raise_for_status=lambda: None,
        json=lambda: {'token': 'secret', 'current': {'status': 'SUCCESS'}})
    def request(*a, **kw):
        assert kw['timeout'] == 10
        calls.append(kw)
        return response
    session = types.SimpleNamespace(post=request, get=request, cookies={'XSRF-TOKEN': 'secret'})
    sonar.sonar_login(session)
    sonar.create_project(session, 'secret', 'key', 'name')
    sonar.generate_project_token(session, 'secret', 'key')
    sonar.wait_for_analysis(session, 'key')
    sonar.extract_metrics(session, 'key')
    assert len(calls) == 5
    assert capsys.readouterr().out == ''


def test_polling_timeout(sonar, monkeypatch):
    ticks = iter([0, 0, 301])
    monkeypatch.setattr(sonar.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(sonar.time, 'sleep', lambda _: None)
    session = types.SimpleNamespace(get=lambda *a, **kw: types.SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {'current': {'status': 'PENDING'}}))
    with pytest.raises(TimeoutError):
        sonar.wait_for_analysis(session, 'key')


def test_timeout_preserves_other_tools_and_agents(lint, monkeypatch):
    for name in ['a', 'b']:
        target = lint.GENERATED_AGENTS_DIR / f'by_tool/run_1/model/{name}'
        target.mkdir(parents=True)
        (target / 'agent.py').write_text('synthetic')
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(argv, 300)
        return subprocess.CompletedProcess(argv, 0, '{}', '')
    monkeypatch.setattr(lint.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['lint', '--llm', 'model'])
    assert lint.main() == 1
    report = json.loads(next(lint.EXPERIMENTS_RESULTS.rglob('03_*.json')).read_text())
    assert len(calls) == 8 and len(report['results']) == 2
    assert [e['status'] for e in report['results']] == ['error', 'ok']


@pytest.mark.parametrize('tools', [{}, {'sonarqube': {'status': 'skipped', 'passed': None}}])
def test_aggregate_requires_evaluated_component(lint, monkeypatch, tools):
    target = lint.GENERATED_AGENTS_DIR / 'by_tool/run_1/model/a'
    target.mkdir(parents=True)
    (target / 'agent.py').write_text('synthetic')
    monkeypatch.setattr(lint.subprocess, 'run', lambda argv, **kw: subprocess.CompletedProcess(argv, 0, '{}', ''))
    monkeypatch.setattr(lint, 'run_sonar', lambda results: results[0].update(tools=tools))
    monkeypatch.setattr(sys, 'argv', ['lint', '--llm', 'model'])
    assert lint.main() == 0
    report = json.loads(next(lint.EXPERIMENTS_RESULTS.rglob('03_*.json')).read_text())
    assert report['results'][0]['passed'] is False


@pytest.mark.parametrize('url', [
    'http://:FAKE_URL_PASSWORD@sonar.invalid', 'http://@sonar.invalid',
    'http://sonar.invalid?key=FAKE_URL_PASSWORD', 'http://sonar.invalid#FAKE_URL_PASSWORD',
])
def test_unsafe_url_rejected_before_sonar_import(lint, monkeypatch, capsys, url):
    import builtins
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert name != 'sonar_scripts', 'Unsafe URL reached Sonar import/setup'
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setenv('SONAR_MODE', 'external')
    monkeypatch.setenv('SONAR_HOST_URL', url)
    for key in ['SONAR_LOGIN', 'SONAR_PASSWORD', 'USER_TOKEN']:
        monkeypatch.setenv(key, 'synthetic')
    target = lint.GENERATED_AGENTS_DIR / 'by_tool/run_1/model/a'
    target.mkdir(parents=True)
    (target / 'agent.py').write_text('synthetic')
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, '{}', '')
    monkeypatch.setattr(lint.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['lint', '--llm', 'model'])
    assert lint.main() == 1
    report_text = next(lint.EXPERIMENTS_RESULTS.rglob('03_*.json')).read_text()
    report = json.loads(report_text)
    sonar = report['results'][0]['tools']['sonarqube']
    assert sonar['status'] == 'error' and sonar['reason_code'] == 'invalid_configuration'
    assert report['status'] == 'error' and report['results'][0]['passed'] is False
    assert len(calls) == 4
    assert 'FAKE_URL_PASSWORD' not in report_text + str(calls) + str(capsys.readouterr())
