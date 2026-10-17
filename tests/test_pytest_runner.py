"""Lane C fixtures: synthetic inputs, explicit plugins, no provider execution."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lane_c_runner', ROOT / 'pipeline/utils/pytest_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
# Reuse exactly the preparation guard as a literal, without importing that test.
tree = ast.parse((ROOT / 'tests/test_pytest_plugin_contract.py').read_text())
GUARD = next(ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
             and t.id == 'BOOTSTRAP' for t in node.targets))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    env = {}
    for name in ('home', 'temp', 'bin'):
        (tmp_path / name).mkdir()
    env.update(HOME=str(tmp_path / 'home'), TMPDIR=str(tmp_path / 'temp'),
               PATH=str(tmp_path / 'bin'), SPEC2AGENT_SKIP_DOTENV='1',
               PYTHON_DOTENV_DISABLED='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               DEEPEVAL_TELEMETRY_OPT_OUT='1')
    for key in tuple(os.environ):
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # tempfile caches its parent in this process; never reuse the host temp HOME.
    monkeypatch.setattr(runner.tempfile, 'tempdir', str(tmp_path / 'temp'))
    agent = tmp_path / 'generated/exp/run_1/model/agent'
    agent.mkdir(parents=True)
    (agent / 'agent.py').write_text('def agent():\n    return 42\n')
    bench = tmp_path / 'bench/agent'
    bench.mkdir(parents=True)
    (bench / 'test_deepeval.py').write_text('from agent import agent\ndef test_ok():\n    assert agent() == 42\n')
    return tmp_path, agent, bench


def invoke(fixture, **kwargs):
    tmp_path, _, _ = fixture
    return runner.run_pytest_for_agents(llm='model', agent_filter=kwargs.pop('agent_filter', 'agent'),
        test_filename='test_deepeval.py', generated_dir=tmp_path / 'generated',
        bench_dir=tmp_path / 'bench', experiments_results=tmp_path / 'results',
        stage_label='evaluation', stage_prefix='07', report_name_stem='test_deepeval',
        experiment='exp', **kwargs)


def payload(outcome='passed', phase='call', exitcode=0):
    test = {'nodeid': 'test_synthetic.py::test_one', 'outcome': outcome,
            'setup': {'outcome': 'passed'}, 'teardown': {'outcome': 'passed'}}
    test[phase] = {'outcome': 'failed' if outcome in ('failed', 'error') else outcome,
                   'longrepr': 'synthetic reason'}
    return {'exitcode': exitcode, 'summary': {'total': 1, 'collected': 1, outcome: 1}, 'tests': [test]}


def fake(monkeypatch, report, rc=0, inspect=None):
    class Process:
        pid = 99999999
        returncode = rc
        def __init__(self, args, **kwargs):
            if inspect:
                inspect(args, kwargs)
            path = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--json-report-file=')))
            if report is not None:
                path.write_text(report if isinstance(report, str) else json.dumps(report))
        def communicate(self, timeout=None):
            return 'stdout sentinel', 'stderr sentinel'
        def poll(self):
            return rc
        def wait(self, timeout=None):
            return rc
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner.os, 'killpg', lambda *_: None)


@pytest.mark.parametrize('outcome,phase,rc,status,passed,reason', [
    ('passed', 'call', 0, 'ok', True, None),
    ('failed', 'call', 1, 'ok', False, None),
    ('error', 'setup', 1, 'error', None, 'pytest_phase_error'),
    ('error', 'teardown', 1, 'error', None, 'pytest_phase_error'),
    ('skipped', 'setup', 0, 'ok', None, 'no_evaluated_tests'),
])
def test_status_and_phase_details(fixture, monkeypatch, outcome, phase, rc, status, passed, reason):
    report = payload(outcome, phase, rc)
    if phase == 'teardown':
        report['tests'][0]['call'] = {'outcome': 'passed'}
    fake(monkeypatch, report, rc)
    result = invoke(fixture)
    entry = result['results'][0]
    assert result['status'] == entry['status'] == status
    assert entry['passed'] is passed and entry.get('reason_code') == reason
    assert entry['returncode'] == rc and entry['pytest_report'] == report
    assert entry['tests'][0][phase] == report['tests'][0][phase]
    assert entry['stdout'] == 'stdout sentinel' and entry['stderr'] == 'stderr sentinel'
    assert entry['tests'][0]['reason'] == 'synthetic reason'


@pytest.mark.parametrize('report,rc,reason', [
    (None, 0, 'pytest_report_invalid'), ('{bad', 0, 'pytest_report_invalid'),
    ([], 0, 'pytest_report_invalid'), ({}, 0, 'pytest_report_invalid'),
    (payload('passed', exitcode=1), 0, 'pytest_report_invalid'),
    (payload('failed', exitcode=0), 0, 'pytest_report_invalid'),
    (payload(), 1, 'pytest_report_invalid'),
    (payload(), 2, 'pytest_exit'), (payload(), 3, 'pytest_exit'),
    (payload(), 4, 'pytest_exit'), (payload(), 5, 'pytest_exit'),
    (payload(), -15, 'pytest_exit'),
])
def test_invalid_or_operational_exit(fixture, monkeypatch, report, rc, reason):
    fake(monkeypatch, report, rc)
    entry = invoke(fixture)['results'][0]
    assert entry['status'] == 'error' and entry['passed'] is None
    assert entry['reason_code'] == reason and entry['returncode'] == rc


def test_private_workspace_env_and_preexisting_sentinels(fixture, monkeypatch):
    _, agent, _ = fixture
    sentinels = [agent / name for name in ('test_deepeval.py', 'test_trace.py', '.pytest_report.json')]
    for path in sentinels:
        path.write_text('sentinel')
    monkeypatch.setenv('PYTEST_ADDOPTS', '--invalid-host-option')
    monkeypatch.setenv('PYTEST_PLUGINS', 'host_plugin')
    monkeypatch.setenv('DEEPEVAL_TELEMETRY_OPT_OUT', '0')
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-preserved-not-used')
    workspaces = []
    def inspect(args, kwargs):
        assert args[:3] == [sys.executable, '-m', 'pytest']
        assert kwargs['cwd'] == agent and kwargs['start_new_session'] is True
        env = kwargs['env']
        assert 'PYTEST_ADDOPTS' not in env and 'PYTEST_PLUGINS' not in env
        assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == env['DEEPEVAL_TELEMETRY_OPT_OUT'] == '1'
        assert env['OPENAI_API_KEY'] == 'synthetic-preserved-not-used'
        assert all(plugin in args for plugin in ('pytest_jsonreport.plugin', 'pytest_asyncio.plugin', 'deepeval.plugins.plugin'))
        config = Path(args[args.index('-c') + 1])
        assert config.parent != agent and config.is_file()
        workspaces.append(config.parent)
    fake(monkeypatch, payload(), inspect=inspect)
    invoke(fixture)
    invoke(fixture)
    assert len(set(workspaces)) == 2 and all(not p.exists() for p in workspaces)
    assert all(p.read_text() == 'sentinel' for p in sentinels)


@pytest.mark.parametrize('missing', ['folder', 'agent', 'test'])
def test_known_missing_target_entry(fixture, monkeypatch, missing):
    _, agent, bench = fixture
    if missing == 'folder':
        (agent / 'agent.py').unlink()
        agent.rmdir()
    else:
        (agent / 'agent.py' if missing == 'agent' else bench / 'test_deepeval.py').unlink()
    fake(monkeypatch, payload())
    report = invoke(fixture)
    assert report['status'] == 'error' and report['total_agents'] == 1
    assert report['results'][0]['reason_code'] == ('missing_test' if missing == 'test' else 'missing_agent')


def test_empty_selection_and_missing_roots_report(fixture):
    tmp_path, _, _ = fixture
    report = invoke(fixture, agent_filter='unknown')
    assert report['results'][0]['reason_code'] == 'missing_agent'
    (tmp_path / 'generated/exp/run_1/model').rename(tmp_path / 'old_model')
    report = invoke(fixture, agent_filter=None)
    assert report['status'] == 'error' and report['total_agents'] == 0
    assert report['reason_code'] == 'empty_targets'
    assert list((tmp_path / 'results/exp/run_1').glob('07_*.json'))


@pytest.mark.parametrize('location', ['agent_file', 'agent_parent', 'test_file', 'bench_parent'])
def test_symlinks_rejected(fixture, monkeypatch, location):
    _, agent, bench = fixture
    path = {'agent_file': agent / 'agent.py', 'agent_parent': agent,
            'test_file': bench / 'test_deepeval.py', 'bench_parent': bench}[location]
    real = path.with_name(path.name + '_real')
    path.rename(real)
    path.symlink_to(real, target_is_directory=real.is_dir())
    fake(monkeypatch, payload())
    entry = invoke(fixture)['results'][0]
    assert entry['status'] == 'error' and entry['reason_code'] == 'invalid_input'


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'bad'])
def test_invalid_timeout_reported(fixture, monkeypatch, value):
    monkeypatch.setenv('PYTEST_TIMEOUT_SECONDS', value)
    entry = invoke(fixture)['results'][0]
    assert entry['reason_code'] == 'invalid_timeout' and entry['passed'] is None


@pytest.mark.parametrize('source,expected_status,passed,rc', [
    ('from agent import agent\nimport pytest\ndef test_ok(): assert agent() == 42\n@pytest.mark.asyncio\nasync def test_async(): assert agent() == 42\n', 'ok', True, 0),
    ('def test_fail(): assert False, "call failure"\n', 'ok', False, 1),
    ('def test_pass(): pass\ndef test_fail(): assert False\n', 'ok', False, 1),
    ('import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError("setup failure")\ndef test_setup(broken): pass\n', 'error', None, 1),
    ('import pytest\n@pytest.fixture\ndef broken():\n yield\n raise RuntimeError("teardown failure")\ndef test_teardown(broken): pass\n', 'error', None, 1),
    ('import pytest\n@pytest.mark.skip(reason="explicit test skip")\ndef test_skip(): pass\n', 'ok', None, 0),
    ('import pytest\n@pytest.mark.xfail(reason="expected failure")\ndef test_xfail(): assert False\n', 'ok', None, 0),
    ('raise RuntimeError("collection failure")\n', 'error', None, 2),
    ('answer = 42\n', 'error', None, 5),
    ('import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError("setup failure")\ndef test_pass(): pass\ndef test_fail(): assert False\ndef test_setup(broken): pass\n', 'error', None, 1),
])
def test_real_guarded_pytest(fixture, monkeypatch, source, expected_status, passed, rc):
    tmp_path, agent, bench = fixture
    (bench / 'test_deepeval.py').write_text(source)
    for name in ('test_deepeval.py', 'test_trace.py', 'conftest.py'):
        (agent / name).write_text('raise AssertionError("preexisting sentinel must not load")\n')
    (agent / '.pytest_report.json').write_text('stale sentinel')
    (agent / 'pytest.ini').write_text('[pytest]\naddopts = --invalid-agent-option\n')
    monkeypatch.setenv('PYTEST_ADDOPTS', '--invalid-host-option')
    monkeypatch.setenv('PYTEST_PLUGINS', 'forbidden_host_plugin')
    monkeypatch.setenv('DEEPEVAL_TELEMETRY_OPT_OUT', '0')
    monkeypatch.setenv('DEEPEVAL_TELEMETRY_ENABLED', '1')
    actual_popen = subprocess.Popen
    guarded = []
    def popen(args, **kwargs):
        assert args[:3] == [sys.executable, '-m', 'pytest']
        attempts = tmp_path / 'blocked.jsonl'
        guarded.append(attempts)
        return actual_popen([args[0], '-c', GUARD, str(attempts), *args[3:]], **kwargs)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    report = invoke(fixture)
    entry = report['results'][0]
    assert report['status'] == entry['status'] == expected_status, entry
    assert entry['passed'] is passed and entry['returncode'] == rc, entry
    assert entry['pytest_report']['exitcode'] == rc
    assert all(not p.exists() for p in guarded)
    assert (agent / '.pytest_report.json').read_text() == 'stale sentinel'
    assert all('preexisting sentinel' in (agent / name).read_text()
               for name in ('test_deepeval.py', 'test_trace.py', 'conftest.py'))
    assert not list((tmp_path / 'temp').glob('spec2agent-pytest-*'))


def test_batch_continues_after_missing_agent(fixture, monkeypatch):
    tmp_path, agent, _ = fixture
    missing = agent.with_name('aaa_missing')
    missing.mkdir()
    fake(monkeypatch, payload())
    report = invoke(fixture, agent_filter=None)
    assert report['status'] == 'error' and report['exit_code'] == 1
    assert [e['agent_folder'] for e in report['results']] == ['aaa_missing', 'agent']
    assert [e['status'] for e in report['results']] == ['error', 'ok']


def test_stale_agent_report_is_never_consumed(fixture, monkeypatch):
    _, agent, _ = fixture
    sentinel = agent / '.pytest_report.json'
    sentinel.write_text(json.dumps(payload()))
    fake(monkeypatch, None)
    entry = invoke(fixture)['results'][0]
    assert entry['reason_code'] == 'pytest_report_invalid' and entry['pytest_report'] is None
    assert json.loads(sentinel.read_text()) == payload()


def test_report_symlink_cannot_reuse_stale_file(fixture, monkeypatch):
    tmp_path, _, _ = fixture
    stale = tmp_path / 'stale.json'
    stale.write_text(json.dumps(payload()))
    def inspect(args, kwargs):
        path = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--json-report-file=')))
        path.symlink_to(stale)
    fake(monkeypatch, None, inspect=inspect)
    assert invoke(fixture)['results'][0]['reason_code'] == 'pytest_report_invalid'


def test_spawn_failure_reported(fixture, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('synthetic spawn failure')
    monkeypatch.setattr(runner.subprocess, 'Popen', fail)
    entry = invoke(fixture)['results'][0]
    assert entry['status'] == 'error' and entry['reason_code'] == 'pytest_spawn'
    assert entry['returncode'] is None and 'spawn failure' in entry['error']


@pytest.mark.parametrize('change', ['summary', 'missing_phase', 'invalid_phase', 'impossible_call', 'empty'])
def test_structural_json_contradictions(fixture, monkeypatch, change):
    report = payload()
    if change == 'summary':
        report['summary']['passed'] = 99
    elif change == 'missing_phase':
        del report['tests'][0]['setup']
    elif change == 'invalid_phase':
        report['tests'][0]['call'] = []
    elif change == 'impossible_call':
        report['tests'][0]['setup']['outcome'] = 'skipped'
    else:
        report = {'exitcode': 0, 'summary': {'total': 0, 'collected': 0}, 'tests': []}
    fake(monkeypatch, report)
    entry = invoke(fixture)['results'][0]
    assert entry['reason_code'] == 'pytest_report_invalid' and entry['passed'] is None


CHILD = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(90)'
WORKER = r'''
import os, pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', CHILD])
pathlib.Path(MARKER).write_text(str(child.pid))
print('worker stdout', flush=True)
print('worker stderr', file=sys.stderr, flush=True)
while True:
    time.sleep(.05)
'''


def worker_source(marker):
    return f'CHILD={CHILD!r}\nMARKER={str(marker)!r}\n' + WORKER


def assert_not_running(pid):
    # A killed grandchild may await init reaping; it must not be executing.
    stat = Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().split(') ', 1)[1].startswith('Z ')


def test_timeout_kills_owned_descendants_and_saves_streams(fixture, monkeypatch):
    tmp_path, _, _ = fixture
    marker = tmp_path / 'child.pid'
    actual_popen = subprocess.Popen
    monkeypatch.setenv('PYTEST_TIMEOUT_SECONDS', '.3')
    def popen(args, **kwargs):
        return actual_popen([sys.executable, '-c', worker_source(marker)], **kwargs)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    started = time.monotonic()
    report = invoke(fixture)
    entry = report['results'][0]
    assert time.monotonic() - started < 5
    assert entry['reason_code'] == 'pytest_timeout' and entry['returncode'] == -signal.SIGKILL
    assert entry['status'] == 'error' and report['exit_code'] == 1
    assert 'worker stdout' in entry['stdout'] and 'worker stderr' in entry['stderr']
    assert_not_running(int(marker.read_text()))
    assert not list((tmp_path / 'temp').glob('spec2agent-pytest-*'))


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_parent_signal_saves_report_and_cleans_descendants(fixture, signum):
    tmp_path, _, _ = fixture
    marker = tmp_path / 'child.pid'
    driver = tmp_path / 'driver.py'
    driver.write_text(f'''
import importlib.util, pathlib, subprocess, sys
spec=importlib.util.spec_from_file_location('runner', {str(ROOT / 'pipeline/utils/pytest_runner.py')!r})
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
actual_popen=subprocess.Popen
def popen(args, **kwargs):
    return actual_popen([sys.executable, '-c', {worker_source(marker)!r}], **kwargs)
r.subprocess.Popen=popen
report=r.run_pytest_for_agents(llm='model',agent_filter='agent',test_filename='test_deepeval.py',
 generated_dir=pathlib.Path({str(tmp_path / 'generated')!r}),bench_dir=pathlib.Path({str(tmp_path / 'bench')!r}),
 experiments_results=pathlib.Path({str(tmp_path / 'results')!r}),stage_label='evaluation',
 stage_prefix='07',report_name_stem='test_deepeval',experiment='exp')
raise SystemExit(report['exit_code'])
''')
    process = subprocess.Popen([sys.executable, str(driver)], env=dict(os.environ),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=6)
        assert process.returncode == 128 + signum, (stdout, stderr)
        report = json.loads(next((tmp_path / 'results').rglob('07_*.json')).read_text())
        entry = report['results'][0]
        assert entry['status'] == 'error' and entry['reason_code'] == 'interrupted'
        assert entry['returncode'] == -signal.SIGKILL
        assert_not_running(int(marker.read_text()))
        assert not list((tmp_path / 'temp').glob('spec2agent-pytest-*'))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize('step', ['07_behavioral_check.py', '08_trace_check.py'])
def test_wrappers_save_missing_target_before_error_exit(fixture, step):
    tmp_path, _, _ = fixture
    env = dict(os.environ, GENERATED_DIR=str(tmp_path / 'generated'),
               BENCH_DIR=str(tmp_path / 'bench'), EXPERIMENTS_RESULTS=str(tmp_path / 'results'))
    result = subprocess.run([sys.executable, str(ROOT / 'pipeline' / step), '--llm', 'model',
                             '--agent', 'unknown', '--experiment', 'exp'], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    report = json.loads(next((tmp_path / 'results').rglob(step[:2] + '_*.json')).read_text())
    assert report['status'] == 'error' and report['results'][0]['reason_code'] == 'missing_agent'


@pytest.mark.parametrize('step', ['07_behavioral_check.py', '08_trace_check.py'])
@pytest.mark.parametrize('code', [0, 1, 143])
def test_wrapper_propagates_aggregate_not_metric_rc(fixture, monkeypatch, step, code):
    utils = types.ModuleType('utils')
    helper = types.ModuleType('utils.pytest_runner')
    helper.run_pytest_for_agents = lambda **kwargs: {'exit_code': code, 'results': [{'returncode': 1}]}
    monkeypatch.setitem(sys.modules, 'utils', utils)
    monkeypatch.setitem(sys.modules, 'utils.pytest_runner', helper)
    monkeypatch.setattr(sys, 'argv', [step, '--llm', 'model', '--agent', 'agent'])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / 'pipeline' / step), run_name='__main__')
    assert result.value.code == code
