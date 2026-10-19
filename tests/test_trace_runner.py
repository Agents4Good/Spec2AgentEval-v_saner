"""Managed 08 trace fixtures; no providers, real bench, or network."""
import importlib.util
import builtins
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trace_runner_base', ROOT / 'tests/test_pytest_runner.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
runner = base.runner
fixture = base.fixture
NODE = 'test_synthetic.py::test_one'
EXEC = 'a' * 32


@pytest.fixture
def managed(fixture):
    tmp, agent, bench = fixture
    (bench / 'test_deepeval.py').rename(bench / 'test_trace.py')
    return tmp, agent, bench


def invoke(managed, trace_enabled=True, **kwargs):
    tmp, _, _ = managed
    return runner.run_pytest_for_agents(llm=kwargs.pop('llm', 'model'),
        agent_filter=kwargs.pop('agent_filter', 'agent'), test_filename='test_trace.py',
        generated_dir=tmp / 'generated', bench_dir=tmp / 'bench',
        experiments_results=tmp / 'results', stage_label='trace_evaluation',
        stage_prefix='08', report_name_stem='trace_eval', experiment=kwargs.pop('experiment', 'exp'),
        trace_enabled=trace_enabled, **kwargs)


def capture(**changes):
    value = dict(schema_version=1, run_id='1' * 32, status='ok', query='synthetic',
                 error=None, runs_collected=0, trajectory=[], tool_calls=[], test_nodeid=NODE)
    value.update(changes)
    return value


def fake(monkeypatch, captures=(), report=None, rc=0, inspect=None):
    def write(args, kwargs):
        if inspect:
            inspect(args, kwargs)
        directory = kwargs['env'].get('TRACE_DIR')
        if directory:
            for name, data in captures:
                path = Path(directory) / name
                path.write_text(data if isinstance(data, str) else json.dumps(data))
    base.fake(monkeypatch, report if report is not None else base.payload(), rc, write)


def test_environment_07_unchanged_and_08_controlled(managed, monkeypatch):
    tmp, agent, _ = managed
    seen = []
    monkeypatch.setenv('PYTHONPATH', 'untrusted-host-path')
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'true')
    def inspect(args, kwargs):
        env = kwargs['env']
        assert kwargs['cwd'] == agent
        assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == env['DEEPEVAL_TELEMETRY_OPT_OUT'] == '1'
        assert all(p in args for p in ('pytest_jsonreport.plugin', 'pytest_asyncio.plugin', 'deepeval.plugins.plugin'))
        seen.append(env)
    fake(monkeypatch, [('named.json', capture())], inspect=inspect)
    plain = invoke(managed, trace_enabled=False)['results'][0]
    assert 'trace_artifacts' not in plain
    assert 'TRACE_DIR' not in seen[0] and 'SPEC2AGENT_TRACE_MANAGED' not in seen[0]
    assert seen[0]['PYTHONPATH'] == 'untrusted-host-path' and seen[0]['LANGSMITH_TRACING'] == 'true'
    monkeypatch.setenv('TRACE_DIR', str(tmp / 'untrusted-host-traces'))
    monkeypatch.setenv('SPEC2AGENT_TRACE_MANAGED', '0')
    traced = invoke(managed)['results'][0]
    assert traced['status'] == 'ok'
    env = seen[1]
    assert env['PYTHONPATH'] == str((ROOT / 'pipeline/utils').absolute())
    assert env['LANGSMITH_TRACING'] == env['LANGCHAIN_TRACING_V2'] == 'false'
    assert env['SPEC2AGENT_TRACE_MANAGED'] == '1'
    assert Path(env['TRACE_DIR']).name == 'traces' and not Path(env['TRACE_DIR']).exists()
    assert not (tmp / 'untrusted-host-traces').exists()


@pytest.mark.parametrize('outcome,rc,capture_status,passed', [
    ('passed', 0, 'ok', True), ('failed', 1, 'ok', False), ('passed', 0, 'error', True)])
def test_published_capture_does_not_define_verdict(managed, monkeypatch, outcome, rc, capture_status, passed):
    tmp, _, _ = managed
    fake(monkeypatch, [('explicit-name.json', capture(status=capture_status))], base.payload(outcome, exitcode=rc), rc)
    report = invoke(managed)
    entry = report['results'][0]
    assert entry['status'] == 'ok' and entry['passed'] is passed
    artifact = entry['trace_artifacts'][0]
    assert artifact['capture_status'] == capture_status and artifact['test_nodeid'] == NODE
    relative = Path(artifact['path'])
    assert not relative.is_absolute() and relative.parts[:3] == ('08_traces', 'model', 'agent')
    assert len(relative.parts[3]) == 32 and relative.name == 'explicit-name.json'
    published = tmp / 'results/exp/run_1' / relative
    assert json.loads(published.read_text())['runs_collected'] == 0
    assert published.stat().st_mode & 0o777 == 0o600


def test_allskip_requires_no_capture(managed, monkeypatch):
    fake(monkeypatch, report=base.payload('skipped', phase='setup'))
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'ok' and entry['passed'] is None
    assert entry['reason_code'] == 'no_evaluated_tests' and entry['trace_artifacts'] == []


@pytest.mark.parametrize('files', [
    [], [('bad.json', '{broken')], [('bad.json', [])],
    [('bad.json', capture(schema_version=2))], [('bad.json', capture(run_id='../escape'))],
    [('bad.json', capture(test_nodeid=None))], [('bad.json', capture(test_nodeid=NODE + ' (call)'))],
    [('bad.json', capture(test_nodeid='foreign_test::test_other'))],
    [('bad.json', capture(trajectory={}))], [('bad.json', capture(tool_calls={}))],
    [('bad.json', capture(runs_collected=-1))], [('bad.json', capture(status='passed'))],
    [('bad.json', capture(schema_version=True))], [('bad.json', capture(runs_collected=True))],
    [('bad.json', capture(query=float('nan')))], [('.hidden.json', capture())],
    [('nonascii-é.json', capture())], [('bad.JSON', capture())],
    [('one.json', capture()), ('two.json', capture())], [('not-json.txt', capture())],
])
def test_invalid_or_missing_trace_preserves_pytest_records(managed, monkeypatch, files):
    fake(monkeypatch, files)
    report = invoke(managed)
    entry = report['results'][0]
    assert entry['status'] == report['status'] == 'error' and entry['passed'] is None
    assert entry['reason_code'] == 'trace_artifacts_invalid'
    assert entry['returncode'] == 0 and entry['pytest_report'] == base.payload()
    assert entry['tests'][0]['call']['outcome'] == 'passed'
    assert entry['trace_artifacts'] == []
    assert not list((managed[0] / 'results').rglob('08_traces/**/*.json'))


def test_multi_capture_and_required_coverage(managed, monkeypatch):
    report = base.payload()
    second = dict(report['tests'][0], nodeid='test_synthetic.py::test_two')
    report['tests'].append(second)
    report['summary'].update(total=2, collected=2, passed=2)
    files = [('one.json', capture()), ('two.json', capture(run_id='2' * 32)),
             ('three.json', capture(run_id='3' * 32, test_nodeid=second['nodeid']))]
    fake(monkeypatch, files, report)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'ok' and len(entry['trace_artifacts']) == 3
    fake(monkeypatch, files[:2], report)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and len(entry['tests']) == 2


@pytest.mark.parametrize('attack', ['file', 'directory', 'destination', 'nested'])
def test_symlink_and_nonregular_rejection(managed, monkeypatch, attack):
    tmp, _, _ = managed
    outside = tmp / 'outside'
    outside.mkdir()
    sentinel = outside / 'sentinel.json'
    sentinel.write_text(json.dumps(capture()))
    def inspect(args, kwargs):
        directory = Path(kwargs['env']['TRACE_DIR'])
        if attack == 'file':
            (directory / 'evil.json').symlink_to(sentinel)
        elif attack == 'directory':
            directory.rmdir()
            directory.symlink_to(outside, target_is_directory=True)
        elif attack == 'nested':
            (directory / 'nested').mkdir()
        else:
            target = tmp / 'results/exp/run_1/08_traces'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(outside, target_is_directory=True)
    fake(monkeypatch, [('valid.json', capture())] if attack == 'destination' else [], inspect=inspect)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['trace_artifacts'] == []
    assert sentinel.read_text() == json.dumps(capture())
    assert set(p.name for p in outside.iterdir()) == {'sentinel.json'}


def test_no_overwrite_and_ignore_paths_inside_json(managed, monkeypatch):
    tmp, _, _ = managed
    monkeypatch.setattr(runner.uuid, 'uuid4', lambda: types.SimpleNamespace(hex=EXEC))
    fake(monkeypatch, [('safe.json', capture(path=str(tmp / 'injected.json')))])
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'ok' and not (tmp / 'injected.json').exists()
    published = tmp / 'results/exp/run_1' / entry['trace_artifacts'][0]['path']
    before = published.read_bytes()
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['trace_artifacts'] == []
    assert published.read_bytes() == before


@pytest.mark.parametrize('reason,rc', [('pytest_timeout', -9), ('interrupted', -15)])
def test_debug_artifacts_never_erase_primary_error(managed, monkeypatch, reason, rc):
    def execute(args, cwd, env, timeout, cancelled):
        target = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--json-report-file=')))
        target.write_text(json.dumps(base.payload()))
        (Path(env['TRACE_DIR']) / 'debug.json').write_text(json.dumps(capture()))
        return rc, 'stdout', 'stderr', reason
    monkeypatch.setattr(runner, '_execute', execute)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['passed'] is None
    assert entry['reason_code'] == reason and entry['returncode'] == rc
    assert len(entry['trace_artifacts']) == 1 and entry['pytest_report'] == base.payload()


def test_phase_error_is_not_replaced_by_missing_trace(managed, monkeypatch):
    report = base.payload('error', phase='teardown', exitcode=1)
    report['tests'][0]['call'] = {'outcome': 'passed'}
    fake(monkeypatch, report=report, rc=1)
    entry = invoke(managed)['results'][0]
    assert entry['reason_code'] == 'pytest_phase_error' and entry['status'] == 'error'
    assert entry['tests'][0]['call']['outcome'] == 'passed'


def test_phase_error_preserves_valid_debug_capture(managed, monkeypatch):
    report = base.payload('error', phase='teardown', exitcode=1)
    report['tests'][0]['call'] = {'outcome': 'passed'}
    fake(monkeypatch, [('debug.json', capture())], report, rc=1)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['reason_code'] == 'pytest_phase_error'
    assert len(entry['trace_artifacts']) == 1 and entry['passed'] is None


def test_pending_pytest_processing_error_is_not_masked(managed, monkeypatch):
    report = base.payload()
    report['tests'][0]['call'] = {'outcome': 'passed', 'crash': 'invalid phase metadata'}
    fake(monkeypatch, report=report)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['reason_code'] == 'pytest_report_invalid'
    assert entry['passed'] is None and entry['pytest_report'] == report
    assert 'trace_error' in entry


def test_wrapper_only_08_enables_trace(managed, monkeypatch):
    received = []
    helper = types.ModuleType('utils.pytest_runner')
    helper.run_pytest_for_agents = lambda **kw: received.append(kw) or {'exit_code': 1}
    monkeypatch.setitem(sys.modules, 'utils', types.ModuleType('utils'))
    monkeypatch.setitem(sys.modules, 'utils.pytest_runner', helper)
    monkeypatch.setattr(sys, 'argv', ['08_trace_check.py', '--llm', 'model', '--agent', 'agent'])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / 'pipeline/08_trace_check.py'), run_name='__main__')
    assert result.value.code == 1 and received[0]['trace_enabled'] is True


@pytest.mark.parametrize('field,value', [('llm', '../outside'), ('agent_filter', 'a/b'), ('experiment', '../outside')])
def test_unsafe_components_rejected_before_filesystem(managed, field, value):
    with pytest.raises(ValueError):
        invoke(managed, **{field: value})


def test_stale_pytest_and_foreign_trace_are_not_consumed(managed, monkeypatch):
    tmp, agent, _ = managed
    (agent / '.pytest_report.json').write_text(json.dumps(base.payload()))
    (agent / 'traces').mkdir()
    (agent / 'traces/stale.json').write_text(json.dumps(capture()))
    def inspect(args, kwargs):
        (Path(kwargs['env']['TRACE_DIR']) / 'fresh.json').write_text(json.dumps(capture()))
    base.fake(monkeypatch, None, inspect=inspect)
    entry = invoke(managed)['results'][0]
    assert entry['reason_code'] == 'pytest_report_invalid' and entry['trace_artifacts'] == []
    assert entry['pytest_report'] is None and (agent / 'traces/stale.json').is_file()


def test_results_ancestor_symlink_rejected_before_runtime(managed):
    tmp, _, _ = managed
    outside = tmp / 'outside'
    outside.mkdir()
    (tmp / 'results').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        invoke(managed)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize('kind', ['hardlink', 'fifo'])
def test_nonprivate_or_nonregular_trace_rejected(managed, monkeypatch, kind):
    tmp, _, _ = managed
    outside = tmp / 'synthetic.json'
    outside.write_text(json.dumps(capture()))
    def inspect(args, kwargs):
        target = Path(kwargs['env']['TRACE_DIR']) / 'trace.json'
        os.link(outside, target) if kind == 'hardlink' else os.mkfifo(target)
    fake(monkeypatch, inspect=inspect)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['trace_artifacts'] == []
    assert outside.read_text() == json.dumps(capture())


def test_safe_diagnostics_and_publication_rollback(managed, monkeypatch):
    tmp, _, _ = managed
    secret = 'synthetic-secret-not-for-error-message'
    files = [('a.json', capture()), ('b.json', capture(run_id='2' * 32))]
    fake(monkeypatch, files)
    actual_open = os.open
    def fail_second(path, *args, **kwargs):
        if Path(path).name == 'b.json' and '08_traces' in Path(path).parts:
            raise OSError(secret)
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(runner.os, 'open', fail_second)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['trace_artifacts'] == []
    assert secret not in entry['error']
    assert not list((tmp / 'results/exp/run_1/08_traces').rglob('*.json'))


def test_execution_uuid_is_validated(managed, monkeypatch):
    monkeypatch.setattr(runner.uuid, 'uuid4', lambda: types.SimpleNamespace(hex='../outside'))
    fake(monkeypatch, [('valid.json', capture())])
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['trace_artifacts'] == []


def tool_call(**changes):
    value = dict(name='synthetic_tool', input={'arbitrary': [None, 42]},
                 start=None, end=None, output={'arbitrary': ['result', False]})
    value.update(changes)
    return value


@pytest.mark.parametrize('corruption', [
    {'error': 42}, {'error': []}, {'error': {}},
    {'trajectory': [42]}, {'trajectory': [{}]},
    {'tool_calls': [42]}, {'tool_calls': [{}]},
    {'tool_calls': [tool_call(name=42)]},
    {'tool_calls': [{k: v for k, v in tool_call().items() if k != 'input'}]},
    {'tool_calls': [{k: v for k, v in tool_call().items() if k != 'start'}]},
    {'tool_calls': [{k: v for k, v in tool_call().items() if k != 'end'}]},
    {'tool_calls': [tool_call(start=42)]}, {'tool_calls': [tool_call(end={})]},
    {'tool_calls': [dict(name='tool', input=None, start=None, end=None, error=42)]},
    {'tool_calls': [tool_call(error='both outcomes')]},
    {'tool_calls': [{k: v for k, v in tool_call().items() if k != 'output'}]},
])
def test_nested_trace_schema_rejects_entire_batch(managed, monkeypatch, corruption):
    fake(monkeypatch, [('a-valid.json', capture()),
                       ('b-corrupt.json', capture(run_id='2' * 32, **corruption))])
    report = invoke(managed)
    entry = report['results'][0]
    assert report['status'] == entry['status'] == 'error' and entry['passed'] is None
    assert entry['reason_code'] == 'trace_artifacts_invalid'
    assert entry['returncode'] == 0 and entry['pytest_report'] == base.payload()
    assert entry['tests'][0]['call']['outcome'] == 'passed'
    assert entry['trace_artifacts'] == []
    assert not list((managed[0] / 'results').rglob('08_traces/**/*.json'))


@pytest.mark.parametrize('call,trace_error', [
    (tool_call(start='synthetic-start', end='synthetic-end'), None),
    (dict(name='tool', input=[None, {'any': 42}], start=None, end=None,
          error='expected synthetic tool failure'), 'expected agent failure'),
])
def test_valid_nested_tool_output_or_error_and_nullable_timestamps(managed, monkeypatch, call, trace_error):
    trace = capture(status='error' if trace_error is not None else 'ok', error=trace_error,
                    trajectory=['compute', 'tool'], tool_calls=[call])
    fake(monkeypatch, [('valid.json', trace)])
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'ok' and entry['passed'] is True
    artifact = entry['trace_artifacts'][0]
    path = managed[0] / 'results/exp/run_1' / artifact['path']
    assert json.loads(path.read_text()) == trace


def test_nested_corruption_preserves_primary_phase_error(managed, monkeypatch):
    report = base.payload('error', phase='teardown', exitcode=1)
    report['tests'][0]['call'] = {'outcome': 'passed'}
    fake(monkeypatch, [('bad.json', capture(tool_calls=[{}]))], report, rc=1)
    entry = invoke(managed)['results'][0]
    assert entry['status'] == 'error' and entry['passed'] is None
    assert entry['reason_code'] == 'pytest_phase_error' and 'trace_error' in entry
    assert entry['pytest_report'] == report and entry['returncode'] == 1
    assert entry['tests'][0]['call']['outcome'] == 'passed' and entry['trace_artifacts'] == []


GRAPH_AGENT = '''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
@tool
def double(value: int) -> int:
    """Synthetic doubling, no provider."""
    return value * 2
class State(TypedDict):
    value: int
def compute(state):
    return {"value": double.invoke({"value": state["value"]})}
builder = StateGraph(State)
builder.add_node("compute", compute)
builder.add_edge(START, "compute")
builder.add_edge("compute", END)
graph = builder.compile()
def agent(value=21):
    return graph.invoke({"value": value})
'''


@pytest.mark.parametrize('agent_source,body,status,passed,count', [
    ('def agent(): return 42\n',
     'def test_capture():\n output, trace = traced(agent)()\n assert output == 42 and trace["runs_collected"] == 0\n', 'ok', True, 1),
    ('def agent(): return 42\n',
     'def test_capture():\n output, trace = traced(agent)()\n assert output == 99\n', 'ok', False, 1),
    ('def agent(): raise RuntimeError("expected synthetic failure")\n',
     'def test_capture():\n output, trace = traced(agent)()\n assert output is None and trace["status"] == "error"\n', 'ok', True, 1),
    ('def agent(value): raise RuntimeError("expected synthetic failure")\n',
     'def test_capture():\n with pytest.raises(RuntimeError): run_traced("query")\n', 'ok', True, 1),
    ('def agent(): return 42\n',
     '@pytest.mark.skip(reason="synthetic skip")\ndef test_capture(): pass\n', 'ok', None, 0),
    ('def agent(): return 42\n', 'def test_capture(): assert agent() == 42\n', 'error', None, 0),
    ('def agent(): return 42\n',
     'def test_capture():\n wrapped = traced(agent)\n assert wrapped()[0] == 42\n assert wrapped()[0] == 42\n', 'ok', True, 2),
    (GRAPH_AGENT,
     'def test_capture():\n output, trace = traced(agent)()\n assert output == {"value": 42}\n assert "compute" in trace["trajectory"]\n assert trace["tool_calls"][0]["name"] == "double"\n', 'ok', True, 1),
])
def test_real_capture_pytest_guarded(managed, monkeypatch, agent_source, body, status, passed, count):
    # Normal execution discovers Lane A's output; absence is explicit, never fake evidence.
    if not (ROOT / 'pipeline/utils/trace_capture.py').is_file():
        pytest.skip('Lane A capture helper not available; real integration not proved')
    tmp, agent, bench = managed
    (agent / 'agent.py').write_text(agent_source)
    (agent / 'test_trace.py').write_text('raise AssertionError("preexisting sentinel")\n')
    (agent / '.pytest_report.json').write_text('stale sentinel')
    (agent / 'traces').mkdir()
    (agent / 'traces/stale.json').write_text('foreign stale sentinel')
    (bench / 'test_trace.py').write_text('import pytest\nfrom trace_capture import traced, run_traced\n'
                                        'from agent import agent\n' + body)
    actual_popen, actual_import = subprocess.Popen, builtins.__import__
    attempts = tmp / 'blocked.jsonl'
    def popen(args, **kwargs):
        assert args[:3] == [sys.executable, '-m', 'pytest']
        assert kwargs['env']['PYTHONPATH'] == str(ROOT / 'pipeline/utils')
        return actual_popen([args[0], '-c', base.GUARD, str(attempts), *args[3:]], **kwargs)
    def parent_import(name, *args, **kwargs):
        assert name != 'trace_capture' and not name.startswith(('langchain', 'langgraph'))
        return actual_import(name, *args, **kwargs)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(builtins, '__import__', parent_import)
    report = invoke(managed)
    entry = report['results'][0]
    assert entry['status'] == status and entry['passed'] is passed, entry
    assert len(entry['trace_artifacts']) == count
    assert not attempts.exists(), attempts.read_text() if attempts.exists() else ''
    nodeids = {test['nodeid'] for test in entry['pytest_report']['tests']}
    for artifact in entry['trace_artifacts']:
        exported = json.loads((tmp / 'results/exp/run_1' / artifact['path']).read_text())
        assert exported['test_nodeid'] == artifact['test_nodeid'] in nodeids
        assert exported['run_id'] == artifact['run_id']
    assert (agent / '.pytest_report.json').read_text() == 'stale sentinel'
    assert (agent / 'traces/stale.json').read_text() == 'foreign stale sentinel'
    assert 'preexisting sentinel' in (agent / 'test_trace.py').read_text()
    assert not list((tmp / 'temp').glob('spec2agent-pytest-*'))
