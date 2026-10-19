"""Installed callback probe only: no product capture module or provider inputs."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypedDict
from langchain_core.tools import tool
from langchain_core.tracers.context import collect_runs as public_collect
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

@tool
def synthetic_increment(value: int) -> int:
    """Increment a synthetic integer without external effects."""
    if value < 0:
        raise ValueError('synthetic tool failure')
    return value + 1

def prepare(state):
    return {'value': state['value'] + 1}

def calculate(state):
    return {'value': synthetic_increment.invoke({'value': state['value']})}

builder = StateGraph(State)
builder.add_node('prepare', prepare)
builder.add_node('calculate', calculate)
builder.add_edge(START, 'prepare')
builder.add_edge('prepare', 'calculate')
builder.add_edge('calculate', END)
graph = builder.compile()

def flatten(runs):
    return [item for run in runs for item in [run, *flatten(run.child_runs)]]

def snapshot(collector):
    def record(run):
        return {'name': run.name, 'run_type': run.run_type,
                'metadata': run.metadata, 'inputs': run.inputs, 'outputs': run.outputs,
                'start': run.start_time.isoformat(), 'end': run.end_time.isoformat(),
                'error': run.error, 'children': [record(child) for child in run.child_runs]}
    return [record(run) for run in collector.traced_runs]

def capture(collect, value):
    with collect() as collector:
        if value > 0:
            assert graph.invoke({'value': value}) == {'value': 3}
        else:
            try:
                graph.invoke({'value': value})
            except ValueError as exc:
                assert str(exc) == 'synthetic tool failure'
            else:
                raise AssertionError('expected synthetic failure')
    return collector

public = capture(public_collect, 1)
public_nodes = {r.metadata.get('langgraph_node') for r in flatten(public.traced_runs)
                if r.run_type == 'chain'} - {None}
selected, collect = 'public', public_collect
result = {'public': snapshot(public), 'public_nodes': sorted(public_nodes)}
if not {'prepare', 'calculate'} <= public_nodes:
    from langchain_core.tracers.context import register_configure_hook
    from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
    variable = ContextVar('synthetic_inheritable_collector', default=None)
    register_configure_hook(variable, inheritable=True)
    @contextmanager
    def collect():
        collector = RunCollectorCallbackHandler()
        token = variable.set(collector)
        try:
            yield collector
        finally:
            variable.reset(token)
    selected = 'inheritable_hook'
    successful = capture(collect, 1)
else:
    successful = public
runs = flatten(successful.traced_runs)
assert successful.traced_runs[0].child_runs
nodes = [r for r in runs if r.run_type == 'chain' and r.metadata.get('langgraph_node')]
assert {'prepare', 'calculate'} <= {r.metadata['langgraph_node'] for r in nodes}
assert all(isinstance(r.metadata['langgraph_step'], int) for r in nodes)
tools = [r for r in runs if r.run_type == 'tool']
assert len(tools) == 1 and tools[0].name == 'synthetic_increment'
assert tools[0].inputs == {'value': 2} and tools[0].outputs == {'output': 3}
assert all(r.start_time <= r.end_time for r in runs)
failed = capture(collect, -3)
assert failed.traced_runs[0].error
assert any(r.run_type == 'tool' and r.error for r in flatten(failed.traced_runs))
result.update(selected=selected, successful=snapshot(successful), failed=snapshot(failed))
print('CALLBACK_PROBE=' + json.dumps(result))
'''


PRODUCT_PROBE = r'''
import importlib, os, pathlib, stat, inspect, asyncio
from langchain_core.messages import ToolMessage
sys.path.insert(0, sys.argv[2])
capture_module = importlib.import_module('trace_capture')
traced, run_traced = capture_module.traced, capture_module.run_traced
base = pathlib.Path.cwd()
os.environ['TRACE_DIR'] = str(base / 'exports')
os.environ['PYTEST_CURRENT_TEST'] = 'test_sample.py::test_value[x (call)] (call)'
out, trace = traced(graph.invoke)({'value': 1})
assert out == {'value': 3} and trace['status'] == 'ok'
assert trace['schema_version'] == 1 and len(trace['run_id']) == 32
assert trace['test_nodeid'] == 'test_sample.py::test_value[x (call)]'
assert trace['trajectory'] == ['prepare', 'calculate', 'synthetic_increment']
assert trace['tool_calls'][0]['input'] == {'value': 2}
assert trace['tool_calls'][0]['output'] == 3
assert capture_module.load_run(base / 'exports' / (trace['run_id'] + '.json')) == (
    trace['trajectory'], trace['tool_calls'])
assert stat.S_IMODE((base / 'exports' / (trace['run_id'] + '.json')).stat().st_mode) == 0o600
out, failed = traced(graph.invoke)({'value': -3})
assert out is None and failed['status'] == 'error'
assert failed['error'] == 'Agent execution failed (ValueError)'
assert 'synthetic tool failure' not in json.dumps(failed)
assert failed['tool_calls'][0]['error'] == 'Tool execution failed'
assert traced(lambda a, b=0: a+b)(2, b=3)[0] == 5
assert traced(lambda: 7)()[1]['runs_collected'] == 0
os.environ['FAKE_API_KEY'] = 'FAKE_SELECTED_KEY'
class Unsafe:
    def __repr__(self):
        raise AssertionError('repr must never run')
    def __str__(self):
        raise AssertionError('str must never run')
cycle = []; cycle.append(cycle)
deep = []; pointer = deep
for _ in range(30):
    child = []; pointer.append(child); pointer = child
query = {'password': 'hidden', 'normal': 'prefix FAKE_SELECTED_KEY suffix',
         'unknown': Unsafe(), 'cycle': cycle, 'deep': deep}
safe = traced(lambda x: x)(query)[1]
encoded = json.dumps(safe)
assert 'FAKE_SELECTED_KEY' not in encoded and 'hidden' not in encoded
assert '<Unsafe>' in encoded and '<cycle>' in encoded and '<max_depth>' in encoded
@tool
def synthetic_echo(value: str) -> str:
    """Return a synthetic string."""
    return value
redacted = traced(synthetic_echo.invoke)({'value': 'xFAKE_SELECTED_KEYy'})[1]
assert 'FAKE_SELECTED_KEY' not in json.dumps(redacted)
@tool(response_format='content_and_artifact')
def synthetic_artifact(value: str) -> tuple:
    """Return known synthetic artifact fields."""
    return 'ignored content', {'password': 'hidden', 'value': value}
artifact = traced(synthetic_artifact.invoke)({
    'name': 'synthetic_artifact', 'args': {'value': 'xFAKE_SELECTED_KEYy'},
    'id': 'synthetic-tool-id', 'type': 'tool_call'})[1]
assert artifact['tool_calls'][0]['output'] == {'password': '***', 'value': 'x***y'}
assert capture_module._tool_output(ToolMessage(content='{"value": 3}', tool_call_id='id')) == {'value': 3}
async def async_agent():
    return 1
async_body_executed = []
async def async_generator_agent():
    async_body_executed.append(True)
    yield 42
exports_before_async = set((base / 'exports').iterdir())
for function in (async_generator_agent, lambda: async_generator_agent()):
    try:
        traced(function)()
    except TypeError:
        pass
    else:
        raise AssertionError('async generator was accepted')
assert not async_body_executed
assert set((base / 'exports').iterdir()) == exports_before_async
for function in (async_agent, lambda: async_agent()):
    try:
        traced(function)()
    except TypeError:
        pass
    else:
        raise AssertionError('async was accepted')
original_path = list(sys.path)
sentinel = object(); original_alias = sys.modules.get('agent', sentinel)
for i in (1, 2):
    directory = base / ('agent' + str(i)); directory.mkdir()
    (directory / 'agent.py').write_text('def agent(value=0):\n    return value + ' + str(i) + '\n')
    state, trajectory, tools = run_traced({'value': 10}, agent_dir=directory,
                                          out_dir=base / 'explicit', name='agent' + str(i))
    assert state == 10+i and trajectory == [] and tools == []
assert sys.path == original_path and sys.modules.get('agent', sentinel) is original_alias
bad = base / 'bad'; bad.mkdir()
(bad / 'agent.py').write_text('def agent(*args):\n    raise ValueError("FAKE_SELECTED_KEY")\n')
try:
    run_traced([], agent_dir=bad, out_dir=base / 'explicit', name='failed')
except ValueError:
    payload = json.loads((base / 'explicit/failed.json').read_text())
    assert payload['status'] == 'error' and 'FAKE_SELECTED_KEY' not in json.dumps(payload)
else:
    raise AssertionError('agent error not rethrown')
for name in ('../escape', '.', '..', '/absolute', 'a/b', 'é'):
    try:
        capture_module.trace_path(name, base / 'explicit')
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe name accepted')
before = (base / 'explicit/failed.json').read_bytes()
try:
    run_traced([], agent_dir=bad, out_dir=base / 'explicit', name='failed')
except FileExistsError:
    pass
else:
    raise AssertionError('export error was hidden as agent error')
assert (base / 'explicit/failed.json').read_bytes() == before
collision = base / 'exports' / 'fixed.json'
collision.write_text('sentinel')
original_export = capture_module._export
def fail_export(*args, **kwargs):
    raise RuntimeError('synthetic instrumentation failure')
capture_module._export = fail_export
try:
    traced(lambda: 1)()
except RuntimeError:
    pass
else:
    raise AssertionError('instrumentation failure hidden')
capture_module._export = original_export
assert collision.read_text() == 'sentinel'
original_dump, original_uuid = capture_module.json.dump, capture_module.uuid.uuid4
capture_module.uuid.uuid4 = lambda: type('SyntheticUUID', (), {'hex': 'a' * 32})()
def fail_dump(*args, **kwargs):
    raise RuntimeError('synthetic serialization failure')
capture_module.json.dump = fail_dump
try:
    traced(lambda: 1)()
except RuntimeError:
    pass
else:
    raise AssertionError('serialization error hidden')
capture_module.json.dump = original_dump
assert not (base / 'exports' / ('a' * 32 + '.json')).exists()
traced(lambda: 1)()
saved = (base / 'exports' / ('a' * 32 + '.json')).read_bytes()
try:
    traced(lambda: 2)()
except FileExistsError:
    pass
else:
    raise AssertionError('automatic collision overwritten')
assert (base / 'exports' / ('a' * 32 + '.json')).read_bytes() == saved
capture_module.uuid.uuid4 = original_uuid
link = base / 'linked'; link.symlink_to(base / 'explicit', target_is_directory=True)
for path in (link, base / 'x/../explicit'):
    try:
        run_traced(1, agent_dir=base / 'agent1', out_dir=path)
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe directory accepted')
agent_link = base / 'agent_link'; agent_link.mkdir()
(agent_link / 'agent.py').symlink_to(base / 'absent-agent.py')
try:
    capture_module.load_agent(agent_link)
except ValueError:
    pass
else:
    raise AssertionError('dangling agent symlink accepted')
trace_link = base / 'explicit/link.json'
trace_link.symlink_to(base / 'missing.json')
try:
    capture_module.load_run(trace_link)
except ValueError:
    pass
else:
    raise AssertionError('dangling JSON symlink accepted')
os.environ['SPEC2AGENT_TRACE_MANAGED'] = '1'
managed = base / 'managed'; os.environ['TRACE_DIR'] = str(managed)
run_traced((1,), agent_dir=base / 'agent1')
assert len(list(managed.glob('*.json'))) == 1
stale_directory = base / 'traces'; stale_directory.mkdir()
stale_path = stale_directory / 'named.json'
stale_path.write_text(json.dumps({'trajectory': ['STALE'], 'tool_calls': []}))
stale_bytes = stale_path.read_bytes()
run_traced((1,), agent_dir=base / 'agent1', name='named')
assert capture_module.trace_path('named') == managed / 'named.json'
assert capture_module.load_run(capture_module.trace_path('named')) == ([], [])
assert stale_path.read_bytes() == stale_bytes
try:
    capture_module.trace_path('named', stale_directory)
except ValueError:
    pass
else:
    raise AssertionError('managed lookup bypass accepted')
for outside in (base / 'outside', None, link):
    try:
        run_traced(1, agent_dir=base / 'agent1', out_dir=outside)
    except ValueError:
        pass
    else:
        raise AssertionError('managed export bypass')
del os.environ['SPEC2AGENT_TRACE_MANAGED']; del os.environ['TRACE_DIR']
run_traced(1, agent_dir=base / 'agent1')
assert (base / 'traces/trace.json').is_file()
assert run_traced([2], agent_dir=base / 'agent1', out_dir=None)[0] == 3
assert traced(lambda: 1)()[1]['test_nodeid'] == 'test_sample.py::test_value[x (call)]'
del os.environ['PYTEST_CURRENT_TEST']
assert traced(lambda: 1)()[1]['test_nodeid'] is None
print('PRODUCT_CAPTURE_CONTRACT=passed')
'''


@pytest.mark.parametrize('product', [False, True])
def test_installed_sync_graph_callbacks_offline(tmp_path, product):
    for name in ('home', 'temp', 'bin', 'cache'):
        (tmp_path / name).mkdir()
    env = {'HOME': str(tmp_path / 'home'), 'TMPDIR': str(tmp_path / 'temp'),
           'XDG_CACHE_HOME': str(tmp_path / 'cache'), 'PATH': str(tmp_path / 'bin'),
           'LANGSMITH_TRACING': 'false', 'LANGCHAIN_TRACING_V2': 'false',
           'SPEC2AGENT_SKIP_DOTENV': '1', 'PYTHON_DOTENV_DISABLED': '1',
           'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'DEEPEVAL_TELEMETRY_OPT_OUT': '1'}
    python = str(ROOT / '.venv/bin/python')
    metadata = "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ('langchain-core','langgraph')}))"
    installed = subprocess.run([python, '-I', '-c', metadata], env=env, cwd=tmp_path,
                               capture_output=True, text=True, timeout=30)
    assert installed.returncode == 0, installed.stderr
    versions = json.loads(installed.stdout)
    assert versions == {'langchain-core': '1.6.6', 'langgraph': '1.2.12'}
    print('CALLBACK_METADATA=' + json.dumps(versions))
    source = ast.parse((ROOT / 'tests/test_pytest_plugin_contract.py').read_text())
    bootstrap = next(ast.literal_eval(node.value) for node in source.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(t, ast.Name) and t.id == 'BOOTSTRAP' for t in node.targets))
    # Keep all network/DNS/audit guards unchanged; replace only pytest invocation.
    guards = bootstrap.split('\nimport pytest\n', 1)[0]
    assert 'sys.addaudithook(audit)' in guards
    attempts = tmp_path / 'blocked.jsonl'
    script = guards + '\n' + PROBE + ('\n' + PRODUCT_PROBE if product else '')
    result = subprocess.run([python, '-I', '-c', script, str(attempts), str(ROOT / 'pipeline/utils')],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert not attempts.exists(), ('BLOCKED: ' + attempts.read_text() + result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end='')
