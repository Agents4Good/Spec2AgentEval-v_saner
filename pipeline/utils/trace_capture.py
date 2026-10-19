"""Synchronous local callback capture; limited known-secret redaction, not a sandbox."""
import functools
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.messages import ToolMessage
from langchain_core.tracers.context import register_configure_hook
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

_collector = ContextVar('spec2agent_trace_collector', default=None)
register_configure_hook(_collector, inheritable=True)
_DEFAULT = object()
_SENSITIVE = ('TOKEN', 'KEY', 'SECRET', 'PASSWORD')


@contextmanager
def collect_runs():
    collector = RunCollectorCallbackHandler()
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


def _safe(path):
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('Trace path traversal forbidden')
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (*path.parents, path)):
        raise ValueError('Trace symlink forbidden')
    return path


def _destination(out_dir):
    managed = os.getenv('SPEC2AGENT_TRACE_MANAGED') == '1'
    if out_dir is _DEFAULT:
        out_dir = os.environ.get('TRACE_DIR') if managed else 'traces'
    if managed:
        root = os.environ.get('TRACE_DIR')
        if not root or out_dir is None or _safe(out_dir) != _safe(root):
            raise ValueError('Managed trace directory required')
    return None if out_dir is None else _safe(out_dir)


def trace_path(name, out_dir=_DEFAULT):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name):
        raise ValueError('Unsafe trace name')
    return _safe(_destination(out_dir) / (name + '.json'))


def _normalizer():
    secrets = sorted({v for k, v in os.environ.items() if v and any(
        word in k.upper() for word in _SENSITIVE)}, key=len, reverse=True)

    def normalize(value, active=None, depth=0):
        active = set() if active is None else active
        if depth > 16:
            return '<max_depth>'
        if type(value) is str:
            for secret in secrets:
                value = value.replace(secret, '***')
            return value
        if value is None or type(value) in (bool, int):
            return value
        if type(value) is float:
            return value if math.isfinite(value) else '<nonfinite>'
        if type(value) not in (dict, list, tuple):
            return '<' + type(value).__name__ + '>'
        if id(value) in active:
            return '<cycle>'
        active.add(id(value))
        try:
            if type(value) is dict:
                result = {}
                for key, item in value.items():
                    label = normalize(key, active, depth + 1) if type(key) is str else '<key>'
                    result[label] = ('***' if type(key) is str and any(
                        word in key.upper() for word in _SENSITIVE) else normalize(item, active, depth + 1))
                return result
            return [normalize(item, active, depth + 1) for item in value]
        finally:
            active.remove(id(value))
    return normalize


def _metadata(run):
    return getattr(run, 'metadata', None) or (run.extra or {}).get('metadata', {}) or {}


def _walk(runs, parent=None):
    for run in runs:
        metadata = _metadata(run)
        key = (metadata.get('langgraph_node'), metadata.get('langgraph_step'))
        yield run, key, parent
        yield from _walk(run.child_runs, key)


def _iso(value):
    return value.isoformat() if value is not None else None


def _tool_output(value):
    if isinstance(value, ToolMessage):
        if value.artifact is not None:
            return value.artifact
        value = value.content
    if type(value) is str:
        try:
            return json.loads(value)
        except (ValueError, RecursionError):
            pass
    return value


def _nodeid():
    value = os.getenv('PYTEST_CURRENT_TEST')
    if value is not None:
        for suffix in (' (setup)', ' (call)', ' (teardown)'):
            if value.endswith(suffix):
                return value[:-len(suffix)]
    return value


def _capture(entrypoint, args, kwargs):
    if inspect.iscoroutinefunction(entrypoint) or inspect.isasyncgenfunction(entrypoint):
        raise TypeError('Async capture unsupported')
    output, error = None, None
    with collect_runs() as collector:
        try:
            output = entrypoint(*args, **kwargs)
        except Exception as exc:
            error = exc
        if error is None and (inspect.isawaitable(output) or inspect.isasyncgen(output)):
            if inspect.iscoroutine(output):
                output.close()
            raise TypeError('Awaitable capture unsupported')
    normalize = _normalizer()
    events, calls = [], []
    for run, key, parent in _walk(collector.traced_runs):
        if run.run_type == 'chain' and key[0] is not None and key != parent:
            events.append((run.start_time, normalize(key[0])))
        if run.run_type != 'tool':
            continue
        events.append((run.start_time, normalize(run.name)))
        call = {'name': normalize(run.name), 'input': normalize(run.inputs),
                'start': _iso(run.start_time), 'end': _iso(run.end_time)}
        value = (run.outputs or {}).get('output')
        if run.error or isinstance(value, ToolMessage) and value.status == 'error':
            call['error'] = 'Tool execution failed'
        else:
            call['output'] = normalize(_tool_output(value))
        calls.append((run.start_time, call))
    trace = {'schema_version': 1, 'run_id': uuid.uuid4().hex,
             'query': normalize({'args': list(args), 'kwargs': kwargs} if kwargs else list(args)),
             'status': 'error' if error is not None else 'ok',
             'error': f'Agent execution failed ({type(error).__name__})' if error is not None else None,
             'runs_collected': len(collector.traced_runs),
             'trajectory': [name for _, name in sorted(events, key=lambda item: item[0])],
             'tool_calls': [call for _, call in sorted(calls, key=lambda item: item[0])],
             'test_nodeid': _nodeid()}
    return output, trace, error


def _export(trace, out_dir, name=None):
    directory = _destination(out_dir)
    path = trace_path(trace['run_id'] if name is None else name, directory)
    for parent in reversed((directory, *directory.parents)):
        _safe(parent)
        if not parent.exists():
            parent.mkdir(mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(trace, stream, ensure_ascii=False, allow_nan=False, indent=2)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def traced(entrypoint):
    if inspect.iscoroutinefunction(entrypoint) or inspect.isasyncgenfunction(entrypoint):
        raise TypeError('Async capture unsupported')

    @functools.wraps(entrypoint)
    def wrapper(*args, **kwargs):
        output, trace, _ = _capture(entrypoint, args, kwargs)
        if os.getenv('TRACE_DIR') or os.getenv('SPEC2AGENT_TRACE_MANAGED') == '1':
            _export(trace, _DEFAULT if os.getenv('SPEC2AGENT_TRACE_MANAGED') == '1' else os.environ['TRACE_DIR'])
        return output, trace
    return wrapper


def load_agent(agent_dir=None):
    directory = _safe(agent_dir if agent_dir is not None else os.getenv('AGENT_DIR', '.'))
    path = _safe(directory / 'agent.py')
    if not path.is_file():
        raise ValueError('Regular agent.py required')
    name = '_spec2agent_' + hashlib.sha256(os.fsencode(path)).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def run_traced(query, out_dir=_DEFAULT, name=_DEFAULT, agent_dir=None):
    directory = _destination(out_dir)
    if name is _DEFAULT:
        name = None if os.getenv('SPEC2AGENT_TRACE_MANAGED') == '1' else 'trace'
    if directory is not None and name is not None:
        trace_path(name, directory)
    module = load_agent(agent_dir)
    if isinstance(query, dict):
        args, kwargs = (), query
    elif isinstance(query, (tuple, list)):
        args, kwargs = tuple(query), {}
    else:
        args, kwargs = (query,), {}
    output, trace, error = _capture(module.agent, args, kwargs)
    trace['query'] = _normalizer()(query)
    if directory is not None:
        _export(trace, directory, name)
    if error is not None:
        raise error
    return output, trace['trajectory'], trace['tool_calls']


def load_run(path):
    path = _safe(path)
    if not path.is_file():
        raise ValueError('Regular trace JSON required')
    run = json.loads(path.read_text(encoding='utf-8'))
    return run['trajectory'], run['tool_calls']
