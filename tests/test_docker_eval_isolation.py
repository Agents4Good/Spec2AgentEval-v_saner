"""Synthetic fake-Docker evidence; never contacts Docker or provider services."""
import json
import ast
import importlib.util
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'docker/eval_staging.py'
STEMS = {'02': 'validation', '03': 'lint', '04': 'tests_check',
         '05': 'spec_judge', '06': 'graph_eval', '07': 'test_deepeval', '08': 'trace_eval'}
FAKE = r'''
import json, os, pathlib, signal, subprocess, sys, time
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(args) + '\n')
state = pathlib.Path(os.environ['STATE'])
if os.environ.get('BLOCK') == ('inspect' if args[:2] == ['image', 'inspect'] else args[0]):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(os.environ['READY']).write_text(str(os.getpid()))
    while True:
        time.sleep(.05)
if args[:2] == ['container', 'ls']:
    if os.environ.get('LIST_FAIL'):
        sys.exit(1)
    if state.exists():
        print(state.read_text())
    sys.exit(0)
if args[0] == 'rm':
    if os.environ.get('RM_FAIL'):
        sys.exit(1)
    state.unlink(missing_ok=True)
    if os.environ.get('STABILIZE'):
        snapshot = json.loads(pathlib.Path(os.environ['SNAPSHOT']).read_text())
        for p in pathlib.Path(snapshot['mounts']['/results']).rglob('08_trace_eval_*.json'):
            p.write_text('stabilized')
    sys.exit(0)
if args[:2] == ['image', 'inspect']:
    sys.exit(int(os.environ.get('INSPECT_CODE', '0')))
if args[0] != 'run':
    sys.exit(0)
state.write_text(args[args.index('--name') + 1])
if os.environ.get('IGNORE_TERM'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
mounts = {}
for i, arg in enumerate(args):
    if arg == '-v':
        source, target, mode = args[i+1].split(':')
        mounts[target] = source
files = {target: sorted(str(p.relative_to(source)) for p in pathlib.Path(source).rglob('*')
                        if p.is_file()) for target, source in mounts.items()}
contents = {target: {name: (pathlib.Path(source) / name).read_text() for name in files[target]}
            for target, source in mounts.items()}
forwarded = {}
for i, arg in enumerate(args):
    if arg == '-e':
        key, separator, value = args[i+1].partition('=')
        forwarded[key] = value if separator else os.environ[key]
pathlib.Path(os.environ['SNAPSHOT']).write_text(json.dumps({'args':args, 'mounts':mounts, 'files':files, 'contents':contents, 'forwarded':forwarded}))
if os.environ.get('EXECUTE_STAGED'):
    # Execute the actual allowlisted stage, not a mock report producer.
    child_env = {'HOME': os.environ['HOME'], 'PATH': '',
                 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'DEEPEVAL_TELEMETRY_OPT_OUT': '1'}
    child_env.update({key: mounts.get(value, value) for key, value in forwarded.items()})
    index = args.index('python3')
    script = args[index + 1].replace('/repo/', mounts['/repo'] + '/', 1)
    prefix = [sys.executable, os.environ['TRACE_DRIVER']] if os.environ.get('TRACE_DRIVER') else [sys.executable]
    child = subprocess.run([*prefix, script, *args[index + 2:]],
                           env=child_env, capture_output=True, text=True, timeout=30)
    print(child.stdout, end='')
    print(child.stderr, end='', file=sys.stderr)
    products = list(pathlib.Path(mounts['/results']).rglob('*.json'))
    pathlib.Path(os.environ['BEFORE_CLEANUP']).write_text(json.dumps({
        'returncode': child.returncode, 'reports': [json.loads(p.read_text()) for p in products]}))
    sys.exit(child.returncode)
step = os.environ['STEP']
relative = pathlib.Path('exp/run_1')
out = pathlib.Path(mounts['/results']) / relative
out.mkdir(parents=True)
product = out / (step + '_' + os.environ['STEM'] + '_model_20260101_010101.json')
attack = os.environ.get('ATTACK', '')
if attack == 'ancestor':
    out.rmdir()
    out.symlink_to(os.environ['OUTSIDE'], target_is_directory=True)
elif attack == 'file':
    product.symlink_to(os.environ['OUTSIDE'] + '/secret')
else:
    product.write_text('report')
if os.environ.get('TRACE_CASE'):
    case = os.environ['TRACE_CASE']
    run_id, execution = 'a' * 32, 'b' * 32
    link = pathlib.Path('08_traces/model/agent') / execution / (run_id + '.json')
    trace = dict(schema_version=1, run_id=run_id, status='error', query='synthetic',
                 error='Agent execution failed (ValueError)', runs_collected=0,
                 trajectory=[], tool_calls=[], test_nodeid='test_trace.py::test_synthetic')
    record = dict(path=str(link), run_id=run_id, capture_status='error', test_nodeid=trace['test_nodeid'])
    source = out / link
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(trace))
    if case in ('foreign_agent', 'foreign_llm', 'foreign_run', 'uuid', 'filename', 'absolute'):
        record['path'] = {'foreign_agent': str(link).replace('/agent/', '/other/'),
            'foreign_llm': str(link).replace('/model/', '/other/'),
            'foreign_run': '../run_2/' + str(link), 'uuid': str(link).replace(execution, 'bad'),
            'filename': str(link).replace(run_id + '.json', '_bad.json'),
            'absolute': str(source)}[case]
    elif case in ('symlink', 'dangling'):
        source.unlink()
        source.symlink_to(os.environ['OUTSIDE'] + ('/missing' if case == 'dangling' else '/secret'))
    elif case == 'ancestor':
        source.unlink(); source.parent.rmdir()
        source.parent.symlink_to(os.environ['OUTSIDE'], target_is_directory=True)
    elif case in ('schema', 'nodeid', 'run_id', 'status'):
        trace[{'schema': 'schema_version', 'nodeid': 'test_nodeid', 'run_id': 'run_id', 'status': 'status'}[case]] = 'invalid'
        source.write_text(json.dumps(trace))
    elif case == 'hardlink':
        source.unlink(); source.hardlink_to(pathlib.Path(os.environ['OUTSIDE']) / 'secret')
    elif case == 'fifo':
        source.unlink(); os.mkfifo(source)
    elif case == 'directory':
        source.unlink(); source.mkdir()
    elif case.startswith('nested_'):
        call = dict(name='synthetic', input=None, start=None, end=None, output={'arbitrary': [42]})
        trace['trajectory'], trace['tool_calls'] = ['synthetic'], [call]
        if case == 'nested_trajectory': trace['trajectory'] = [42]
        elif case == 'nested_notdict': trace['tool_calls'] = [42]
        elif case == 'nested_name': call['name'] = 42
        elif case == 'nested_input': del call['input']
        elif case == 'nested_startmissing': del call['start']
        elif case == 'nested_endmissing': del call['end']
        elif case == 'nested_start': call['start'] = 42
        elif case == 'nested_end': call['end'] = {}
        elif case == 'nested_both': call['error'] = 'failure'
        elif case == 'nested_neither': del call['output']
        elif case == 'nested_error': del call['output']; call['error'] = None
        elif case == 'nested_traceerror': trace['error'] = 42
        elif case == 'nested_validerror': del call['output']; call['error'] = 'failure'
        elif case == 'nested_validoutput': call['start'] = 'synthetic-start'; call['end'] = 'synthetic-end'; trace['error'] = None
        source.write_text(json.dumps(trace))
    # An unreferenced orphan must never be copied, even if it looks valid.
    (source.parent / 'orphan.json').write_text(json.dumps(trace))
    product.write_text(json.dumps(dict(model='model', results=[dict(agent_folder='agent', model='model',
        status='ok', passed=True, trace_artifacts=[] if case == 'orphan' else [record])], status='ok')))
(pathlib.Path(mounts['/results']) / 'agent.py').write_text('evil')
(pathlib.Path(mounts['/results']) / 'unapproved.json').write_text('evil')
(out / '02_validation_other_20260101_010101.json').write_text('evil')
(out / '99_fake_model_20260101_010101.json').write_text('evil')
data = pathlib.Path(mounts['/data']) / relative / 'model/agent'
(data / 'agent.py').write_text('changed')
(data / 'trace.jsonl').write_text('trace')
(data / 'agent.log').write_text('log')
(data / '.pytest_report.json').write_text('pytest diagnostic')
print('fake stdout', flush=True)
print('fake stderr', file=sys.stderr, flush=True)
if os.environ.get('WAIT'):
    while True:
        time.sleep(.05)
sys.exit(int(os.environ.get('EXIT_CODE', '0')))
'''


@pytest.fixture
def harness(tmp_path):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    fake = bindir / 'docker'
    fake.write_text(f'#!{sys.executable}\n' + FAKE)
    fake.chmod(0o755)
    # No inherited PATH, credentials, HOME, or dotenv inputs.
    home = tmp_path / 'home'
    home.mkdir()
    generated = tmp_path / 'generated'
    target = generated / 'exp/run_1/model/agent'
    target.mkdir(parents=True)
    for name in ('agent.py', 'test_unit.py', '.env', 'credentials.json', 'session.log'):
        (target / name).write_text('# synthetic')
    for other in ('exp/run_1/model/other', 'exp/run_2/model/agent', 'other/run_1/model/agent'):
        directory = generated / other
        directory.mkdir(parents=True)
        (directory / 'agent.py').write_text('forbidden')
    bench = tmp_path / 'bench'
    (bench / 'agent').mkdir(parents=True)
    for name in ('spec.md', 'spec_adherence.py', 'eval_2_graph.py', 'test_deepeval.py',
                 'test_trace.py', '.env', 'rubric.md', 'unrelated.py'):
        (bench / 'agent' / name).write_text('# fake fixture')
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'private.json').write_text('never mounted')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'secret').write_text('synthetic secret')
    env = dict(PATH=str(bindir), HOME=str(home), SPEC2AGENT_SKIP_DOTENV='1',
               PYTHON_DOTENV_DISABLED='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               DEEPEVAL_TELEMETRY_OPT_OUT='1',
               BENCH_DIR=str(bench), GENERATED_DIR=str(generated),
               EXPERIMENTS_RESULTS=str(results), CALLS=str(tmp_path / 'calls'),
               SNAPSHOT=str(tmp_path / 'snapshot'), OUTSIDE=str(outside),
               STATE=str(tmp_path / 'container-state'), READY=str(tmp_path / 'ready'),
               OPENAI_API_KEY='fake-openai', GEMINI_API_KEY='fake-gemini',
               SONAR_HOST_URL='fake-sonar', SONAR_LOGIN='fake-login',
               SONAR_PASSWORD='fake-password', USER_TOKEN='fake-token',
               SONAR_COMMAND='fake-command', ANTHROPIC_API_KEY='forbidden',
               HTTPS_PROXY='forbidden', TMPDIR=str(tmp_path))
    return tmp_path, env


def command(env, step='02', extra=()):
    env.update(STEP=step, STEM=STEMS[step])
    return [sys.executable, str(HELPER), '--step', step, '--llm', 'model',
            '--agent', 'agent', '--experiment', 'exp', '--run', '1', *extra]


def run(harness, step='02', extra=()):
    directory, env = harness
    result = subprocess.run(command(env, step, extra), env=env, capture_output=True, text=True)
    snapshot = directory / 'snapshot'
    data = json.loads(snapshot.read_text()) if snapshot.exists() else None
    if data:
        assert all(not Path(source).exists() for source in data['mounts'].values())
    assert not list(directory.glob('spec2agent-eval-*'))
    assert not Path(env['STATE']).exists()
    return result, data


@pytest.mark.parametrize('step', STEMS)
def test_allowlisted_mounts_env_network_and_products(harness, step):
    directory, env = harness
    result, snapshot = run(harness, step)
    assert result.returncode == 0, result.stderr
    args, files = snapshot['args'], snapshot['files']
    assert args[args.index('--network') + 1] == ('none' if step in ('02', '04', '06') else 'bridge')
    assert args[args.index('--user') + 1] == f'{os.getuid()}:{os.getgid()}'
    assert len(snapshot['mounts']) == 4
    assert files['/results'] == []
    data_files = ['exp/run_1/model/agent/agent.py']
    if step == '04':
        data_files.append('exp/run_1/model/agent/test_unit.py')
    assert files['/data'] == data_files
    assert set(snapshot['contents']['/data'].values()) == {'# synthetic'}
    expected_bench = {'05': ['spec.md', 'spec_adherence.py'], '06': ['eval_2_graph.py'],
                      '07': ['test_deepeval.py'], '08': ['test_trace.py']}.get(step, [])
    assert files['/bench'] == sorted('agent/' + f for f in expected_bench)
    assert len(files['/repo']) == (3 if step == '08' else 2 if step in ('03', '07') else 1)
    if step in ('07', '08'):
        expected = {f'pipeline/{step}_' + ('behavioral_check.py' if step == '07' else 'trace_check.py'),
                    'pipeline/utils/pytest_runner.py'}
        if step == '08':
            expected.add('pipeline/utils/trace_capture.py')
        assert set(files['/repo']) == expected
    forwarded = snapshot['forwarded']
    secret_names = {k for k in forwarded if k in env and k not in ('GENERATED_DIR', 'BENCH_DIR',
                    'EXPERIMENTS_RESULTS', 'SPEC2AGENT_SKIP_DOTENV', 'PYTHON_DOTENV_DISABLED')}
    assert secret_names == (set(('SONAR_HOST_URL', 'SONAR_LOGIN', 'SONAR_PASSWORD',
                                'USER_TOKEN', 'SONAR_COMMAND')) if step == '03' else
                            {'OPENAI_API_KEY'} if step in ('05', '07', '08') else set())
    assert forwarded['PYTHON_DOTENV_DISABLED'] == '1'
    sensitive = ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'SONAR_HOST_URL', 'SONAR_LOGIN',
                 'SONAR_PASSWORD', 'USER_TOKEN', 'SONAR_COMMAND', 'ANTHROPIC_API_KEY', 'HTTPS_PROXY')
    assert all(env[key] not in ' '.join(args) for key in sensitive)
    for key in secret_names:
        assert forwarded[key] == env[key] and key in args
    assert 'fake stdout' in result.stdout and 'fake stderr' in result.stderr
    reports = list((directory / 'results/exp/run_1').glob(f'{step}_*.json'))
    assert len(reports) == 1 and reports[0].read_text() == 'report'
    assert (directory / 'generated/exp/run_1/model/agent/agent.py').read_text() == '# synthetic'
    assert not (directory / 'results/agent.py').exists()
    assert not list((directory / 'results').rglob('*_other_*.json'))
    assert not list((directory / 'results').rglob('99_*.json'))
    artifacts = list((directory / 'results').rglob('trace.jsonl'))
    assert len(artifacts) == (1 if step in ('07', '08') else 0)
    assert len(list((directory / 'results').rglob('.pytest_report.json'))) == (1 if step in ('07', '08') else 0)


@pytest.mark.parametrize('mode', ['Z', 'none', 'disable', 'invalid'])
def test_selinux(harness, mode):
    harness[1]['EVAL_SELINUX'] = mode
    result, snapshot = run(harness)
    if mode == 'invalid':
        assert result.returncode != 0 and snapshot is None
        return
    assert result.returncode == 0
    args = snapshot['args']
    mounts = [args[i+1] for i, a in enumerate(args) if a == '-v']
    assert all(m.endswith(',Z') == (mode == 'Z') for m in mounts)
    assert ('label=disable' in args) == (mode == 'disable')
    assert ('WARNING' in result.stderr) == (mode == 'disable')


@pytest.mark.parametrize('network', ['none', 'custom_net', 'host', 'container:other', 'bad;id'])
def test_network_policy(harness, network):
    harness[1]['EVAL_NETWORK'] = network
    result, snapshot = run(harness, '07')
    assert (result.returncode == 0) == (network in ('none', 'custom_net'))
    if snapshot:
        assert snapshot['args'][snapshot['args'].index('--network')+1] == network


@pytest.mark.parametrize('step', ['02', '04', '06'])
def test_offline_cannot_be_overridden(harness, step):
    harness[1]['EVAL_NETWORK'] = 'custom_net'
    result, snapshot = run(harness, step)
    assert result.returncode == 0
    assert snapshot['args'][snapshot['args'].index('--network')+1] == 'none'


@pytest.mark.parametrize('provider', ['gemini', 'none', 'bad'])
@pytest.mark.parametrize('step', ['05', '07', '08'])
def test_provider_selection(harness, provider, step):
    harness[1]['EVAL_PROVIDER'] = provider
    result, snapshot = run(harness, step)
    assert (result.returncode == 0) == (provider != 'bad')
    if snapshot:
        joined = ' '.join(snapshot['args'])
        assert 'OPENAI_API_KEY' not in joined
        assert ('GEMINI_API_KEY' in joined) == (provider == 'gemini')
        assert 'fake-gemini' not in joined and 'fake-openai' not in joined
        allowed = {'GENERATED_DIR', 'EXPERIMENTS_RESULTS', 'BENCH_DIR',
                   'SPEC2AGENT_SKIP_DOTENV', 'PYTHON_DOTENV_DISABLED'}
        if provider == 'gemini':
            allowed.add('GEMINI_API_KEY')
            assert snapshot['forwarded']['GEMINI_API_KEY'] == 'fake-gemini'
        assert set(snapshot['forwarded']) == allowed


@pytest.mark.parametrize('field,value', [('llm', '../escape'), ('agent', 'a;id'),
                                       ('experiment', '/tmp'), ('run', '01'), ('run', '-1')])
def test_invalid_components(harness, field, value):
    result, snapshot = run(harness, extra=(f'--{field}', value))
    assert result.returncode != 0 and snapshot is None


@pytest.mark.parametrize('location', ['input', 'ancestor', 'bench', 'output', 'destination'])
def test_symlinks_rejected(harness, location):
    directory, env = harness
    if location == 'input':
        source = directory / 'generated/exp/run_1/model/agent/agent.py'
        source.unlink()
        source.symlink_to(directory / 'outside/secret')
    elif location == 'ancestor':
        source = directory / 'generated/exp/run_1/model/agent'
        source.rename(source.with_name('real'))
        source.symlink_to(source.with_name('real'), target_is_directory=True)
    elif location == 'bench':
        source = directory / 'bench/agent/spec.md'
        source.unlink()
        source.symlink_to(directory / 'outside/secret')
    elif location == 'output':
        env['ATTACK'] = 'file'
    else:
        (directory / 'results/exp').symlink_to(directory / 'outside', target_is_directory=True)
    result, _ = run(harness, '05')
    assert result.returncode != 0
    assert (directory / 'outside/secret').read_text() == 'synthetic secret'


def test_output_ancestor_rejected(harness):
    harness[1]['ATTACK'] = 'ancestor'
    result, _ = run(harness)
    assert result.returncode != 0


def test_failure_publication_and_no_overwrite(harness):
    harness[1]['EXIT_CODE'] = '17'
    result, _ = run(harness, '08')
    assert result.returncode == 17
    assert list((harness[0] / 'results').rglob('trace.jsonl'))
    harness[1]['EXIT_CODE'] = '0'
    result, _ = run(harness, '08')
    assert result.returncode != 0  # operational publication error
    assert len(list((harness[0] / 'results').rglob('08_trace_eval_*.json'))) == 1


def test_inspect_failure_build_fallback(harness):
    harness[1]['INSPECT_CODE'] = '1'
    result, _ = run(harness)
    assert result.returncode == 0
    calls = [json.loads(line) for line in (harness[0] / 'calls').read_text().splitlines()]
    assert calls[1][0] == 'build' and calls[2][0] == 'run'


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize('refused', [False, True])
def test_interrupt_cleanup_and_failure_products(harness, signum, refused):
    directory, env = harness
    env['WAIT'] = '1'
    env['IGNORE_TERM'] = '1'
    if refused:
        env['RM_FAIL'] = '1'
    process = subprocess.Popen(command(env, '08'), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not (directory / 'snapshot').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (directory / 'snapshot').exists()
        # stdout readiness means products have been written before interruption.
        assert process.stdout.readline().strip() == b'fake stdout'
        started = time.monotonic()
        process.send_signal(signum)
        process.communicate(timeout=10)
        assert time.monotonic() - started < 5
        if refused:
            assert process.returncode == 1
            assert Path(env['STATE']).exists()
            assert list(directory.glob('spec2agent-eval-*'))
            assert not list((directory / 'results').rglob('trace.jsonl'))
            return
        assert process.returncode == 128 + signum
        assert not list(directory.glob('spec2agent-eval-*'))
        assert list((directory / 'results').rglob('trace.jsonl'))
        calls = [json.loads(line) for line in (directory / 'calls').read_text().splitlines()]
        assert calls[-2][:2] == ['rm', '-f']
        assert calls[-1][:2] == ['container', 'ls']
        assert not Path(env['STATE']).exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_disconnected_client_teardown_before_publication(harness):
    directory, env = harness
    env.update(EXIT_CODE='125', STABILIZE='1')
    result, _ = run(harness, '08')
    assert result.returncode == 125
    report = next((directory / 'results').rglob('08_trace_eval_*.json'))
    assert report.read_text() == 'stabilized'


@pytest.mark.parametrize('failure', ['RM_FAIL', 'LIST_FAIL'])
def test_unconfirmed_teardown_preserves_private_staging(harness, failure):
    directory, env = harness
    env.update(EXIT_CODE='125', **{failure: '1'})
    result = subprocess.run(command(env, '08'), env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert 'absence unconfirmed' in result.stderr
    snapshot = json.loads((directory / 'snapshot').read_text())
    private = Path(snapshot['mounts']['/data']).parent
    assert private.is_dir() and private.stat().st_mode & 0o777 == 0o700
    assert f'PRIVATE STAGING PRESERVED: {private}' in result.stderr
    assert not list((directory / 'results').rglob('08_trace_eval_*.json'))
    assert not list((directory / 'results').rglob('trace.jsonl'))
    if failure == 'RM_FAIL':
        assert Path(env['STATE']).exists()  # independent of client exit


def test_failed_rm_but_confirmed_absent_is_safe(harness):
    # A daemon may report rm failure when --rm already removed the container.
    directory, env = harness
    env['RM_FAIL'] = '1'
    fake = directory / 'bin/docker'
    fake.write_text(fake.read_text().replace("state.write_text(args[args.index('--name') + 1])",
                                             "pass  # simulate auto-removed container"))
    result, _ = run(harness)
    assert result.returncode == 0


def test_blocked_teardown_is_bounded_and_preserves_staging(harness):
    directory, env = harness
    env.update(BLOCK='rm', EXIT_CODE='125')
    started = time.monotonic()
    result = subprocess.run(command(env, '08'), env=env, capture_output=True,
                            text=True, timeout=8)
    assert time.monotonic() - started < 7
    assert result.returncode == 1
    assert 'PRIVATE STAGING PRESERVED' in result.stderr
    assert Path(env['STATE']).exists()
    assert list(directory.glob('spec2agent-eval-*'))
    assert not list((directory / 'results').rglob('08_trace_eval_*.json'))
    with pytest.raises(ProcessLookupError):
        os.kill(int(Path(env['READY']).read_text()), 0)


@pytest.mark.parametrize('stage', ['inspect', 'build'])
@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_cancel_blocked_setup_never_starts_next_operation(harness, stage, signum):
    directory, env = harness
    env.update(BLOCK=stage, INSPECT_CODE='1')
    process = subprocess.Popen(command(env), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not Path(env['READY']).exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert Path(env['READY']).exists()
        started = time.monotonic()
        process.send_signal(signum)
        process.communicate(timeout=5)
        assert time.monotonic() - started < 4
        assert process.returncode == 128 + signum
        calls = [json.loads(line) for line in (directory / 'calls').read_text().splitlines()]
        assert all(call[0] != 'run' for call in calls)
        if stage == 'inspect':
            assert all(call[0] != 'build' for call in calls)
        assert not list(directory.glob('spec2agent-eval-*'))
        pid = int(Path(env['READY']).read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize('refused', [False, True])
def test_wait_exception_still_tears_down(harness, monkeypatch, refused):
    directory, env = harness
    env.update(WAIT='1', IGNORE_TERM='1')
    if refused:
        env['RM_FAIL'] = '1'
    spec = importlib.util.spec_from_file_location('eval_staging_test', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    real_popen = subprocess.Popen

    class FailedWait:
        def __init__(self, process):
            self.process = process
            self.pid = process.pid
            self.failed = False

        def poll(self):
            return self.process.poll()

        def wait(self, timeout=None):
            if not self.failed:
                self.failed = True
                # Wait until the independent fake workload is registered.
                deadline = time.monotonic() + 5
                while not (directory / 'snapshot').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                raise RuntimeError('synthetic wait exception')
            return self.process.wait(timeout=timeout)

    def popen(args, **kwargs):
        process = real_popen(args, **kwargs)
        return FailedWait(process) if args[:2] == ['docker', 'run'] else process

    argv = command(env, '08')[1:]
    for key in tuple(os.environ):
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(helper.tempfile, 'tempdir', str(directory))
    monkeypatch.setattr(subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='synthetic wait exception'):
        helper.main()
    if refused:
        assert Path(env['STATE']).exists()
        assert list(directory.glob('spec2agent-eval-*'))
        assert not list((directory / 'results').rglob('08_trace_eval_*.json'))
        return
    assert not Path(env['STATE']).exists()
    assert not list(directory.glob('spec2agent-eval-*'))
    assert not list((directory / 'results').rglob('08_trace_eval_*.json'))


def test_unreaped_client_preserves_staging_even_if_name_currently_absent(harness, monkeypatch):
    directory, env = harness
    spec = importlib.util.spec_from_file_location('eval_staging_unreaped', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(sys, 'argv', command(env)[1:])
    monkeypatch.setattr(os, 'environ', env)
    monkeypatch.setattr(helper.tempfile, 'tempdir', str(directory))
    checked = []

    def invoke(args, cancelled, **kwargs):
        if args[:2] == ['docker', 'run']:
            raise helper.ClientStillAlive('synthetic unreaped client')
        return 0

    monkeypatch.setattr(helper, 'invoke', invoke)
    monkeypatch.setattr(helper, 'teardown', lambda name: checked.append(name) or True)
    with pytest.raises(RuntimeError, match='synthetic unreaped client'):
        helper.main()
    assert checked and list(directory.glob('spec2agent-eval-*'))


@pytest.mark.parametrize('step', STEMS)
def test_only_stage_specific_operational_config_is_forwarded(harness, step):
    harness[1].update(SONAR_MODE='disabled', SONAR_REQUIRED='0', PYTEST_TIMEOUT_SECONDS='17')
    result, snapshot = run(harness, step)
    assert result.returncode == 0, result.stderr
    expected = {'SONAR_MODE': 'disabled', 'SONAR_REQUIRED': '0'} if step == '03' else (
        {'PYTEST_TIMEOUT_SECONDS': '17'} if step in ('07', '08') else {})
    keys = {'SONAR_MODE', 'SONAR_REQUIRED', 'PYTEST_TIMEOUT_SECONDS'}
    assert {key: value for key, value in snapshot['forwarded'].items() if key in keys} == expected
    for key, value in expected.items():
        assert f'{key}={value}' in snapshot['args']
        assert key not in snapshot['args']
    assert 'DEEPEVAL_TELEMETRY_OPT_OUT' not in snapshot['forwarded']


@pytest.mark.parametrize('step', STEMS)
@pytest.mark.parametrize('missing', ['file', 'folder'])
def test_missing_agent_real_staged_script_publishes_error(harness, step, missing):
    directory, env = harness
    target = directory / 'generated/exp/run_1/model/agent'
    if missing == 'file':
        (target / 'agent.py').unlink()
    else:
        shutil.rmtree(target)
    env.update(EXECUTE_STAGED='1', BEFORE_CLEANUP=str(directory / 'before-cleanup'),
               EVAL_PROVIDER='none', SONAR_MODE='disabled')
    for key in ('SONAR_HOST_URL', 'SONAR_LOGIN', 'SONAR_PASSWORD', 'USER_TOKEN', 'SONAR_COMMAND',
                'OPENAI_API_KEY', 'GEMINI_API_KEY'):
        env.pop(key, None)
    result, snapshot = run(harness, step)
    assert result.returncode == 1, result.stderr
    assert 'exp/run_1/model/agent/agent.py' not in snapshot['files']['/data']
    before = json.loads((directory / 'before-cleanup').read_text())
    assert before['returncode'] == 1 and len(before['reports']) == 1
    reports = list((directory / 'results/exp/run_1').glob(f'{step}_*.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report == before['reports'][0]
    assert report['status'] == 'error' and len(report['results']) == 1
    assert report['results'][0]['status'] == 'error'
    assert report['results'][0]['reason_code'] == 'missing_agent'
    assert not (target / 'agent.py').exists()
    assert target.is_dir() == (missing == 'file')


@pytest.mark.parametrize('step', ['02', '04'])
@pytest.mark.parametrize('location', ['file', 'ancestor'])
def test_missing_optional_input_still_rejects_dangling_symlinks(harness, step, location):
    directory, env = harness
    target = directory / 'generated/exp/run_1/model/agent'
    if location == 'file':
        target = target / 'agent.py'
        target.unlink()
    else:
        shutil.rmtree(target)
    target.symlink_to(directory / 'nonexistent')
    result, snapshot = run(harness, step)
    assert result.returncode == 1 and snapshot is None
    assert 'symlink forbidden' in result.stderr
    assert target.is_symlink()


@pytest.mark.parametrize('case', ['valid', 'orphan', 'nested_validoutput', 'nested_validerror',
    'nested_trajectory', 'nested_notdict', 'nested_name', 'nested_input', 'nested_startmissing',
    'nested_endmissing', 'nested_start', 'nested_end', 'nested_both', 'nested_neither', 'nested_error', 'nested_traceerror',
    'foreign_agent', 'foreign_llm', 'foreign_run',
    'uuid', 'filename', 'absolute', 'symlink', 'dangling', 'ancestor', 'schema', 'nodeid', 'run_id', 'status', 'hardlink', 'fifo', 'directory'])
def test_manifest_trace_publication(harness, case):
    directory, env = harness
    env['TRACE_CASE'] = case
    result, _ = run(harness, '08')
    valid = case in ('valid', 'orphan', 'nested_validoutput', 'nested_validerror')
    assert (result.returncode == 0) is valid, result.stderr
    traces = list((directory / 'results').rglob('08_traces/**/*.json'))
    assert len(traces) == (1 if valid and case != 'orphan' else 0)
    reports = list((directory / 'results').rglob('08_trace_eval_*.json'))
    assert len(reports) == int(valid)
    if valid and case != 'orphan':
        report = json.loads(reports[0].read_text())
        record = report['results'][0]['trace_artifacts'][0]
        trace = json.loads(traces[0].read_text())
        assert traces[0] == reports[0].parent / record['path']
        assert (trace['run_id'], trace['test_nodeid'], trace['status']) == (record['run_id'], record['test_nodeid'], record['capture_status'])
        assert trace['status'] == 'error' and report['results'][0]['passed'] is True
        assert traces[0].stat().st_mode & 0o777 == 0o600
    assert (directory / 'outside/secret').read_text() == 'synthetic secret'


def test_trace_failure_products_and_no_overwrite(harness):
    directory, env = harness
    env.update(TRACE_CASE='valid', EXIT_CODE='17')
    result, _ = run(harness, '08')
    assert result.returncode == 17
    trace = next((directory / 'results').rglob('08_traces/**/*.json'))
    previous = trace.read_bytes()
    result, _ = run(harness, '08')
    assert result.returncode == 1
    assert trace.read_bytes() == previous


@pytest.mark.parametrize('location', ['ancestor', 'file', 'dangling'])
def test_trace_destination_symlinks_rejected(harness, location):
    directory, env = harness
    env['TRACE_CASE'] = 'valid'
    target = directory / 'results/exp/run_1/08_traces'
    if location != 'ancestor':
        target = target / 'model/agent' / ('b' * 32) / ('a' * 32 + '.json')
    target.parent.mkdir(parents=True)
    target.symlink_to(directory / ('missing' if location == 'dangling' else 'outside'),
                      target_is_directory=location == 'ancestor')
    result, _ = run(harness, '08')
    assert result.returncode == 1
    assert target.is_symlink()
    assert not list((directory / 'results').rglob('08_trace_eval_*.json'))
    assert sorted(p.name for p in (directory / 'outside').iterdir()) == ['secret']


def test_trace_unconfirmed_teardown_never_publishes(harness):
    directory, env = harness
    env.update(TRACE_CASE='valid', RM_FAIL='1')
    result = subprocess.run(command(env, '08'), env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert not list((directory / 'results').rglob('08_traces/**/*.json'))


def test_staged08_real_capture_guarded(harness):
    directory, env = harness
    tree = ast.parse((ROOT / 'tests/test_pytest_plugin_contract.py').read_text())
    guard = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'BOOTSTRAP' for t in node.targets))
    attempts = directory / 'blocked.jsonl'
    driver = directory / 'trace_driver.py'
    driver.write_text('import pathlib, runpy, subprocess, sys\n'
        'script = sys.argv.pop(1)\nsys.argv[0] = script\n'
        'sys.path.insert(0, str(pathlib.Path(script).parent))\n'
        'actual = subprocess.Popen\n'
        'def guarded(args, **kwargs):\n'
        ' assert args[:3] == [sys.executable, "-m", "pytest"]\n'
        ' assert kwargs["env"]["PYTHONPATH"] == str(pathlib.Path(script).parent / "utils")\n'
        f' return actual([args[0], "-c", {guard!r}, {str(attempts)!r}, *args[3:]], **kwargs)\n'
        'subprocess.Popen = guarded\nrunpy.run_path(script, run_name="__main__")\n')
    (directory / 'generated/exp/run_1/model/agent/agent.py').write_text('def agent():\n raise ValueError("synthetic failure")\n')
    (directory / 'bench/agent/test_trace.py').write_text(
        'from trace_capture import traced\nfrom agent import agent\n'
        'def test_expected_failure():\n output, trace = traced(agent)()\n'
        ' assert output is None and trace["status"] == "error"\n')
    env.update(EXECUTE_STAGED='1', TRACE_DRIVER=str(driver), EVAL_PROVIDER='none',
               BEFORE_CLEANUP=str(directory / 'before-cleanup'))
    result, snapshot = run(harness, '08')
    assert result.returncode == 0, result.stdout + result.stderr
    assert not attempts.exists(), attempts.read_text() if attempts.exists() else ''
    report_path = next((directory / 'results').rglob('08_trace_eval_*.json'))
    entry = json.loads(report_path.read_text())['results'][0]
    assert entry['status'] == 'ok' and entry['passed'] is True
    assert len(entry['trace_artifacts']) == 1
    record = entry['trace_artifacts'][0]
    trace = json.loads((report_path.parent / record['path']).read_text())
    assert trace['status'] == record['capture_status'] == 'error'
    assert trace['test_nodeid'] == record['test_nodeid']
    assert trace['run_id'] == record['run_id']
    assert snapshot['files']['/repo'] == ['pipeline/08_trace_check.py', 'pipeline/utils/pytest_runner.py', 'pipeline/utils/trace_capture.py']


def test_trace_publication_rollback_only_owned_files(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('trace_staging_rollback', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir(); destination.mkdir()
    sentinel = destination / 'sentinel.json'
    sentinel.write_text('preexisting')
    prior = destination / '08_traces/model/agent' / ('c' * 32) / 'previous.json'
    prior.parent.mkdir(parents=True)
    prior.write_text('previous execution')
    relative = Path('08_traces/model/agent') / ('b' * 32) / 'trace.json'
    trace = dict(schema_version=1, run_id='a' * 32, status='ok', query=None, error=None,
                 runs_collected=0, trajectory=[], tool_calls=[], test_nodeid='test_synthetic')
    (source / relative).parent.mkdir(parents=True)
    (source / relative).write_text(json.dumps(trace))
    report = source / '08_trace_eval_model_20260101_010101.json'
    report.write_text(json.dumps(dict(model='model', results=[dict(model='model', agent_folder='agent',
        trace_artifacts=[dict(path=str(relative), run_id=trace['run_id'], capture_status='ok', test_nodeid='test_synthetic')])])))
    actual = os.open
    def fail_report(path, *args, **kwargs):
        if Path(path).name == report.name:
            assert (destination / relative).exists()  # traces precede report
            raise OSError('FAKE_SELECTED_KEY')
        return actual(path, *args, **kwargs)
    monkeypatch.setattr(helper.os, 'open', fail_report)
    with pytest.raises(ValueError, match='^trace/report publication failed$'):
        helper.publish_trace_reports([report], source, destination, 'model', 'agent')
    assert sentinel.read_text() == 'preexisting'
    assert list(destination.rglob('08_traces/**/*.json')) == [prior]
    assert prior.read_text() == 'previous execution'
    assert not (destination / report.name).exists()
