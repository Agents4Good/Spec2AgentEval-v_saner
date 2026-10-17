"""Offline plugin-only probe; not evidence for the future evaluation runner."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / '.venv/bin/python') if (ROOT / '.venv/bin/python').exists() else sys.executable
PLUGINS = ('pytest_jsonreport.plugin', 'pytest_asyncio.plugin', 'deepeval.plugins.plugin')
METADATA = r'''
import importlib.metadata as m, json
result = {}
for name in ('pytest', 'pytest-json-report', 'pytest-asyncio', 'deepeval'):
    try:
        d = m.distribution(name)
        result[name] = {'version': d.version, 'plugins':
                       [e.value for e in d.entry_points if e.group == 'pytest11']}
    except m.PackageNotFoundError:
        result[name] = {'missing': True}
print(json.dumps(result))
'''
BOOTSTRAP = r'''
import json, pathlib, socket, sys
attempts = pathlib.Path(sys.argv[1])
def deny(kind):
    frames, frame = [], sys._getframe(1)
    while frame is not None and len(frames) < 16:
        frames.append({'file': frame.f_code.co_filename, 'line': frame.f_lineno,
                       'function': frame.f_code.co_name})
        frame = frame.f_back
    with attempts.open('a') as out:
        out.write(json.dumps({'blocked': kind, 'stack': frames}) + '\n')
    raise RuntimeError('BLOCKED plugin probe attempted ' + kind)
def blocked(*args, **kwargs):
    return deny('network/DNS')
for name in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
    if hasattr(socket.socket, name):
        setattr(socket.socket, name, blocked)
for name in ('create_connection', 'getaddrinfo', 'gethostbyname',
             'gethostbyname_ex', 'gethostbyaddr'):
    setattr(socket, name, blocked)
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec'):
        deny('external process: ' + event)
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname',
                 'socket.gethostbyaddr', 'socket.sendto', 'socket.sendmsg'):
        deny('network/DNS audit: ' + event)
sys.addaudithook(audit)
# Guard is installed BEFORE importing pytest or any third-party plugin.
sys.path.insert(0, str(pathlib.Path.cwd()))
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
'''


def test_explicit_plugins_offline_with_agent_async_and_json(tmp_path):
    home, temp, fake_bin = (tmp_path / name for name in ('home', 'temp', 'bin'))
    for directory in (home, temp, fake_bin):
        directory.mkdir()
    env = {'HOME': str(home), 'TMPDIR': str(temp), 'PATH': str(fake_bin),
           'SPEC2AGENT_SKIP_DOTENV': '1', 'PYTHON_DOTENV_DISABLED': '1',
           'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'DEEPEVAL_TELEMETRY_OPT_OUT': '1'}
    metadata = subprocess.run([PYTHON, '-I', '-c', METADATA], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=30)
    assert metadata.returncode == 0, metadata.stderr
    installed = json.loads(metadata.stdout)
    print('PLUGIN_METADATA=' + json.dumps(installed, sort_keys=True))
    missing = [name for name, value in installed.items() if value.get('missing')]
    if missing:
        pytest.fail('BLOCKED: missing installed distributions: ' + ', '.join(missing)
                    + '; no plugins imported, no installation, no compatibility evidence')
    expected = {'pytest': '9.', 'pytest-json-report': '1.5.',
                'pytest-asyncio': '1.3.', 'deepeval': '3.9.5'}
    for name, version in expected.items():
        assert installed[name]['version'].startswith(version), (
            f'BLOCKED unexpected version: {name}={installed[name]["version"]}')
    for name, plugin in zip(('pytest-json-report', 'pytest-asyncio', 'deepeval'), PLUGINS):
        assert plugin in installed[name]['plugins'], f'BLOCKED missing entrypoint: {plugin}'
    config = tmp_path / 'pytest.ini'
    config.write_text('[pytest]\nasyncio_mode = strict\n')
    (tmp_path / 'agent.py').write_text('def agent():\n    return 42\n')
    test = tmp_path / 'test_synthetic.py'
    test.write_text('import pytest\nfrom agent import agent\n'
                    'def test_agent():\n    assert agent() == 42\n'
                    '@pytest.mark.asyncio\nasync def test_async_agent():\n'
                    '    assert agent() == 42\n')
    report, attempts = tmp_path / 'report.json', tmp_path / 'blocked.jsonl'
    args = [PYTHON, '-I', '-c', BOOTSTRAP, str(attempts), '-c', str(config),
            '--confcutdir', str(tmp_path), '--noconftest', '-q',
            '--json-report', f'--json-report-file={report}']
    for plugin in PLUGINS:
        args += ['-p', plugin]
    result = subprocess.run([*args, str(test)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert not attempts.exists(), ('BLOCKED plugin network/process attempt: '
                                    + attempts.read_text() + '\nSUBPROCESS:\n'
                                    + result.stdout + result.stderr)
    assert result.returncode == 0, 'BLOCKED plugin load/run:\n' + result.stdout + result.stderr
    payload = json.loads(report.read_text())
    assert payload['exitcode'] == 0 and payload['summary']['passed'] == 2
    assert {test['outcome'] for test in payload['tests']} == {'passed'}
