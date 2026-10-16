"""Phase-3 core lane A: --backend local|docker + --model tests.

Fake transport/context plus a fake docker executable, all under a
fake-bin-only PATH with absolute interpreter shebangs. The real
``adapters.docker_transport`` module does not exist yet (transport lane
B); tests inject a fake module exporting the SAME DockerTransportError
class the collector catches, or force ImportError deterministically.
No dotenv, no network, no real Docker/CLI/credentials.
"""
import importlib.util
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from adapters.base_adapter import DockerTransportError, transport_error_token  # noqa: E402
from adapters.claude import ClaudeCodeAdapter  # noqa: E402
from adapters.codex import CodexAdapter  # noqa: E402
from adapters.open_code import OpenCodeAdapter  # noqa: E402

SENTINEL = "BACKEND-SENTINEL-5c1e88aa"

CLAUDE_OK = {
    "result": "agent code here",
    "usage": {
        "input_tokens": 100,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 10,
        "output_tokens": 50,
    },
    "num_turns": 3,
    "total_cost_usd": 0.01,
}

CODEX_OK = [
    {"type": "turn.completed", "thread_id": "thr-1", "usage": {
        "input_tokens": 200, "output_tokens": 80,
        "cached_input_tokens": 30, "cache_write_input_tokens": 5,
        "reasoning_output_tokens": 7}},
]

OPENCODE_OK = [
    {"type": "step_finish", "sessionID": "s1",
     "part": {"id": "p1", "messageID": "m1",
              "tokens": {"input": 12, "output": 6, "reasoning": 1,
                         "cache": {"read": 2, "write": 3}, "total": 19},
              "cost": 0.004}},
]


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setenv("SPEC2AGENT_SKIP_DOTENV", "1")


def _stage():
    """Import pipeline/01_generate.py in-process (dotenv already bypassed)."""
    name = "phase3_gen01"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "pipeline" / "01_generate.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _write_exe(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _isolated_path(monkeypatch, bindir: Path) -> None:
    monkeypatch.setenv("PATH", str(bindir))


def _spec_and_root(tmp_path: Path, name: str = "agent1"):
    spec_dir = tmp_path / "specs" / name
    spec_dir.mkdir(parents=True)
    (spec_dir / "spec.md").write_text("# fake spec\nDo the thing.\n", encoding="utf-8")
    return spec_dir, tmp_path / "gen"


def _install_local_claude(bindir: Path, argv_log: Path | None = None) -> None:
    lines = [f"#!{sys.executable}", "import json, os, sys"]
    if argv_log is not None:
        lines.append(f"open({str(argv_log)!r}, 'w').write('\\n'.join(sys.argv))")
    lines += [f"sys.stdout.buffer.write({json.dumps(CLAUDE_OK).encode()!r})",
              "sys.stderr.buffer.write(b'')",
              "open('agent.py', 'w').write('# fake\\n')",
              "sys.exit(0)"]
    _write_exe(bindir / "claude", "\n".join(lines) + "\n")


def _install_local_opencode(bindir: Path, argv_log: Path | None = None) -> None:
    stdout = b"\n".join(json.dumps(line).encode() for line in OPENCODE_OK) + b"\n"
    lines = [f"#!{sys.executable}", "import sys"]
    if argv_log is not None:
        lines.append(f"open({str(argv_log)!r}, 'w').write('\\n'.join(sys.argv))")
    lines += [f"sys.stdout.buffer.write({stdout!r})",
              "sys.stderr.buffer.write(b'')",
              "open('agent.py', 'w').write('# fake\\n')",
              "sys.exit(0)"]
    _write_exe(bindir / "opencode", "\n".join(lines) + "\n")


def _install_fake_docker(bindir: Path, *, stdout: bytes, stderr: bytes = b"",
                         exit_code: int = 0, argv_log: Path | None = None,
                         name: str = "docker") -> Path:
    lines = [f"#!{sys.executable}", "import sys"]
    if argv_log is not None:
        lines.append(f"open({str(argv_log)!r}, 'w').write('\\n'.join(sys.argv))")
    lines += [f"sys.stdout.buffer.write({bytes(stdout)!r})",
              f"sys.stderr.buffer.write({bytes(stderr)!r})",
              "open('agent.py', 'w').write('# fake\\n')",
              f"sys.exit({exit_code})"]
    target = bindir / name
    _write_exe(target, "\n".join(lines) + "\n")
    return target


class FakeTransport:
    """Stand-in for B's DockerTransport: config via class behavior dict."""

    behavior: dict = {}
    instances: list = []

    def __init__(self, spec_file):
        self.spec_file = Path(spec_file)
        self.log_paths: dict = {}
        self.entered = False
        self.cleaned = False
        self.seen: dict = {}
        type(self).instances.append(self)

    @contextmanager
    def prepare(self, argv, *, cli, output_dir):
        cfg = type(self).behavior
        self.entered = True
        self.seen = {"argv": list(argv), "cli": cli,
                     "output_dir": str(output_dir)}
        logs = Path(output_dir) / "logs"
        setup = logs / "docker_setup.log"
        # Decoy numerics: must never leak into parsed usage.
        fd = os.open(str(setup), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("docker setup ok\ninspect tokens=999999 build_ms=42\n")
        self.log_paths["docker_setup.log"] = str(setup)
        try:
            if cfg.get("fail_code") is not None:
                raise DockerTransportError(cfg["fail_code"])
            yield [str(cfg["docker_bin"]), "run", "--workdir", "/out",
                   "fake-image"] + [str(a) for a in argv]
        finally:
            self.cleaned = True


def _inject_fake_transport(monkeypatch, transport_cls=FakeTransport):
    """Point the stage at a fake transport (module attr, auto-undone)."""
    monkeypatch.setattr(_stage(), "DockerTransport", transport_cls)


def _configure(monkeypatch, **cfg):
    monkeypatch.setattr(FakeTransport, "behavior", dict(cfg))
    monkeypatch.setattr(FakeTransport, "instances", [])


# --- local behavior preserved -------------------------------------------

def test_local_default_unchanged(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_local_claude(bindir)
    spec_dir, root = _spec_and_root(tmp_path)
    adapter_probe = ClaudeCodeAdapter(spec_dir / "spec.md", tmp_path / "probe")
    assert adapter_probe.transport is None
    result = gen01.generate("claude", spec_dir, root)
    assert result["status"] == "ok"
    assert result["error"] is None
    assert result["exit_code"] == 0
    assert result["usage"]["input_tokens"] == 130
    assert result["parse_status"] == "ok"


def test_model_override_preserves_alias_effort(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    argv_log = tmp_path / "argv.txt"
    _install_local_claude(bindir, argv_log=argv_log)
    spec_dir, root = _spec_and_root(tmp_path)
    before = dict(gen01.ADAPTERS["claude-opus-4-6-low"][1])
    result = gen01.generate("claude-opus-4-6-low", spec_dir, root,
                            model="custom-m")
    assert result["status"] == "ok"
    lines = argv_log.read_text(encoding="utf-8").splitlines()
    assert "--model" in lines and "custom-m" in lines
    assert "--effort" in lines and "low" in lines  # alias effort preserved
    assert gen01.ADAPTERS["claude-opus-4-6-low"][1] == before  # registry copy


# --- docker success paths ------------------------------------------------

def test_docker_claude_success(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    argv_log = tmp_path / "docker_argv.txt"
    docker_bin = _install_fake_docker(
        bindir, stdout=json.dumps(CLAUDE_OK).encode(), argv_log=argv_log)
    _inject_fake_transport(monkeypatch)
    _configure(monkeypatch, docker_bin=docker_bin)
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("claude", spec_dir, root, backend="docker")
    assert result["status"] == "ok"
    assert result["exit_code"] == 0
    assert result["usage"]["input_tokens"] == 130
    assert result["parse_status"] == "ok"
    assert "999999" not in json.dumps(result["usage"])  # no setup parsing
    assert "999999" not in json.dumps(result["warnings"])
    transport = FakeTransport.instances[0]
    assert transport.entered and transport.cleaned
    assert transport.seen["cli"] == "claude"
    assert transport.seen["argv"][0] == "claude"  # canonical argv passed in
    observed = argv_log.read_text(encoding="utf-8").splitlines()
    assert observed[1:5] == ["run", "--workdir", "/out", "fake-image"]
    assert "claude" in observed and "--output-format" in observed
    setup_path = Path(result["log_paths"]["docker_setup.log"])
    assert setup_path.is_file() and "999999" in setup_path.read_text()
    assert Path(result["log_paths"]["session.log"]).is_file()


def test_docker_exit9_preserves_usage(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    stdout = b"\n".join(json.dumps(line).encode() for line in CODEX_OK) + b"\n"
    docker_bin = _install_fake_docker(bindir, stdout=stdout, exit_code=9)
    _inject_fake_transport(monkeypatch)
    _configure(monkeypatch, docker_bin=docker_bin)
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("codex", spec_dir, root, backend="docker")
    assert result["status"] == "error"
    assert "cli_exit_9" in result["error"]
    assert result["exit_code"] == 9
    assert result["usage"]["input_tokens"] == 200
    assert Path(result["generated_file"]).is_file()  # artifact retained
    assert FakeTransport.instances[0].cleaned


def test_docker_malformed_run_stdout(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    docker_bin = _install_fake_docker(bindir, stdout=b"not usage\n")
    _inject_fake_transport(monkeypatch)
    _configure(monkeypatch, docker_bin=docker_bin)
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("opencode", spec_dir, root, backend="docker")
    assert result["status"] == "error"
    assert "parse_error" in result["error"]
    assert result["usage"]["input_tokens"] is None


# --- docker failure paths -------------------------------------------------

def test_docker_setup_failure_reports_code(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _inject_fake_transport(monkeypatch)
    _configure(monkeypatch, docker_bin="/nonexistent",
               fail_code="docker_setup_failed")
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("claude", spec_dir, root, backend="docker")
    assert result["status"] == "error"
    assert "docker_setup_failed" in result["error"]
    assert "Traceback" not in result["error"]
    assert result["exit_code"] is None
    assert result["usage"]["input_tokens"] is None
    assert not Path(result["generated_file"]).exists()  # no artifact
    transport = FakeTransport.instances[0]
    assert transport.cleaned  # context cleanup ran
    assert Path(result["log_paths"]["docker_setup.log"]).is_file()


def test_docker_unknown_code_falls_back(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _inject_fake_transport(monkeypatch)
    # Final transport constructor is lenient: unknown codes construct and
    # the collector remaps them through its own fallback.
    _configure(monkeypatch, docker_bin="/nonexistent",
               fail_code="bogus!!")
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("claude", spec_dir, root, backend="docker")
    assert result["status"] == "error"
    assert "docker_setup_failed" in result["error"]
    assert "bogus" not in result["error"]


def test_docker_transport_ctor_failure_keeps_code(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)

    class CtorBoom:
        def __init__(self, spec_file):
            raise DockerTransportError("docker_auth_invalid")

    _inject_fake_transport(monkeypatch, transport_cls=CtorBoom)
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("codex", spec_dir, root, backend="docker")
    assert result["status"] == "error"
    assert result["error"] == "docker_auth_invalid"


def test_docker_rejects_copilot_and_gemini(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    spec_dir, root = _spec_and_root(tmp_path)
    for llm in ("copilot", "gemini"):
        result = gen01.generate(llm, spec_dir, root, backend="docker")
        assert result["status"] == "error"
        assert result["error"] == "docker_unsupported_adapter"
    # Rejected before reservation: no output directories created.
    assert not (root / "copilot").exists()
    assert not (root / "gemini").exists()


def test_docker_binary_missing_unavailable(tmp_path, monkeypatch):
    """Actual docker-unavailable runtime: no `docker` binary on PATH.

    The real transport maps the missing binary to the typed code; the
    entry stays a canonical operational error with no raw exception.
    """
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()  # empty: no docker binary, no injection either
    _isolated_path(monkeypatch, bindir)
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("claude", spec_dir, root, backend="docker")
    assert result["status"] == "error"
    assert "docker_unavailable" in result["error"]
    assert "Traceback" not in result["error"]
    assert result["exit_code"] is None


def test_transport_cleanup_on_run_error(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    spec_dir = tmp_path / "s"
    spec_dir.mkdir()
    (spec_dir / "spec.md").write_text("# s\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    adapter = ClaudeCodeAdapter(spec_dir / "spec.md", out)
    transport = FakeTransport(tmp_path / "s" / "spec.md")
    FakeTransport.behavior = {"docker_bin": "/nonexistent-docker-binary"}
    adapter.transport = transport
    assert adapter.run() == str(out)  # returns, never raises
    assert adapter.exit_code is None
    assert "spawn_failed" in (adapter.error or "")
    assert transport.cleaned


class CleanupFailTransport(FakeTransport):
    """Fake whose context exit raises after a completed collection."""

    @contextmanager
    def prepare(self, argv, *, cli, output_dir):
        with FakeTransport.prepare(self, argv, cli=cli,
                                   output_dir=output_dir) as run_argv:
            yield run_argv
        raise DockerTransportError("docker_persistence_failed")


def test_transport_cleanup_failure_keeps_evidence(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    stdout = b"\n".join(json.dumps(line).encode() for line in CODEX_OK) + b"\n"
    docker_bin = _install_fake_docker(bindir, stdout=stdout, exit_code=9)
    _inject_fake_transport(monkeypatch, transport_cls=CleanupFailTransport)
    _configure(monkeypatch, docker_bin=docker_bin)
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("codex", spec_dir, root, backend="docker")
    assert result["status"] == "error"
    assert result["error"] == "cli_exit_9"  # process outcome not masked
    assert "docker_persistence_failed" in result["warnings"]
    assert result["exit_code"] == 9
    assert result["usage"]["input_tokens"] == 200
    assert Path(result["log_paths"]["session.log"]).read_bytes() == stdout
    assert CleanupFailTransport.instances[0].cleaned


def test_transport_error_token_unit():
    assert transport_error_token(DockerTransportError("docker_build_failed")) \
        == "docker_build_failed"
    assert transport_error_token(types.SimpleNamespace(code="bogus!!")) \
        == "docker_setup_failed"
    assert transport_error_token(types.SimpleNamespace(code=None)) \
        == "docker_setup_failed"
    assert transport_error_token(ValueError("x")) == "docker_setup_failed"
    assert transport_error_token(None) == "docker_setup_failed"


# --- opencode execution dir -----------------------------------------------

def test_opencode_execution_dir_local_and_docker(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    local_argv = tmp_path / "local_argv.txt"
    _install_local_opencode(bindir, argv_log=local_argv)
    docker_argv = tmp_path / "docker_argv.txt"
    stdout = b"\n".join(json.dumps(line).encode() for line in OPENCODE_OK) + b"\n"
    docker_bin = _install_fake_docker(bindir, stdout=stdout,
                                      argv_log=docker_argv)
    _inject_fake_transport(monkeypatch)
    _configure(monkeypatch, docker_bin=docker_bin)

    spec_dir, root = _spec_and_root(tmp_path, name="local-agent")
    result = gen01.generate("opencode", spec_dir, root, backend="local")
    assert result["status"] == "ok"
    local_lines = local_argv.read_text(encoding="utf-8").splitlines()
    assert local_lines[local_lines.index("--auto") + 1] == \
        str(root / "opencode" / "local-agent")

    spec_dir2, root2 = _spec_and_root(tmp_path, name="docker-agent")
    result2 = gen01.generate("opencode", spec_dir2, root2, backend="docker")
    assert result2["status"] == "ok"
    docker_lines = docker_argv.read_text(encoding="utf-8").splitlines()
    assert docker_lines[docker_lines.index("--auto") + 1] == "/out"


def test_opencode_auto_literal_out_under_host_resolve(tmp_path, monkeypatch):
    """Gate3 finding 2: host-resolved /out must not leak into --auto."""
    gen01 = _stage()
    target = tmp_path / "simulated-host-out-target"
    target.mkdir()
    real_resolve = Path.resolve

    def fake_resolve(self, *args, **kwargs):
        if str(self) == "/out":
            return target
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    assert Path("/out").resolve() == target  # host models the symlink

    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    local_argv = tmp_path / "local_argv.txt"
    _install_local_opencode(bindir, argv_log=local_argv)
    docker_argv = tmp_path / "docker_argv.txt"
    stdout = b"\n".join(json.dumps(line).encode() for line in OPENCODE_OK) + b"\n"
    docker_bin = _install_fake_docker(bindir, stdout=stdout,
                                      argv_log=docker_argv)
    _inject_fake_transport(monkeypatch)
    _configure(monkeypatch, docker_bin=docker_bin)

    spec_dir, root = _spec_and_root(tmp_path, name="local-agent")
    assert gen01.generate("opencode", spec_dir, root,
                          backend="local")["status"] == "ok"
    local_lines = local_argv.read_text(encoding="utf-8").splitlines()
    assert local_lines[local_lines.index("--auto") + 1] == \
        str((root / "opencode" / "local-agent").resolve())

    spec_dir2, root2 = _spec_and_root(tmp_path, name="docker-agent")
    assert gen01.generate("opencode", spec_dir2, root2,
                          backend="docker")["status"] == "ok"
    docker_lines = docker_argv.read_text(encoding="utf-8").splitlines()
    assert docker_lines[docker_lines.index("--auto") + 1] == "/out"


# --- real transport compatibility --------------------------------------------

_CRED_ENV = (
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_API_KEY", "CODEX_ACCESS_TOKEN",
    "GEN_OPENCODE_API_KEY_ENV", "CLAUDE_CONFIG_DIR", "CODEX_HOME",
)


def test_real_transport_docker_success(tmp_path, monkeypatch):
    """Core lane against B's final transport: fake `docker` binary only.

    No sys.modules injection here, so the stage imports the real
    DockerTransport. Credential env is scrubbed and HOME is synthetic, so
    only credential-less mode can engage; the fake docker answers image
    inspect and emits canonical Claude JSON on run.
    """
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    for name in _CRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "synth-home"))
    monkeypatch.setenv("GEN_IMAGE", "fake-image")
    argv_log = tmp_path / "docker_argv.txt"
    _write_exe(bindir / "docker", f"#!{sys.executable}\n"
               "import os, sys\n"
               f"open({str(argv_log)!r}, 'w').write('\\n'.join(sys.argv))\n"
               "if sys.argv[1:2] == ['image']:\n"
               "    sys.exit(0)\n"
               "if sys.argv[1:2] == ['build']:\n"
               "    sys.exit(0)\n"
               "assert sys.argv[1:2] == ['run'], sys.argv\n"
               f"assert os.path.abspath(os.getcwd()) != {str(ROOT)!r}, "
               "'refusing repo-root artifact'\n"
               f"sys.stdout.buffer.write({json.dumps(CLAUDE_OK).encode()!r})\n"
               "sys.stderr.buffer.write(b'')\n"
               "open('agent.py', 'w').write('# fake\\n')\n"
               "sys.exit(0)\n")
    spec_dir, root = _spec_and_root(tmp_path)
    result = gen01.generate("claude", spec_dir, root, backend="docker")
    assert result["status"] == "ok"
    assert result["exit_code"] == 0
    assert result["usage"]["input_tokens"] == 130
    assert result["parse_status"] == "ok"
    setup = Path(result["log_paths"]["docker_setup.log"])
    assert setup.is_file()
    observed = argv_log.read_text(encoding="utf-8").splitlines()
    assert observed[1] == "run"
    assert "--workdir" in observed and "/out" in observed
    assert "fake-image" in observed and "claude" in observed
    for secret in _CRED_ENV[:5]:
        assert secret not in observed
    assert not (ROOT / "agent.py").exists()  # never litter the repo root


# --- stage flag plumbing ----------------------------------------------------

def _stage_env(monkeypatch, tmp_path):
    bench = tmp_path / "specs" / "agent1"
    bench.mkdir(parents=True)
    (bench / "spec.md").write_text("# spec\n", encoding="utf-8")
    monkeypatch.setenv("BENCH_DIR", str(tmp_path / "specs"))
    monkeypatch.setenv("GENERATED_DIR", str(tmp_path / "gen"))
    monkeypatch.setenv("EXPERIMENTS_RESULTS", str(tmp_path / "results"))


def test_main_local_backend_ok(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_local_claude(bindir)
    _stage_env(monkeypatch, tmp_path)
    assert gen01.main(["--llm", "claude", "--run", "1",
                       "--experiment", "t"]) is None  # success: no exit
    reports = list((tmp_path / "results" / "t" / "run_1").glob("01_generation_*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["results"][0]["status"] == "ok"


def test_main_docker_backend_reports_error(tmp_path, monkeypatch):
    gen01 = _stage()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _inject_fake_transport(monkeypatch)
    _configure(monkeypatch, docker_bin="/nonexistent",
               fail_code="docker_setup_failed")
    _stage_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        gen01.main(["--llm", "claude", "--run", "1", "--experiment", "t",
                    "--backend", "docker"])
    assert exc.value.code == 1  # report written first, then operational nonzero
    reports = list((tmp_path / "results" / "t" / "run_1").glob("01_generation_*.json"))
    assert len(reports) == 1
    entry = json.loads(reports[0].read_text())["results"][0]
    assert entry["status"] == "error"
    assert "docker_setup_failed" in entry["error"]
