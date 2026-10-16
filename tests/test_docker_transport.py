"""Phase-3 Docker transport tests (lane B).

Offline tests for the frozen transport interface: credential boundaries
per CLI with synthetic HOME/config/env only, setup cleanup, typed error
codes, exclusive setup logs, thin-wrapper forwarding and Node22 image
configuration. No real Docker, provider, auth store, secrets or dotenv
reads. Fake `docker` uses an absolute interpreter with fake-bin-only
PATH; no live image build/pull/run is claimed.
"""
import importlib.util
import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "docker_transport_under_test",
    str(ROOT / "pipeline" / "adapters" / "docker_transport.py"))
dt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dt)

TIMEOUT = 120
SECRET = "SENTINEL-CRED-9f8e7d6c"
IMAGE = "test-gen:local"

CLAUDE_ARGV = ["claude", "--dangerously-skip-permissions",
               "--model", "claude-sonnet-4-5",
               "-p", "PROMPT-PLACEHOLDER", "--output-format", "json"]
CODEX_ARGV = ["codex", "exec", "--json", "--yolo",
              "--model", "gpt-6-luna", "PROMPT-PLACEHOLDER"]
OPENCODE_ARGV = ["opencode", "run", "PROMPT-PLACEHOLDER",
                 "--model", "opencode/longcat-2.5-preview-free",
                 "--format", "json", "--auto", "/out"]

PROMPT = "Implement the following instructions in the current directory:\n\n# spec\n"

CRED_VARS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
             "CLAUDE_CODE_OAUTH_TOKEN", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN",
             "OPENAI_API_KEY", "GEMINI_API_KEY", "GEN_OPENCODE_API_KEY_ENV",
             "CLAUDE_CONFIG_DIR", "CODEX_HOME")


def write_fake_docker(path, manifest=None):
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "m = os.environ.get('FAKE_DOCKER_MANIFEST')\n"
        "if m:\n"
        "    open(m, 'a', encoding='utf-8').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "mode = os.environ.get('FAKE_DOCKER_MODE', 'inspect_ok')\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['image', 'inspect']:\n"
        "    sys.exit(0 if mode == 'inspect_ok' else 1)\n"
        "if args[:1] == ['build']:\n"
        "    sys.exit(0 if mode == 'build_ok' else 1)\n"
        "sys.exit(0)\n",
        encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolated process env: fake bin PATH, synthetic HOME, clean creds."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    spec = tmp_path / "specs" / "aaa" / "spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("# spec\n", encoding="utf-8")
    out = tmp_path / "gen" / "t" / "run_1" / "claude" / "aaa"
    out.mkdir(parents=True)
    tmproot = tmp_path / "tmproot"
    tmproot.mkdir()
    for var in CRED_VARS + ("GEN_IMAGE", "FAKE_DOCKER_MANIFEST",
                            "FAKE_DOCKER_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GEN_IMAGE", IMAGE)
    monkeypatch.setenv("SPEC2AGENT_SKIP_DOTENV", "1")
    monkeypatch.setattr(dt, "_default_temp_root", lambda: str(tmproot))
    manifest = tmp_path / "manifest.log"
    monkeypatch.setenv("FAKE_DOCKER_MANIFEST", str(manifest))
    return SimpleNamespace(bin=bindir, home=home, spec=spec, out=out,
                           tmproot=tmproot, manifest=manifest)


def claude_argv(prompt=PROMPT):
    return [a.replace("PROMPT-PLACEHOLDER", prompt) for a in CLAUDE_ARGV]


def codex_argv(prompt=PROMPT):
    return [a.replace("PROMPT-PLACEHOLDER", prompt) for a in CODEX_ARGV]


def opencode_argv(prompt=PROMPT, model="opencode/longcat-2.5-preview-free"):
    argv = [a.replace("PROMPT-PLACEHOLDER", prompt) for a in OPENCODE_ARGV]
    argv[argv.index("--model") + 1] = model
    return argv


def mounts_of(final):
    """Map container path -> host path for -v mounts in yielded argv."""
    mounts = {}
    items = final[final.index("docker") + 1:]
    for i, item in enumerate(items):
        if item == "-v" and i + 1 < len(items):
            host, container, _mode = items[i + 1].split(":")
            mounts[container] = host
    return mounts


def full_text(*paths):
    return "\n".join(Path(p).read_text(encoding="utf-8") for p in paths
                     if Path(p).exists())


# --- error type ----------------------------------------------------------------

def test_error_code_whitelist():
    for code in dt.ERROR_CODES:
        err = dt.DockerTransportError(code)
        assert err.code == code
        assert str(err) == code
    # Lenient construction for collector-side fallback mapping; the
    # transport itself only ever raises the agreed codes above.
    unknown = dt.DockerTransportError("bogus!!")
    assert unknown.code == "bogus!!" and str(unknown) == "bogus!!"


# --- credentials -----------------------------------------------------------------

def test_claude_env_forwarded_by_name_only(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    t = dt.DockerTransport(iso.spec)

    with t.prepare(claude_argv(), cli="claude", output_dir=iso.out) as final:
        text = " ".join(final)
        pairs = [final[i:i + 2] for i in range(len(final) - 1)
                 if final[i] == "-e"]
        assert ["-e", "ANTHROPIC_API_KEY"] in pairs
        assert SECRET not in text
        assert "CLAUDE_CONFIG_DIR=/home/agent/.claude" in text
        assert "CODEX_HOME" not in text
        mounts = mounts_of(final)
        assert set(mounts) == {"/in", "/out", "/home/agent"}
        assert mounts["/out"] == str(iso.out)
        assert Path(mounts["/in"], "spec.md").read_bytes() == b"# spec\n"
        assert Path(mounts["/in"], "prompt.txt").read_text() == PROMPT
        cfg_dir = Path(mounts["/home/agent"]) / ".claude"
        assert cfg_dir.is_dir()
        assert stat.S_IMODE(cfg_dir.stat().st_mode) == 0o700
    assert SECRET not in full_text(iso.manifest, *t.log_paths.values())
    assert list(iso.tmproot.iterdir()) == []


def test_claude_env_precedence_first_nonempty(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "other")
    t = dt.DockerTransport(iso.spec)
    with t.prepare(claude_argv(), cli="claude", output_dir=iso.out) as final:
        assert ["-e", "ANTHROPIC_API_KEY"] in [
            final[i:i + 2] for i in range(len(final) - 1) if final[i] == "-e"]
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    t2 = dt.DockerTransport(iso.spec)
    with t2.prepare(claude_argv(), cli="claude", output_dir=iso.out) as final:
        pairs = [final[i:i + 2] for i in range(len(final) - 1)
                 if final[i] == "-e"]
        assert ["-e", "ANTHROPIC_AUTH_TOKEN"] in pairs


def test_claude_file_snapshot_modes_and_safety(iso):
    write_fake_docker(iso.bin / "docker")
    src_dir = iso.home / ".claude"
    src_dir.mkdir()
    src = src_dir / ".credentials.json"
    src.write_bytes(b'{"token": "%s"}' % SECRET.encode())
    src_mode = stat.S_IMODE(src.stat().st_mode)
    t = dt.DockerTransport(iso.spec)

    with t.prepare(claude_argv(), cli="claude", output_dir=iso.out) as final:
        pairs = [final[i:i + 2] for i in range(len(final) - 1)
                 if final[i] == "-e"]
        assert not any(p[1] in CRED_VARS for p in pairs)
        mounts = mounts_of(final)
        host_home = Path(mounts["/home/agent"])
        snap = host_home / ".claude" / ".credentials.json"
        assert snap.read_bytes() == src.read_bytes()
        assert stat.S_IMODE(snap.stat().st_mode) == 0o600
        assert stat.S_IMODE(host_home.stat().st_mode) == 0o700
        assert SECRET not in " ".join(final)
    assert src.read_bytes().endswith(SECRET.encode() + b'"}')
    assert stat.S_IMODE(src.stat().st_mode) == src_mode
    assert SECRET not in full_text(iso.manifest, *t.log_paths.values())
    assert list(iso.tmproot.iterdir()) == []


def test_codex_env_and_file(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", SECRET)
    t = dt.DockerTransport(iso.spec)
    with t.prepare(codex_argv(), cli="codex", output_dir=iso.out) as final:
        pairs = [final[i:i + 2] for i in range(len(final) - 1)
                 if final[i] == "-e"]
        assert ["-e", "CODEX_ACCESS_TOKEN"] in pairs
        assert "CODEX_HOME=/home/agent/.codex" in " ".join(final)
        assert "CLAUDE_CONFIG_DIR" not in " ".join(final)
        assert SECRET not in " ".join(final)
        cfg_dir = Path(mounts_of(final)["/home/agent"]) / ".codex"
        assert cfg_dir.is_dir()
        assert stat.S_IMODE(cfg_dir.stat().st_mode) == 0o700
    monkeypatch.delenv("CODEX_ACCESS_TOKEN")
    src_dir = iso.home / ".codex"
    src_dir.mkdir()
    (src_dir / "auth.json").write_text(f'{{"k": "{SECRET}"}}',
                                       encoding="utf-8")
    t2 = dt.DockerTransport(iso.spec)
    with t2.prepare(codex_argv(), cli="codex", output_dir=iso.out) as final:
        mounts = mounts_of(final)
        snap = Path(mounts["/home/agent"]) / ".codex" / "auth.json"
        assert stat.S_IMODE(snap.stat().st_mode) == 0o600
        assert SECRET not in " ".join(final)
    assert SECRET not in full_text(iso.manifest, *t2.log_paths.values())


@pytest.mark.parametrize("cli,argv,cfg", [
    ("claude", claude_argv(), ".claude"),
    ("codex", codex_argv(), ".codex"),
])
def test_none_mode_creates_config_dir(iso, cli, argv, cfg):
    """Advertised config dirs exist writable even with no credentials."""
    write_fake_docker(iso.bin / "docker")
    t = dt.DockerTransport(iso.spec)
    with t.prepare(argv, cli=cli, output_dir=iso.out) as final:
        cfg_dir = Path(mounts_of(final)["/home/agent"]) / cfg
        assert cfg_dir.is_dir()
        assert stat.S_IMODE(cfg_dir.stat().st_mode) == 0o700
        assert list(cfg_dir.iterdir()) == []
    assert list(iso.tmproot.iterdir()) == []


def test_opencode_default_no_credentials(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    t = dt.DockerTransport(iso.spec)
    with t.prepare(opencode_argv(), cli="opencode",
                   output_dir=iso.out) as final:
        text = " ".join(final)
        assert "OPENAI_API_KEY" not in text
        assert "ANTHROPIC_API_KEY" not in text
        assert "GEMINI_API_KEY" not in text
        assert "CODEX_HOME" not in text and "CLAUDE_CONFIG_DIR" not in text
        assert SECRET not in text
        host_home = Path(mounts_of(final)["/home/agent"])
        assert list(host_home.iterdir()) == []
    assert SECRET not in full_text(iso.manifest, *t.log_paths.values())


def test_opencode_opt_in_valid(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    monkeypatch.setenv("GEN_OPENCODE_API_KEY_ENV", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    t = dt.DockerTransport(iso.spec)
    with t.prepare(opencode_argv(model="openai/gpt-x"), cli="opencode",
                   output_dir=iso.out) as final:
        pairs = [final[i:i + 2] for i in range(len(final) - 1)
                 if final[i] == "-e"]
        assert ["-e", "OPENAI_API_KEY"] in pairs
        assert SECRET not in " ".join(final)


@pytest.mark.parametrize("env_value,model,opt", [
    ("OPENAI_API_KEY", "anthropic/claude-x", "OPENAI_API_KEY"),
    ("OPENAI_API_KEY", "openai/gpt-x", "OPENAI_API_KEY"),
    ("SOME_OTHER_KEY", "openai/gpt-x", "SOME_OTHER_KEY"),
])
def test_opencode_opt_in_rejected(iso, monkeypatch, env_value, model, opt):
    write_fake_docker(iso.bin / "docker")
    monkeypatch.setenv("GEN_OPENCODE_API_KEY_ENV", opt)
    if env_value != "OPENAI_API_KEY" or model == "openai/gpt-x":
        monkeypatch.setenv(env_value, "" if model == "openai/gpt-x"
                           and opt == "OPENAI_API_KEY" else SECRET)
    else:
        monkeypatch.setenv(env_value, SECRET)
    t = dt.DockerTransport(iso.spec)
    with pytest.raises(dt.DockerTransportError) as exc:
        with t.prepare(opencode_argv(model=model), cli="opencode",
                       output_dir=iso.out):
            pass
    assert exc.value.code == "docker_auth_invalid"
    assert list(iso.tmproot.iterdir()) == []


# --- image / availability ----------------------------------------------------------

def test_image_present_no_build(iso):
    write_fake_docker(iso.bin / "docker")
    t = dt.DockerTransport(iso.spec)
    with t.prepare(claude_argv(), cli="claude", output_dir=iso.out):
        pass
    calls = [json.loads(line) for line in
             iso.manifest.read_text(encoding="utf-8").splitlines()]
    assert calls == [["image", "inspect", IMAGE]]
    assert list(iso.tmproot.iterdir()) == []


def test_missing_image_auto_build(iso, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_MODE", "build_ok")
    write_fake_docker(iso.bin / "docker")
    t = dt.DockerTransport(iso.spec)
    with t.prepare(claude_argv(), cli="claude", output_dir=iso.out):
        pass
    calls = [json.loads(line) for line in
             iso.manifest.read_text(encoding="utf-8").splitlines()]
    assert calls[0] == ["image", "inspect", IMAGE]
    assert calls[1][0] == "build"
    assert calls[1][calls[1].index("-f") + 1].endswith("docker/Dockerfile.gen")
    assert calls[1][calls[1].index("-t") + 1] == IMAGE
    assert Path(calls[1][-1]).name == "docker"
    assert not any(c[:1] == ["run"] for c in calls)


def test_build_failure_cleans_up(iso, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_MODE", "build_fail")
    write_fake_docker(iso.bin / "docker")
    t = dt.DockerTransport(iso.spec)
    with pytest.raises(dt.DockerTransportError) as exc:
        with t.prepare(claude_argv(), cli="claude", output_dir=iso.out):
            pass
    assert exc.value.code == "docker_build_failed"
    assert t.log_paths == {}
    assert list(iso.tmproot.iterdir()) == []


def test_docker_unavailable(iso):
    t = dt.DockerTransport(iso.spec)
    with pytest.raises(dt.DockerTransportError) as exc:
        with t.prepare(claude_argv(), cli="claude", output_dir=iso.out):
            pass
    assert exc.value.code == "docker_unavailable"
    assert list(iso.tmproot.iterdir()) == []


# --- validation / argv -----------------------------------------------------------------

@pytest.mark.parametrize("cli,argv", [
    ("copilot", ["copilot", "x"]),
    ("gemini", ["agy", "x"]),
    ("claude", []),
    ("claude", "not-a-list"),
    ("claude", ["claude", "-p", 42]),
])
def test_invalid_command_rejected(iso, cli, argv):
    write_fake_docker(iso.bin / "docker")
    t = dt.DockerTransport(iso.spec)
    with pytest.raises(dt.DockerTransportError) as exc:
        with t.prepare(argv, cli=cli, output_dir=iso.out):
            pass
    assert exc.value.code == "docker_setup_failed"


def test_yielded_argv_shape(iso):
    write_fake_docker(iso.bin / "docker")
    argv = claude_argv()
    t = dt.DockerTransport(iso.spec)
    with t.prepare(argv, cli="claude", output_dir=iso.out) as final:
        assert final[:3] == ["docker", "run", "--rm"]
        assert "--user" in final
        uid_gid = final[final.index("--user") + 1]
        assert uid_gid == f"{os.getuid()}:{os.getgid()}"
        assert final[final.index("--workdir") + 1] == "/out"
        img_idx = final.index(IMAGE)
        assert final[img_idx + 1:] == argv
        assert "--env-file" not in final
        for i, item in enumerate(final):
            if item == "-v":
                assert ":ro" in final[i + 1] or ":rw" in final[i + 1]
                host = final[i + 1].split(":")[0]
                assert "plano" not in host  # no stray repo-adjacent mounts


def test_no_repo_home_or_auth_mounts(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    t = dt.DockerTransport(iso.spec)
    with t.prepare(claude_argv(), cli="claude", output_dir=iso.out) as final:
        volumes = [final[i + 1] for i, item in enumerate(final[:-1])
                   if item == "-v"]
        assert len(volumes) == 3
        assert not any(v.startswith(str(ROOT)) for v in volumes)
        assert not any(v.startswith(os.environ["HOME"]) for v in volumes)
        assert not any(v.startswith("/auth") or ":/auth" in v for v in volumes)


@pytest.mark.parametrize("cli,argv", [
    ("claude", claude_argv()),
    ("codex", codex_argv()),
    ("opencode", opencode_argv()),
])
def test_prompt_staging_exact(iso, cli, argv):
    write_fake_docker(iso.bin / "docker")
    t = dt.DockerTransport(iso.spec)
    with t.prepare(argv, cli=cli, output_dir=iso.out) as final:
        in_dir = Path(mounts_of(final)["/in"])
        assert (in_dir / "spec.md").read_bytes() == b"# spec\n"
        assert (in_dir / "prompt.txt").read_text(encoding="utf-8") == PROMPT


# --- setup logs --------------------------------------------------------------------------

def test_cleanup_recovers_inaccessible_credentials_after_fake_cli(iso):
    write_fake_docker(iso.bin / "docker")
    src = iso.home / ".claude" / ".credentials.json"
    src.parent.mkdir()
    src.write_text(SECRET)
    t = dt.DockerTransport(iso.spec)
    with t.prepare(claude_argv(), cli="claude", output_dir=iso.out) as final:
        home = Path(mounts_of(final)["/home/agent"])
        cli = iso.bin / "lock-home"
        cli.write_text(
            f"#!{sys.executable}\nimport os, sys\n"
            "os.chmod(sys.argv[1] + '/.claude', 0)\n"
            "os.chmod(sys.argv[1], 0)\n")
        cli.chmod(0o755)
        assert subprocess.run([str(cli), str(home)], timeout=TIMEOUT).returncode == 0
    assert list(iso.tmproot.iterdir()) == []
    assert src.read_text() == SECRET


def test_unexpected_provisioning_failure_cleans_snapshot(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    src = iso.home / ".claude" / ".credentials.json"
    src.parent.mkdir()
    src.write_text(SECRET)
    t = dt.DockerTransport(iso.spec)
    provision = t._provision_credentials

    def fail_after_copy(*args):
        provision(*args)
        assert (args[2] / ".claude" / ".credentials.json").read_text() == SECRET
        raise RuntimeError(SECRET)

    monkeypatch.setattr(t, "_provision_credentials", fail_after_copy)
    with pytest.raises(dt.DockerTransportError) as exc:
        with t.prepare(claude_argv(), cli="claude", output_dir=iso.out):
            pytest.fail("setup must not yield")
    assert str(exc.value) == "docker_setup_failed"
    assert SECRET not in str(exc.value)
    assert list(iso.tmproot.iterdir()) == []
    assert t.log_paths == {}


def test_cleanup_failure_attempts_both_owned_trees(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    remove = dt._remove_owned_tree
    attempted = []

    def fail_first(path):
        attempted.append(path)
        if len(attempted) == 1:
            raise PermissionError(SECRET)
        remove(path)

    monkeypatch.setattr(dt, "_remove_owned_tree", fail_first)
    try:
        with pytest.raises(dt.DockerTransportError) as exc:
            with dt.DockerTransport(iso.spec).prepare(
                    claude_argv(), cli="claude", output_dir=iso.out):
                pass
        assert str(exc.value) == "docker_cleanup_failed"
        assert len(attempted) == 2
        assert attempted[0].name.startswith("gen-in-")
        assert attempted[1].name.startswith("gen-home-")
        assert not attempted[1].exists()
    finally:
        for path in iso.tmproot.iterdir():
            remove(path)


@pytest.mark.parametrize("replace_root", [False, True])
def test_cleanup_never_follows_outside_symlinks(iso, replace_root):
    write_fake_docker(iso.bin / "docker")
    outside = iso.home / "outside"
    outside.mkdir(mode=0o700)
    sentinel = outside / "sentinel"
    sentinel.write_text(SECRET)
    sentinel.chmod(0o400)
    outside.chmod(0o500)
    before = (outside.stat().st_mode, sentinel.stat().st_mode)
    with dt.DockerTransport(iso.spec).prepare(
            claude_argv(), cli="claude", output_dir=iso.out) as final:
        home = Path(mounts_of(final)["/home/agent"])
        if replace_root:
            (home / ".claude").rmdir()
            home.rmdir()
            home.symlink_to(outside, target_is_directory=True)
        else:
            (home / "link").symlink_to(outside, target_is_directory=True)
            (home / "file-link").symlink_to(sentinel)
    assert sentinel.read_text() == SECRET
    assert (outside.stat().st_mode, sentinel.stat().st_mode) == before
    assert list(iso.tmproot.iterdir()) == []


@pytest.mark.parametrize("exit_code,malformed,expected_error", [
    (0, False, "docker_cleanup_failed"),
    (9, False, "cli_exit_9"),
    (0, True, "parse_error"),
])
def test_real_cleanup_failure_retains_collection(
        iso, monkeypatch, exit_code, malformed, expected_error):
    """Use the real transport/collector; only Docker and deletion are faked."""
    monkeypatch.syspath_prepend(str(ROOT / "pipeline"))
    from adapters.base_adapter import BaseAdapter
    from adapters import docker_transport as transport_module

    stdout = (b"not json\r\n" if malformed else
              b'{"usage":{"input_tokens":12,"output_tokens":3,"cache_read_input_tokens":0,"cache_creation_input_tokens":0},"num_turns":1,"total_cost_usd":0.01}\r\n')
    stderr = b"\xff-cleanup-marker\n"
    docker = iso.bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if sys.argv[1:3] == ['image', 'inspect']: sys.exit(0)\n"
        f"sys.stdout.buffer.write({stdout!r})\n"
        f"sys.stderr.buffer.write({stderr!r})\n"
        f"sys.exit({exit_code})\n")
    docker.chmod(0o755)
    remove = transport_module._remove_owned_tree
    attempted = []

    def fail_first(path):
        attempted.append(path)
        if len(attempted) == 1:
            raise PermissionError(SECRET)
        remove(path)

    monkeypatch.setattr(transport_module, "_default_temp_root", lambda: str(iso.tmproot))
    monkeypatch.setattr(transport_module, "_remove_owned_tree", fail_first)
    adapter = BaseAdapter(iso.spec, iso.out)
    adapter.transport = transport_module.DockerTransport(iso.spec)
    try:
        adapter._collect(claude_argv(), cli="claude", model="test", tag="test")
        assert adapter.exit_code == exit_code
        assert adapter.error == expected_error
        assert "docker_cleanup_failed" in adapter.warnings
        assert adapter.parse_status == ("error" if malformed else "ok")
        assert adapter.usage["input_tokens"] == (None if malformed else 12)
        assert adapter.usage["process_duration_s"] is not None
        assert Path(adapter.log_paths["session.log"]).read_bytes() == stdout
        assert Path(adapter.log_paths["cli_stderr.log"]).read_bytes() == stderr
        assert SECRET not in str(adapter.warnings) + str(adapter.error)
        assert len(attempted) == 2 and not attempted[1].exists()
    finally:
        for path in iso.tmproot.iterdir():
            remove(path)


def test_setup_log_exclusive_and_safe(iso, monkeypatch):
    write_fake_docker(iso.bin / "docker")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    logs = iso.out / "logs"
    logs.mkdir()
    (logs / "docker_setup.log").write_bytes(b"OLD\n")
    t = dt.DockerTransport(iso.spec)
    with t.prepare(claude_argv(), cli="claude", output_dir=iso.out):
        pass
    assert set(t.log_paths) == {"docker_setup.log"}
    advertised = Path(t.log_paths["docker_setup.log"])
    assert advertised.name == "docker_setup-1.log"
    assert advertised.is_file()
    assert (logs / "docker_setup.log").read_bytes() == b"OLD\n"
    body = advertised.read_text(encoding="utf-8")
    assert SECRET not in body
    assert "ANTHROPIC_API_KEY" in body  # name only, never the value
    assert PROMPT not in body


# --- thin wrapper ----------------------------------------------------------------------------

STUB_STAGE = """#!{python}
import json, os, sys
capture = os.environ["STUB_CAPTURE"]
payload = {{"argv": sys.argv[1:], "env": {{k: os.environ.get(k) for k in (
    "BENCH_DIR", "GENERATED_DIR", "EXPERIMENTS_RESULTS", "GEN_IMAGE")}},
    "backend_marker": "stub-stage01"}}
open(capture, "w", encoding="utf-8").write(json.dumps(payload))
"""


def install_wrapper_tree(tmp_path):
    tree = tmp_path / "wtree"
    (tree / "docker").mkdir(parents=True)
    (tree / "pipeline").mkdir(parents=True)
    (tree / "docker" / "docker_run_gen.sh").write_bytes(
        (ROOT / "docker" / "docker_run_gen.sh").read_bytes())
    (tree / "docker" / "docker_run_gen.sh").chmod(0o755)
    stub = tree / "pipeline" / "01_generate.py"
    stub.write_text(STUB_STAGE.format(python=sys.executable), encoding="utf-8")
    stub.chmod(0o755)
    return tree


def wrapper_env(tmp_path):
    env = dict(os.environ)
    env["SPEC2AGENT_SKIP_DOTENV"] = "1"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, target in (("bash", "/bin/bash"), ("dirname", "/usr/bin/dirname")):
        (bindir / name).symlink_to(target)
    python = bindir / "python3"
    python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
    python.chmod(0o755)
    env["PATH"] = str(bindir)
    home = tmp_path / "home"
    home.mkdir()
    env["HOME"] = str(home)
    for var in CRED_VARS:
        env.pop(var, None)
    for var in ("BENCH_DIR", "GENERATED_DIR", "EXPERIMENTS_RESULTS",
                "GEN_IMAGE", "STUB_CAPTURE"):
        env.pop(var, None)
    return env


def test_wrapper_forwards_to_stage(tmp_path):
    tree = install_wrapper_tree(tmp_path)
    env = wrapper_env(tmp_path)
    capture = tmp_path / "capture.json"
    env["STUB_CAPTURE"] = str(capture)
    proc = subprocess.run(
        [str(tree / "docker" / "docker_run_gen.sh"),
         "--llm", "claude", "--agent", "aaa", "--run", "2",
         "--experiment", "by_tool", "--model", "claude-opus-4-6"],
        env=env, capture_output=True, text=True, timeout=TIMEOUT,
        cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert payload["argv"] == ["--backend", "docker", "--llm", "claude",
                               "--agent", "aaa", "--run", "2",
                               "--experiment", "by_tool",
                               "--model", "claude-opus-4-6"]
    assert payload["env"]["BENCH_DIR"] == str(tree / "specs")
    assert payload["env"]["GENERATED_DIR"] == str(tree / "generated_agents")
    assert payload["env"]["EXPERIMENTS_RESULTS"] == \
        str(tree / "experiments_results")
    assert payload["env"]["GEN_IMAGE"] == "spec2agent-gen:infra"


def test_wrapper_gen_image_passthrough(tmp_path):
    tree = install_wrapper_tree(tmp_path)
    env = wrapper_env(tmp_path)
    capture = tmp_path / "capture.json"
    env["STUB_CAPTURE"] = str(capture)
    env["GEN_IMAGE"] = "custom:img"
    proc = subprocess.run(
        [str(tree / "docker" / "docker_run_gen.sh"),
         "--llm", "codex", "--agent", "aaa"],
        env=env, capture_output=True, text=True, timeout=TIMEOUT,
        cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert payload["env"]["GEN_IMAGE"] == "custom:img"
    assert "--model" not in payload["argv"]


@pytest.mark.parametrize("args", [
    ["--agent", "aaa"],
    ["--llm", "claude"],
    ["--llm", "copilot", "--agent", "aaa"],
    ["--llm", "gemini", "--agent", "aaa"],
    ["--llm", "bogus", "--agent", "aaa"],
    ["--llm", "claude", "--agent", "aaa", "--nope", "x"],
])
def test_wrapper_validation_minimal(tmp_path, args):
    tree = install_wrapper_tree(tmp_path)
    env = wrapper_env(tmp_path)
    env["STUB_CAPTURE"] = str(tmp_path / "capture.json")
    proc = subprocess.run(
        [str(tree / "docker" / "docker_run_gen.sh"), *args],
        env=env, capture_output=True, text=True, timeout=TIMEOUT,
        cwd=str(tmp_path))
    assert proc.returncode == 2


# --- image configuration -----------------------------------------------------------------------

def test_dockerfile_gen_node22_and_pins():
    src = (ROOT / "docker" / "Dockerfile.gen").read_text(encoding="utf-8")
    assert "setup_22.x" in src
    assert "setup_20.x" not in src
    assert "CLAUDE_CODE_VERSION=2.1.299" in src
    assert "CODEX_VERSION=0.162.1" in src
    assert "OPENCODE_VERSION=1.19.0" in src
    assert "python:3.11-slim" in src
    assert "WORKDIR /out" in src
    assert "USER agent" in src
    assert not [line for line in src.splitlines()
                if line.strip().upper().startswith("COPY")]
    # Two images preserved: the evaluation image is a separate file.
    assert (ROOT / "docker" / "Dockerfile.eval").is_file()
