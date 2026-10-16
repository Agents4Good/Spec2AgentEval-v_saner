"""Phase-3 Docker generation end-to-end tests (integration/docs lane).

Real chain: ``docker_run_gen.sh`` wrapper -> stage01 ``--backend docker``
-> real ``DockerTransport`` -> ``_collect`` subprocess -> host parser ->
canonical report. The only fake is the ``docker`` executable (absolute
interpreter shebang, fake-bin-only PATH with individually symlinked
shell/python helpers). Synthetic HOME/config, scrubbed credential env
plus per-test sentinels; never real credentials/.env/installed docker.
No image build, provider auth, or paid calls.
"""
import concurrent.futures
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "docker" / "docker_run_gen.sh"

TIMEOUT = 180
SECRET = "SENTINEL-E2E-7d3c9a"
CRED_SECRET = "SENTINEL-CRED-e2e51f"

PROMPT = "Implement the following instructions in the current directory:\n\n# spec\n"

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
    {"type": "thread.started", "thread_id": "thr-1"},
    {"type": "turn.completed", "thread_id": "thr-1", "usage": {
        "input_tokens": 200,
        "output_tokens": 80,
        "cached_input_tokens": 30,
        "cache_write_input_tokens": 5,
        "reasoning_output_tokens": 10,
    }},
]

CODEX_PARTIAL = {"type": "turn.completed", "thread_id": "thr-1", "usage": {
    "input_tokens": 200, "cached_input_tokens": 30}}

OPENCODE_OK = {
    "type": "step_finish",
    "sessionID": "ses-1",
    "part": {
        "id": "prt-1",
        "messageID": "msg-1",
        "tokens": {
            "input": 300,
            "output": 90,
            "reasoning": 5,
            "cache": {"read": 15, "write": 7},
            "total": 402,
        },
        "cost": 0.02,
    },
}

SCRUB_VARS = (
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY",
    "GEN_OPENCODE_API_KEY_ENV", "CLAUDE_CONFIG_DIR", "CODEX_HOME",
    "GEN_IMAGE", "FAKE_DOCKER_MANIFEST", "FAKE_DOCKER_MODE", "FAKE_RUN_EXIT",
    "FAKE_RUN_STDOUT", "FAKE_RUN_STDERR", "FAKE_RUN_ARTIFACT",
    "SONAR_HOST_URL", "SONAR_LOGIN", "SONAR_PASSWORD", "USER_TOKEN",
    "SONAR_COMMAND",
)

FAKE_DOCKER = """#!{python}
import json, os, sys

FIXTURES = json.loads(os.environ.get("FAKE_FIXTURES", "{}"))

def manifest(entry):
    m = os.environ.get("FAKE_DOCKER_MANIFEST")
    if m:
        with open(m, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\\n")

args = sys.argv[1:]
manifest(args)
mode = os.environ.get("FAKE_DOCKER_MODE", "ok")

if args[:2] == ["image", "inspect"]:
    sys.exit(0 if mode in ("ok",) else 1)
if args[:1] == ["build"]:
    sys.exit(0 if mode in ("build_ok", "missing_image") else 1)

if args[:1] == ["run"]:
    out = None
    for i, a in enumerate(args):
        if a == "-v" and args[i + 1].endswith(":/out:rw"):
            out = args[i + 1].split(":")[0]
    llm = os.path.basename(os.path.dirname(out or "")) if out else ""
    if "claude" in llm:
        key = "claude"
    elif "codex" in llm:
        key = "codex"
    elif "opencode" in llm:
        key = "opencode"
    else:
        key = None
    if key is not None and out:
        os.makedirs(out, exist_ok=True)
        if os.environ.get("FAKE_RUN_ARTIFACT", "yes") == "yes":
            with open(os.path.join(out, "agent.py"), "w",
                      encoding="utf-8") as f:
                f.write("# generated agent\\n")
        stdout_mode = os.environ.get("FAKE_RUN_STDOUT", "valid")
        if stdout_mode == "malformed":
            sys.stdout.write("NOT JSON {{{ broken\\n")
        elif stdout_mode == "partial":
            sys.stdout.write(json.dumps(FIXTURES["partial"]) + "\\n")
        elif stdout_mode == "crlf":
            sys.stdout.write("\\r\\n".join(
                json.dumps(line) for line in FIXTURES["codex"]) + "\\r\\n")
        elif stdout_mode == "empty":
            pass
        else:
            payload = FIXTURES[key]
            if isinstance(payload, list):
                sys.stdout.write("\\n".join(
                    json.dumps(line) for line in payload) + "\\n")
            else:
                sys.stdout.write(json.dumps(payload) + "\\n")
        stderr_mode = os.environ.get("FAKE_RUN_STDERR", "empty")
        if stderr_mode == "marker":
            sys.stderr.write("container marker\\n")
        elif stderr_mode == "invalid":
            sys.stderr.buffer.write(b"\\xff\\xfe-bad\\n")
        elif stderr_mode == "secret":
            sys.stderr.write("leak-%s\\n" % os.environ.get("SECRET", ""))
    sys.exit(int(os.environ.get("FAKE_RUN_EXIT", "0")))

sys.exit(0)
"""


def _stage_python():
    """Project interpreter for stage01 subprocesses (dotenv self-contained)."""
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def base_env(tmp_path, monkeypatch):
    """Isolated env: tmp roots, synthetic HOME, scrubbed secrets, fake PATH."""
    for var in SCRUB_VARS:
        monkeypatch.delenv(var, raising=False)
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    link = bindir / "python3"
    # Fresh exec-wrapper (never a symlink): a symlinked venv interpreter
    # loses its prefix, so re-exec the absolute test interpreter instead.
    link.write_text(
        f"#!/bin/sh\nexec {shlex.quote(_stage_python())} \"$@\"\n",
        encoding="utf-8")
    link.chmod(0o755)
    for helper in ("bash", "dirname"):
        target = shutil.which(helper, path=os.pathsep.join(
            [os.path.dirname(sys.executable), "/usr/bin", "/bin"]))
        if target:
            link = bindir / helper
            if not link.exists():
                os.symlink(target, link)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    bench = tmp_path / "specs"
    env = {
        "PATH": str(bindir),
        "HOME": str(home),
        "SPEC2AGENT_SKIP_DOTENV": "1",
        "BENCH_DIR": str(bench),
        "GENERATED_DIR": str(tmp_path / "generated_agents"),
        "EXPERIMENTS_RESULTS": str(tmp_path / "experiments_results"),
        "GEN_IMAGE": "test-gen:e2e",
        "TERM": "dumb",
    }
    for var in ("SYSTEMROOT", "SYSTEMDRIVE", "TMPDIR", "TEMP", "TMP",
                "LANG", "LC_ALL"):
        if var in os.environ:
            env[var] = os.environ[var]
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return dict(env)


def write_spec(bench, name):
    agent_dir = bench / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "spec.md").write_text("# spec\n", encoding="utf-8")
    return agent_dir


def install_fake_docker(bindir):
    path = bindir / "docker"
    path.write_text(FAKE_DOCKER.replace("{python}", sys.executable),
                    encoding="utf-8")
    path.chmod(0o755)
    return path


def run_wrapper(env, tmp_path, *args):
    return subprocess.run(
        [str(WRAPPER), *args], env=env, capture_output=True, text=True,
        timeout=TIMEOUT, cwd=str(tmp_path))


def gen_run(env, tmp_path, llm, agent, extra=()):
    return run_wrapper(env, tmp_path, "--llm", llm, "--agent", agent,
                       "--run", "1", "--experiment", "t", *extra)


def reports(tmp_path):
    root = tmp_path / "experiments_results" / "t" / "run_1"
    return sorted(root.glob("01_generation_*.json")) if root.is_dir() else []


def manifest_calls(tmp_path):
    path = tmp_path / "manifest.log"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()]


def check_ok_entry(report, llm, agent):
    assert report["stage"] == "generation" and report["model"] == llm
    assert len(report["report_id"]) == 32
    entry = report["results"][0]
    assert entry["agent"] == agent and entry["llm"] == llm
    assert entry["status"] == "ok" and entry["error"] is None
    return entry


# --- success paths ------------------------------------------------------------------

@pytest.mark.parametrize("llm,exp_in,exp_out,exp_req", [
    ("claude", 130, 50, 3),
    ("codex", 200, 80, 1),
    ("opencode", 322, 90, 1),
])
def test_e2e_success(tmp_path, monkeypatch, llm, exp_in, exp_out, exp_req):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    manifest = tmp_path / "manifest.log"
    monkeypatch.setenv("FAKE_DOCKER_MANIFEST", str(manifest))
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, llm, "aaa")

    assert proc.returncode == 0, proc.stderr
    reps = reports(tmp_path)
    assert len(reps) == 1
    report = json.loads(reps[0].read_text(encoding="utf-8"))
    entry = check_ok_entry(report, llm, "aaa")
    assert entry["exit_code"] == 0 and entry["parse_status"] == "ok"
    assert entry["usage"]["input_tokens"] == exp_in
    assert entry["usage"]["output_tokens"] == exp_out
    assert entry["usage"]["requests"] == exp_req
    artifact = Path(entry["generated_file"])
    assert artifact.is_file() and not artifact.is_symlink()
    logs = {k: Path(v) for k, v in entry["log_paths"].items()}
    assert "docker_setup.log" in logs and logs["docker_setup.log"].is_file()
    assert logs["session.log"].is_file()
    calls = manifest_calls(tmp_path)
    assert calls[0][:2] == ["image", "inspect"]
    runs = [c for c in calls if c[:1] == ["run"]]
    assert len(runs) == 1 and "build" not in [c[0] for c in calls]


def test_e2e_alias_model_override(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    manifest = tmp_path / "manifest.log"
    monkeypatch.setenv("FAKE_DOCKER_MANIFEST", str(manifest))
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "claude-opus-4-6-low", "aaa",
                   extra=("--model", "custom-m"))

    assert proc.returncode == 0, proc.stderr
    runs = [c for c in manifest_calls(tmp_path) if c[:1] == ["run"]]
    assert len(runs) == 1
    assert "--model" in runs[0] and "custom-m" in runs[0]
    assert "--effort" in runs[0] and "low" in runs[0]
    report = json.loads(reports(tmp_path)[0].read_text(encoding="utf-8"))
    assert check_ok_entry(report, "claude-opus-4-6-low", "aaa")


# --- failure paths ---------------------------------------------------------------------

def test_e2e_exit9_with_artifact(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    monkeypatch.setenv("FAKE_RUN_EXIT", "9")
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    report = json.loads(reports(tmp_path)[0].read_text(encoding="utf-8"))
    entry = report["results"][0]
    assert entry["status"] == "error" and entry["exit_code"] == 9
    assert "cli_exit_9" in entry["error"]
    assert entry["usage"]["input_tokens"] == 130
    assert Path(entry["generated_file"]).is_file()


def test_e2e_missing_artifact(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    monkeypatch.setenv("FAKE_RUN_ARTIFACT", "no")
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "codex", "aaa")

    assert proc.returncode == 1
    entry = json.loads(reports(tmp_path)[0].read_text(
        encoding="utf-8"))["results"][0]
    assert "artifact_missing" in entry["error"]
    assert entry["usage"]["input_tokens"] == 200


@pytest.mark.parametrize("llm", ["claude", "codex", "opencode"])
def test_e2e_malformed_fatal(tmp_path, monkeypatch, llm):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    monkeypatch.setenv("FAKE_RUN_STDOUT", "malformed")
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, llm, "aaa")

    assert proc.returncode == 1
    entry = json.loads(reports(tmp_path)[0].read_text(
        encoding="utf-8"))["results"][0]
    assert entry["status"] == "error"
    assert entry["parse_status"] == "error" and "parse_error" in entry["error"]
    assert entry["usage"]["input_tokens"] is None


def test_e2e_partial_ok(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    monkeypatch.setenv("FAKE_RUN_STDOUT", "partial")
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "codex", "aaa")

    assert proc.returncode == 0, proc.stderr
    entry = json.loads(reports(tmp_path)[0].read_text(
        encoding="utf-8"))["results"][0]
    assert entry["status"] == "ok" and entry["parse_status"] == "partial"


def test_e2e_invalid_stderr_bytes_and_crlf(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    monkeypatch.setenv("FAKE_RUN_STDOUT", "crlf")
    monkeypatch.setenv("FAKE_RUN_STDERR", "invalid")
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "codex", "aaa")

    assert proc.returncode == 0, proc.stderr
    report = json.loads(reports(tmp_path)[0].read_text(encoding="utf-8"))
    entry = check_ok_entry(report, "codex", "aaa")
    assert entry["usage"]["input_tokens"] == 200
    logs = {k: Path(v) for k, v in entry["log_paths"].items()}
    assert logs["cli_stderr.log"].read_bytes() == b"\xff\xfe-bad\n"


def test_e2e_missing_docker(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    proc = gen_run(env, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    entry = json.loads(reports(tmp_path)[0].read_text(
        encoding="utf-8"))["results"][0]
    assert entry["status"] == "error"
    assert "docker_unavailable" in entry["error"]
    assert entry["exit_code"] is None


def test_e2e_build_failure(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_DOCKER_MODE", "build_fail")
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    entry = json.loads(reports(tmp_path)[0].read_text(
        encoding="utf-8"))["results"][0]
    assert "docker_build_failed" in entry["error"]


# --- boundaries ---------------------------------------------------------------------------

def test_e2e_rerun_launches_nothing(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    manifest = tmp_path / "manifest.log"
    monkeypatch.setenv("FAKE_DOCKER_MANIFEST", str(manifest))
    env = dict(os.environ)

    assert gen_run(env, tmp_path, "claude", "aaa").returncode == 0
    before = manifest.read_text(encoding="utf-8")
    second = gen_run(env, tmp_path, "claude", "aaa")

    assert second.returncode == 1
    assert manifest.read_text(encoding="utf-8") == before
    reps = reports(tmp_path)
    assert len(reps) == 2
    assert "output_exists" in json.loads(
        max(reps, key=lambda p: p.stat().st_mtime).read_text(
            encoding="utf-8"))["results"][0]["error"]


def test_e2e_unsafe_output_no_launch(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    manifest = tmp_path / "manifest.log"
    monkeypatch.setenv("FAKE_DOCKER_MANIFEST", str(manifest))
    stale = tmp_path / "generated_agents" / "t" / "run_1" / "claude" / "aaa"
    stale.mkdir(parents=True)
    (stale / "agent.py").write_text("# stale\n", encoding="utf-8")
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    assert not manifest.exists()
    entry = json.loads(reports(tmp_path)[0].read_text(
        encoding="utf-8"))["results"][0]
    assert "output_exists" in entry["error"]


def test_e2e_contention_single_winner(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    env = dict(os.environ)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        procs = list(pool.map(
            lambda _: gen_run(env, tmp_path, "opencode", "aaa"), range(3)))

    assert sorted(p.returncode for p in procs) == [0, 1, 1]
    reps = reports(tmp_path)
    assert len(reps) == 3
    assert len({p.name for p in reps}) == 3
    statuses = [json.loads(p.read_text(encoding="utf-8"))["results"][0]["status"]
                for p in reps]
    assert sorted(statuses) == ["error", "error", "ok"]


# --- credentials -------------------------------------------------------------------------------

def test_e2e_selected_credential_manifest(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    manifest = tmp_path / "manifest.log"
    monkeypatch.setenv("FAKE_DOCKER_MANIFEST", str(manifest))
    monkeypatch.setenv("ANTHROPIC_API_KEY", CRED_SECRET)
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "claude", "aaa")

    assert proc.returncode == 0, proc.stderr
    text = manifest.read_text(encoding="utf-8")
    assert CRED_SECRET not in text
    assert '"-e", "ANTHROPIC_API_KEY"' in text
    report = json.loads(reports(tmp_path)[0].read_text(encoding="utf-8"))
    assert CRED_SECRET not in json.dumps(report)
    setup = Path(report["results"][0]["log_paths"]["docker_setup.log"])
    assert CRED_SECRET not in setup.read_text(encoding="utf-8")


def test_e2e_opencode_no_creds_despite_junk(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    manifest = tmp_path / "manifest.log"
    monkeypatch.setenv("FAKE_DOCKER_MANIFEST", str(manifest))
    monkeypatch.setenv("OPENAI_API_KEY", CRED_SECRET)
    monkeypatch.setenv("ANTHROPIC_API_KEY", CRED_SECRET)
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "opencode", "aaa")

    assert proc.returncode == 0, proc.stderr
    text = manifest.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in text and "ANTHROPIC_API_KEY" not in text
    assert CRED_SECRET not in text
    assert CRED_SECRET not in json.dumps(
        json.loads(reports(tmp_path)[0].read_text(encoding="utf-8")))


def test_e2e_sentinel_absent_from_report(tmp_path, monkeypatch):
    env = base_env(tmp_path, monkeypatch)
    write_spec(tmp_path / "specs", "aaa")
    install_fake_docker(tmp_path / "bin")
    monkeypatch.setenv("FAKE_FIXTURES", json.dumps(
        {"claude": CLAUDE_OK, "codex": CODEX_OK, "opencode": OPENCODE_OK,
         "partial": CODEX_PARTIAL}))
    monkeypatch.setenv("FAKE_RUN_STDERR", "secret")
    monkeypatch.setenv("SECRET", SECRET)
    env = dict(os.environ)

    proc = gen_run(env, tmp_path, "claude", "aaa")

    assert proc.returncode == 0, proc.stderr
    report = json.loads(reports(tmp_path)[0].read_text(encoding="utf-8"))
    assert SECRET not in json.dumps(report)
    logs = report["results"][0]["log_paths"]
    logged = "".join(Path(p).read_text(encoding="utf-8", errors="replace")
                      for p in logs.values())
    assert SECRET in logged


# --- image configuration --------------------------------------------------------------------------

def test_dockerfile_gen_node22_and_pins():
    src = (ROOT / "docker" / "Dockerfile.gen").read_text(encoding="utf-8")
    assert "setup_22.x" in src and "setup_20.x" not in src
    assert "CLAUDE_CODE_VERSION=2.1.299" in src
    assert "CODEX_VERSION=0.162.1" in src
    assert "OPENCODE_VERSION=1.19.0" in src
    assert "WORKDIR /out" in src and "USER agent" in src
    assert "python:3.11-slim" in src
    assert not [line for line in src.splitlines()
                if line.strip().upper().startswith("COPY")]
    assert (ROOT / "docker" / "Dockerfile.eval").is_file()
