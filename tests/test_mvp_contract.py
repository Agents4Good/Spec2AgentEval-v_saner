"""Contrato MVP Docker: (a) gen isolada, (b) eval offline, (c) ordem do pipe.

Phase-3 migration: the Docker generation path is the thin wrapper over
stage01 ``--backend docker``. The fake docker answers inspect/build and
materializes ``agent.py`` plus valid per-CLI usage on ``run``; assertions
target the canonical ``01_generation_*`` report (the obsolete flat
``docker_gen_report.json`` contract was deliberately removed). Eval and
pipe-barrier behavior contracts are unchanged.

Hardened harness: fake-bin-only PATH with individually symlinked
shell/python helpers, forced ``SPEC2AGENT_SKIP_DOTENV=1``, synthetic HOME,
scrubbed credential env. No real Docker, credentials, or .env reads.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"

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
        "reasoning_output_tokens": 7,
    }},
]

OPENCODE_OK = {
    "type": "step_finish",
    "sessionID": "s1",
    "part": {"id": "p1", "messageID": "m1",
             "tokens": {"input": 12, "output": 6, "reasoning": 1,
                        "cache": {"read": 2, "write": 3}, "total": 19},
             "cost": 0.004},
}

SCRUB_VARS = (
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY",
    "GEN_OPENCODE_API_KEY_ENV", "CLAUDE_CONFIG_DIR", "CODEX_HOME",
    "GEN_IMAGE", "FAKE_DOCKER_LOG", "FAKE_FIXTURES",
    "SONAR_HOST_URL", "SONAR_LOGIN", "SONAR_PASSWORD", "USER_TOKEN",
    "SONAR_COMMAND",
)

# Fake docker: inspect/build succeed; gen `run` (identified by the /out
# mount) materializes agent.py and emits the matching valid CLI fixture;
# anything else exits 0 after logging (eval path behavior preserved).
FAKE_DOCKER = """#!{python}
import json, os, sys

def log(entry):
    with open(os.environ["FAKE_DOCKER_LOG"], "a", encoding="utf-8") as f:
        f.write(" ".join(entry) + "\\n")

args = sys.argv[1:]
log(args)
if args[:2] == ["image", "inspect"] or args[:1] == ["build"]:
    sys.exit(0)
if args[:1] == ["run"]:
    out = None
    for i, a in enumerate(args):
        if a == "-v" and args[i + 1].endswith(":/out:rw"):
            out = args[i + 1].split(":")[0]
    llm = os.path.basename(os.path.dirname(out or "")) if out else ""
    fixtures = json.loads(os.environ.get("FAKE_FIXTURES", "{}"))
    key = None
    if "claude" in llm:
        key = "claude"
    elif "codex" in llm:
        key = "codex"
    elif "opencode" in llm:
        key = "opencode"
    if key is not None and out and key in fixtures:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "agent.py"), "w",
                  encoding="utf-8") as f:
            f.write("# fake agent\\n")
        payload = fixtures[key]
        if isinstance(payload, list):
            sys.stdout.write("\\n".join(
                json.dumps(line) for line in payload) + "\\n")
        else:
            sys.stdout.write(json.dumps(payload) + "\\n")
sys.exit(0)
"""


def _stage_python():
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def run(cmd, env, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


def base_env(tmp_path, monkeypatch):
    for var in SCRUB_VARS:
        monkeypatch.delenv(var, raising=False)
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    py = bindir / "python3"
    py.write_text(f"#!/bin/sh\nexec {shlex.quote(_stage_python())} \"$@\"\n",
                  encoding="utf-8")
    py.chmod(0o755)
    for helper in ("bash", "dirname", "cut", "seq", "sleep", "mkdir", "id"):
        target = shutil.which(helper)
        link = bindir / helper
        if target and not link.exists():
            os.symlink(target, link)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    bench = tmp_path / "specs" / "011_list_issues"
    bench.mkdir(parents=True)
    (bench / "spec.md").write_text("# spec fake\n")
    for key, value in {
        "PATH": str(bindir),
        "HOME": str(home),
        "SPEC2AGENT_SKIP_DOTENV": "1",
        "BENCH_DIR": str(tmp_path / "specs"),
        "GENERATED_DIR": str(tmp_path / "generated_agents"),
        "EXPERIMENTS_RESULTS": str(tmp_path / "experiments_results"),
        "GEN_IMAGE": "test-gen:mvp",
        "FAKE_DOCKER_LOG": str(bindir / "FAKE_LOG"),
        "FAKE_FIXTURES": json.dumps(
            {"claude": CLAUDE_OK, "codex": CODEX_OK,
             "opencode": OPENCODE_OK}),
        "TERM": "dumb",
    }.items():
        monkeypatch.setenv(key, value)
    (bindir / "FAKE_LOG").write_text("")
    return dict(os.environ)


def fake_docker_bin(tmp_path):
    bindir = tmp_path / "bin"
    path = bindir / "docker"
    path.write_text(FAKE_DOCKER.replace("{python}", sys.executable),
                    encoding="utf-8")
    path.chmod(0o755)
    return bindir


def test_gen_isolated_no_repo_mount(tmp_path, monkeypatch):
    """(a) gen via wrapper+stage01: canonical report, /in:ro, sem repo."""
    env = base_env(tmp_path, monkeypatch)
    bindir = fake_docker_bin(tmp_path)

    assert not [l for l in (DOCKER / "Dockerfile.gen").read_text().splitlines()
                 if l.strip().upper().startswith("COPY")]  # sem COPY do repo
    assert "WORKDIR /out" in (DOCKER / "Dockerfile.gen").read_text()
    gen_src = (DOCKER / "docker_run_gen.sh").read_text()
    assert "/repo" not in gen_src  # wrapper itself never mounts anything
    assert "--backend docker" in gen_src and "01_generate.py" in gen_src
    assert "docker_gen_report.json" not in gen_src  # contrato obsoleto removido

    p = run(["bash", str(DOCKER / "docker_run_gen.sh"), "--llm", "opencode",
             "--agent", "011_list_issues", "--run", "1", "--experiment", "t"], env=env)
    assert p.returncode == 0, p.stderr
    out = tmp_path / "generated_agents" / "t" / "run_1" / "opencode" / "011_list_issues"
    assert (out / "agent.py").exists()
    assert not (out / "docker_gen_report.json").exists()
    reps = sorted((tmp_path / "experiments_results" / "t" / "run_1").glob(
        "01_generation_opencode_*.json"))
    assert len(reps) == 1
    entry = json.loads(reps[0].read_text(encoding="utf-8"))["results"][0]
    assert entry["status"] == "ok" and entry["exit_code"] == 0
    assert entry["usage"]["input_tokens"] == 12 + 2 + 3
    log = (bindir / "FAKE_LOG").read_text()
    assert ":/in:ro" in log and "/repo" not in log  # spec :ro, repo ausente


def test_eval_offline_uses_network_none(tmp_path, monkeypatch):
    """(b) 02/04/06 offline (--network none); 03 online; repo sempre :ro."""
    env = base_env(tmp_path, monkeypatch)
    bindir = fake_docker_bin(tmp_path)

    for step, offline in [("02", True), ("03", False)]:
        (bindir / "FAKE_LOG").write_text("")
        p = run(["bash", str(DOCKER / "docker_run_eval.sh"), "--step", step,
                 "--llm", "opencode", "--agent", "011_list_issues",
                 "--run", "1", "--experiment", "t"], env=env)
        assert p.returncode == 0, p.stderr
        log = (bindir / "FAKE_LOG").read_text()
        assert ("--network none" in log) == offline
        assert ":/repo:ro" in log and f"{step}_" in log


def test_pipe_order_docker(tmp_path, monkeypatch):
    """(c) USE_DOCKER=1: 01 em paralelo -> barreira -> 02..08, tudo via docker."""
    src = (ROOT / "pipe.sh").read_text()
    assert 'USE_DOCKER=${USE_DOCKER:-0}' in src  # default 0, fluxo atual intacto
    i_par = src.index('bash -c "$(gen_cmd')
    i_wait = src.index('Aguarda todas as gerações')
    i_eval = src.index('eval_cmd 02_static_check.py')
    assert i_par < i_wait < i_eval  # 01-paralelo -> barreira -> 02..08

    env = base_env(tmp_path, monkeypatch)
    bindir = fake_docker_bin(tmp_path)
    env.update(USE_DOCKER="1", RUNS="1", RUNS_FROM="1", RUNS_TO="1")
    p = run(["bash", str(ROOT / "pipe.sh")], env=env, cwd=tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    lines = (bindir / "FAKE_LOG").read_text().splitlines()
    gens = [i for i, l in enumerate(lines) if ":/in:ro" in l]
    evals = [i for i, l in enumerate(lines) if ":/repo:ro" in l]
    assert len(gens) == 3 and len(evals) == 3 * 7  # 3 llms x (1 gen + 7 eval)
    assert max(gens) < min(evals)  # barreira: toda gen antes de qualquer eval
    reps = sorted((tmp_path / "experiments_results" / "by_tool" / "run_1").glob(
        "01_generation_*.json"))
    assert len(reps) == 3  # canonical reports for all 3 llms
    for rep in reps:
        results = json.loads(rep.read_text(encoding="utf-8"))["results"]
        assert results and all(r["status"] == "ok" for r in results)
