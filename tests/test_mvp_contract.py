"""Contrato MVP Docker: (a) gen isolada, (b) eval offline, (c) ordem do pipe."""
import json
import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"

FAKE_DOCKER = """#!/bin/bash
echo "$@" >> "$FAKE_DOCKER_LOG"
for a in "$@"; do
  case "$a" in *:/out*|*:/out:rw) h="${a%%:*}"; mkdir -p "$h"; touch "$h/agent.py" ;; esac
done
exit 0
"""


def run(cmd, env, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


def base_env(tmp_path):
    env = dict(os.environ)
    bench = tmp_path / "specs" / "011_list_issues"
    bench.mkdir(parents=True)
    (bench / "spec.md").write_text("# spec fake\n")
    env["BENCH_DIR"] = str(tmp_path / "specs")
    env["GENERATED_DIR"] = str(tmp_path / "generated_agents")
    env["EXPERIMENTS_RESULTS"] = str(tmp_path / "experiments_results")
    return env


def fake_docker_bin(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "docker").write_text(FAKE_DOCKER)
    (bindir / "docker").chmod(0o755)
    (bindir / "FAKE_LOG").write_text("")
    return bindir


def test_gen_isolated_no_repo_mount(tmp_path):
    """(a) gen monta spec :ro em /in, escreve agent.py, nunca monta o repo."""
    env = base_env(tmp_path)
    bindir = fake_docker_bin(tmp_path)
    env["PATH"] = f"{bindir}{os.pathsep}{os.environ['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(bindir / "FAKE_LOG")

    assert not [l for l in (DOCKER / "Dockerfile.gen").read_text().splitlines()
                 if l.strip().upper().startswith("COPY")]  # sem COPY do repo
    assert "WORKDIR /out" in (DOCKER / "Dockerfile.gen").read_text()
    gen_src = (DOCKER / "docker_run_gen.sh").read_text()
    assert "/repo" not in gen_src and ":ro" in gen_src and "/in" in gen_src

    p = run(["bash", str(DOCKER / "docker_run_gen.sh"), "--llm", "opencode",
             "--agent", "011_list_issues", "--run", "1", "--experiment", "t"], env=env)
    assert p.returncode == 0, p.stderr
    out = tmp_path / "generated_agents" / "t" / "run_1" / "opencode" / "011_list_issues"
    assert (out / "agent.py").exists()
    rep = json.loads((out / "docker_gen_report.json").read_text())
    assert isinstance(rep["duration_s"], int) and "tokens" in rep
    log = (bindir / "FAKE_LOG").read_text()
    assert ":/in:ro" in log and "/repo" not in log  # spec :ro, repo ausente


def test_eval_offline_uses_network_none(tmp_path):
    """(b) 02/04/06 offline (--network none); 03 online; repo sempre :ro."""
    env = base_env(tmp_path)
    bindir = fake_docker_bin(tmp_path)
    env["PATH"] = f"{bindir}{os.pathsep}{os.environ['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(bindir / "FAKE_LOG")

    for step, offline in [("02", True), ("03", False)]:
        (bindir / "FAKE_LOG").write_text("")
        p = run(["bash", str(DOCKER / "docker_run_eval.sh"), "--step", step,
                 "--llm", "opencode", "--agent", "011_list_issues",
                 "--run", "1", "--experiment", "t"], env=env)
        assert p.returncode == 0, p.stderr
        log = (bindir / "FAKE_LOG").read_text()
        assert ("--network none" in log) == offline
        assert ":/repo:ro" in log and f"{step}_" in log


def test_pipe_order_docker(tmp_path):
    """(c) USE_DOCKER=1: 01 em paralelo -> barreira -> 02..08, tudo via docker."""
    src = (ROOT / "pipe.sh").read_text()
    assert 'USE_DOCKER=${USE_DOCKER:-0}' in src  # default 0, fluxo atual intacto
    i_par = src.index('bash -c "$(gen_cmd')
    i_wait = src.index('Aguarda todas as gerações')
    i_eval = src.index('eval_cmd 02_static_check.py')
    assert i_par < i_wait < i_eval  # 01-paralelo -> barreira -> 02..08

    env = base_env(tmp_path)
    bindir = fake_docker_bin(tmp_path)
    env.update(PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
               FAKE_DOCKER_LOG=str(bindir / "FAKE_LOG"),
               USE_DOCKER="1", RUNS="1", RUNS_FROM="1", RUNS_TO="1", TERM="dumb")
    p = run(["bash", str(ROOT / "pipe.sh")], env=env, cwd=tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    lines = (bindir / "FAKE_LOG").read_text().splitlines()
    gens = [i for i, l in enumerate(lines) if ":/in:ro" in l]
    evals = [i for i, l in enumerate(lines) if ":/repo:ro" in l]
    assert len(gens) == 3 and len(evals) == 3 * 7  # 3 llms x (1 gen + 7 eval)
    assert max(gens) < min(evals)  # barreira: toda gen antes de qualquer eval
