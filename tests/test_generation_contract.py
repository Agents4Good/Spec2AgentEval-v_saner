"""Phase-2 generation contract tests (lane B).

Real ``pipeline/01_generate.py`` subprocess runs with fake CLI binaries in
an isolated PATH (temporary ``bin`` dir only, so a real installed CLI can
never resolve) and temporary bench/generated/results roots. No dotenv
(``SPEC2AGENT_SKIP_DOTENV=1``), no network, no Docker, no credentials.

Covers the phase2 plan matrix: canonical success for all 3 CLIs, exit-9
with artifact retained, missing artifact, spawn failure (canonical return
path and legacy raise path), malformed/partial usage, separate stderr,
symlink/directory/stale/unsafe artifacts, rerun immutability, concurrent
contention, report uniqueness/collision/no-clobber atomicity, durations,
Claude aliases, Copilot/Gemini legacy smoke, and sentinel leakage checks.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = ROOT / "pipeline" / "01_generate.py"

TIMEOUT = 120
SECRET = "SENTINEL-9f8e7d6c-ZZ9"
SPEC_MARKER = "SPEC-MARKER-4b2c1a"

ENTRY_KEYS = {
    "agent", "llm", "spec_file", "output_dir", "generated_file", "status",
    "error", "exit_code", "parse_status", "warnings", "usage", "log_paths",
}

CLAUDE_BIN = "claude"
CODEX_BIN = "codex"
OPENCODE_BIN = "opencode"
COPILOT_BIN = "copilot"
GEMINI_BIN = "agy"

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

GEMINI_OK = {
    "usage": {"input_tokens": 50, "output_tokens": 25, "cache_read_tokens": 5},
    "num_turns": 2,
    "duration_seconds": 1.5,
}


def write_fake(path, *, stdout="", stderr="", code=0, artifact="file",
               agent_body="# generated agent\n", extra=""):
    """Write a fake CLI executable scripted by stdout/stderr/exit/artifact."""
    prog = [extra] if extra else []
    if artifact == "file":
        prog.append(f"open('agent.py','w',encoding='utf-8').write({agent_body!r})")
    elif artifact == "symlink":
        prog.append("open('real_target.py','w',encoding='utf-8').write('x\\n')")
        prog.append("os.symlink('real_target.py','agent.py')")
    elif artifact == "dir":
        prog.append("os.makedirs('agent.py',exist_ok=True)")
    elif artifact != "none":
        raise ValueError(artifact)
    prog.append(f"sys.stdout.write({stdout!r})")
    prog.append(f"sys.stderr.write({stderr!r})")
    prog.append(f"sys.exit({code})")
    path.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    + "\n".join(prog) + "\n", encoding="utf-8")
    path.chmod(0o755)


def fresh_env(tmp_path):
    """Isolated env: tmp roots, SKIP_DOTENV, PATH with only the fake bin dir."""
    env = dict(os.environ)
    for key in ("BENCH_DIR", "GENERATED_DIR", "EXPERIMENTS_RESULTS"):
        env.pop(key, None)
    bench = tmp_path / "specs"
    gen = tmp_path / "generated"
    res = tmp_path / "results"
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    env["BENCH_DIR"] = str(bench)
    env["GENERATED_DIR"] = str(gen)
    env["EXPERIMENTS_RESULTS"] = str(res)
    env["SPEC2AGENT_SKIP_DOTENV"] = "1"
    env["PATH"] = str(bindir)
    return env, {"bench": bench, "gen": gen, "res": res, "bin": bindir}


def write_spec(bench, name):
    agent_dir = bench / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "spec.md").write_text(
        f"# {name}\n\n{SPEC_MARKER}-{name}\n", encoding="utf-8")
    return agent_dir


def valid_stdout(binary):
    if binary == CLAUDE_BIN:
        return json.dumps(CLAUDE_OK)
    if binary == CODEX_BIN:
        return "\n".join(json.dumps(line) for line in CODEX_OK) + "\n"
    if binary == OPENCODE_BIN:
        return json.dumps(OPENCODE_OK) + "\n"
    raise ValueError(binary)


def run_gen(env, tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(GEN_SCRIPT), *args],
        env=env, capture_output=True, text=True, timeout=TIMEOUT,
        cwd=str(tmp_path))


def reports_for(dirs, experiment, run):
    root = dirs["res"] / experiment / f"run_{run}"
    if not root.is_dir():
        return []
    return sorted(root.glob("01_generation_*.json"))


def run_single(env, dirs, tmp_path, llm, agent, run=1, experiment="t"):
    proc = run_gen(env, tmp_path, "--llm", llm, "--agent", agent,
                   "--run", str(run), "--experiment", experiment)
    reps = reports_for(dirs, experiment, run)
    assert len(reps) == 1, f"expected one report, got {reps}\n{proc.stdout}\n{proc.stderr}"
    return proc, json.loads(reps[0].read_text(encoding="utf-8")), reps[0]


def check_envelope(report, *, llm, agent):
    for key in ("run_id", "report_id", "run", "experiment", "stage", "model",
                "bench_dir", "generated_dir", "total_agents", "results"):
        assert key in report, f"missing envelope key {key}"
    assert report["stage"] == "generation"
    assert report["model"] == llm
    assert len(report["report_id"]) == 32
    assert all(c in "0123456789abcdef" for c in report["report_id"])
    assert report["total_agents"] == len(report["results"]) == 1
    entry = report["results"][0]
    assert set(entry.keys()) == ENTRY_KEYS
    assert entry["agent"] == agent
    assert entry["llm"] == llm
    return entry


# --- canonical success -----------------------------------------------------

@pytest.mark.parametrize("binary,llm", [
    (CLAUDE_BIN, "claude"), (CODEX_BIN, "codex"), (OPENCODE_BIN, "opencode")])
def test_canonical_success(tmp_path, binary, llm):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / binary, stdout=valid_stdout(binary))

    proc, report, path = run_single(env, dirs, tmp_path, llm, "aaa")

    assert proc.returncode == 0, proc.stderr
    assert f"01_generation_{llm}_" in path.name
    assert path.name.endswith(f"_{report['report_id']}.json")
    entry = check_envelope(report, llm=llm, agent="aaa")
    assert entry["status"] == "ok"
    assert entry["error"] is None
    assert entry["exit_code"] == 0
    assert entry["parse_status"] == "ok"
    assert entry["warnings"] == []
    usage = entry["usage"]
    assert usage["cli_duration_s"] is None
    assert usage["source"]["cli_version"] is None
    assert isinstance(usage["process_duration_s"], float)
    assert isinstance(usage["wall_duration_s"], float)
    assert usage["process_duration_s"] >= 0
    assert usage["wall_duration_s"] >= usage["process_duration_s"] >= 0
    text = path.read_text(encoding="utf-8")
    assert SPEC_MARKER not in text  # prompt/spec content never leaks

    artifact = Path(entry["generated_file"])
    assert artifact.is_file() and not artifact.is_symlink()
    logs = {k: Path(v) for k, v in entry["log_paths"].items()}
    assert set(logs) == {"session.log", "cli_stderr.log", "credits.log",
                         "01-generate.log"}
    assert logs["session.log"].read_text(encoding="utf-8") == valid_stdout(binary)
    assert all(p.is_file() for p in logs.values())


def test_canonical_usage_numerics(tmp_path):
    env, dirs = fresh_env(tmp_path)
    for agent, binary in (("a1", CLAUDE_BIN), ("a2", CODEX_BIN), ("a3", OPENCODE_BIN)):
        write_spec(dirs["bench"], agent)
        write_fake(dirs["bin"] / binary, stdout=valid_stdout(binary))
    expected = {
        "claude": {"input_tokens": 130, "output_tokens": 50, "requests": 3},
        "codex": {"input_tokens": 200, "output_tokens": 80, "requests": 1},
        "opencode": {"input_tokens": 322, "output_tokens": 90, "requests": 1},
    }
    for llm, agent, run in (("claude", "a1", 1), ("codex", "a2", 2),
                         ("opencode", "a3", 3)):
        proc, report, _ = run_single(env, dirs, tmp_path, llm, agent, run=run)
        assert proc.returncode == 0, proc.stderr
        usage = report["results"][0]["usage"]
        for field, value in expected[llm].items():
            assert usage[field] == value, (llm, field, usage)


# --- failure paths ---------------------------------------------------------

@pytest.mark.parametrize("binary,llm", [
    (CLAUDE_BIN, "claude"), (CODEX_BIN, "codex"), (OPENCODE_BIN, "opencode")])
def test_exit9_with_artifact_retained(tmp_path, binary, llm):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / binary, stdout=valid_stdout(binary), code=9)

    proc, report, _ = run_single(env, dirs, tmp_path, llm, "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm=llm, agent="aaa")
    assert entry["status"] == "error"
    assert entry["exit_code"] == 9
    assert "cli_exit_9" in entry["error"]
    # Artifact and usage survive the operational failure.
    assert Path(entry["generated_file"]).is_file()
    assert entry["usage"].get("input_tokens") is not None
    assert entry["parse_status"] in ("ok", "partial")


@pytest.mark.parametrize("binary,llm", [
    (CLAUDE_BIN, "claude"), (CODEX_BIN, "codex"), (OPENCODE_BIN, "opencode")])
def test_missing_artifact(tmp_path, binary, llm):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / binary, stdout=valid_stdout(binary),
               artifact="none")

    proc, report, _ = run_single(env, dirs, tmp_path, llm, "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm=llm, agent="aaa")
    assert entry["status"] == "error"
    assert "artifact_missing" in entry["error"]
    assert entry["usage"].get("input_tokens") is not None  # usage retained


def test_spawn_failure_canonical_returns_metadata(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    # No fake binary installed: nothing named "claude" resolves in PATH.

    proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm="claude", agent="aaa")
    assert entry["status"] == "error"
    assert "spawn_failed" in entry["error"]
    assert entry["exit_code"] is None
    assert "Traceback" not in json.dumps(report)
    logs = [Path(v) for v in entry["log_paths"].values()]
    assert logs, "spawn failure must still record log paths"


def test_spawn_failure_legacy_raise_path(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    # Copilot adapter still raises on spawn failure; B must convert safely.

    proc, report, _ = run_single(env, dirs, tmp_path, "copilot", "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm="copilot", agent="aaa")
    assert entry["status"] == "error"
    assert "spawn_failed" in entry["error"]
    assert entry["exit_code"] is None
    assert "Traceback" not in json.dumps(report)
    assert "FileNotFoundError" not in json.dumps(report)


@pytest.mark.parametrize("binary,llm", [
    (CLAUDE_BIN, "claude"), (CODEX_BIN, "codex"), (OPENCODE_BIN, "opencode")])
def test_malformed_stdout_is_fatal(tmp_path, binary, llm):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / binary, stdout="NOT JSON {{{ broken\n")

    proc, report, _ = run_single(env, dirs, tmp_path, llm, "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm=llm, agent="aaa")
    assert entry["status"] == "error"
    assert entry["parse_status"] == "error"
    assert "parse_error" in entry["error"]
    assert entry["usage"].get("input_tokens") is None  # nulls, never zeros


def test_partial_usage_is_not_failure(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    partial = {"type": "turn.completed", "thread_id": "thr-1", "usage": {
        "input_tokens": 200, "cached_input_tokens": 30}}
    write_fake(dirs["bin"] / CODEX_BIN, stdout=json.dumps(partial) + "\n")

    proc, report, _ = run_single(env, dirs, tmp_path, "codex", "aaa")

    assert proc.returncode == 0, proc.stderr
    entry = check_envelope(report, llm="codex", agent="aaa")
    assert entry["status"] == "ok"
    assert entry["error"] is None
    assert entry["parse_status"] == "partial"
    assert entry["warnings"], "partial must carry a diagnosis"


@pytest.mark.parametrize("binary,llm", [
    (CLAUDE_BIN, "claude"), (CODEX_BIN, "codex"), (OPENCODE_BIN, "opencode")])
def test_stderr_kept_separate(tmp_path, binary, llm):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    stdout = valid_stdout(binary) if binary != CODEX_BIN else valid_stdout(binary)
    write_fake(dirs["bin"] / binary, stdout=stdout,
               stderr="warn marker line\n")

    proc, report, _ = run_single(env, dirs, tmp_path, llm, "aaa")

    assert proc.returncode == 0, proc.stderr
    entry = report["results"][0]
    logs = {k: Path(v) for k, v in entry["log_paths"].items()}
    assert logs["session.log"].read_text(encoding="utf-8") == stdout
    assert logs["cli_stderr.log"].read_text(encoding="utf-8") == "warn marker line\n"
    assert "warn marker" not in logs["credits.log"].read_text(encoding="utf-8")
    assert "warn marker" not in json.dumps(report)


# --- artifact safety -------------------------------------------------------

def test_symlink_artifact_rejected(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               artifact="symlink")

    proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm="claude", agent="aaa")
    assert "artifact_symlink" in entry["error"]


def test_directory_artifact_rejected(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               artifact="dir")

    proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm="claude", agent="aaa")
    assert "artifact_not_regular" in entry["error"]


def test_preexisting_output_rejected_before_cli(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    marker = dirs["bin"] / "invoked.log"
    stale_dir = dirs["gen"] / "t" / "run_1" / "claude" / "aaa"
    stale_dir.mkdir(parents=True)
    stale_artifact = stale_dir / "agent.py"
    stale_artifact.write_text("# stale\n", encoding="utf-8")
    before = stale_artifact.read_bytes()
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               extra=f"open({str(marker)!r},'a').write('started\\n')")

    proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm="claude", agent="aaa")
    assert "output_exists" in entry["error"]
    assert not marker.exists(), "CLI must not start when the target exists"
    assert stale_artifact.read_bytes() == before, "stale artifact untouched"


def test_symlinked_parent_rejected_before_cli(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    marker = dirs["bin"] / "invoked.log"
    link_parent = dirs["gen"] / "t" / "run_1" / "claude"
    link_parent.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link_parent.symlink_to(outside, target_is_directory=True)
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               extra=f"open({str(marker)!r},'a').write('started\\n')")

    proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm="claude", agent="aaa")
    assert "output_unsafe" in entry["error"]
    assert not marker.exists(), "CLI must not start on an unsafe target"
    assert not (outside / "aaa").exists()


# --- isolation / atomicity ---------------------------------------------------

def test_rerun_never_overwrites(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    marker = dirs["bin"] / "invoked.log"
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               extra=f"open({str(marker)!r},'a').write('started\\n')")

    proc1, report1, path1 = run_single(env, dirs, tmp_path, "claude", "aaa")
    assert proc1.returncode == 0, proc1.stderr
    artifact = Path(report1["results"][0]["generated_file"])
    session_log = Path(report1["results"][0]["log_paths"]["session.log"])
    artifact_bytes = artifact.read_bytes()
    session_bytes = session_log.read_bytes()
    report_bytes = path1.read_bytes()

    proc2 = run_gen(env, tmp_path, "--llm", "claude", "--agent", "aaa",
                    "--run", "1", "--experiment", "t")
    reps2 = reports_for(dirs, "t", 1)
    assert len(reps2) == 2
    path2 = next(p for p in reps2 if p != path1)
    report2 = json.loads(path2.read_text(encoding="utf-8"))

    assert proc2.returncode == 1
    assert "output_exists" in report2["results"][0]["error"]
    assert path2 != path1, "second run must publish its own report file"
    assert path1.read_bytes() == report_bytes, "first report never clobbered"
    assert artifact.read_bytes() == artifact_bytes
    assert session_log.read_bytes() == session_bytes
    assert marker.read_text(encoding="utf-8") == "started\n"


def test_concurrent_contention_single_winner(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN))

    def attempt():
        return run_gen(env, tmp_path, "--llm", "claude", "--agent", "aaa",
                       "--run", "1", "--experiment", "t")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        procs = list(pool.map(lambda _: attempt(), range(4)))

    reps = reports_for(dirs, "t", 1)
    assert len(reps) == 4, "every attempt publishes its own report"
    entries = [json.loads(p.read_text(encoding="utf-8"))["results"][0]
               for p in reps]
    ok = [e for e in entries if e["status"] == "ok"]
    taken = [e for e in entries
             if e["status"] == "error" and "output_exists" in e["error"]]
    assert len(ok) == 1, entries
    assert len(taken) == 3, entries
    assert sorted(p.returncode for p in procs) == [0, 1, 1, 1]
    artifact = Path(ok[0]["generated_file"])
    assert artifact.is_file() and not artifact.is_symlink()
    assert len({p.name for p in reps}) == 4, "report names unique"


def test_parallel_reports_unique_and_atomic(tmp_path):
    env, dirs = fresh_env(tmp_path)
    agents = ["a1", "a2", "a3"]
    for agent in agents:
        write_spec(dirs["bench"], agent)
    write_fake(dirs["bin"] / CODEX_BIN, stdout=valid_stdout(CODEX_BIN))

    def attempt(agent):
        return run_gen(env, tmp_path, "--llm", "codex", "--agent", agent,
                       "--run", "1", "--experiment", "t")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        procs = list(pool.map(attempt, agents))

    assert all(p.returncode == 0 for p in procs)
    reps = reports_for(dirs, "t", 1)
    assert len(reps) == 3
    assert len({p.name for p in reps}) == 3
    for path in reps:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["report_id"] in path.name
        check_envelope(payload, llm="codex", agent=payload["results"][0]["agent"])
    assert {p["results"][0]["agent"] for p in
            (json.loads(r.read_text(encoding="utf-8")) for r in reps)} == set(agents)


# --- aliases / legacy smoke --------------------------------------------------

def test_claude_alias_forwards_model_and_effort(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    capture = ("import json as _j; _a=sys.argv[1:]; _d={}; "
               "_d['effort_low']=('--effort' in _a and 'low' in _a); "
               "_d['model']=_a[_a.index('--model')+1] if '--model' in _a else None; "
               "open('argv_seen.json','w').write(_j.dumps(_d))")
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               extra=capture)

    proc, report, _ = run_single(env, dirs, tmp_path,
                                 "claude-opus-4-6-low", "aaa")

    assert proc.returncode == 0, proc.stderr
    assert report["results"][0]["status"] == "ok"
    seen = json.loads((dirs["gen"] / "t" / "run_1" / "claude-opus-4-6-low"
                       / "aaa" / "argv_seen.json").read_text(encoding="utf-8"))
    assert seen == {"effort_low": True, "model": "claude-opus-4-6"}


def test_all_adapter_choices_still_accepted(tmp_path):
    env = dict(os.environ)
    env["SPEC2AGENT_SKIP_DOTENV"] = "1"
    proc = run_gen(env, tmp_path, "--help")
    assert proc.returncode == 0
    for choice in ("copilot", "gemini", "claude", "codex", "opencode",
                   "claude-opus-4-6-low", "claude-opus-4-8-low",
                   "claude-opus-5-5-low"):
        assert choice in proc.stdout
    bad = run_gen(env, tmp_path, "--llm", "bogus")
    assert bad.returncode == 2


def test_copilot_gemini_legacy_shapes_preserved(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "a1")
    write_spec(dirs["bench"], "a2")
    write_fake(dirs["bin"] / COPILOT_BIN, stdout=f"done {SECRET}\n",
               stderr=f"credits {SECRET}\n")
    gemini_stdout = dict(GEMINI_OK)
    gemini_stdout["unrelated_extra"] = SECRET
    write_fake(dirs["bin"] / GEMINI_BIN, stdout=json.dumps(gemini_stdout))

    proc_c, rep_c, _ = run_single(env, dirs, tmp_path, "copilot", "a1",
                                   run=1)
    proc_g, rep_g, _ = run_single(env, dirs, tmp_path, "gemini", "a2",
                                   run=2)

    assert proc_c.returncode == 0, proc_c.stderr
    assert proc_g.returncode == 0, proc_g.stderr
    entry_c = check_envelope(rep_c, llm="copilot", agent="a1")
    entry_g = check_envelope(rep_g, llm="gemini", agent="a2")
    assert entry_c["status"] == "ok" and entry_g["status"] == "ok"
    # Legacy metric shapes pass through untouched (never strict-parsed).
    assert entry_c["usage"]["requests"] == 0
    assert set(entry_c["usage"]) <= {"changes_added", "changes_removed",
                                     "requests", "time_spent",
                                     "models_breakdown"}
    assert "input_tokens" not in entry_c["usage"]
    assert entry_g["usage"]["requests"] == 2
    assert "models_breakdown" in entry_g["usage"]
    # Legacy parser status is informational only: never a failure.
    assert entry_c["parse_status"] is None
    assert entry_g["parse_status"] is None
    assert entry_c["exit_code"] is None
    # Legacy logs are discovered even though the adapters expose no mapping.
    logs_c = {k: Path(v) for k, v in entry_c["log_paths"].items()}
    assert {"session.log", "credits.log", "01-generate.log"} <= set(logs_c)
    assert all(p.is_file() for p in logs_c.values())
    logs_g = {k: Path(v) for k, v in entry_g["log_paths"].items()}
    assert {"session.log", "cli_stderr.log", "01_generate.log"} <= set(logs_g)
    assert all(p.is_file() for p in logs_g.values())
    assert SECRET not in json.dumps(rep_c)
    assert SECRET not in json.dumps(rep_g)


# --- sentinel leakage ----------------------------------------------------------

@pytest.mark.parametrize("binary,llm", [
    (CLAUDE_BIN, "claude"), (CODEX_BIN, "codex"), (OPENCODE_BIN, "opencode")])
def test_sentinel_never_reaches_report(tmp_path, binary, llm):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    stdout = valid_stdout(binary)
    if binary == CLAUDE_BIN:
        payload = json.loads(stdout)
        payload["result"] = f"code containing {SECRET}"
        stdout = json.dumps(payload)
    write_fake(dirs["bin"] / binary, stdout=stdout,
               stderr=f"stderr carrying {SECRET}\n")

    proc, report, _ = run_single(env, dirs, tmp_path, llm, "aaa")

    assert proc.returncode == 0, proc.stderr
    text = json.dumps(report)
    assert SECRET not in text, "raw CLI streams must stay in logs, not reports"
    logs = report["results"][0]["log_paths"]
    logged = "".join(Path(p).read_text(encoding="utf-8") for p in logs.values())
    assert SECRET in logged, "check is meaningful: secret reached the logs"


# --- config / selector safety ----------------------------------------------------

def test_missing_env_friendly_without_report(tmp_path):
    env = dict(os.environ)
    for key in ("BENCH_DIR", "GENERATED_DIR", "EXPERIMENTS_RESULTS"):
        env.pop(key, None)
    env["SPEC2AGENT_SKIP_DOTENV"] = "1"
    env["PATH"] = str(tmp_path)

    proc = run_gen(env, tmp_path, "--llm", "claude")

    assert proc.returncode == 2
    assert "BENCH_DIR" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("args", [
    ["--llm", "claude", "--experiment", "../evil"],
    ["--llm", "claude", "--agent", "a/b"],
    ["--llm", "claude", "--run", "0"],
])
def test_unsafe_selectors_rejected(tmp_path, args):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")

    proc = run_gen(env, tmp_path, *args)

    assert proc.returncode == 2, proc.stderr
    assert not (tmp_path / "evil").exists()
    assert reports_for(dirs, "../evil", 1) == []


def test_tmp_env_wins_with_dotenv_bypass(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN))

    proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")

    assert proc.returncode == 0, proc.stderr
    assert Path(report["bench_dir"]).resolve() == dirs["bench"].resolve()
    assert Path(report["generated_dir"]).resolve() == \
        (dirs["gen"] / "t" / "run_1").resolve()


# --- gate-2 regressions ------------------------------------------------------

def test_run_root_symlink_rejected(tmp_path):
    """A symlinked experiment/run ancestor must fail before any CLI starts."""
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    marker = dirs["bin"] / "invoked.log"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("OUTSIDE-SENTINEL\n", encoding="utf-8")
    run_root = dirs["gen"] / "t" / "run_1"
    run_root.parent.mkdir(parents=True)
    run_root.symlink_to(outside, target_is_directory=True)
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               extra=f"open({str(marker)!r},'a').write('started\\n')")

    proc = run_gen(env, tmp_path, "--llm", "claude", "--agent", "aaa",
                   "--run", "1", "--experiment", "t")

    assert proc.returncode == 2
    assert "unsafe" in proc.stderr
    assert not marker.exists(), "CLI must not start on a redirected root"
    assert sorted(p.name for p in outside.iterdir()) == ["sentinel.txt"]
    assert reports_for(dirs, "t", 1) == []


def test_output_dir_replaced_post_run(tmp_path):
    """A post-run symlink swap is rejected even with an artifact inside."""
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    evil = tmp_path / "evil"
    swap = (
        f"import os as _o; _e={str(evil)!r}; _o.makedirs(_e,exist_ok=True); "
        f"open(_o.path.join(_e,'agent.py'),'w').write('# evil\\n'); "
        f"_o.chdir('..'); _o.rename('aaa','aaa.bak'); _o.symlink(_e,'aaa')"
    )
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               extra=swap)

    proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")

    assert proc.returncode == 1
    entry = check_envelope(report, llm="claude", agent="aaa")
    assert entry["status"] == "error"
    assert "output_replaced" in entry["error"]
    assert "spawn_failed" not in entry["error"]
    assert "# evil" not in json.dumps(report)


@pytest.mark.parametrize("code", [0, 9])
def test_persistence_fault_retains_metadata(tmp_path, code):
    """Read-only logs keep exit/usage evidence and never read as spawn."""
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN),
               code=code, extra="os.chmod('logs',0o555)")
    logs_dir = dirs["gen"] / "t" / "run_1" / "claude" / "aaa" / "logs"

    try:
        proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")
    finally:
        if logs_dir.is_dir() and not logs_dir.is_symlink():
            os.chmod(logs_dir, 0o755)

    assert proc.returncode == 1
    entry = check_envelope(report, llm="claude", agent="aaa")
    assert entry["status"] == "error"
    assert entry["exit_code"] == code
    assert entry["usage"].get("input_tokens") == 130
    assert entry["parse_status"] == "ok"
    assert "spawn_failed" not in entry["error"]
    if code == 9:
        assert "cli_exit_9" in entry["error"]
    else:
        assert "persistence_failed" in entry["error"]
    assert any("persistence_failed" in w for w in entry["warnings"])


COLLISION_SHIM = """\
import os as _os
import uuid as _uuid

_real_uuid4 = _uuid.uuid4
_FIXED = "f" * 32
_calls = [0]


class _Fixed(object):
    hex = _FIXED


def _patched():
    _calls[0] += 1
    if _calls[0] <= 2:
        return _Fixed()
    return _real_uuid4()


_uuid.uuid4 = _patched

_target_dir = _os.environ.get("GEN01_COLLISION_DIR")
if _target_dir:
    _target = _os.path.join(
        _target_dir, "01_generation_claude_%s.json" % _FIXED)
    _tmp = "%s.tmp-%d" % (_target, _os.getpid())
    try:
        _os.link(_target, _tmp)
    except OSError:
        pass
"""


def test_report_uuid_collision_keeps_old_report(tmp_path):
    """A forced UUID collision plus a leftover tmp hardlink changes nothing."""
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN))
    run_results = dirs["res"] / "t" / "run_1"
    run_results.mkdir(parents=True)
    fixed_name = "01_generation_claude_%s.json" % ("f" * 32)
    old_report = run_results / fixed_name
    old_report.write_bytes(b'{"old": "OLD-REPORT-SENTINEL"}\n')
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(COLLISION_SHIM, encoding="utf-8")
    env["PYTHONPATH"] = str(shim)
    env["GEN01_COLLISION_DIR"] = str(run_results)

    proc = run_gen(env, tmp_path, "--llm", "claude", "--agent", "aaa",
                   "--run", "1", "--experiment", "t")

    assert proc.returncode == 0, proc.stderr
    assert old_report.read_bytes() == b'{"old": "OLD-REPORT-SENTINEL"}\n'
    reps = reports_for(dirs, "t", 1)
    assert len(reps) == 2
    assert old_report in reps
    new = next(p for p in reps if p != old_report)
    payload = json.loads(new.read_text(encoding="utf-8"))
    check_envelope(payload, llm="claude", agent="aaa")
    assert payload["results"][0]["status"] == "ok"


def test_stale_tmp_files_untouched(tmp_path):
    env, dirs = fresh_env(tmp_path)
    write_spec(dirs["bench"], "aaa")
    write_fake(dirs["bin"] / CLAUDE_BIN, stdout=valid_stdout(CLAUDE_BIN))
    run_results = dirs["res"] / "t" / "run_1"
    run_results.mkdir(parents=True)
    stale = run_results / ".tmp-publish-stale"
    stale.write_bytes(b"STALE-TMP-SENTINEL\n")

    proc, report, _ = run_single(env, dirs, tmp_path, "claude", "aaa")

    assert proc.returncode == 0, proc.stderr
    assert stale.read_bytes() == b"STALE-TMP-SENTINEL\n"
    assert report["results"][0]["status"] == "ok"
