"""Shared helpers for the behavioral/trace evaluation runners (stages 07, 08).

Why this module exists
----------------------
The old runners captured only pytest's quiet stdout, which truncates failure
reasons (``AssertionError: ...``). The aggregator then had to reconstruct test
status via regex and still lost per-test reasons. We now ask pytest to also
write a structured JSON report via the ``pytest-json-report`` plugin and embed
it into the per-agent entry. Nothing is lost, and the aggregator can display
the full ``longrepr`` for every failure.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PYTEST_REPORT_FILENAME = ".pytest_report.json"


def _load_report(report_path: Path) -> dict[str, Any] | None:
    try:
        if not _safe(report_path).is_file():
            return None
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_tests(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten pytest-json-report tests into the shape the frontend consumes."""
    normalized = []
    for test in report.get("tests") or []:
        nodeid = test.get("nodeid") or ""
        short = nodeid.split("::", 1)[-1] if "::" in nodeid else nodeid
        outcome = test.get("outcome") or ""
        # longrepr is on the phase that failed/errored; call for actual test,
        # setup for fixture errors, teardown for cleanup errors.
        reason = ""
        for phase in ("call", "setup", "teardown"):
            info = test.get(phase) or {}
            lr = info.get("longrepr") or (info.get("crash") or {}).get("message")
            if lr:
                reason = lr if isinstance(lr, str) else str(lr)
                break
        normalized.append(
            {
                "name": short,
                "qualified_name": nodeid,
                "status": outcome,
                "passed": outcome == "passed",
                "reason": reason,
                "duration": test.get("duration"),
                **{phase: test.get(phase) for phase in ("setup", "call", "teardown")},
            }
        )
    return normalized


def _safe(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("selected input or ancestor is a symlink")
    return path


def _execute(args, cwd, env, timeout, cancelled):
    process = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    deadline, reason = time.monotonic() + timeout, None
    try:
        while True:
            if cancelled() or time.monotonic() >= deadline:
                reason = "interrupted" if cancelled() else "pytest_timeout"
                break
            try:
                process.communicate(timeout=min(.1, max(.001, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        # Kill the whole owned group, even when its leader already exited.
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        stdout, stderr = process.communicate(timeout=1)
    return process.returncode, stdout, stderr, reason


def _classify(report, rc):
    if (not isinstance(report, dict) or type(report.get("exitcode")) is not int
            or report["exitcode"] != rc or not isinstance(report.get("summary"), dict)
            or not isinstance(report.get("tests"), list)):
        raise ValueError("invalid JSON report structure or exitcode")
    tests = report["tests"]
    summary = report["summary"]
    if (type(summary.get("total")) is not int or summary["total"] != len(tests)
            or type(summary.get("collected")) is not int or summary["collected"] < len(tests)):
        raise ValueError("JSON report total contradicts tests")
    failures, evaluated, phase_error = False, [], False
    for test in tests:
        if not isinstance(test, dict) or not isinstance(test.get("nodeid"), str):
            raise ValueError("invalid JSON test")
        outcome = test.get("outcome")
        if outcome not in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
            raise ValueError("invalid test outcome")
        for phase in ("setup", "call", "teardown"):
            info = test.get(phase)
            if info is None and phase == "call":
                continue
            if not isinstance(info, dict) or info.get("outcome") not in ("passed", "failed", "skipped"):
                raise ValueError("invalid or missing pytest phase")
            if info["outcome"] == "failed":
                failures = True
                phase_error |= phase != "call"
        call = test.get("call", {})
        if test["setup"]["outcome"] != "passed" and call:
            raise ValueError("call present after unsuccessful setup")
        if call.get("outcome") in ("passed", "failed") and outcome not in ("xfailed", "xpassed"):
            evaluated.append(call["outcome"] == "passed")
        if outcome == "passed" and call.get("outcome") != "passed":
            raise ValueError("passed test without passed call")
        if outcome in ("failed", "error") and not any(
                test.get(p, {}).get("outcome") == "failed" for p in ("setup", "call", "teardown")):
            raise ValueError("failed test without failing phase")
        if outcome in ("passed", "skipped", "xfailed", "xpassed") and any(
                test.get(p, {}).get("outcome") == "failed" for p in ("setup", "call", "teardown")):
            raise ValueError("test outcome contradicts failing phase")
    for outcome in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
        if summary.get(outcome, 0) != sum(t["outcome"] == outcome for t in tests):
            raise ValueError("JSON summary contradicts test outcomes")
    if not tests:
        raise ValueError("empty test report is not an evaluated run")
    if (rc == 0 and failures) or (rc == 1 and not failures):
        raise ValueError("pytest returncode contradicts phases")
    if phase_error:
        return "error", None, "pytest_phase_error"
    return "ok", all(evaluated) if evaluated else None, None if evaluated else "no_evaluated_tests"


def _trace_component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("invalid trace path component")
    return value


def _publish_traces(directory, report, results_root, llm, agent):
    """Validate the complete private set before publishing any captured bytes."""
    sources = sorted(_safe(directory).iterdir())
    if not sources and not isinstance(report, dict):
        return []
    # Debug exports may survive cancellation: validate the report's own recorded
    # exitcode, without replacing the actual process rc or its primary error.
    _classify(report, report.get("exitcode") if isinstance(report, dict) else None)
    nodes = {test["nodeid"] for test in report["tests"]}
    required = {test["nodeid"] for test in report["tests"]
                if test.get("call", {}).get("outcome") in ("passed", "failed")
                and test["outcome"] not in ("xfailed", "xpassed")}
    validated, run_ids, associated = [], set(), set()
    for source in sources:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.json", source.name):
            raise ValueError("invalid trace filename")
        info = _safe(source).stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("trace is not a private regular file")
        with source.open("rb") as stream:
            content = stream.read()
        data = json.loads(content)
        json.dumps(data, allow_nan=False)  # Reject non-standard NaN/Infinity JSON.
        fields = {"schema_version", "run_id", "status", "query", "error", "runs_collected",
                  "trajectory", "tool_calls", "test_nodeid"}
        if (not isinstance(data, dict) or not fields.issubset(data)
                or type(data["schema_version"]) is not int or data["schema_version"] != 1
                or not isinstance(data["run_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", data["run_id"])
                or data["run_id"] in run_ids or data["status"] not in ("ok", "error")
                or type(data["runs_collected"]) is not int or data["runs_collected"] < 0
                or not isinstance(data["trajectory"], list) or not isinstance(data["tool_calls"], list)
                or not isinstance(data["test_nodeid"], str) or data["test_nodeid"] not in nodes):
            raise ValueError("invalid trace schema or association")
        if (data["error"] is not None and not isinstance(data["error"], str)
                or any(not isinstance(name, str) for name in data["trajectory"])):
            raise ValueError("invalid trace error or trajectory")
        for call in data["tool_calls"]:
            if (not isinstance(call, dict) or not {"name", "input", "start", "end"}.issubset(call)
                    or not isinstance(call["name"], str)
                    or any(call[key] is not None and not isinstance(call[key], str) for key in ("start", "end"))
                    or ("output" in call) == ("error" in call)
                    or "error" in call and not isinstance(call["error"], str)):
                raise ValueError("invalid trace tool call")
        run_ids.add(data["run_id"])
        associated.add(data["test_nodeid"])
        validated.append((source.name, content, data))
    if required - associated:
        raise ValueError("missing trace for evaluated test")
    if not validated:
        return []
    execution = uuid.uuid4().hex
    if not isinstance(execution, str) or not re.fullmatch(r"[0-9a-f]{32}", execution):
        raise ValueError("invalid trace execution identity")
    relative = Path("08_traces") / _trace_component(llm) / _trace_component(agent) / execution
    destination = _safe(results_root / relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _safe(destination.parent)
    destination.mkdir(mode=0o700)  # An existing execution directory is never reused.
    created, artifacts = [], []
    try:
        for name, content, data in validated:
            path = _safe(destination / name)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            created.append(path)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
            artifacts.append(dict(path=(relative / name).as_posix(), run_id=data["run_id"],
                                  test_nodeid=data["test_nodeid"], capture_status=data["status"]))
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return artifacts


def run_pytest_for_agents(
    *,
    llm: str,
    agent_filter: str | None,
    test_filename: str,
    generated_dir: Path,
    bench_dir: Path,
    stage_label: str,
    stage_prefix: str,
    report_name_stem: str,
    experiments_results: Path,
    run: int = 1,
    experiment: str = "by_tool",
    trace_enabled: bool = False,
) -> dict[str, Any]:
    """Run ``test_filename`` for every agent generated by ``llm`` and build a report.

    Parameters
    ----------
    llm
        LLM folder under ``generated_dir`` (e.g. ``"claude"``).
    agent_filter
        If provided, run only this agent folder (e.g. ``"001_list_issues_agent"``).
    test_filename
        Test file to copy from ``bench_dir/<agent>/`` into a private workspace;
        pytest keeps the agent directory as cwd for its imports.
    stage_label
        Human-readable stage used in the top-level ``"stage"`` field.
    stage_prefix
        Filename prefix for the written report (e.g. ``"07"`` or ``"08"``).
    report_name_stem
        Second segment of the filename (e.g. ``"test_deepeval"`` or
        ``"trace_eval"``).
    """

    if trace_enabled:
        for value in (llm, experiment, *([agent_filter] if agent_filter is not None else [])):
            _trace_component(value)
        if type(run) is not int or run < 1:
            raise ValueError("invalid trace run component")
    generated_root = generated_dir / experiment / f"run_{run}"
    experiments_root = experiments_results / experiment / f"run_{run}"
    if trace_enabled:
        experiments_root = _safe(experiments_root)

    model_folder = generated_root / llm
    results: list[dict[str, Any]] = []
    selection_error = None
    interrupted = None
    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        targets = ([model_folder / agent_filter] if agent_filter else
                   sorted(p for p in _safe(model_folder).iterdir() if p.is_dir() or p.is_symlink()))
    except (OSError, ValueError) as exc:
        targets, selection_error = [], str(exc)

    try:
        for agent_folder in targets:
            entry: dict[str, Any] = {
                "model": llm,
                "agent_folder": agent_folder.name,
                "agent_file": "agent.py",
                "test_file": test_filename,
                "passed": None,
                "returncode": None,
                "stdout": None,
                "stderr": None,
                "error": None,
                "summary": None,
                "tests": [],
                "pytest_report": None,
                "status": "error",
                "reason_code": None,
            }
            if trace_enabled:
                entry["trace_artifacts"] = []
            try:
                entry["reason_code"] = "invalid_input"
                if trace_enabled:
                    _trace_component(agent_folder.name)
                if Path(test_filename).name != test_filename or Path(test_filename).suffix != ".py":
                    raise ValueError("test filename must be a Python basename")
                agent_file = _safe(agent_folder / "agent.py")
                source_test = _safe(bench_dir / agent_folder.name / test_filename)
                if not agent_file.is_file():
                    entry["reason_code"] = "missing_agent"
                    raise FileNotFoundError("agent.py not found")
                if not source_test.is_file():
                    entry["reason_code"] = "missing_test"
                    raise FileNotFoundError(f"{test_filename} not found in specification directory")
                entry["reason_code"] = "invalid_timeout"
                timeout = float(os.environ.get("PYTEST_TIMEOUT_SECONDS", "300"))
                if not math.isfinite(timeout) or timeout <= 0:
                    raise ValueError("PYTEST_TIMEOUT_SECONDS must be finite and positive")
                with tempfile.TemporaryDirectory(prefix="spec2agent-pytest-") as temp:
                    workspace = Path(temp)
                    target_test, report_path = workspace / test_filename, workspace / PYTEST_REPORT_FILENAME
                    shutil.copyfile(source_test, target_test)
                    config = workspace / "pytest.ini"
                    config.write_text("[pytest]\nasyncio_mode = strict\n", encoding="utf-8")
                    env = os.environ.copy()
                    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
                        env.pop(key, None)
                    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", DEEPEVAL_TELEMETRY_OPT_OUT="1")
                    if trace_enabled:
                        (workspace / "traces").mkdir(mode=0o700)
                        env.update(TRACE_DIR=str(workspace / "traces"), SPEC2AGENT_TRACE_MANAGED="1",
                                   PYTHONPATH=str(Path(__file__).absolute().parent),
                                   LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")
                    args = [sys.executable, "-m", "pytest", str(target_test), "-v", "--tb=long", "-rN",
                            "-c", str(config), "--confcutdir", str(workspace), "--noconftest",
                            "--json-report", f"--json-report-file={report_path}",
                            "--json-report-omit=keywords,warnings,streams,collectors"]
                    for plugin in ("pytest_jsonreport.plugin", "pytest_asyncio.plugin", "deepeval.plugins.plugin"):
                        args += ["-p", plugin]
                    entry["reason_code"] = "pytest_spawn"
                    rc, stdout, stderr, reason = _execute(args, _safe(agent_folder), env, timeout,
                                                         lambda: interrupted is not None)
                    entry.update(returncode=rc, stdout=stdout, stderr=stderr)
                    report = _load_report(report_path)
                    entry["pytest_report"] = report
                    if isinstance(report, dict):
                        entry["summary"] = report.get("summary")
                    try:
                        entry["reason_code"] = reason or ("pytest_exit" if rc not in (0, 1) else "pytest_report_invalid")
                        if reason or rc not in (0, 1):
                            raise ValueError(f"pytest did not complete normally ({entry['reason_code']}; rc={rc})")
                        status, passed, reason = _classify(report, rc)
                        entry.update(status=status, passed=passed, reason_code=reason)
                        entry["tests"] = _normalize_tests(report)
                        if status == "error":
                            entry["error"] = "pytest setup or teardown failed"
                    finally:
                        if trace_enabled:
                            primary_failure = sys.exc_info()[0] is not None
                            try:
                                entry["trace_artifacts"] = _publish_traces(
                                    workspace / "traces", report, experiments_root, llm, agent_folder.name)
                            except Exception as exc:
                                diagnostic = f"trace validation/publication failed ({type(exc).__name__})"
                                if primary_failure or entry["status"] == "error":
                                    entry["trace_error"] = diagnostic
                                else:
                                    entry.update(status="error", passed=None, reason_code="trace_artifacts_invalid",
                                                 error=diagnostic)
            except Exception as exc:
                entry.update(status="error", passed=None)
                entry["reason_code"] = entry["reason_code"] or "pytest_report_invalid"
                entry["error"] = str(exc)
            results.append(entry)
            if interrupted:
                break
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_report = {
        "run_id": run_id,
        "run": run,
        "experiment": experiment,
        "stage": stage_label,
        "model": llm,
        "test": test_filename,
        "agent_filter": agent_filter,
        "bench_dir": str(bench_dir),
        "generated_dir": str(generated_root),
        "total_agents": len(results),
        "results": results,
        "status": "error" if not results or any(e["status"] == "error" for e in results) else "ok",
    }
    if not results:
        final_report.update(reason_code="empty_targets", error=selection_error or "no agents selected")
    final_report["exit_code"] = 128 + interrupted if interrupted else int(final_report["status"] == "error")

    if trace_enabled:
        _safe(experiments_root)
    report_path = (
        experiments_root
        / f"{stage_prefix}_{report_name_stem}_{llm}_{run_id}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"{stage_label} completed. Report saved to {report_path}")
    return final_report
