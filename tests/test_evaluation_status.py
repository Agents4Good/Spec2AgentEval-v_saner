"""Synthetic operational-status contracts; evaluator sources are never imported."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["02_static_check", "04_tests_check", "05_spec_llm_judge_check", "06_graph_eval_check"]


@pytest.fixture
def stage(tmp_path, monkeypatch):
    for key in list(os.environ):
        monkeypatch.delenv(key)
    for key, value in {
        "HOME": str(tmp_path / "home"), "PATH": "",
        "GENERATED_DIR": str(tmp_path / "generated"),
        "BENCH_DIR": str(tmp_path / "bench"),
        "EXPERIMENTS_RESULTS": str(tmp_path / "results"),
        "SPEC2AGENT_SKIP_DOTENV": "1", "PYTHON_DOTENV_DISABLED": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    }.items():
        monkeypatch.setenv(key, value)

    def run(script, selected="target", contents="def agent(): pass\n", judge=None, fault=None,
            missing=None, graph_fault=None, batch=False, batch_success=False,
            exception_message=None):
        folder = tmp_path / "generated/by_tool/run_1/fake/target"
        if contents is not None:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "agent.py").write_text(contents)
        bench = tmp_path / "bench/target"
        bench.mkdir(parents=True, exist_ok=True)
        for name in ("spec.md", "spec_adherence.py", "eval_2_graph.py"):
            (bench / name).touch()
        if missing:
            (bench / missing).unlink()
        if batch:
            (folder.parent / "absent").mkdir()
        if batch_success:
            other = folder.parent / "other"
            other.mkdir()
            (other / "agent.py").write_text("# synthetic sibling")
            other_bench = bench.parent / "other"
            other_bench.mkdir()
            for name in ("spec.md", "spec_adherence.py", "eval_2_graph.py"):
                (other_bench / name).touch()
        spec = importlib.util.spec_from_file_location("synthetic_stage", ROOT / "pipeline" / f"{script}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def loader(path):
            if path.parent.name == "target" and (
                    fault is True or (fault == "evaluator" and path.name != "agent.py")):
                raise RuntimeError(exception_message or "synthetic load failure")
            def evaluate(*args, **kwargs):
                if graph_fault == "evaluate" and path.parent.name == "target":
                    raise RuntimeError(exception_message or "synthetic evaluation failure")
                return judge
            def build():
                if graph_fault == "build":
                    raise RuntimeError("synthetic build failure")
                return object()
            obj = SimpleNamespace(run_static_eval=evaluate, validate_architecture=evaluate)
            if graph_fault != "missing":
                obj.build_graph = build
            return obj
        if script.startswith("05"):
            monkeypatch.setattr(module, "load_module", loader)
        if script.startswith("06"):
            monkeypatch.setattr(module, "load_module_from_path", loader)
        monkeypatch.setattr(sys, "argv", [script, "--llm", "fake"] +
                            (["--agent", selected] if selected else []))
        code = module.main()
        reports = list((tmp_path / "results/by_tool/run_1").glob("*.json"))
        return code, json.loads(reports[-1].read_text()), module
    return run


@pytest.mark.parametrize("script", SCRIPTS)
def test_missing_requested_target_is_reported(stage, script):
    code, report, _ = stage(script, contents=None)
    assert code == 1
    assert report["status"] == "error"
    assert report["total_agents"] == 1
    assert report["results"][0]["reason_code"] == "missing_agent"


@pytest.mark.parametrize("script", SCRIPTS)
def test_empty_discovery_is_error(stage, script):
    code, report, _ = stage(script, selected=None, contents=None)
    assert code == 1
    assert report["status"] == "error"
    assert report["reason_code"] in {"missing_generated_root", "missing_model_root", "empty_selection"}


@pytest.mark.parametrize("contents", ["invalid python !", "x = 1", "import missing_synthetic_dependency_xyz"])
def test_static_metric_failure_is_operational_ok(stage, contents):
    code, report, _ = stage(SCRIPTS[0], contents=contents)
    assert code == 0
    assert report["status"] == "ok"
    assert report["results"][0]["final_status"] == "FAILED"


def test_no_tests_is_valid_result(stage):
    code, report, _ = stage(SCRIPTS[1])
    assert code == 0
    assert report["results"][0]["tests"] == {"exists": False, "count": 0, "files": []}


@pytest.mark.parametrize("script", SCRIPTS[2:])
@pytest.mark.parametrize("judge", [False, {"passed": False}, None])
def test_judge_is_opaque(stage, script, judge):
    code, report, _ = stage(script, judge=judge)
    entry = report["results"][0]
    assert code == 0 and entry["status"] == "ok"
    assert entry["judge"] == judge
    assert "passed" not in entry


@pytest.mark.parametrize("script", SCRIPTS[2:])
def test_load_exception_saved(stage, script):
    code, report, _ = stage(script, fault=True)
    assert code == 1
    assert report["results"][0]["status"] == "error"
    assert isinstance(report["results"][0]["error"], str)


@pytest.mark.parametrize("script,missing,reason", [
    (SCRIPTS[2], "spec.md", "missing_spec"),
    (SCRIPTS[2], "spec_adherence.py", "missing_evaluator"),
    (SCRIPTS[3], "eval_2_graph.py", "missing_evaluator"),
])
def test_required_inputs(stage, script, missing, reason):
    code, report, _ = stage(script, missing=missing)
    assert code == 1
    assert report["results"][0]["reason_code"] == reason


@pytest.mark.parametrize("fault,reason", [("missing", "missing_build_graph"),
                                          ("build", "graph_build_error"),
                                          ("evaluate", "evaluation_error")])
def test_graph_errors(stage, fault, reason):
    code, report, _ = stage(SCRIPTS[3], graph_fault=fault)
    assert code == 1
    assert report["results"][0]["reason_code"] == reason


def test_static_evaluation_exception(stage):
    code, report, _ = stage(SCRIPTS[2], graph_fault="evaluate")
    assert code == 1
    assert report["results"][0]["reason_code"] == "evaluation_error"


def test_graph_evaluator_import_exception(stage):
    code, report, _ = stage(SCRIPTS[3], fault="evaluator")
    assert code == 1
    assert report["results"][0]["reason_code"] == "evaluation_error"


@pytest.mark.parametrize("script", SCRIPTS)
def test_batch_continues_preserving_success(stage, script):
    code, report, _ = stage(script, selected=None, batch=True, judge={"negative": True})
    assert code == 1
    entries = {entry["agent_folder"]: entry for entry in report["results"]}
    assert entries["absent"]["reason_code"] == "missing_agent"
    assert entries["target"]["status"] == "ok"
    assert report["total_agents"] == 2


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("root_kind", ["empty", "missing_model"])
def test_empty_existing_roots(stage, script, root_kind, tmp_path):
    root = tmp_path / "generated/by_tool/run_1"
    (root / "fake" if root_kind == "empty" else root).mkdir(parents=True)
    code, report, _ = stage(script, selected=None, contents=None)
    assert code == 1
    assert report["reason_code"] == ("empty_selection" if root_kind == "empty" else "missing_model_root")


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_nonzero_after_report(stage, script, tmp_path):
    # stage fixture supplies an entirely synthetic environment; no real inputs.
    result = subprocess.run(
        [str(Path(sys.executable).absolute()), str(ROOT / "pipeline" / f"{script}.py"),
         "--llm", "fake", "--agent", "target"],
        env=dict(os.environ), capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    reports = list((tmp_path / "results/by_tool/run_1").glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["status"] == "error"


@pytest.mark.parametrize("script", SCRIPTS[2:])
@pytest.mark.parametrize("boundary", ["import", "call"])
def test_evaluation_error_does_not_persist_sensitive_message(stage, script, boundary, capsys, tmp_path):
    url = "https://provider.invalid/?key=FAKE_SELECTED_KEY"
    judge = {"passed": False, "detail": "opaque sibling verdict"}
    code, report, _ = stage(
        script, selected=None, batch_success=True, judge=judge,
        fault="evaluator" if boundary == "import" else None,
        graph_fault="evaluate" if boundary == "call" else None,
        exception_message=f"request failed: {url}")
    assert code == 1 and report["status"] == "error"
    entries = {entry["agent_folder"]: entry for entry in report["results"]}
    failed = entries["target"]
    assert failed["status"] == "error" and failed["reason_code"] == "evaluation_error"
    label = "Spec" if script.startswith("05") else "Graph"
    assert failed["error"] == f"{label} evaluation failed (RuntimeError)"
    assert failed["judge"] is None
    assert entries["other"]["status"] == "ok" and entries["other"]["judge"] == judge
    assert len(entries) == 2
    captured = capsys.readouterr()
    persisted = "".join(p.read_text() for p in (tmp_path / "results").rglob("*.json"))
    for forbidden in ("FAKE_SELECTED_KEY", url, "request failed:"):
        assert forbidden not in persisted
        assert forbidden not in captured.out + captured.err
