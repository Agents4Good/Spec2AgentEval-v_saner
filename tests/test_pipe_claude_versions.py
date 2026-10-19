"""Synthetic orchestration tests: never invoke repository pipeline or wrappers."""

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "pipe_claude_versions.sh",
                    root / "pipe_claude_versions.sh")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (root / "docker").mkdir()
    recorder = root / "record.py"
    recorder.write_text('''import json, os, sys, time
from pathlib import Path
mode, *args = sys.argv[1:]
if mode == "local":
    step = Path(args.pop(0)).name[:2]
elif mode == "gen":
    step = "01"
else:
    assert args.pop(0) == "--step"
    step = args.pop(0)
log = Path(os.environ["CALL_LOG"])
with log.open("a") as stream:
    stream.write(json.dumps([mode, step, args]) + "\\n")
if step == "01":
    time.sleep(0.03)
    Path(os.environ["GEN_DONE"]).touch()
else:
    assert Path(os.environ["GEN_DONE"]).exists(), "evaluation before generation"
sys.exit(17 if os.environ.get("FAIL_STEP") == step else 0)
''')

    def executable(path, text):
        path.write_text("#!/bin/bash\n" + text)
        path.chmod(0o755)

    runner = f"exec {shlex.quote(sys.executable)} {shlex.quote(str(recorder))}"
    executable(bin_dir / "python3", runner + ' local "$@"\n')
    executable(root / "docker/docker_run_gen.sh", runner + ' gen "$@"\n')
    executable(root / "docker/docker_run_eval.sh", runner + ' eval "$@"\n')
    executable(bin_dir / "sleep", 'exec /bin/sleep 0.001\n')
    executable(bin_dir / "tput", 'exit 0\n')

    def run(**overrides):
        log = root / "calls.jsonl"
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(root), "LC_ALL": "C.UTF-8",
               "CALL_LOG": str(log), "GEN_DONE": str(root / "generation-done"),
               "EXPERIMENT": "experiment with spaces; literal $value"}
        env.update(overrides)
        result = subprocess.run(["bash", str(root / "pipe_claude_versions.sh")],
                                env=env, capture_output=True, text=True, timeout=15)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls, env["EXPERIMENT"]

    return run


def expected_calls(docker, runs, experiment, last=8):
    return [[("gen" if step == 1 else "eval") if docker else "local",
             f"{step:02}", ["--llm", "claude-opus-5-5-low", "--agent",
                            "001_list_issues_agent", "--run", str(run),
                            "--experiment", experiment]]
            for run in runs for step in range(1, last + 1)]


@pytest.mark.parametrize("docker", [False, True])
def test_single_run_dispatch(sandbox, docker):
    options = {"RUN": "1"}
    if docker:
        options["USE_DOCKER"] = "1"
    result, calls, experiment = sandbox(**options)
    assert result.returncode == 0, result.stderr
    assert calls == expected_calls(docker, [1], experiment)
    assert "RUN 1 (selected 1..1)" in result.stdout
    assert "RUN 1 / 3" not in result.stdout


@pytest.mark.parametrize("options,runs", [({"RUNS": "2"}, [1, 2]),
    ({"RUNS": "4", "RUNS_FROM": "2", "RUNS_TO": "3"}, [2, 3])])
@pytest.mark.parametrize("docker", [False, True])
def test_run_ranges(sandbox, options, runs, docker):
    result, calls, experiment = sandbox(**options, USE_DOCKER=str(int(docker)))
    assert result.returncode == 0, result.stderr
    assert calls == expected_calls(docker, runs, experiment)


@pytest.mark.parametrize("docker", [False, True])
@pytest.mark.parametrize("step", ["01", "04"])
def test_failure_stops_pipeline(sandbox, docker, step):
    result, calls, experiment = sandbox(RUNS="2", USE_DOCKER=str(int(docker)),
                                         FAIL_STEP=step)
    assert result.returncode == (1 if step == "01" else 17)
    assert calls == expected_calls(docker, [1], experiment, last=int(step))
