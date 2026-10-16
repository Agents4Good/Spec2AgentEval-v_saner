"""Lane A: adapter subprocess contract tests.

Real fake-CLI subprocesses (executables in a temp bin dir) with a
fake-bin-only PATH; the real installed CLIs are never invoked. Fake
scripts use an absolute interpreter shebang so no system PATH entry is
needed. Covers the three canonical adapters only; Copilot/Gemini get a
light untouched smoke test. Nothing here reads .env.
"""
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from adapters.claude import ClaudeCodeAdapter  # noqa: E402
from adapters.codex import CodexAdapter  # noqa: E402
from adapters.open_code import OpenCodeAdapter  # noqa: E402
from utils.generation_usage import USAGE_FIELDS  # noqa: E402

SENTINEL = "SECRET-SENTINEL-7f3a2b9d"

CLAUDE_PAYLOAD = {
    "result": "fake implementation",
    "usage": {
        "input_tokens": 100,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 10,
        "output_tokens": 50,
    },
    "num_turns": 2,
    "total_cost_usd": 0.0123,
    "duration_api_ms": 1234,
    "duration_ms": 1500,
}

CODEX_LINES = [
    {"type": "turn.completed", "thread_id": "t1",
     "usage": {"input_tokens": 10, "output_tokens": 5,
               "cached_input_tokens": 2, "reasoning_output_tokens": 1}},
    {"type": "turn.completed", "thread_id": "t1",
     "usage": {"input_tokens": 30, "output_tokens": 12,
               "cached_input_tokens": 4, "reasoning_output_tokens": 3}},
]

OPENCODE_LINES = [
    {"type": "step_finish", "sessionID": "s1",
     "part": {"id": "p1", "messageID": "m1",
              "tokens": {"input": 10, "output": 5, "reasoning": 1,
                         "cache": {"read": 2, "write": 3}, "total": 16},
              "cost": 0.001}},
    {"type": "step_finish", "sessionID": "s1",
     "part": {"id": "p2", "messageID": "m1",
              "tokens": {"input": 20, "output": 7, "reasoning": 2,
                         "cache": {"read": 4, "write": 5}, "total": 29},
              "cost": 0.002}},
]


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setenv("SPEC2AGENT_SKIP_DOTENV", "1")


def _write_exe(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _isolated_path(monkeypatch, bindir: Path) -> None:
    """Fake-bin-only PATH: no system dir, no real CLI resolvable."""
    monkeypatch.setenv("PATH", str(bindir))


def _spec_and_out(tmp_path: Path, name: str = "agent1"):
    spec_dir = tmp_path / "specs" / name
    spec_dir.mkdir(parents=True)
    (spec_dir / "spec.md").write_text("# fake spec\nDo the thing.\n", encoding="utf-8")
    out = tmp_path / "gen" / name
    out.mkdir(parents=True)
    return spec_dir / "spec.md", out


def _script_head(argv_log: bool) -> list:
    lines = [f"#!{sys.executable}", "import os, sys"]
    if argv_log:
        lines.append('open(os.environ["ARGV_LOG"], "w").write(" ".join(sys.argv[1:]))')
    return lines


def _install_claude(bindir: Path, payload=None, stderr: bytes = b"note: fake stderr\n",
                    exit_code: int = 0, argv_log: bool = False,
                    stdout_suffix: bytes = b"") -> bytes:
    payload = CLAUDE_PAYLOAD if payload is None else payload
    raw = json.dumps(payload).encode("utf-8") + bytes(stdout_suffix)
    lines = _script_head(argv_log) + [
        f"sys.stdout.buffer.write({raw!r})",
        f"sys.stderr.buffer.write({bytes(stderr)!r})",
        "open('agent.py', 'w').write('# fake\\n')",
        f"sys.exit({exit_code})",
    ]
    _write_exe(bindir / "claude", "\n".join(lines) + "\n")
    return raw


def _install_codex(bindir: Path, lines=None, stderr: bytes = b"",
                   exit_code: int = 0, raw: bytes | None = None) -> bytes:
    lines = CODEX_LINES if lines is None else lines
    if raw is None:
        raw = "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8")
    raw = bytes(raw)
    body = _script_head(False) + [
        f"sys.stdout.buffer.write({raw!r})",
        f"sys.stderr.buffer.write({bytes(stderr)!r})",
        "open('agent.py', 'w').write('# fake\\n')",
        f"sys.exit({exit_code})",
    ]
    _write_exe(bindir / "codex", "\n".join(body) + "\n")
    return raw


def _install_opencode(bindir: Path, lines=None, stderr: bytes = b"",
                      exit_code: int = 0, raw: bytes | None = None) -> bytes:
    lines = OPENCODE_LINES if lines is None else lines
    if raw is None:
        raw = "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8")
    raw = bytes(raw)
    body = _script_head(False) + [
        f"sys.stdout.buffer.write({raw!r})",
        f"sys.stderr.buffer.write({bytes(stderr)!r})",
        "open('agent.py', 'w').write('# fake\\n')",
        f"sys.exit({exit_code})",
    ]
    _write_exe(bindir / "opencode", "\n".join(body) + "\n")
    return raw


def _assert_contract_types(adapter, expect_exit: int | None):
    assert isinstance(adapter.usage, dict)
    assert adapter.exit_code == expect_exit
    assert adapter.parse_status in ("ok", "partial", "error")
    assert isinstance(adapter.warnings, list)
    assert all(isinstance(w, str) for w in adapter.warnings)
    assert adapter.error is None or isinstance(adapter.error, str)
    assert isinstance(adapter.log_paths, dict)
    for key in ("session.log", "cli_stderr.log", "credits.log", "01-generate.log"):
        assert key in adapter.log_paths, adapter.log_paths
        assert Path(adapter.log_paths[key]).is_file()
    assert set(adapter.durations) == {"process_duration_s", "wall_duration_s"}


def test_claude_success_ok(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    stdout = _install_claude(bindir)
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    assert adapter.run() == str(out)
    _assert_contract_types(adapter, 0)
    assert adapter.parse_status == "ok"
    assert adapter.error is None
    assert adapter.usage["input_tokens"] == 130
    assert adapter.usage["output_tokens"] == 50
    assert adapter.usage["cache_read_tokens"] == 20
    assert adapter.usage["cache_write_tokens"] == 10
    assert adapter.usage["total_tokens"] == 180
    assert adapter.usage["requests"] == 2
    assert adapter.usage["cost_usd"] == 0.0123
    assert adapter.usage["cli_duration_s"] is None
    assert adapter.usage["source"] == {"cli": "claude", "cli_version": None, "provider": None}
    assert Path(adapter.log_paths["session.log"]).read_bytes() == stdout
    assert Path(adapter.log_paths["cli_stderr.log"]).read_bytes() == b"note: fake stderr\n"
    credits = json.loads(Path(adapter.log_paths["credits.log"]).read_text(encoding="utf-8"))
    assert credits == adapter.usage


def test_codex_success_ok(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    stdout = _install_codex(bindir)
    spec, out = _spec_and_out(tmp_path)
    adapter = CodexAdapter(spec, out)
    assert adapter.run() == str(out)
    _assert_contract_types(adapter, 0)
    assert adapter.parse_status == "ok"
    assert adapter.error is None
    assert adapter.usage["input_tokens"] == 30
    assert adapter.usage["output_tokens"] == 12
    assert adapter.usage["cache_read_tokens"] == 4
    assert adapter.usage["reasoning_tokens"] == 3
    assert adapter.usage["total_tokens"] == 42
    assert adapter.usage["requests"] == 2
    assert Path(adapter.log_paths["session.log"]).read_bytes() == stdout


def test_opencode_success_ok(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    stdout = _install_opencode(bindir)
    spec, out = _spec_and_out(tmp_path)
    adapter = OpenCodeAdapter(spec, out)
    assert adapter.run() == str(out)
    _assert_contract_types(adapter, 0)
    assert adapter.parse_status == "ok"
    assert adapter.error is None
    assert adapter.usage["input_tokens"] == 44
    assert adapter.usage["output_tokens"] == 12
    assert adapter.usage["cache_read_tokens"] == 6
    assert adapter.usage["cache_write_tokens"] == 8
    assert adapter.usage["reasoning_tokens"] == 3
    assert adapter.usage["requests"] == 2
    assert Path(adapter.log_paths["session.log"]).read_bytes() == stdout


def test_nonzero_exit_preserves_parsed_usage(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    stdout = _install_claude(bindir, exit_code=9, stderr=b"boom\n")
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    assert adapter.run() == str(out)
    _assert_contract_types(adapter, 9)
    assert adapter.error == "cli_exit_9"
    assert "boom" not in adapter.error
    assert adapter.usage["input_tokens"] == 130
    assert adapter.parse_status == "ok"
    assert Path(adapter.log_paths["session.log"]).read_bytes() == stdout
    assert Path(adapter.log_paths["cli_stderr.log"]).read_bytes() == b"boom\n"


def test_invalid_stderr_byte_preserved_with_nonzero(tmp_path, monkeypatch):
    """Gate2 finding 2: byte 0xff on stderr + exit 9 keeps usage and exit."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    raw_err = b"warn \xff binary \r\n noise\n"
    stdout = _install_claude(bindir, stderr=raw_err, exit_code=9,
                             stdout_suffix=b"\r\n")
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    assert adapter.run() == str(out)
    _assert_contract_types(adapter, 9)
    assert adapter.error == "cli_exit_9"
    assert adapter.usage["input_tokens"] == 130
    assert adapter.usage["output_tokens"] == 50
    assert adapter.parse_status == "ok"
    assert Path(adapter.log_paths["session.log"]).read_bytes() == stdout
    assert b"\r\n" in Path(adapter.log_paths["session.log"]).read_bytes()
    assert Path(adapter.log_paths["cli_stderr.log"]).read_bytes() == raw_err
    credits_raw = Path(adapter.log_paths["credits.log"]).read_bytes()
    assert b"\xff" not in credits_raw
    assert json.loads(credits_raw.decode("utf-8")) == adapter.usage


def test_invalid_stdout_utf8_is_safe_parse_error(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    raw = b"\xff\xfe not utf8\n"
    _install_codex(bindir, raw=raw, stderr=b"ok\n")
    spec, out = _spec_and_out(tmp_path)
    adapter = CodexAdapter(spec, out)
    assert adapter.run() == str(out)
    _assert_contract_types(adapter, 0)
    assert adapter.parse_status == "error"
    assert adapter.error == "parse_error"
    assert adapter.usage["input_tokens"] is None
    assert "stdout is not valid UTF-8; no usage data" in adapter.warnings
    assert all("Traceback" not in w for w in adapter.warnings)
    assert Path(adapter.log_paths["session.log"]).read_bytes() == raw
    assert Path(adapter.log_paths["cli_stderr.log"]).read_bytes() == b"ok\n"


def test_malformed_stdout_is_parse_error(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_codex(bindir, raw=b"not json {{{\n", stderr=b"warn\n")
    spec, out = _spec_and_out(tmp_path)
    adapter = CodexAdapter(spec, out)
    assert adapter.run() == str(out)
    _assert_contract_types(adapter, 0)
    assert adapter.parse_status == "error"
    assert adapter.error == "parse_error"
    assert adapter.usage["input_tokens"] is None
    assert adapter.usage["requests"] is None
    assert adapter.warnings
    assert Path(adapter.log_paths["session.log"]).read_bytes() == b"not json {{{\n"


def test_partial_usage_is_not_failure(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    payload = {"result": "x",
               "usage": {"input_tokens": 100, "output_tokens": 50},
               "num_turns": 1, "total_cost_usd": 0.01}
    _install_claude(bindir, payload=payload)
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    adapter.run()
    _assert_contract_types(adapter, 0)
    assert adapter.parse_status == "partial"
    assert adapter.error is None
    assert adapter.usage["input_tokens"] is None
    assert adapter.usage["output_tokens"] == 50
    assert adapter.warnings


def test_spawn_failure_records_metadata(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()  # empty: no fake, no interpreter dir, no real CLI
    _isolated_path(monkeypatch, bindir)
    assert os.environ["PATH"] == str(bindir)
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out, model="m")
    assert adapter.run() == str(out)  # returns, never raises
    _assert_contract_types(adapter, None)
    assert adapter.error is not None and "spawn_failed" in adapter.error
    assert "Traceback" not in adapter.error
    assert "FileNotFoundError" not in adapter.error
    assert adapter.parse_status == "error"
    assert adapter.usage["input_tokens"] is None
    assert Path(adapter.log_paths["session.log"]).read_bytes() == b""
    assert Path(adapter.log_paths["cli_stderr.log"]).read_bytes() == b""


def test_stderr_kept_separate_from_usage(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    marker = b"STDERR-MARKER-123\n"
    stdout = _install_opencode(bindir, stderr=marker)
    spec, out = _spec_and_out(tmp_path)
    adapter = OpenCodeAdapter(spec, out)
    adapter.run()
    _assert_contract_types(adapter, 0)
    assert adapter.error is None
    assert Path(adapter.log_paths["session.log"]).read_bytes() == stdout
    assert Path(adapter.log_paths["cli_stderr.log"]).read_bytes() == marker
    credits = Path(adapter.log_paths["credits.log"]).read_text(encoding="utf-8")
    assert marker.decode().strip() not in credits
    assert adapter.usage["input_tokens"] == 44


def test_logs_never_overwrite_existing(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    stdout = _install_claude(bindir)
    spec, out = _spec_and_out(tmp_path)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "session.log").write_bytes(b"OLD-SESSION")
    (logs / "cli_stderr.log").write_bytes(b"OLD-STDERR")
    (logs / "credits.log").write_bytes(b"OLD-CREDITS")
    adapter = ClaudeCodeAdapter(spec, out)
    adapter.run()
    assert (logs / "session.log").read_bytes() == b"OLD-SESSION"
    assert (logs / "cli_stderr.log").read_bytes() == b"OLD-STDERR"
    assert (logs / "credits.log").read_bytes() == b"OLD-CREDITS"
    assert Path(adapter.log_paths["session.log"]).read_bytes() == stdout
    assert Path(adapter.log_paths["session.log"]) != logs / "session.log"
    assert Path(adapter.log_paths["cli_stderr.log"]) != logs / "cli_stderr.log"
    assert Path(adapter.log_paths["credits.log"]) != logs / "credits.log"


def test_persistence_failure_preserves_status_and_usage(tmp_path, monkeypatch):
    """Gate2 finding 3: credits.log OSError keeps exit 9 + parsed usage."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_claude(bindir, exit_code=9)
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    real_write = adapter._write_new

    def fail_credits(path, data):
        if Path(path).name == "credits.log":
            raise OSError("injected persistence fault")
        return real_write(path, data)

    monkeypatch.setattr(adapter, "_write_new", fail_credits)
    assert adapter.run() == str(out)
    assert adapter.exit_code == 9
    assert adapter.usage["input_tokens"] == 130
    assert adapter.parse_status == "ok"
    assert adapter.error == "cli_exit_9"
    assert "spawn_failed" not in (adapter.error or "")
    assert "persistence_failed: credits.log" in adapter.warnings
    assert "credits.log" not in adapter.log_paths
    assert Path(adapter.log_paths["session.log"]).is_file()
    assert Path(adapter.log_paths["cli_stderr.log"]).is_file()

    # Clean run with only a persistence fault and no process error.
    _install_claude(bindir, exit_code=0)
    spec2, out2 = _spec_and_out(tmp_path, name="agent2")
    adapter2 = ClaudeCodeAdapter(spec2, out2)
    real_write2 = adapter2._write_new

    def fail_credits2(path, data):
        if Path(path).name == "credits.log":
            raise OSError("injected persistence fault")
        return real_write2(path, data)

    monkeypatch.setattr(adapter2, "_write_new", fail_credits2)
    adapter2.run()
    assert adapter2.exit_code == 0
    assert adapter2.usage["input_tokens"] == 130
    assert adapter2.parse_status == "ok"
    assert adapter2.error == "persistence_failed"
    assert "persistence_failed: credits.log" in adapter2.warnings


def test_exclusive_writer_race(tmp_path):
    """Gate2 finding 4: barrier-controlled writers land on distinct files."""
    import threading

    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    target = adapter.logs_dir / "race.log"
    barrier = threading.Barrier(2)
    results = {}

    def worker(key, payload):
        barrier.wait()
        results[key] = adapter._write_new(target, payload)

    threads = [threading.Thread(target=worker, args=("a", b"A" * 16)),
               threading.Thread(target=worker, args=("b", b"B" * 16))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["a"] != results["b"]
    assert results["a"].read_bytes() == b"A" * 16
    assert results["b"].read_bytes() == b"B" * 16


def test_fake_bin_only_isolation(tmp_path, monkeypatch):
    """Gate2 finding 6: absolute shebang + PATH with only the fake bin."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_claude(bindir)
    assert (bindir / "claude").read_text(encoding="utf-8").splitlines()[0] == f"#!{sys.executable}"
    assert os.environ["PATH"] == str(bindir)
    assert shutil.which("claude") == str(bindir / "claude")
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    assert adapter.run() == str(out)
    assert adapter.exit_code == 0


def test_durations_monotonic_and_mirrored(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_codex(bindir)
    spec, out = _spec_and_out(tmp_path)
    adapter = CodexAdapter(spec, out)
    adapter.run()
    proc = adapter.usage["process_duration_s"]
    wall = adapter.usage["wall_duration_s"]
    assert isinstance(proc, float) and isinstance(wall, float)
    assert proc >= 0 and wall >= 0
    assert proc <= wall + 1e-6
    assert adapter.durations == {"process_duration_s": proc, "wall_duration_s": wall}


def test_usage_schema_and_null_provenance(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_claude(bindir)
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out, model="claude-sonnet-4-5")
    adapter.run()
    assert set(adapter.usage) == set(USAGE_FIELDS)
    assert adapter.usage["source"] == {"cli": "claude", "cli_version": None, "provider": None}
    assert adapter.usage["cli_duration_s"] is None
    assert adapter.usage["requested_model"] == "claude-sonnet-4-5"


def test_constructor_defaults_and_effort_flag(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    argv_log = str(tmp_path / "argv.txt")
    monkeypatch.setenv("ARGV_LOG", argv_log)
    spec, out = _spec_and_out(tmp_path)

    assert ClaudeCodeAdapter(spec, out).model == "claude-sonnet-4-5"
    assert ClaudeCodeAdapter(spec, out).effort is None
    assert CodexAdapter(spec, out).model == "gpt-6-luna"
    assert OpenCodeAdapter(spec, out).model == "opencode/longcat-2.5-preview-free"

    _install_claude(bindir, argv_log=True)
    ClaudeCodeAdapter(spec, out, effort="low").run()
    argv = Path(os.environ["ARGV_LOG"]).read_text(encoding="utf-8")
    assert "--effort low" in argv

    _install_claude(bindir, argv_log=True)
    ClaudeCodeAdapter(spec, out).run()
    argv = Path(argv_log).read_text(encoding="utf-8")
    assert "--effort" not in argv


def test_copilot_gemini_smoke_untouched(tmp_path, monkeypatch):
    from adapters.copilot import CopilotAdapter
    from adapters.gemini import GeminiAdapter

    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _write_exe(bindir / "copilot", f"#!{sys.executable}\n"
               "import sys\n"
               "sys.stdout.buffer.write('Changes +13 -0\\n'.encode())\n"
               "sys.stderr.buffer.write(('Tokens \\u2191 1.0k (0.5k cached, 0.1k written) "
               "\\u2022 \\u2193 0.2k (0.1k reasoning)\\n6s\\n').encode('utf-8'))\n")
    _write_exe(bindir / "agy", f"#!{sys.executable}\n"
               "import json, sys\n"
               "sys.stdout.buffer.write(json.dumps({'usage': {'input_tokens': 10, "
               "'output_tokens': 5, 'cache_read_tokens': 1}, 'num_turns': 1, "
               "'duration_seconds': 2.5}).encode())\n")

    spec, out = _spec_and_out(tmp_path, name="agent-copilot")
    copilot = CopilotAdapter(spec, out)
    assert copilot.run() == str(out)
    assert isinstance(copilot.usage, dict) and "requests" in copilot.usage

    spec2, out2 = _spec_and_out(tmp_path, name="agent-gemini")
    gemini = GeminiAdapter(spec2, out2)
    assert gemini.run() == str(out2)
    assert gemini.usage.get("requests") == 1


def test_sentinel_never_in_metadata(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    payload = dict(CLAUDE_PAYLOAD, result=f"contains {SENTINEL} inside")
    _install_claude(bindir, payload=payload, stderr=f"warn {SENTINEL}\n".encode())
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    adapter.run()
    blob = json.dumps({"usage": adapter.usage, "error": adapter.error,
                       "warnings": adapter.warnings, "status": adapter.parse_status})
    assert SENTINEL not in blob
    assert SENTINEL in Path(adapter.log_paths["session.log"]).read_text(encoding="utf-8")
    assert SENTINEL in Path(adapter.log_paths["cli_stderr.log"]).read_text(encoding="utf-8")
    step = Path(adapter.log_paths["01-generate.log"]).read_text(encoding="utf-8")
    assert SENTINEL not in step


def _fail_step_calls(adapter, monkeypatch, fail_on):
    """Wrap _log_step to report failure on selected 1-based call numbers."""
    real = adapter._log_step
    seen = []

    def selective(message):
        seen.append(message)
        if len(seen) in fail_on:
            return False
        return real(message)

    monkeypatch.setattr(adapter, "_log_step", selective)
    return seen


def test_step_log_initial_only_failure_on_success(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    stdout = _install_claude(bindir)
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    _fail_step_calls(adapter, monkeypatch, {1})
    assert adapter.run() == str(out)
    assert adapter.exit_code == 0
    assert adapter.usage["input_tokens"] == 130
    assert adapter.parse_status == "ok"
    # Initial diagnostic persists even though the final write succeeded.
    assert adapter.error == "persistence_failed"
    assert "persistence_failed: 01-generate.log" in adapter.warnings
    # Final write landed, so the existing file is advertised.
    assert Path(adapter.log_paths["01-generate.log"]).is_file()


def test_step_log_final_only_failure_on_exit9(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_claude(bindir, exit_code=9)
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    _fail_step_calls(adapter, monkeypatch, {2})
    adapter.run()
    assert adapter.exit_code == 9
    assert adapter.usage["input_tokens"] == 130
    assert adapter.parse_status == "ok"
    assert adapter.error == "cli_exit_9"  # CLI diagnostic keeps priority
    assert "persistence_failed: 01-generate.log" in adapter.warnings
    assert Path(adapter.log_paths["01-generate.log"]).is_file()


def test_step_log_all_failures_on_success(tmp_path, monkeypatch):
    """Review evidence case: denied step log + ok CLI must not read success."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_claude(bindir)
    spec, out = _spec_and_out(tmp_path)
    adapter = ClaudeCodeAdapter(spec, out)
    (adapter.logs_dir / "01-generate.log").mkdir()  # blocks every append
    assert adapter.run() == str(out)
    assert adapter.exit_code == 0
    assert adapter.usage["input_tokens"] == 130
    assert adapter.parse_status == "ok"
    assert adapter.error == "persistence_failed"
    assert "persistence_failed: 01-generate.log" in adapter.warnings
    assert "01-generate.log" not in adapter.log_paths  # never written, never advertised
    assert Path(adapter.log_paths["session.log"]).is_file()
    assert Path(adapter.log_paths["cli_stderr.log"]).is_file()
    assert Path(adapter.log_paths["credits.log"]).is_file()


def test_step_log_all_failures_on_parse_error(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _isolated_path(monkeypatch, bindir)
    _install_codex(bindir, raw=b"not json {{{\n")
    spec, out = _spec_and_out(tmp_path)
    adapter = CodexAdapter(spec, out)
    (adapter.logs_dir / "01-generate.log").mkdir()  # blocks every append
    adapter.run()
    assert adapter.exit_code == 0
    assert adapter.usage["input_tokens"] is None
    assert adapter.parse_status == "error"
    assert adapter.error == "parse_error"  # parse diagnostic keeps priority
    assert "persistence_failed: 01-generate.log" in adapter.warnings
    assert "01-generate.log" not in adapter.log_paths
