"""Shared subprocess collector for generation adapters (phase 2, lane A).

One small helper owns the spawn-to-exit process window, overall wall
timing, separate exact stdout/stderr log files and direct parser calls so
the three canonical CLI adapters stay thin and uniform.

Raw streams are captured and persisted as bytes: stdout is decoded as
explicit UTF-8 strict for parsing only, stderr stays opaque. Collected
metadata (exit code, parsed usage) is retained before any fallible log
persistence, and persistence faults surface as safe diagnostics without
dropping execution evidence.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from utils.generation_usage import parse_generation_usage

SPAWN_FAILED = "spawn_failed"
COLLECT_FAILED = "collect_failed"
PARSE_ERROR = "parse_error"
PERSISTENCE_FAILED = "persistence_failed"

_NON_UTF8_STDOUT = "stdout is not valid UTF-8; no usage data"


class BaseAdapter:
    def __init__(self, spec_path: Path, output_dir: Path):
        self.spec_file = Path(spec_path)
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.usage: dict = {}
        self.exit_code: int | None = None
        self.parse_status: str = "error"
        self.warnings: list = []
        self.error: str | None = None
        self.log_paths: dict = {}
        self.durations: dict = {"process_duration_s": None, "wall_duration_s": None}

    def check_spec(self):
        if not self.spec_file.exists():
            raise FileNotFoundError(f"spec.md not found: {self.spec_file}")

    def run(self):
        raise NotImplementedError

    def _write_new(self, path: Path, data) -> Path:
        """Persist without touching any existing entry, exclusively.

        Every candidate (the target itself, then ``-N`` suffixed siblings)
        is created with ``O_CREAT | O_EXCL`` in a single loop: pre-existing
        files, symlinks and directories are never overwritten, truncated or
        followed, and concurrent writers always land on distinct files.
        Accepts bytes (written exactly) or str (UTF-8 encoded).
        """
        if isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = bytes(data)
        path = Path(path)
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        for i in range(1000):
            cand = path if i == 0 else parent / f"{stem}-{i}{suffix}"
            try:
                fd = os.open(str(cand), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                continue
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(payload)
            except OSError:
                try:
                    os.unlink(str(cand))
                except OSError:
                    pass
                raise
            return cand
        raise FileExistsError(f"no free log name for {path}")

    def _log_step(self, message: str) -> bool:
        """Append a step-log line; True when written.

        A failure never raises and never advertises the path: the caller
        folds it into a safe persistence diagnostic instead.
        """
        try:
            step_log = self.logs_dir / "01-generate.log"
            with step_log.open("a", encoding="utf-8") as f:
                f.write(message + "\n")
        except OSError:
            return False
        self.log_paths["01-generate.log"] = str(step_log)
        return True

    def _collect(self, cmd: list, *, cli: str, model, tag: str) -> dict:
        """Run ``cmd``, persist separate raw logs, parse stdout directly.

        Sets ``usage`` (normalized schema plus numeric durations),
        ``exit_code``, ``parse_status``, ``warnings``, ``error`` (stable
        safe diagnostic, never raw output), ``log_paths`` and ``durations``.
        Exit status and parsed metadata are retained before fallible
        persistence; a log-write fault keeps all collected evidence and
        records a safe ``persistence_failed`` diagnostic instead of
        masquerading as a spawn failure. Expected process failures never
        raise.
        """
        wall_start = time.monotonic()
        self.log_paths = {}
        step_log_failed = False
        if not self._log_step(f"[{tag}] starting cli={cli}"):
            step_log_failed = True

        stdout_b: bytes = b""
        stderr_b: bytes = b""
        exit_code = None
        process_duration = None
        error = None
        try:
            proc_start = time.monotonic()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.output_dir,
                shell=False,
            )
            out, err = process.communicate()
            process_duration = time.monotonic() - proc_start
            exit_code = process.returncode
            if isinstance(out, (bytes, bytearray)):
                stdout_b = bytes(out)
            if isinstance(err, (bytes, bytearray)):
                stderr_b = bytes(err)
        except (FileNotFoundError, OSError):
            process_duration = None
            prog = cmd[0] if cmd else cli
            error = f"{SPAWN_FAILED}: {prog}"
            if not self._log_step(f"[{tag}] spawn failed cli={cli}"):
                step_log_failed = True
        except Exception:
            process_duration = None
            error = COLLECT_FAILED
            if not self._log_step(f"[{tag}] collect failed cli={cli}"):
                step_log_failed = True

        # Retain exit status immediately, before any fallible persistence.
        self.exit_code = exit_code

        try:
            stdout_text = stdout_b.decode("utf-8")
            utf8_ok = True
        except UnicodeDecodeError:
            stdout_text = ""
            utf8_ok = False

        parsed = parse_generation_usage(cli, stdout_text, model=model, cli_version=None)
        usage = parsed["usage"]
        warnings = list(parsed["warnings"])
        if not utf8_ok:
            warnings.append(_NON_UTF8_STDOUT)
        wall_duration = time.monotonic() - wall_start
        usage["process_duration_s"] = process_duration
        usage["wall_duration_s"] = wall_duration

        # Retain parsed metadata before fallible persistence.
        self.usage = usage
        self.parse_status = parsed["parse_status"]
        self.warnings = warnings
        if error is None:
            if exit_code is not None and exit_code != 0:
                error = f"cli_exit_{exit_code}"
            elif parsed["parse_status"] == "error":
                error = PARSE_ERROR
        self.error = error
        self.durations = {
            "process_duration_s": process_duration,
            "wall_duration_s": wall_duration,
        }

        # Persist each raw log independently; a fault keeps collected
        # metadata and still attempts the remaining logs.
        for name, payload in (
            ("session.log", stdout_b),
            ("cli_stderr.log", stderr_b),
            ("credits.log", json.dumps(usage, indent=2, ensure_ascii=False)),
        ):
            try:
                written = self._write_new(self.logs_dir / name, payload)
            except OSError:
                self.warnings.append(f"{PERSISTENCE_FAILED}: {name}")
                if self.error is None:
                    self.error = PERSISTENCE_FAILED
                continue
            self.log_paths[name] = str(written)

        if not self._log_step(f"[{tag}] finished exit={exit_code} parse={self.parse_status}"):
            step_log_failed = True
        if step_log_failed:
            marker = f"{PERSISTENCE_FAILED}: 01-generate.log"
            if marker not in self.warnings:
                self.warnings.append(marker)
            if self.error is None:
                self.error = PERSISTENCE_FAILED
        return self.usage
