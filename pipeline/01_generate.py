# pipeline/steps/01_generate.py
"""Stage 01: run the generation adapter and publish a canonical report.

Phase-2 contract (lane B):

- Canonical per-agent output directories are reserved atomically before
  any CLI starts. A pre-existing, symlinked or otherwise unsafe target is
  rejected as an operational error; reruns never overwrite prior runs
  (choose a new ``--run`` for another execution).
- ``agent.py`` must be a fresh regular symlink-free file contained in the
  output directory, validated after the run even on CLI failure.
- Success and error entries share one stable key set. Adapter metadata
  (``usage``, ``exit_code``, ``parse_status``, ``warnings``, ``error``,
  ``log_paths``) is read with ``getattr`` fallbacks so untouched legacy
  adapters (copilot/gemini) keep working with their metric shapes, and
  their parser status is never treated as a canonical failure.
- Reports carry a UUID identity and are published atomically with
  no-clobber semantics in the same directory: a real collision gets a new
  identity, prior reports are never overwritten.
- The report is always written first; any operational error then exits
  nonzero. Diagnostics in the report are stable safe tokens, never raw
  exception strings, prompts, or CLI stdout/stderr (those stay in logs).

``SPEC2AGENT_SKIP_DOTENV=1`` bypasses ``.env`` loading (isolated tests).
"""
import argparse
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

if os.getenv("SPEC2AGENT_SKIP_DOTENV") != "1":
    load_dotenv(override=True)

from adapters.copilot import CopilotAdapter
from adapters.gemini import GeminiAdapter
from adapters.claude import ClaudeCodeAdapter
from adapters.codex import CodexAdapter
from adapters.open_code import OpenCodeAdapter

ADAPTERS: dict[str, tuple[type, dict]] = {
    "copilot":   (CopilotAdapter,     {}),
    "gemini":    (GeminiAdapter,      {}),
    "claude":    (ClaudeCodeAdapter,  {}),
    "codex":     (CodexAdapter,       {}),
    "opencode":  (OpenCodeAdapter,    {}),

    # Claude version comparison experiment — all at effort "low"
    "claude-opus-4-6-low": (ClaudeCodeAdapter, {"model": "claude-opus-4-6", "effort": "low"}),
    "claude-opus-4-8-low": (ClaudeCodeAdapter, {"model": "claude-opus-4-8", "effort": "low"}),
    "claude-opus-5-5-low": (ClaudeCodeAdapter, {"model": "claude-opus-5-5", "effort": "low"}),
}

# parse_status "error" is an operational failure only for adapters that
# actually run the numeric usage parser. Untouched legacy adapters inherit
# a default "error" status without ever parsing, so it stays informational
# for them and is normalized to null in their entries.
_CANONICAL_ADAPTERS = (ClaudeCodeAdapter, CodexAdapter, OpenCodeAdapter)

# Binary names used only to build the stable spawn_failed diagnostic when
# a legacy adapter raises instead of returning failure metadata.
_SPAWN_PROG = {
    CopilotAdapter: "copilot",
    GeminiAdapter: "agy",
    ClaudeCodeAdapter: "claude",
    CodexAdapter: "codex",
    OpenCodeAdapter: "opencode",
}

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Raw log filenames the legacy adapters write; used only as a getattr
# fallback when an adapter exposes no log_paths mapping of its own.
_KNOWN_LOG_FILES = (
    "01-generate.log",
    "01_generate.log",
    "session.log",
    "credits.log",
    "cli_stderr.log",
)

ENTRY_KEYS = (
    "agent",
    "llm",
    "spec_file",
    "output_dir",
    "generated_file",
    "status",
    "error",
    "exit_code",
    "parse_status",
    "warnings",
    "usage",
    "log_paths",
)


def _load_config() -> tuple[Path, Path, Path]:
    missing = [k for k in ("BENCH_DIR", "GENERATED_DIR", "EXPERIMENTS_RESULTS")
               if not os.getenv(k)]
    if missing:
        print(
            "[generate] missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in BENCH_DIR, "
              "GENERATED_DIR and EXPERIMENTS_RESULTS.",
            file=sys.stderr,
        )
        sys.exit(2)
    return (
        Path(os.environ["BENCH_DIR"]),
        Path(os.environ["GENERATED_DIR"]),
        Path(os.environ["EXPERIMENTS_RESULTS"]),
    )


def _lexical_nodes_ok(base: Path, *parts: str) -> bool:
    """Reject symlink components beneath the configured base, lexically.

    Checked before any resolve() so a symlinked experiment/run ancestor can
    never redirect the run outside the trusted tree.
    """
    node = base
    for part in parts:
        node = node / part
        if os.path.lexists(node) and os.path.islink(node):
            return False
    return True


def _resolve_root(base: Path, experiment: str, run: int) -> Path | None:
    """Lexically validate a run root, then resolve it; None when unsafe."""
    run_dir = f"run_{run}"
    if not _lexical_nodes_ok(base, experiment, run_dir):
        return None
    anchor = base.resolve(strict=False)
    resolved = (base / experiment / run_dir).resolve(strict=False)
    try:
        resolved.relative_to(anchor)
    except ValueError:
        return None
    return resolved


def get_all_specs(bench_dir: Path) -> list[Path]:
    return sorted([p for p in bench_dir.iterdir()
                   if p.is_dir() and (p / "spec.md").exists()])


def _reserve_output_dir(output_dir: Path, root: Path) -> tuple[str | None, tuple | None]:
    """Atomically claim a fresh output directory, or return an error token.

    Any pre-existing entry (file, directory or symlink) is rejected so a
    rerun can never overwrite prior artifacts or logs. Symlinks anywhere
    in the reserved subtree and paths escaping the run root are unsafe.
    On success also returns the reserved directory identity (dev, inode)
    so a post-run replacement is detectable even when the new destination
    still contains the artifact path.
    """
    if os.path.lexists(output_dir):
        return "output_exists", None
    chain = [output_dir, *output_dir.parents]
    if root not in chain:
        return "output_unsafe", None
    for component in chain:
        if component == root:
            break
        if os.path.islink(component):
            return "output_unsafe", None
    try:
        resolved_out = output_dir.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
        resolved_out.relative_to(resolved_root)
    except (ValueError, OSError):
        return "output_unsafe", None
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Lost a concurrent reservation race: same no-overwrite outcome.
        return "output_exists", None
    except OSError:
        return "output_unwritable", None
    # Re-walk after creation: a symlink planted between the check and the
    # mkdir must not redirect the fresh directory into another tree.
    for component in chain:
        if component == root:
            break
        if os.path.islink(component):
            return "output_unsafe", None
    try:
        st = os.stat(output_dir)
    except OSError:
        return "output_unwritable", None
    return None, (st.st_dev, st.st_ino)


def _output_intact(output_dir: Path, generated_root: Path, identity) -> bool:
    """Reject a replaced output directory after the CLI ran.

    A symlink swap (or any directory replacement) is detected via lstat
    plus the reserved dev/inode identity, even when the artifact path still
    resolves inside the changed destination. Ancestors up to the run root
    are re-checked lexically; this bounded check is not a sandbox.
    """
    try:
        if os.path.islink(output_dir):
            return False
        st = os.stat(output_dir)
    except OSError:
        return False
    if identity is not None and (st.st_dev, st.st_ino) != identity:
        return False
    node = output_dir.parent
    while True:
        if os.path.islink(node):
            return False
        if node == generated_root:
            return True
        parent = node.parent
        if parent == node:
            return False
        node = parent


def _snapshot(path: Path):
    """Stat signature of an existing regular file, else None."""
    try:
        if os.path.lexists(path) and not os.path.islink(path):
            st = os.stat(path)
            return (st.st_mtime_ns, st.st_size)
    except OSError:
        pass
    return None


def _validate_artifact(artifact: Path, output_dir: Path, pre_sig) -> str | None:
    """Require a fresh regular contained agent.py, or return an error token."""
    if not os.path.lexists(artifact):
        return "artifact_missing"
    if os.path.islink(artifact):
        return "artifact_symlink"
    if not artifact.is_file():
        return "artifact_not_regular"
    try:
        artifact.resolve(strict=True).relative_to(output_dir.resolve(strict=True))
    except (ValueError, OSError):
        return "artifact_uncontained"
    if pre_sig is not None and _snapshot(artifact) == pre_sig:
        return "artifact_stale"
    return None


def _default_log_paths(logs_dir: Path) -> dict:
    """Fallback log mapping for adapters without their own log_paths."""
    found = {}
    try:
        entries = set(os.listdir(logs_dir))
    except OSError:
        return found
    for name in _KNOWN_LOG_FILES:
        if name in entries:
            candidate = logs_dir / name
            try:
                if candidate.is_file() and not os.path.islink(candidate):
                    found[name] = str(candidate)
            except OSError:
                continue
    return found


def _adapter_metadata(adapter, logs_dir: Path) -> dict:
    """Read canonical adapter attributes with legacy-safe fallbacks."""
    usage = getattr(adapter, "usage", None)
    usage = dict(usage) if isinstance(usage, dict) else {}

    exit_code = getattr(adapter, "exit_code", None)
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        exit_code = None

    parse_status = getattr(adapter, "parse_status", None)
    if parse_status not in ("ok", "partial", "error"):
        parse_status = None
    elif not isinstance(adapter, _CANONICAL_ADAPTERS):
        # Legacy adapters never run the numeric parser; their inherited
        # default status must not fail the job nor mislead readers.
        parse_status = None

    warnings = getattr(adapter, "warnings", None)
    warnings = [w for w in warnings if isinstance(w, str)] \
        if isinstance(warnings, list) else []

    log_paths = getattr(adapter, "log_paths", None)
    if not (isinstance(log_paths, dict) and log_paths):
        # Untouched legacy adapters inherit an empty mapping; discover
        # the logs they actually wrote instead of publishing nothing.
        log_paths = _default_log_paths(logs_dir)
    else:
        log_paths = {str(k): str(v) for k, v in log_paths.items()}

    adapter_error = getattr(adapter, "error", None)
    if not (isinstance(adapter_error, str) and adapter_error):
        adapter_error = None

    return {
        "usage": usage,
        "exit_code": exit_code,
        "parse_status": parse_status,
        "warnings": warnings,
        "log_paths": log_paths,
        "adapter_error": adapter_error,
    }


def _base_entry(llm: str, spec_folder: Path, generated_root: Path) -> dict:
    output_dir = generated_root / llm / spec_folder.name
    return {
        "agent": spec_folder.name,
        "llm": llm,
        "spec_file": str((spec_folder / "spec.md").resolve()),
        "output_dir": os.path.abspath(output_dir),
        "generated_file": os.path.abspath(output_dir / "agent.py"),
    }


def _error_entry(base: dict, *, error: str, usage=None, exit_code=None,
                 parse_status=None, warnings=None, log_paths=None) -> dict:
    return {
        **base,
        "status": "error",
        "error": error,
        "exit_code": exit_code,
        "parse_status": parse_status,
        "warnings": list(warnings) if warnings else [],
        "usage": dict(usage) if isinstance(usage, dict) else {},
        "log_paths": dict(log_paths) if isinstance(log_paths, dict) else {},
    }


def generate(llm: str, spec_folder: Path, generated_root: Path) -> dict:
    base = _base_entry(llm, spec_folder, generated_root)
    output_dir = Path(base["output_dir"])
    artifact = output_dir / "agent.py"
    logs_dir = output_dir / "logs"

    if not (spec_folder / "spec.md").exists():
        return _error_entry(base, error="spec_not_found")

    token, identity = _reserve_output_dir(output_dir, generated_root)
    if token is not None:
        return _error_entry(base, error=token)

    pre_sig = _snapshot(artifact)
    adapter_class, adapter_kwargs = ADAPTERS[llm]

    adapter = None
    exc_kind: str | None = None  # None | "fnf" | "collect"
    try:
        adapter = adapter_class(spec_folder / "spec.md", output_dir, **adapter_kwargs)
        adapter.run()
    except FileNotFoundError:
        # Legacy adapters raise on spawn failure while canonical ones
        # record it as metadata; the final classification happens below
        # once output integrity is known, and the spec is re-checked
        # instead of trusting messages.
        exc_kind = "fnf"
    except Exception:
        # Arbitrary OSError (e.g. a persistence fault inside an adapter
        # that raises) is a collection failure, never a spawn failure.
        exc_kind = "collect"

    # Retain whatever the adapter actually collected: exit code, parser
    # status, usage, warnings and log paths survive into error entries.
    meta = _adapter_metadata(adapter, logs_dir)
    intact = _output_intact(output_dir, generated_root, identity)

    tokens: list[str] = []
    if meta["adapter_error"]:
        tokens.append(meta["adapter_error"])
    elif exc_kind == "fnf":
        if not (spec_folder / "spec.md").exists():
            tokens.append("spec_not_found")
        elif intact:
            tokens.append(f"spawn_failed: {_SPAWN_PROG.get(adapter_class, llm)}")
        else:
            # The tree was swapped mid-run, so post-spawn writes fail with
            # FileNotFoundError inside the replaced destination: a
            # collection failure, never a missing binary.
            tokens.append("collect_failed")
    elif exc_kind == "collect":
        tokens.append("collect_failed")

    if not intact:
        # The directory was swapped after reservation (e.g. replaced by a
        # symlink): nothing inside it can be trusted, not even an artifact
        # that still resolves within the changed destination.
        tokens.append("output_replaced")
    else:
        artifact_token = _validate_artifact(artifact, output_dir, pre_sig)
        if artifact_token is not None:
            tokens.append(artifact_token)

    failed_root_cause = any(
        t == "collect_failed" or t.startswith(("spawn_failed", "persistence_failed"))
        for t in tokens
    )

    exit_code = meta["exit_code"]
    if isinstance(exit_code, int) and exit_code != 0:
        derived = f"cli_exit_{exit_code}"
        if derived not in tokens:
            tokens.append(derived)

    if exc_kind is None and meta["parse_status"] == "error" \
            and isinstance(adapter, _CANONICAL_ADAPTERS) \
            and not failed_root_cause \
            and "parse_error" not in tokens:
        tokens.append("parse_error")

    if not tokens:
        return {
            **base,
            "status": "ok",
            "error": None,
            "exit_code": exit_code,
            "parse_status": meta["parse_status"],
            "warnings": meta["warnings"],
            "usage": meta["usage"],
            "log_paths": meta["log_paths"],
        }
    return _error_entry(base, error="; ".join(tokens), usage=meta["usage"],
                        exit_code=exit_code, parse_status=meta["parse_status"],
                        warnings=meta["warnings"], log_paths=meta["log_paths"])


def _publish_report(experiments_root: Path, llm: str, report: dict) -> tuple[Path, str]:
    """Write the report atomically without ever overwriting a prior one.

    The payload is written to an exclusively created temporary sibling
    (mkstemp: O_CREAT|O_EXCL, unique name), so a leftover temporary file —
    even a hardlink to an already-published report — is never opened or
    truncated. The temp file is then hard-linked to its final UUID name:
    ``os.link`` fails when the target already exists, so an actual UUID
    collision retries with a fresh identity instead of clobbering the
    previous report.
    """
    experiments_root.mkdir(parents=True, exist_ok=True)
    for _ in range(16):
        report_id = uuid.uuid4().hex
        name = f"01_generation_{llm}_{report_id}.json"
        target = experiments_root / name
        payload_text = json.dumps({**report, "report_id": report_id},
                                  indent=2, ensure_ascii=False) + "\n"
        try:
            fd, tmp_name = tempfile.mkstemp(dir=str(experiments_root),
                                            prefix=".tmp-publish-")
        except OSError as e:
            print(f"[generate] failed to allocate report temp file: {type(e).__name__}",
                  file=sys.stderr)
            sys.exit(1)
        try:
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload_text)
            except OSError as e:
                print(f"[generate] failed to write report temp file: {type(e).__name__}",
                      file=sys.stderr)
                sys.exit(1)
            try:
                os.link(tmp_name, target)
            except FileExistsError:
                continue
            return target, report_id
        finally:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    print("[generate] could not publish a unique report: "
          "repeated name collisions", file=sys.stderr)
    sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True, choices=list(ADAPTERS.keys()))
    parser.add_argument("--agent", default=None, help="Filter a specific agent (folder name in bench/)")
    parser.add_argument("--run", type=int, default=1, help="Run index (1..N). Writes under run_<N>/ subpath.")
    parser.add_argument("--experiment", default="by_tool", help="Experiment slug (e.g., by_tool, by_claude_version).")
    args = parser.parse_args(argv)

    if not _SAFE_COMPONENT.fullmatch(args.experiment or ""):
        parser.error(f"unsafe --experiment selector: {args.experiment!r}")
    if args.agent is not None and not _SAFE_COMPONENT.fullmatch(args.agent):
        parser.error(f"unsafe --agent selector: {args.agent!r}")
    if args.run is None or args.run < 1:
        parser.error(f"--run must be an index >= 1, got {args.run!r}")

    bench_dir, generated_base, results_base = _load_config()
    if not bench_dir.is_dir():
        print(f"[generate] bench directory not found: {bench_dir}",
              file=sys.stderr)
        sys.exit(2)

    # Validate the lexical experiment/run path beneath each configured base
    # BEFORE resolving: a symlinked ancestor must never redirect the run
    # outside the trusted tree (the resolve below would erase the evidence).
    generated_root = _resolve_root(generated_base, args.experiment, args.run)
    experiments_root = _resolve_root(results_base, args.experiment, args.run)
    if generated_root is None:
        print(f"[generate] unsafe generated path: symlink component under "
              f"{generated_base}/{args.experiment}/run_{args.run}",
              file=sys.stderr)
        sys.exit(2)
    if experiments_root is None:
        print(f"[generate] unsafe results path: symlink component under "
              f"{results_base}/{args.experiment}/run_{args.run}",
              file=sys.stderr)
        sys.exit(2)

    specs = get_all_specs(bench_dir)

    if args.agent:
        specs = [s for s in specs if s.name == args.agent]
        if not specs:
            print(f"[generate] agent '{args.agent}' not found in {bench_dir}",
                  file=sys.stderr)
            sys.exit(2)

    results = []

    for spec_folder in specs:
        print(f"[generate] experiment={args.experiment} run={args.run} {args.llm} ← {spec_folder.name}")
        result = generate(args.llm, spec_folder, generated_root)
        if result["status"] == "error":
            print(f"[generate] {args.llm} ← {spec_folder.name}: "
                  f"error ({result['error']})")
        results.append(result)

    final_report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "run": args.run,
        "experiment": args.experiment,
        "stage": "generation",
        "model": args.llm,
        "bench_dir": str(bench_dir.resolve()),
        "generated_dir": str(generated_root),
        "total_agents": len(results),
        "results": results,
    }

    try:
        report_path, _ = _publish_report(experiments_root, args.llm, final_report)
    except OSError as e:
        print(f"[generate] failed to publish report: {type(e).__name__}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[generate] Report saved → {report_path}")

    if any(r["status"] == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
