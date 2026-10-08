"""Aggregate experiments_results/*.json into the shapes the frontend expects.

Reads /Users/deboralls/Spec2AgentEval-v_saner/experiments_results/ and writes
JSON files under website/public/. For each stage (01, 02, 03, 05, 07, 08),
picks the most recent run per LLM and reshapes it for the matching panel.

Behavioral stages (07, 08) are parsed from pytest stdout via regex, extracting
test names, status (passed/failed/error/skipped), and the failure reason.

Run:
    python3 website/scripts/aggregate.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
WEBSITE_DIR = HERE.parent.parent
REPO_ROOT = WEBSITE_DIR.parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments_results"
PUBLIC_DIR = WEBSITE_DIR / "public"

STAGE_PATTERNS = {
    "01": re.compile(r"^01_generation_([^_]+)_(\d+_\d+)\.json$"),
    "02": re.compile(r"^02_validation_([^_]+)_(\d+_\d+)\.json$"),
    "03": re.compile(r"^03_lint_([^_]+)_(\d+_\d+)\.json$"),
    "05": re.compile(r"^05_spec_judge_([^_]+)_(\d+_\d+)\.json$"),
    "07": re.compile(r"^07_test_deepeval_([^_]+)_(\d+_\d+)\.json$"),
    "08": re.compile(r"^08_trace_eval_([^_]+)_(\d+_\d+)\.json$"),
}

RUN_DIR_RE = re.compile(r"^run_(\d+)$")

# Mutated by main() per experiment. All aggregators read runs from here.
EXPERIMENTS_DIR: Path = EXPERIMENTS_ROOT


def discover_experiments() -> list[tuple[str, Path]]:
    """Return [(experiment_slug, path), ...] under ``experiments_results/``.

    An experiment is a direct child dir that is NOT itself a ``run_<N>`` folder.
    If none are found but ``experiments_results/`` contains ``run_<N>/``
    directly, falls back to a single experiment keyed by its folder name
    (treats the root as the sole experiment, keeps legacy layouts working).
    """
    experiments: list[tuple[str, Path]] = []
    if not EXPERIMENTS_ROOT.exists():
        return experiments
    for child in sorted(EXPERIMENTS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if RUN_DIR_RE.match(child.name):
            # Legacy: run folders live directly under experiments_results/.
            return [(EXPERIMENTS_ROOT.name, EXPERIMENTS_ROOT)]
        experiments.append((child.name, child))
    return experiments


def discover_run_dirs() -> list[tuple[int, Path]]:
    """Return [(run_index, Path), ...] inside the current experiment dir."""
    runs: list[tuple[int, Path]] = []
    if EXPERIMENTS_DIR.exists():
        for child in EXPERIMENTS_DIR.iterdir():
            if not child.is_dir():
                continue
            m = RUN_DIR_RE.match(child.name)
            if m:
                runs.append((int(m.group(1)), child))
    if runs:
        return sorted(runs, key=lambda t: t[0])
    return [(1, EXPERIMENTS_DIR)]


def latest_per_model_in(run_dir: Path, stage: str) -> dict[str, Path]:
    """Return {model: Path} within a run folder, picking the newest timestamp."""
    pattern = STAGE_PATTERNS[stage]
    buckets: dict[str, tuple[str, Path]] = {}
    if not run_dir.exists():
        return {}
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        m = pattern.match(path.name)
        if not m:
            continue
        model, timestamp = m.group(1), m.group(2)
        current = buckets.get(model)
        if current is None or timestamp > current[0]:
            buckets[model] = (timestamp, path)
    return {model: info[1] for model, info in buckets.items()}


def latest_per_model(stage: str) -> dict[str, Path]:
    """Latest file per model across the newest run directory.

    Dimension panels (02/03/05/07/08) use this to render their drill-down from
    the most recent run. The leaderboard aggregator looks at every run.
    """
    runs = discover_run_dirs()
    if not runs:
        return {}
    _, latest_run_dir = runs[-1]
    return latest_per_model_in(latest_run_dir, stage)


def load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage 01 — generation: time + tokens
# ---------------------------------------------------------------------------

NUMBER = r"([\d.]+)([kKmM]?)"

TOKEN_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "input": [re.compile(rf"{NUMBER}\s*in\b", re.IGNORECASE)],
    "output": [re.compile(rf"{NUMBER}\s*out\b", re.IGNORECASE)],
    "cache_read": [
        re.compile(rf"{NUMBER}\s*cache\s*read\b", re.IGNORECASE),
        re.compile(rf"{NUMBER}\s*cached\b", re.IGNORECASE),
    ],
    "cache_created": [
        re.compile(rf"{NUMBER}\s*cache\s*created\b", re.IGNORECASE),
        re.compile(rf"{NUMBER}\s*written\b", re.IGNORECASE),
    ],
    "reasoning": [re.compile(rf"{NUMBER}\s*reasoning\b", re.IGNORECASE)],
}
COST_PATTERN = re.compile(r"\$\s*([\d.]+)")

TIME_UNIT_PATTERNS = {
    "h": re.compile(r"(\d+(?:\.\d+)?)\s*h\b", re.IGNORECASE),
    "m": re.compile(r"(\d+(?:\.\d+)?)\s*m(?![sx])\b", re.IGNORECASE),  # minutes, not ms/mx
    "s": re.compile(r"(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE),
}


def to_number(value: str, suffix: str) -> float:
    base = float(value)
    if suffix.lower() == "k":
        base *= 1_000
    elif suffix.lower() == "m":
        base *= 1_000_000
    return base


def parse_tokens(breakdown: dict[str, str]) -> dict[str, float | None]:
    totals: dict[str, float] = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_created": 0,
        "reasoning": 0,
    }
    cost = 0.0
    had_cost = False
    for raw in breakdown.values():
        text = str(raw)
        for key, patterns in TOKEN_PATTERNS.items():
            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    totals[key] += to_number(m.group(1), m.group(2))
                    break  # first matching alias wins per entry
        c = COST_PATTERN.search(text)
        if c:
            cost += float(c.group(1))
            had_cost = True
    return {
        **totals,
        "total": totals["input"] + totals["output"],
        "cost_usd": cost if had_cost else None,
    }


def parse_time_seconds(raw: str | None) -> float | None:
    """Parse strings like '2m 14s', '185.67s', '1h 30m', '90.93', '2 min 10 s'."""
    if not raw:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    total = 0.0
    found = False
    hours = TIME_UNIT_PATTERNS["h"].search(text)
    if hours:
        total += float(hours.group(1)) * 3600
        found = True
    minutes = TIME_UNIT_PATTERNS["m"].search(text)
    if minutes:
        total += float(minutes.group(1)) * 60
        found = True
    seconds = TIME_UNIT_PATTERNS["s"].search(text)
    if seconds:
        total += float(seconds.group(1))
        found = True
    if not found:
        bare = re.search(r"(\d+(?:\.\d+)?)", text)
        if bare:
            total = float(bare.group(1))
            found = True
    return total if found else None


def _merge_runs(
    per_run: list[tuple[int, list[dict[str, Any]]]],
    key_fn,
    mean_number_fields: list[str] | None = None,
    mean_bool_fields: list[str] | None = None,
    mean_nested_tokens: bool = False,
) -> list[dict[str, Any]]:
    """Group entries from multiple runs by key and attach a ``runs`` array.

    The merged entry uses the LAST run's values as the primary view (so panels
    that read ``entry.foo`` keep working unchanged). ``mean_number_fields`` and
    ``mean_bool_fields`` override those keys on the primary with the mean
    across runs. ``runs`` carries one dict per run with the original per-run
    payload plus a ``run`` index.
    """
    mean_number_fields = mean_number_fields or []
    mean_bool_fields = mean_bool_fields or []

    merged: dict[Any, dict[str, Any]] = {}
    for run_idx, entries in per_run:
        for entry in entries:
            key = key_fn(entry)
            bucket = merged.setdefault(key, {"_runs": []})
            bucket["_runs"].append((run_idx, entry))

    out: list[dict[str, Any]] = []
    for bucket in merged.values():
        run_entries = sorted(bucket["_runs"], key=lambda t: t[0])
        last = dict(run_entries[-1][1])  # primary = latest run
        last["runs"] = [{"run": r, **e} for r, e in run_entries]
        last["runs_count"] = len(run_entries)

        for field in mean_number_fields:
            vals = [e.get(field) for _, e in run_entries if isinstance(e.get(field), (int, float))]
            if vals:
                last[field] = round(sum(vals) / len(vals), 2)

        for field in mean_bool_fields:
            vals = [bool(e.get(field)) for _, e in run_entries if e.get(field) is not None]
            if vals:
                last[field] = round(sum(1 for v in vals if v) / len(vals), 2)

        if mean_nested_tokens:
            # tokens dict → average each numeric key across runs
            token_dicts = [e.get("tokens") or {} for _, e in run_entries]
            keys = set().union(*[set(td.keys()) for td in token_dicts]) if token_dicts else set()
            merged_tokens: dict[str, float | None] = {}
            for k in keys:
                vals = [td.get(k) for td in token_dicts if isinstance(td.get(k), (int, float))]
                merged_tokens[k] = round(sum(vals) / len(vals), 2) if vals else None
            last["tokens"] = merged_tokens

        out.append(last)
    return out


def aggregate_01(run_dir: Path | None = None) -> dict[str, Any]:
    # Single-run variant (used by leaderboard which iterates runs itself).
    if run_dir is not None:
        files = latest_per_model_in(run_dir, "01")
        results = []
        for model, path in sorted(files.items()):
            payload = load(path)
            for entry in payload.get("results", []):
                usage = entry.get("usage") or {}
                tokens = parse_tokens(usage.get("models_breakdown") or {})
                raw_time = usage.get("time_spent")
                results.append({
                    "model": model,
                    "agent_folder": entry.get("agent"),
                    "status": entry.get("status"),
                    "execution_time": raw_time,
                    "execution_seconds": parse_time_seconds(raw_time),
                    "requests": usage.get("requests"),
                    "tokens": tokens,
                })
        return {"stage": "generation", "results": results}

    # Multi-run merged variant (used by dimension JSON output).
    per_run = [(idx, aggregate_01(rd)["results"]) for idx, rd in discover_run_dirs()]
    merged = _merge_runs(
        per_run,
        key_fn=lambda e: (e["model"], e["agent_folder"]),
        mean_number_fields=["execution_seconds", "requests"],
        mean_nested_tokens=True,
    )
    return {"stage": "generation", "results": merged}


# ---------------------------------------------------------------------------
# Stage 02 — validation / executability
# ---------------------------------------------------------------------------

CHECK_KEYS = (
    "file_exists",
    "syntax_valid",
    "dependency_error",
    "has_entrypoint",
    "entrypoint_callable",
)


def aggregate_02(run_dir: Path | None = None) -> dict[str, Any]:
    if run_dir is not None:
        files = latest_per_model_in(run_dir, "02")
        results = []
        for model, path in sorted(files.items()):
            payload = load(path)
            for entry in payload.get("results", []):
                checks = entry.get("checks") or {}
                flat = {
                    key: bool(((checks.get(key) or {}).get("passed")))
                    for key in CHECK_KEYS
                }
                errors = []
                for key in CHECK_KEYS:
                    detail = (checks.get(key) or {}).get("error")
                    if detail:
                        errors.append(f"{key}: {detail}")
                results.append({
                    "model": model,
                    "agent_folder": entry.get("agent_folder") or entry.get("agent"),
                    **flat,
                    "final_status": entry.get("final_status"),
                    "_source_error": "\n".join(errors) if errors else None,
                })
        return {"stage": "validation", "results": results}

    per_run = [(idx, aggregate_02(rd)["results"]) for idx, rd in discover_run_dirs()]
    merged = _merge_runs(
        per_run,
        key_fn=lambda e: (e["model"], e["agent_folder"]),
        mean_bool_fields=list(CHECK_KEYS),
    )
    return {"stage": "validation", "results": merged}


# ---------------------------------------------------------------------------
# Stage 03 — lint
# ---------------------------------------------------------------------------

def _summarize_ruff(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_code: dict[str, int] = defaultdict(int)
    sample = []
    for item in items[:200]:
        code = item.get("code") or "UNKNOWN"
        by_code[code] += 1
        if len(sample) < 20:
            sample.append({
                "code": code,
                "message": item.get("message"),
                "row": (item.get("location") or {}).get("row"),
            })
    return {"total": len(items), "by_code": dict(by_code), "sample": sample}


def _summarize_pylint(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = defaultdict(int)
    sample = []
    for item in items[:200]:
        t = item.get("type") or "unknown"
        by_type[t] += 1
        if len(sample) < 20:
            sample.append({
                "symbol": item.get("symbol"),
                "message": item.get("message"),
                "line": item.get("line"),
                "type": t,
            })
    return {"total": len(items), "by_type": dict(by_type), "sample": sample}


def _summarize_radon_cc(payload: Any) -> dict[str, Any]:
    functions: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for file, items in payload.items():
            if not isinstance(items, list):
                continue
            for item in items:
                functions.append({
                    "file": file,
                    "name": item.get("name"),
                    "rank": item.get("rank"),
                    "complexity": item.get("complexity"),
                })
    functions.sort(key=lambda f: -(f.get("complexity") or 0))
    return {"total": len(functions), "top": functions[:10]}


def _summarize_radon_mi(payload: Any) -> dict[str, Any]:
    files = []
    if isinstance(payload, dict):
        for file, info in payload.items():
            if isinstance(info, dict):
                files.append({"file": file, "mi": info.get("mi"), "rank": info.get("rank")})
    return {"files": files}


def aggregate_03(run_dir: Path | None = None) -> dict[str, Any]:
    if run_dir is not None:
        files = latest_per_model_in(run_dir, "03")
        results = []
        for model, path in sorted(files.items()):
            payload = load(path)
            for entry in payload.get("results", []):
                tools = entry.get("tools") or {}
                summary: dict[str, Any] = {}
                for name, tool in tools.items():
                    stdout = tool.get("stdout")
                    if name == "ruff" and isinstance(stdout, list):
                        summary["ruff"] = _summarize_ruff(stdout)
                    elif name == "pylint" and isinstance(stdout, list):
                        summary["pylint"] = _summarize_pylint(stdout)
                    elif name == "radon_cc":
                        summary["radon_cc"] = _summarize_radon_cc(stdout)
                    elif name == "radon_mi":
                        summary["radon_mi"] = _summarize_radon_mi(stdout)
                results.append({
                    "model": model,
                    "agent": entry.get("agent"),
                    "passed": entry.get("passed"),
                    "summary": summary,
                })
        return {"stage": "lint", "results": results}

    per_run = [(idx, aggregate_03(rd)["results"]) for idx, rd in discover_run_dirs()]
    merged = _merge_runs(
        per_run,
        key_fn=lambda e: (e["model"], e["agent"]),
        mean_bool_fields=["passed"],
    )
    return {"stage": "lint", "results": merged}


# ---------------------------------------------------------------------------
# Stage 05 — spec judge
# ---------------------------------------------------------------------------

def aggregate_05(run_dir: Path | None = None) -> dict[str, Any]:
    if run_dir is not None:
        files = latest_per_model_in(run_dir, "05")
        results = []
        for model, path in sorted(files.items()):
            payload = load(path)
            for entry in payload.get("results", []):
                judge = entry.get("judge") or {}
                scores = judge.get("scores") or {}
                details = [{"key": k, "score": int(bool(v))} for k, v in scores.items()]
                total = judge.get("total_score")
                max_score = judge.get("max_score")
                percent = round((total / max_score) * 100) if (total is not None and max_score) else None
                results.append({
                    "model": model,
                    "agent_folder": entry.get("agent_folder") or entry.get("agent"),
                    "score_percent": percent,
                    "total_score": total,
                    "max_score": max_score,
                    "passed": judge.get("passed"),
                    "summary": judge.get("summary", ""),
                    "details": details,
                })
        return {"stage": "spec_adherence", "results": results}

    per_run = [(idx, aggregate_05(rd)["results"]) for idx, rd in discover_run_dirs()]
    merged = _merge_runs(
        per_run,
        key_fn=lambda e: (e["model"], e["agent_folder"]),
        mean_number_fields=["score_percent", "total_score"],
        mean_bool_fields=["passed"],
    )
    return {"stage": "spec_adherence", "results": merged}


# ---------------------------------------------------------------------------
# Stages 07 / 08 — pytest stdout parsing
# ---------------------------------------------------------------------------

STATUS_LINE_RE = re.compile(
    r"^(FAILED|ERROR|PASSED|SKIPPED|XFAIL|XPASS)\s+"
    r"(\S+::[^\n]+?)"
    r"(?:\s+-\s+(.+?))?\s*$",
    re.MULTILINE,
)

TOTALS_RE = re.compile(
    r"(\d+ (?:failed|passed|skipped|error|xfailed|xpassed)[\w\s,]+?)\s*in\s+[\d.]+s"
)

# Blocks that start with a line like "______ test_name ______" (10+ underscores)
# or "______ ERROR at setup of test_name ______".
BLOCK_RE = re.compile(
    r"^_{10,}\s+(?:ERROR at (?:setup|teardown) of\s+)?([\w\.\[\]\->:]+?)\s+_{10,}$",
    re.MULTILINE,
)


def _extract_block_reasons(stdout: str) -> dict[str, str]:
    """For each ___ test_name ___ block in the FAILURES/ERRORS sections, grab
    the final `E ...` line as the reason."""
    reasons: dict[str, str] = {}
    matches = list(BLOCK_RE.finditer(stdout))
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(stdout)
        block = stdout[start:end]
        e_lines = [
            line.strip()[2:].strip()
            for line in block.splitlines()
            if re.match(r"^E\s", line)
        ]
        if e_lines:
            reasons[name] = e_lines[-1][:280]
    return reasons


def parse_pytest_stdout(stdout: str | None) -> dict[str, Any]:
    if not stdout:
        return {"tests": [], "totals": {}}

    reasons = _extract_block_reasons(stdout)

    # Keep only the short-summary tail to avoid matching listings inside tracebacks.
    tail_idx = stdout.rfind("short test summary info")
    scan = stdout[tail_idx:] if tail_idx != -1 else stdout

    tests: dict[str, dict[str, Any]] = {}
    for m in STATUS_LINE_RE.finditer(scan):
        status, qualified_name, reason = m.group(1), m.group(2), m.group(3)
        _, _, short_name = qualified_name.partition("::")
        reason_text = (reason or "").strip()
        if not reason_text or reason_text.endswith("..."):
            reason_text = reasons.get(short_name, reason_text)
        tests[qualified_name] = {
            "name": short_name,
            "qualified_name": qualified_name,
            "status": status.lower(),
            "passed": status == "PASSED",
            "reason": reason_text,
        }

    totals: dict[str, int] = {}
    tail = TOTALS_RE.search(stdout)
    if tail:
        for chunk in tail.group(1).split(","):
            m = re.match(r"\s*(\d+)\s+(\w+)", chunk)
            if m:
                totals[m.group(2)] = int(m.group(1))

    # Infer PASSED tests not surfaced in the summary from the totals.
    observed = {t["status"]: 0 for t in tests.values()}
    for t in tests.values():
        observed[t["status"]] = observed.get(t["status"], 0) + 1
    passed_known = observed.get("passed", 0)
    passed_total = totals.get("passed", 0)
    silent_passed = max(0, passed_total - passed_known)
    for i in range(silent_passed):
        synthetic = f"__passed__#{i + 1}"
        tests[synthetic] = {
            "name": synthetic,
            "qualified_name": synthetic,
            "status": "passed",
            "passed": True,
            "reason": "",
        }

    return {"tests": list(tests.values()), "totals": totals}


DEEPEVAL_METRIC_RE = re.compile(
    r"([A-Za-z0-9_ \[\]\-]+?)\s*"
    r"\(score:\s*([-\d.]+),\s*"
    r"threshold:\s*([-\d.]+),\s*"
    r"strict:\s*(?:True|False),\s*"
    r"error:\s*(.+?),\s*"
    r"reason:\s*(.+?)\)"
    r"(?=,\s*[A-Za-z0-9_]|\s*failed\.|\s*$)",
    re.DOTALL,
)
ASSERTION_LINE_RE = re.compile(r"^E\s+AssertionError:\s*(.+)$", re.MULTILINE)
LAST_ASSERTION_RE = re.compile(r"AssertionError:\s*(.+?)(?:\n|$)")


def _parse_deepeval_metrics(block: str) -> list[dict[str, Any]]:
    metrics = []
    for m in DEEPEVAL_METRIC_RE.finditer(block):
        name = m.group(1).strip().rstrip(",")
        try:
            score = float(m.group(2))
        except ValueError:
            score = None
        try:
            threshold = float(m.group(3))
        except ValueError:
            threshold = None
        error = m.group(4).strip()
        reason = m.group(5).strip()
        metrics.append({
            "name": name,
            "score": score,
            "threshold": threshold,
            "error": None if error in {"None", "null"} else error,
            "reason": reason,
            "passed": score is not None and threshold is not None and score >= threshold,
        })
    return metrics


def _summarize_reason(longrepr: str) -> tuple[str, list[dict[str, Any]]]:
    """Return (short message, deepeval metrics) extracted from a pytest longrepr."""
    if not longrepr:
        return "", []

    metrics: list[dict[str, Any]] = []
    # Deepeval raises `AssertionError: Metrics: <...> failed.`. The traceback can
    # include the source literal `Metrics: {failed_metrics_str}` too, so skip
    # blocks that don't contain a real `(score: …)` payload.
    for match in re.finditer(r"Metrics:\s*(.+?)\s*failed\.", longrepr, re.DOTALL):
        parsed = _parse_deepeval_metrics(match.group(1))
        if parsed:
            metrics.extend(parsed)

    # Prefer the last `E   AssertionError:` line pytest renders — it's the actual message.
    e_lines = ASSERTION_LINE_RE.findall(longrepr)
    short = (e_lines[-1] if e_lines else "").strip()
    if not short:
        last = LAST_ASSERTION_RE.findall(longrepr)
        short = last[-1].strip() if last else ""
    if not short:
        # Fall back to the last non-empty line.
        lines = [ln.strip() for ln in longrepr.strip().splitlines() if ln.strip()]
        short = lines[-1] if lines else ""

    # When deepeval failed, replace the (very long) raw sentence by a tidy summary.
    if metrics:
        parts = [
            f"{m['name']} score={m['score']:.2f}/{m['threshold']:.2f}"
            if m.get("score") is not None and m.get("threshold") is not None
            else m["name"]
            for m in metrics
        ]
        short = f"deepeval: {' · '.join(parts)}"

    return short[:600], metrics


def _tests_from_runner(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Prefer the structured report produced by pytest-json-report."""
    tests = entry.get("tests")
    if isinstance(tests, list) and tests:
        totals: dict[str, int] = {}
        normalized: list[dict[str, Any]] = []
        for t in tests:
            status = (t.get("status") or "").lower()
            if status:
                totals[status] = totals.get(status, 0) + 1
            qualified = t.get("qualified_name") or t.get("nodeid") or t.get("name") or ""
            short = t.get("name") or (qualified.split("::", 1)[-1] if "::" in qualified else qualified)
            longrepr = t.get("reason") or ""
            summary, metrics = _summarize_reason(longrepr)
            normalized.append({
                "name": short,
                "qualified_name": qualified,
                "status": status,
                "passed": status == "passed",
                "reason": summary or longrepr[:600],
                "metrics": metrics,
                "traceback": longrepr[:6000],
            })
        return {"tests": normalized, "totals": totals}
    # Fallback for legacy runs that only have the pytest summary line.
    summary = entry.get("summary")
    if isinstance(summary, dict) and summary.get("total") is not None:
        return {"tests": [], "totals": {k: v for k, v in summary.items() if isinstance(v, int)}}
    return None


def _collect_behavioral(stage: str, run_dir: Path | None = None) -> list[dict[str, Any]]:
    agents = []
    files = latest_per_model_in(run_dir, stage) if run_dir else latest_per_model(stage)
    for model, path in sorted(files.items()):
        payload = load(path)
        for entry in payload.get("results", []):
            structured = _tests_from_runner(entry)
            if structured is not None:
                parsed = structured
            else:
                parsed = parse_pytest_stdout(entry.get("stdout"))
            agents.append({
                "stage": stage,
                "model": model,
                "agent_folder": entry.get("agent_folder") or entry.get("agent"),
                "test_file": entry.get("test_file"),
                "returncode": entry.get("returncode"),
                "passed": entry.get("passed"),
                "tests": parsed["tests"],
                "totals": parsed["totals"],
                "_source_error": entry.get("stderr") or entry.get("error") or None,
            })
    return agents


def _merge_behavioral_runs(per_run: list[tuple[int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Group behavioral per-stage entries by (model, agent) and attach runs[]."""
    buckets: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for run_idx, entries in per_run:
        for entry in entries:
            key = (entry["model"], entry["agent_folder"])
            buckets.setdefault(key, []).append((run_idx, entry))

    merged: list[dict[str, Any]] = []
    for runs_for_key in buckets.values():
        runs_for_key.sort(key=lambda t: t[0])
        last = dict(runs_for_key[-1][1])  # primary = latest run (keeps tests/reasons for drill-down)

        # Attach per-run breakdown (totals + rate + tests per run)
        run_entries = []
        for r_idx, entry in runs_for_key:
            totals = entry.get("totals") or {}
            passed = totals.get("passed", 0)
            failed = totals.get("failed", 0) + totals.get("error", 0) + totals.get("errors", 0)
            skipped = totals.get("skipped", 0) + totals.get("xfail", 0) + totals.get("xpass", 0)
            total = passed + failed + skipped
            run_entries.append({
                "run": r_idx,
                "totals": totals,
                "rate": round((passed / total) * 100, 1) if total else None,
                "passed": passed,
                "total": total,
                "tests": entry.get("tests") or [],
            })
        last["runs"] = run_entries
        last["runs_count"] = len(runs_for_key)

        # Mean rate (based on each run's rate; ignores runs with 0 total)
        rates = [r["rate"] for r in run_entries if r["rate"] is not None]
        last["rate_mean"] = round(sum(rates) / len(rates), 1) if rates else None

        merged.append(last)
    return merged


def aggregate_behavioral(run_dir: Path | None = None) -> dict[str, Any]:
    if run_dir is not None:
        return {
            "stages": {
                "07_deepeval": _collect_behavioral("07", run_dir),
                "08_trace": _collect_behavioral("08", run_dir),
            }
        }

    runs = discover_run_dirs()
    return {
        "stages": {
            "07_deepeval": _merge_behavioral_runs([(idx, _collect_behavioral("07", rd)) for idx, rd in runs]),
            "08_trace": _merge_behavioral_runs([(idx, _collect_behavioral("08", rd)) for idx, rd in runs]),
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Derived outputs used by Overview and Leaderboard
# ---------------------------------------------------------------------------

def _pct(part: int, total: int) -> float | None:
    return round((part / total) * 100, 1) if total else None


def _models_for_stage(stage: str) -> set[str]:
    return set(latest_per_model(stage).keys())


def aggregate_overview_stats() -> dict[str, Any]:
    agents: set[str] = set()
    for stage in ("01", "02", "03", "05", "07", "08"):
        for path in latest_per_model(stage).values():
            payload = load(path)
            for entry in payload.get("results", []):
                name = entry.get("agent_folder") or entry.get("agent")
                if name:
                    agents.add(name)
    models = _models_for_stage("01") | _models_for_stage("02")
    return {
        "agentSpecifications": len(agents),
        "codeGenTools": len(models),
        "evaluationFacets": 5,
        "totalGenerations": len(models) * len(agents),
    }


LEADERBOARD_METRICS = (
    "executability",
    "structure",
    "spec_graph",
    "spec_prompt",
    "behavioral",
    "lint",
)


def _leaderboard_rows_for_run(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Compute one row per (model, agent) from a single run's data."""
    exec_data = aggregate_02(run_dir)
    lint_data = aggregate_03(run_dir)
    spec_data = aggregate_05(run_dir)
    behavioral = aggregate_behavioral(run_dir)

    exec_index = {(r["model"], r["agent_folder"]): r for r in exec_data["results"]}
    lint_index = {(r["model"], r["agent"]): r for r in lint_data["results"]}
    spec_index = {(r["model"], r["agent_folder"]): r for r in spec_data["results"]}

    behavioral_counts: dict[tuple[str, str], dict[str, int]] = {}
    for stage_results in behavioral["stages"].values():
        for entry in stage_results:
            key = (entry["model"], entry["agent_folder"])
            totals = entry["totals"] or {}
            passed = totals.get("passed", 0)
            failed = totals.get("failed", 0) + totals.get("error", 0) + totals.get("errors", 0)
            counts = behavioral_counts.setdefault(key, {"passed": 0, "total": 0})
            counts["passed"] += passed
            counts["total"] += passed + failed

    keys = set(exec_index) | set(lint_index) | set(spec_index) | set(behavioral_counts)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for model, agent in keys:
        exec_row = exec_index.get((model, agent))
        executability = (
            _pct(sum(1 for k in CHECK_KEYS if exec_row.get(k)), len(CHECK_KEYS))
            if exec_row else None
        )
        lint_row = lint_index.get((model, agent))
        lint = 100.0 if (lint_row and lint_row.get("passed")) else (0.0 if lint_row else None)
        spec_row = spec_index.get((model, agent))
        spec_graph = spec_row.get("score_percent") if spec_row else None
        counts = behavioral_counts.get((model, agent)) or {}
        behavioral_pct = _pct(counts.get("passed", 0), counts.get("total", 0))
        rows[(model, agent)] = {
            "executability": executability,
            "structure": executability,
            "spec_graph": spec_graph,
            "spec_prompt": None,
            "behavioral": behavioral_pct,
            "behavioral_passed": counts.get("passed"),
            "behavioral_total": counts.get("total"),
            "lint": lint,
        }
    return rows


def _mean_ignore_none(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 1)


def aggregate_leaderboard() -> list[dict[str, Any]]:
    runs = discover_run_dirs()
    per_run: list[tuple[int, dict[tuple[str, str], dict[str, Any]]]] = []
    for run_idx, run_dir in runs:
        per_run.append((run_idx, _leaderboard_rows_for_run(run_dir)))

    all_keys: set[tuple[str, str]] = set()
    for _, rows in per_run:
        all_keys |= set(rows)

    merged: list[dict[str, Any]] = []
    for model, agent in sorted(all_keys):
        per_run_entries: list[dict[str, Any]] = []
        for run_idx, rows in per_run:
            row = rows.get((model, agent))
            if row is None:
                continue
            per_run_entries.append({"run": run_idx, **row})

        if not per_run_entries:
            continue

        means: dict[str, float | None] = {}
        for metric in LEADERBOARD_METRICS:
            means[metric] = _mean_ignore_none([e.get(metric) for e in per_run_entries])

        # Behavioral counts are sums per run, so mean across runs is informative
        beh_passed_values = [e.get("behavioral_passed") for e in per_run_entries if e.get("behavioral_passed") is not None]
        beh_total_values = [e.get("behavioral_total") for e in per_run_entries if e.get("behavioral_total") is not None]
        beh_passed_mean = round(sum(beh_passed_values) / len(beh_passed_values), 1) if beh_passed_values else None
        beh_total_mean = round(sum(beh_total_values) / len(beh_total_values), 1) if beh_total_values else None

        merged.append({
            "agent": agent,
            "model": model,
            **means,
            "behavioral_passed": beh_passed_mean,
            "behavioral_total": beh_total_mean,
            "runs": per_run_entries,
            "runs_count": len(per_run_entries),
        })
    return merged


# Per-experiment outputs. The filename is prefixed with the experiment slug
# (e.g., ``by_tool__02_executability_aggregated.json``).
PER_EXPERIMENT_OUTPUTS = {
    "01_resource_aggregated.json": aggregate_01,
    "02_executability_aggregated.json": aggregate_02,
    "03_lint_aggregated.json": aggregate_03,
    "05_spec_adherence_aggregated.json": aggregate_05,
    "behavioral_aggregated.json": aggregate_behavioral,
    "leaderboard_data.json": aggregate_leaderboard,
}


def aggregate_overview_across_experiments(experiments: list[tuple[str, Path]]) -> dict[str, Any]:
    """Union agents and models across all experiments for the shared Overview."""
    global EXPERIMENTS_DIR
    agents: set[str] = set()
    models: set[str] = set()
    for _slug, exp_path in experiments:
        EXPERIMENTS_DIR = exp_path
        for stage in ("01", "02", "03", "05", "07", "08"):
            for path in latest_per_model(stage).values():
                payload = load(path)
                for entry in payload.get("results", []):
                    name = entry.get("agent_folder") or entry.get("agent")
                    if name:
                        agents.add(name)
        models |= _models_for_stage("01") | _models_for_stage("02")
    return {
        "agentSpecifications": len(agents),
        "codeGenTools": len(models),
        "evaluationFacets": 5,
        "totalGenerations": len(models) * len(agents),
    }


def main() -> int:
    global EXPERIMENTS_DIR
    if not EXPERIMENTS_ROOT.exists():
        print(f"not found: {EXPERIMENTS_ROOT}", file=sys.stderr)
        return 1
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments()
    if not experiments:
        print(f"no experiments found under {EXPERIMENTS_ROOT}", file=sys.stderr)
        return 1

    # Per-experiment aggregated JSONs.
    for slug, exp_path in experiments:
        EXPERIMENTS_DIR = exp_path
        for filename, builder in PER_EXPERIMENT_OUTPUTS.items():
            payload = builder()
            out = PUBLIC_DIR / f"{slug}__{filename}"
            with out.open("w") as f:
                json.dump(payload, f, indent=2)
            print(f"wrote {out.relative_to(REPO_ROOT)}")

    # Shared outputs across experiments.
    overview = aggregate_overview_across_experiments(experiments)
    with (PUBLIC_DIR / "overviewStats.json").open("w") as f:
        json.dump(overview, f, indent=2)
    print(f"wrote {(PUBLIC_DIR / 'overviewStats.json').relative_to(REPO_ROOT)}")

    # Index file telling the front which experiments exist.
    index = {"experiments": [{"slug": s} for s, _ in experiments]}
    with (PUBLIC_DIR / "experiments.json").open("w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {(PUBLIC_DIR / 'experiments.json').relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
