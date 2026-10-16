"""Numeric generation-usage parser (phase 1 contract).

Closed semantics (pins: Claude 2.1.299, Codex 0.162.1, OpenCode 1.19.0;
source URLs in docs/generation_usage.md):
- Codex turn.completed.usage is THREAD-CUMULATIVE: LAST turn.completed of a
  single unambiguous thread wins; never sum, never backfill. Turn count is
  independent of cumulative usage.
- OpenCode step_finish parts sum with per-field completeness; native input
  is cache-exclusive (inclusive adds read+write); native output stays
  reasoning-exclusive; tokens.total is optional passthrough, never derived;
  part.cost is an estimate.
- Claude single result object; input_tokens excludes both cache buckets, so
  inclusive input needs base+read+write all known; total_cost_usd is a
  client-side estimate; durations are provenance only.

Null means unknown; zero is preserved. Reported costs are ``estimated``.
Warnings carry field names, counts and line indexes only.
"""

import json
import math

SUPPORTED_CLIS = ("claude", "codex", "opencode")

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_ERROR = "error"

COST_ESTIMATED = "estimated"
COST_UNAVAILABLE = "unavailable"

# Field order is part of the documented schema; keep stable.
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "total_tokens",
    "requests",
    "cost_usd",
    "cost_source",
    "cli_duration_s",
    "process_duration_s",
    "wall_duration_s",
    "source",
    "requested_model",
    "raw_fields",
)

_UNPARSEABLE = object()

_SESSION_KEYS = ("sessionID", "session_id", "sessionId", "session")
_MESSAGE_KEYS = ("messageID", "message_id", "messageId")
_THREAD_KEYS = ("thread_id", "threadId")

_CLAUDE_RECOGNIZED = {"usage", "num_turns", "total_cost_usd",
                      "duration_api_ms", "duration_ms"}
_CLAUDE_TTL_KEYS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
_CLAUDE_MODEL_KEYS = ("inputTokens", "outputTokens", "cacheReadInputTokens",
                      "cacheCreationInputTokens", "costUSD")
_OPENCODE_FIELDS = ("input", "output", "reasoning", "read", "write",
                    "total", "cost")


def _blank_usage(cli, model, cli_version):
    return {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "requests": None,
        "cost_usd": None,
        "cost_source": COST_UNAVAILABLE,
        "cli_duration_s": None,
        "process_duration_s": None,
        "wall_duration_s": None,
        "source": {"cli": cli, "cli_version": cli_version, "provider": None},
        "requested_model": model,
        "raw_fields": {},
    }


# One strict decoder: single JSON object or JSONL with explicit shape.


def _decode(text):
    """Decode stdout; malformed counts non-blank lines that are not objects."""
    result = {"single": None, "events": [], "malformed": 0}
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        parsed = _UNPARSEABLE
    if parsed is not _UNPARSEABLE:
        if isinstance(parsed, dict):
            result["single"] = parsed
            result["events"] = [parsed]
        return result
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            result["malformed"] += 1
            continue
        if isinstance(item, dict):
            result["events"].append(item)
        else:
            result["malformed"] += 1
    return result


# Two numeric validators: exact integer counts, finite non-negative numbers.


def _valid_count(value, label, warnings):
    """Exact int >= 0 or None; never bool/negative/non-integer."""
    if isinstance(value, bool):
        warnings.append(f"invalid {label}: boolean is not a valid count; using null")
        return None
    if not isinstance(value, int):
        warnings.append(f"invalid {label}: expected integer count; using null")
        return None
    if value < 0:
        warnings.append(f"invalid {label}: negative count; using null")
        return None
    return value


def _valid_number(value, label, warnings):
    """Finite non-negative number as float or None; huge ints rejected
    without leaking OverflowError."""
    if isinstance(value, bool):
        warnings.append(f"invalid {label}: boolean is not a valid number; using null")
        return None
    if isinstance(value, int):
        if value < 0:
            warnings.append(f"invalid {label}: negative number; using null")
            return None
        try:
            converted = float(value)
        except (OverflowError, ValueError):
            warnings.append(f"invalid {label}: not representable; using null")
            return None
        if not math.isfinite(converted):
            warnings.append(f"invalid {label}: not representable; using null")
            return None
        return converted
    if isinstance(value, float):
        if not math.isfinite(value):
            warnings.append(f"invalid {label}: non-finite number; using null")
            return None
        if value < 0:
            warnings.append(f"invalid {label}: negative number; using null")
            return None
        return value
    warnings.append(f"invalid {label}: expected number; using null")
    return None


# Complete-sum helpers: a field total exists only with full coverage.


def _complete_sum(values):
    """Exact int sum iff no gaps; entries arrive already validated."""
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def _complete_fsum(values):
    """Float sum iff no gaps and the aggregate itself stays finite."""
    total = 0.0
    for value in values:
        if value is None:
            return None
        total += value
    if not math.isfinite(total):
        return None
    return total


def _scoped_ids(obj, keys):
    """Distinct non-empty session/thread ids, exact values, no coercion."""
    seen = []
    if not isinstance(obj, dict):
        return seen
    for key in keys:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            if value != "" and value not in seen:
                seen.append(value)
        elif isinstance(value, int):
            if value not in seen:
                seen.append(value)
    return seen


def _take(obj, name, warnings, expected, kind="int", label=None):
    """One required/optional metric from a decoded object (silent when
    absent/null and not expected)."""
    tag = label or name
    if not isinstance(obj, dict) or name not in obj or obj[name] is None:
        if expected:
            warnings.append(f"missing {tag}; using null")
        return None
    if kind == "number":
        return _valid_number(obj[name], tag, warnings)
    return _valid_count(obj[name], tag, warnings)


# Claude: single result object, complete components only.


def _parse_claude(obj, usage, warnings):
    if not isinstance(obj, dict) or not any(k in obj for k in _CLAUDE_RECOGNIZED):
        warnings.append("unexpected claude payload: no recognized usage fields")
        return STATUS_ERROR
    raw = {}

    inner = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    if inner is None and "usage" in obj:
        warnings.append("invalid usage object; using nulls for token fields")
    base = _take(inner, "input_tokens", warnings, True)
    cache_read = _take(inner, "cache_read_input_tokens", warnings, True)
    cache_create = _take(inner, "cache_creation_input_tokens", warnings, True)
    output = _take(inner, "output_tokens", warnings, True)
    for key, value in (("input_tokens", base), ("output_tokens", output),
                       ("cache_read_input_tokens", cache_read),
                       ("cache_creation_input_tokens", cache_create)):
        if value is not None:
            raw[key] = value

    # input_tokens excludes both cache buckets: inclusive total needs all
    # three components known.
    if base is not None and cache_read is not None and cache_create is not None:
        usage["input_tokens"] = base + cache_read + cache_create
    else:
        usage["input_tokens"] = None
        warnings.append("incomplete claude input: requires base input plus "
                        "cache read plus cache write; using null")
    usage["output_tokens"] = output
    usage["cache_read_tokens"] = cache_read
    usage["cache_write_tokens"] = cache_create
    usage["reasoning_tokens"] = None  # no separate reasoning field (U4)
    if usage["input_tokens"] is not None and usage["output_tokens"] is not None:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    else:
        usage["total_tokens"] = None

    usage["requests"] = _take(obj, "num_turns", warnings, True)
    if isinstance(obj.get("num_turns"), int) and not isinstance(obj.get("num_turns"), bool) \
            and obj["num_turns"] >= 0:
        raw["num_turns"] = obj["num_turns"]

    # Client-side estimate (includes subagents; usage itself does not).
    cost = _take(obj, "total_cost_usd", warnings, True, kind="number")
    usage["cost_usd"] = float(cost) if cost is not None else None
    if cost is not None:
        usage["cost_source"] = COST_ESTIMATED
        raw["total_cost_usd"] = float(cost)

    # Durations are provenance only (API wait vs unverified wall), so
    # cli_duration_s stays null absent a trusted CLI semantic.
    for key in ("duration_api_ms", "duration_ms"):
        value = obj.get(key)
        if value is None:
            continue
        valid = _valid_number(value, key, warnings)
        if valid is not None:
            raw[key] = valid
    usage["cli_duration_s"] = None

    # Per-TTL cache detail: allowlisted breakdown of the aggregate write
    # bucket, never added back (no double count).
    detail = inner.get("cache_creation") if isinstance(inner, dict) else None
    if isinstance(detail, dict):
        kept = {}
        for key in _CLAUDE_TTL_KEYS:
            if key in detail and detail[key] is not None:
                value = _valid_count(detail[key], f"cache_creation.{key}", warnings)
                if value is not None:
                    kept[key] = value
        if kept:
            raw["cache_creation"] = kept

    # modelUsage numerics (whole-tree scope, may include subagents): kept
    # observed per model, no whole-tree normalization assumed.
    model_usage = obj.get("modelUsage")
    if isinstance(model_usage, dict):
        kept_models = {}
        for model_name, entry in model_usage.items():
            if not isinstance(entry, dict):
                warnings.append("invalid modelUsage entry; skipping model")
                continue
            kept_entry = {}
            for key in _CLAUDE_MODEL_KEYS:
                if key in entry and entry[key] is not None:
                    if key == "costUSD":
                        value = _valid_number(entry[key], "modelUsage numeric", warnings)
                    else:
                        value = _valid_count(entry[key], "modelUsage numeric", warnings)
                    if value is not None:
                        kept_entry[key] = value
            if kept_entry:
                kept_models[model_name] = kept_entry
        if kept_models:
            raw["modelUsage"] = kept_models

    usage["raw_fields"] = raw
    return STATUS_PARTIAL if warnings else STATUS_OK


# Codex: last cumulative snapshot of one unambiguous thread.


def _parse_codex(events, usage, warnings):
    turns = [e for e in events if e.get("type") == "turn.completed"]
    if not turns:
        warnings.append("unexpected codex payload: no turn.completed events")
        return STATUS_ERROR
    threads = []
    for event in events:
        for found in _scoped_ids(event, _THREAD_KEYS):
            if found not in threads:
                threads.append(found)
    if len(threads) > 1:
        warnings.append("multiple codex threads without proven relation; refusing to attribute")
        return STATUS_ERROR
    raw = {"events_observed": len(events), "turn_completed": len(turns),
           "threads_observed": len(threads)}

    # Last snapshot is the thread-cumulative total; earlier snapshots never
    # backfill fields missing here.
    last = turns[-1]
    inner = last.get("usage") if isinstance(last.get("usage"), dict) else None
    if inner is None:
        warnings.append("latest turn.completed without usable usage; using nulls")
    usage["input_tokens"] = _take(inner, "input_tokens", warnings, True)
    usage["output_tokens"] = _take(inner, "output_tokens", warnings, True)
    usage["cache_read_tokens"] = _take(inner, "cached_input_tokens", warnings, True)
    write_defaulted = False
    if not isinstance(inner, dict):
        usage["cache_write_tokens"] = None
    elif "cache_write_input_tokens" not in inner:
        # Absent key in VALID usage defaults to 0 ONLY per pinned SDK
        # (`cache_write ??= 0`). Missing/invalid usage or an explicit null
        # stays unknown with diagnosis instead.
        usage["cache_write_tokens"] = 0
        write_defaulted = True
    else:
        usage["cache_write_tokens"] = _take(inner, "cache_write_input_tokens", warnings, True)
    usage["reasoning_tokens"] = _take(inner, "reasoning_output_tokens", warnings, False)
    if usage["input_tokens"] is not None and usage["output_tokens"] is not None:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    else:
        usage["total_tokens"] = None
    # Turn count is independent of cumulative usage; identical content
    # proves nothing, so no content dedup exists.
    usage["requests"] = len(turns)
    usage["cost_usd"] = None  # no cost anywhere in the exec envelope
    usage["cost_source"] = COST_UNAVAILABLE
    usage["cli_duration_s"] = None  # wall clock belongs to the harness
    for raw_key, value in (("input_tokens", usage["input_tokens"]),
                           ("cached_input_tokens", usage["cache_read_tokens"]),
                           ("cache_write_input_tokens", usage["cache_write_tokens"]),
                           ("output_tokens", usage["output_tokens"]),
                           ("reasoning_output_tokens", usage["reasoning_tokens"])):
        if value is None:
            continue
        if raw_key == "cache_write_input_tokens" and write_defaulted:
            continue  # default is provenance, not an observed measurement
        raw[raw_key] = value
    if write_defaulted:
        raw["cache_write_defaulted"] = True
    usage["raw_fields"] = raw
    return STATUS_PARTIAL if warnings else STATUS_OK


# OpenCode: step_finish parts with scoped identity and completeness.


def _step_values(part, warnings, tag):
    """Validated per-part metrics; gaps stay None to refuse subtotals."""
    tokens = part.get("tokens") if isinstance(part, dict) else None
    tokens = tokens if isinstance(tokens, dict) else None
    cache = tokens.get("cache") if isinstance(tokens, dict) else None
    cache = cache if isinstance(cache, dict) else None
    return {
        "input": _take(tokens, "input", warnings, False, label=f"opencode {tag} tokens.input"),
        "output": _take(tokens, "output", warnings, False, label=f"opencode {tag} tokens.output"),
        "reasoning": _take(tokens, "reasoning", warnings, False,
                            label=f"opencode {tag} tokens.reasoning"),
        "read": _take(cache, "read", warnings, False, label=f"opencode {tag} cache.read"),
        "write": _take(cache, "write", warnings, False, label=f"opencode {tag} cache.write"),
        "total": _take(tokens, "total", warnings, False, label=f"opencode {tag} tokens.total"),
        # Only part.cost: no speculative tokens.cost fallback exists.
        "cost": _take(part, "cost", warnings, False, kind="number",
                      label=f"opencode {tag} cost"),
    }


def _step_key(event, part):
    """Event-local scoped identity (session, message, part) or None.

    The session must be explicitly present on THAT event/part: a unique
    stream-wide session is never borrowed, so incompletely identified steps
    never merge. All three components must be non-empty strings, exact."""
    session = None
    for found in _scoped_ids(event, _SESSION_KEYS) + _scoped_ids(part, _SESSION_KEYS):
        if session is None:
            session = found
        elif found != session:
            return None
    message = None
    if isinstance(part, dict):
        for key in _MESSAGE_KEYS:
            value = part.get(key)
            if isinstance(value, str) and value != "":
                message = value
                break
    pid = part.get("id") if isinstance(part, dict) else None
    if not isinstance(session, str) or session == "":
        return None
    if not isinstance(message, str) or message == "":
        return None
    if not isinstance(pid, str) or pid == "":
        return None
    return (session, message, pid)


def _parse_opencode(events, usage, warnings):
    steps = [e for e in events if e.get("type") == "step_finish"]
    if not steps:
        warnings.append("unexpected opencode payload: no step_finish events")
        return STATUS_ERROR
    sessions = []
    for event in steps:
        part = event.get("part")
        for found in _scoped_ids(event, _SESSION_KEYS) + _scoped_ids(part, _SESSION_KEYS):
            if found not in sessions:
                sessions.append(found)
    if len(sessions) > 1:
        warnings.append("multiple opencode sessions without proven relation; refusing to attribute")
        return STATUS_ERROR

    # Merge compatible retransmits on event-local scoped identity: equal
    # metrics dedup safely, a later completion filling gaps merges
    # deliberately, differing valid values conflict -> error. Without an
    # explicit session on the event itself there is no scope: no dedup.
    merged = []
    by_key = {}
    duplicates = 0
    for index, event in enumerate(steps):
        tag = f"step {index}"
        part = event.get("part")
        if not isinstance(part, dict):
            warnings.append(f"step_finish without usable part at {tag}")
            merged.append({field: None for field in _OPENCODE_FIELDS})
            continue
        values = _step_values(part, warnings, tag)
        if all(value is None for value in values.values()):
            warnings.append(f"step_finish without usable tokens at {tag}")
        key = _step_key(event, part)
        if key is None or key not in by_key:
            if key is not None:
                by_key[key] = len(merged)
            merged.append(values)
            continue
        slot = by_key[key]
        current = merged[slot]
        combined = {}
        conflict = False
        for field in _OPENCODE_FIELDS:
            old, new = current[field], values[field]
            if old is not None and new is not None and old != new:
                conflict = True
                break
            combined[field] = old if old is not None else new
        if conflict:
            warnings.append("conflicting metrics for repeated step identity; refusing to attribute")
            return STATUS_ERROR
        merged[slot] = combined
        duplicates += 1
        warnings.append(f"compatible retransmit merged at {tag}")

    series = {field: [entry[field] for entry in merged] for field in _OPENCODE_FIELDS}
    raw = {"events_observed": len(events), "step_finish": len(merged),
           "duplicates_merged": duplicates, "sessions_observed": len(sessions)}
    coverage = {}
    for field in _OPENCODE_FIELDS:
        valid = sum(1 for value in series[field] if value is not None)
        coverage[field] = {"valid": valid, "total": len(merged)}
    raw["coverage"] = coverage

    def summed(field, label, expected):
        total = _complete_sum(series[field])
        if total is None:
            valid = coverage[field]["valid"]
            if expected or valid > 0:
                warnings.append(f"incomplete opencode {label}: "
                                f"{valid}/{len(merged)} steps valid; using null")
        return total

    raw_input = summed("input", "input", True)
    raw_read = summed("read", "cache read", False)
    raw_write = summed("write", "cache write", False)
    # Native input is cache-exclusive; inclusive adds read+write, needing
    # all three sums complete.
    if raw_input is not None and raw_read is not None and raw_write is not None:
        usage["input_tokens"] = raw_input + raw_read + raw_write
    else:
        usage["input_tokens"] = None
        warnings.append("incomplete opencode inclusive input: requires complete "
                        "raw input plus cache read plus cache write; using null")
        if raw_input is not None:
            # Known native subtotal, explicitly labelled incomplete: kept as
            # provenance, never as the normalized inclusive total.
            raw["input_incomplete_sum"] = raw_input
    # Native output already excludes reasoning: kept native, reasoning
    # preserved separately (cross-provider output comparison is limited).
    usage["output_tokens"] = summed("output", "output", True)
    usage["cache_read_tokens"] = raw_read
    usage["cache_write_tokens"] = raw_write
    usage["reasoning_tokens"] = summed("reasoning", "reasoning", False)
    # tokens.total is optional native passthrough: full coverage only,
    # never derived.
    native_totals = series["total"]
    if all(value is None for value in native_totals):
        usage["total_tokens"] = None
    else:
        usage["total_tokens"] = _complete_sum(native_totals)
        if usage["total_tokens"] is None:
            warnings.append("incomplete opencode native total; using null")
        else:
            raw["tokens_total_sum"] = usage["total_tokens"]
    usage["requests"] = len(merged)
    # part.cost is a pricing-table estimate summed only with full coverage.
    native_costs = series["cost"]
    if all(value is None for value in native_costs):
        usage["cost_usd"] = None
        usage["cost_source"] = COST_UNAVAILABLE
    else:
        cost_total = _complete_fsum(native_costs)
        if cost_total is None:
            usage["cost_usd"] = None
            usage["cost_source"] = COST_UNAVAILABLE
            warnings.append("incomplete opencode cost; using null")
        else:
            usage["cost_usd"] = cost_total
            usage["cost_source"] = COST_ESTIMATED
            raw["cost_sum"] = cost_total
    usage["cli_duration_s"] = None  # only emit-time timestamps exist
    usage["raw_fields"] = raw
    return STATUS_PARTIAL if warnings else STATUS_OK


# Entrypoint.


def _recovery_raw(cli, events, malformed):
    """Observation counts preserved on strict-malformed error; never totals."""
    key = "turn_completed" if cli == "codex" else "step_finish"
    want = "turn.completed" if cli == "codex" else "step_finish"
    return {"events_observed": len(events), "malformed_lines": malformed,
            f"recovered_{key}": sum(1 for e in events if e.get("type") == want)}


def parse_generation_usage(cli, stdout, model=None, cli_version=None):
    """Parse CLI stdout into ``{"usage", "parse_status", "warnings"}``.

    - ``cli``: one of ``claude`` / ``codex`` / ``opencode`` (case-insensitive).
    - ``stdout``: CLI stdout text, or an already-decoded JSON object for the
      single-object shapes. Empty, malformed, unexpected top-level shapes and
      unknown CLIs are ``parse_status == "error"`` with null metrics. Any
      malformed non-blank JSONL line is also ``error`` (strict); recovered
      observations survive in ``raw_fields`` only.
    - ``model``: requested model identity, echoed as
      ``usage["requested_model"]``; the parser never guesses an effective
      model from event payloads.
    - ``parse_status``: ``ok`` (complete), ``partial`` (valid syntax with
      nulls/diagnoses), ``error`` (no usable normalized data; strict).
    """
    warnings = []
    cli_norm = cli.strip().lower() if isinstance(cli, str) else None
    if cli_norm not in SUPPORTED_CLIS:
        usage = _blank_usage(cli if isinstance(cli, str) else None, model, cli_version)
        warnings.append("unsupported cli for numeric usage parser; legacy parser retained")
        return {"usage": usage, "parse_status": STATUS_ERROR, "warnings": warnings}

    usage = _blank_usage(cli_norm, model, cli_version)

    if stdout is None or (isinstance(stdout, str) and not stdout.strip()):
        warnings.append("empty stdout; no usage data")
        return {"usage": usage, "parse_status": STATUS_ERROR, "warnings": warnings}

    if isinstance(stdout, dict):
        decoded = {"single": stdout, "events": [stdout], "malformed": 0}
    elif not isinstance(stdout, str):
        warnings.append("unexpected input type; expected stdout text or decoded object")
        return {"usage": usage, "parse_status": STATUS_ERROR, "warnings": warnings}
    else:
        decoded = _decode(stdout)

    if cli_norm == "claude":
        if decoded["single"] is None:
            warnings.append("claude stdout is not a single result object; "
                            "no regex fallback for numerics")
            return {"usage": usage, "parse_status": STATUS_ERROR, "warnings": warnings}
        status = _parse_claude(decoded["single"], usage, warnings)
        return {"usage": usage, "parse_status": status, "warnings": warnings}

    if decoded["malformed"]:
        usage["raw_fields"] = _recovery_raw(cli_norm, decoded["events"], decoded["malformed"])
        warnings.append(f"malformed JSONL line(s): {decoded['malformed']}; "
                        "recovered observations kept in raw_fields only")
        return {"usage": usage, "parse_status": STATUS_ERROR, "warnings": warnings}
    if cli_norm == "codex":
        status = _parse_codex(decoded["events"], usage, warnings)
    else:
        status = _parse_opencode(decoded["events"], usage, warnings)
    return {"usage": usage, "parse_status": status, "warnings": warnings}
