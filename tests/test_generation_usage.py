"""Phase 1: numeric generation-usage parser tests (synthetic fixtures only).

All fixtures under tests/fixtures/generation_usage are synthetic and
source-derived, NOT real CLI captures. Closed semantics: Codex
thread-cumulative (last snapshot wins), OpenCode per-field completeness with
scoped identity, Claude complete components with estimated cost. No network,
no credentials, no APIs.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from utils.generation_usage import (  # noqa: E402
    parse_generation_usage,
    USAGE_FIELDS,
)
from utils import utils as legacy  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "generation_usage"
SENTINEL = "SENTINEL_SECRET_ABC123"


def fixture(name):
    return (FIX / name).read_text(encoding="utf-8")


def dump(result):
    return json.dumps(result, default=str)


# ---------------------------------------------------------------- Claude

def test_claude_full_exact():
    r = parse_generation_usage("claude", fixture("claude_full.json"), model="req-model")
    assert set(r.keys()) == {"usage", "parse_status", "warnings"}
    assert r["parse_status"] == "ok", r["warnings"]
    u = r["usage"]
    assert u["input_tokens"] == 9500  # 1000 + 8000 + 500
    assert u["output_tokens"] == 250
    assert u["cache_read_tokens"] == 8000
    assert u["cache_write_tokens"] == 500
    assert u["total_tokens"] == 9750
    assert u["requests"] == 3
    assert u["cost_usd"] == pytest.approx(0.042)
    assert u["cost_source"] == "estimated"  # client-side estimate, not billed
    assert u["cli_duration_s"] is None  # no trusted CLI semantic
    assert u["raw_fields"]["duration_api_ms"] == pytest.approx(12500.0)
    assert u["raw_fields"]["duration_ms"] == pytest.approx(13200.0)
    assert "duration_source" not in u["raw_fields"]
    assert u["source"] == {"cli": "claude", "cli_version": None, "provider": None}
    assert u["requested_model"] == "req-model"
    assert u["process_duration_s"] is None and u["wall_duration_s"] is None


def test_claude_ttl_detail_not_double_counted():
    r = parse_generation_usage("claude", fixture("claude_full.json"))
    raw = r["usage"]["raw_fields"]
    assert raw["cache_creation"] == {"ephemeral_5m_input_tokens": 500,
                                     "ephemeral_1h_input_tokens": 0}
    # Detail (500) is part of the aggregate write (500): input unchanged.
    assert r["usage"]["input_tokens"] == 9500
    assert r["usage"]["cache_write_tokens"] == 500


def test_claude_model_usage_allowlist():
    r = parse_generation_usage("claude", fixture("claude_full.json"))
    models = r["usage"]["raw_fields"]["modelUsage"]
    assert set(models["req-model"]) == {"inputTokens", "outputTokens",
                                       "cacheReadInputTokens",
                                       "cacheCreationInputTokens", "costUSD"}
    assert models["req-model"]["costUSD"] == pytest.approx(0.042)


def test_claude_zero_cost_preserved_estimated():
    r = parse_generation_usage("claude", fixture("claude_zero_cost.json"))
    u = r["usage"]
    assert u["cost_usd"] == 0.0 and not isinstance(u["cost_usd"], bool)
    assert u["cost_source"] == "estimated"  # zero estimated != unknown
    assert u["input_tokens"] == 120
    assert r["parse_status"] == "ok", r["warnings"]


def test_claude_missing_cost_partial():
    r = parse_generation_usage("claude", fixture("claude_missing_cost.json"))
    assert r["parse_status"] == "partial"
    assert r["usage"]["cost_usd"] is None
    assert r["usage"]["cost_source"] == "unavailable"
    assert any("total_cost_usd" in w for w in r["warnings"])
    assert r["usage"]["input_tokens"] == 510


def test_claude_partial_input17_needs_all_components():
    r = parse_generation_usage("claude", fixture("claude_partial_input17.json"))
    assert r["parse_status"] == "partial"
    u = r["usage"]
    # Base alone is not enough: read/write unknown -> input null, not 17.
    assert u["input_tokens"] is None
    assert u["output_tokens"] is None
    assert u["total_tokens"] is None
    assert any("output_tokens" in w for w in r["warnings"])


def test_claude_invalid_numerics_null_input():
    r = parse_generation_usage("claude", fixture("claude_invalid_numerics.json"))
    assert r["parse_status"] == "partial"
    u = r["usage"]
    # Bool base poisons the sum even though cache read (100) is valid.
    assert u["input_tokens"] is None
    assert u["output_tokens"] is None
    assert u["total_tokens"] is None
    assert any("input_tokens" in w for w in r["warnings"])
    assert any("output_tokens" in w for w in r["warnings"])


def test_claude_nonfinite_dict_input():
    payload = {
        "num_turns": 1,
        "total_cost_usd": float("inf"),
        "duration_ms": 100,
        "usage": {"input_tokens": 5, "output_tokens": float("nan")},
    }
    r = parse_generation_usage("claude", payload)
    assert r["parse_status"] == "partial"
    assert r["usage"]["input_tokens"] is None  # read/write missing
    assert r["usage"]["output_tokens"] is None
    assert r["usage"]["cost_usd"] is None
    assert r["usage"]["cost_source"] == "unavailable"
    assert any("non-finite" in w for w in r["warnings"])


def test_claude_huge_cost_rejected_without_overflow():
    payload = {"num_turns": 1, "total_cost_usd": 10 ** 10000, "duration_ms": 50,
               "usage": {"input_tokens": 1, "output_tokens": 1,
                         "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0}}
    r = parse_generation_usage("claude", payload)  # must not raise
    assert r["usage"]["cost_usd"] is None
    assert r["usage"]["cost_source"] == "unavailable"
    assert any("total_cost_usd" in w for w in r["warnings"])


def test_claude_zero_inputs_stay_zero_not_null():
    payload = {
        "num_turns": 1, "total_cost_usd": 0.0, "duration_ms": 50,
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0},
    }
    r = parse_generation_usage("claude", payload)
    u = r["usage"]
    assert u["input_tokens"] == 0 and u["input_tokens"] is not None
    assert u["output_tokens"] == 0 and u["output_tokens"] is not None
    assert u["total_tokens"] == 0 and u["total_tokens"] is not None
    assert r["parse_status"] == "ok", r["warnings"]


# ---------------------------------------------------------------- Codex (cumulative)

def test_codex_last_snapshot_wins_never_sums():
    r = parse_generation_usage("codex", fixture("codex_multi_turn.jsonl"))
    assert r["parse_status"] == "ok", r["warnings"]
    u = r["usage"]
    assert u["input_tokens"] == 3000  # last snapshot, not 1000+3000
    assert u["output_tokens"] == 400
    assert u["cache_read_tokens"] == 2200
    assert u["cache_write_tokens"] == 10
    assert u["reasoning_tokens"] == 150
    assert u["total_tokens"] == 3400  # input + output convention
    assert u["requests"] == 2  # turns counted independently
    assert u["cost_usd"] is None and u["cost_source"] == "unavailable"
    assert u["cli_duration_s"] is None
    assert u["raw_fields"]["events_observed"] == 6
    assert u["raw_fields"]["turn_completed"] == 2
    assert u["raw_fields"]["threads_observed"] == 1


def test_codex_single_json_object():
    r = parse_generation_usage("codex", fixture("codex_single_json.json"))
    assert r["parse_status"] == "ok", r["warnings"]
    assert r["usage"]["input_tokens"] == 400
    assert r["usage"]["cache_read_tokens"] == 50
    assert r["usage"]["cache_write_tokens"] == 5
    assert r["usage"]["reasoning_tokens"] == 20
    assert r["usage"]["total_tokens"] == 460


def test_codex_single_decoded_dict_missing_cache_warns():
    r = parse_generation_usage(
        "codex",
        {"type": "turn.completed",
         "usage": {"input_tokens": 7, "output_tokens": 3}},
    )
    assert r["parse_status"] == "partial"
    assert r["usage"]["input_tokens"] == 7
    assert r["usage"]["cache_read_tokens"] is None
    assert r["usage"]["cache_write_tokens"] == 0  # absent write, backed default
    assert r["usage"]["total_tokens"] == 10


def test_codex_partial_input17():
    r = parse_generation_usage("codex", fixture("codex_partial_input17.jsonl"))
    assert r["parse_status"] == "partial"
    u = r["usage"]
    assert u["input_tokens"] == 17
    assert u["output_tokens"] is None  # missing in latest, never backfilled
    assert u["total_tokens"] is None


def test_codex_last_incomplete_no_backfill():
    stdout = "\n".join([
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 100, "cached_input_tokens": 10,
            "output_tokens": 50}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 200}}),
    ])
    r = parse_generation_usage("codex", stdout)
    assert r["parse_status"] == "partial"
    assert r["usage"]["input_tokens"] == 200
    assert r["usage"]["output_tokens"] is None  # earlier 50 not reused
    assert r["usage"]["total_tokens"] is None
    assert r["usage"]["requests"] == 2


def test_codex_earlier_invalid_later_valid():
    stdout = "\n".join([
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": False, "output_tokens": -1}}),
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 300, "cached_input_tokens": 20,
            "output_tokens": 40}}),
    ])
    r = parse_generation_usage("codex", stdout)
    assert r["parse_status"] == "ok", r["warnings"]  # earlier ignored entirely
    assert r["usage"]["input_tokens"] == 300
    assert r["usage"]["requests"] == 2


def test_codex_duplicate_content_counts_turns_last_wins():
    # No reliable turn ids: identical content is two turns; usage is the last.
    r = parse_generation_usage("codex", fixture("codex_duplicate_content.jsonl"))
    assert r["parse_status"] == "ok", r["warnings"]
    assert r["usage"]["requests"] == 2
    assert r["usage"]["input_tokens"] == 500  # not 1000
    assert r["usage"]["output_tokens"] == 50


def test_codex_last_invalid_nulls():
    r = parse_generation_usage("codex", fixture("codex_invalid.jsonl"))
    assert r["parse_status"] == "partial"
    u = r["usage"]
    assert u["input_tokens"] is None
    assert u["output_tokens"] is None
    assert u["cache_read_tokens"] == 10  # valid in latest, kept
    assert u["requests"] == 2
    assert any("input_tokens" in w for w in r["warnings"])


def test_codex_malformed_is_strict_error_with_recovery():
    r = parse_generation_usage("codex", fixture("codex_malformed.jsonl"))
    assert r["parse_status"] == "error"
    assert r["usage"]["input_tokens"] is None  # no deceptive totals
    assert r["usage"]["requests"] is None
    raw = r["usage"]["raw_fields"]
    assert raw["malformed_lines"] == 1
    assert raw["recovered_turn_completed"] == 2
    assert any("malformed" in w for w in r["warnings"])


def test_codex_all_malformed_is_error():
    r = parse_generation_usage("codex", "not json {{{\n[1, 2]\n")
    assert r["parse_status"] == "error"
    assert r["usage"]["input_tokens"] is None
    assert r["warnings"]


def test_codex_no_turn_events_is_error():
    r = parse_generation_usage("codex", '{"type": "other.event", "x": 1}')
    assert r["parse_status"] == "error"


def test_codex_new_thread_is_error():
    r = parse_generation_usage("codex", fixture("codex_new_thread.jsonl"))
    assert r["parse_status"] == "error"
    assert r["usage"]["input_tokens"] is None
    assert any("thread" in w for w in r["warnings"])


def test_codex_thread_mismatch_is_error():
    stdout = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "thr-Y"}),
        json.dumps({"type": "turn.completed", "thread_id": "thr-X",
                    "usage": {"input_tokens": 1, "output_tokens": 1}}),
    ])
    assert parse_generation_usage("codex", stdout)["parse_status"] == "error"


# ---------------------------------------------------------------- OpenCode (scoped + complete)

def test_opencode_missing_write_nulls_inclusive_input():
    r = parse_generation_usage("opencode", fixture("opencode_multi_step.jsonl"))
    u = r["usage"]
    assert u["input_tokens"] is None  # write unknown: no inclusive total
    assert u["output_tokens"] == 300  # native output complete
    assert u["cache_read_tokens"] == 3000
    assert u["cache_write_tokens"] is None
    assert u["total_tokens"] is None  # no native totals observed
    assert u["requests"] == 2
    assert u["cost_usd"] is None and u["cost_source"] == "unavailable"
    assert r["parse_status"] == "partial"
    assert any("inclusive input" in w for w in r["warnings"])


def test_opencode_complete_is_ok_estimated_passthrough():
    r = parse_generation_usage("opencode", fixture("opencode_full_metrics.jsonl"))
    assert r["parse_status"] == "ok", r["warnings"]
    u = r["usage"]
    assert u["input_tokens"] == 1460  # (400+100+20) + (600+300+40)
    assert u["output_tokens"] == 300  # native, reasoning-exclusive
    assert u["cache_read_tokens"] == 400
    assert u["cache_write_tokens"] == 60
    assert u["reasoning_tokens"] == 120
    assert u["total_tokens"] == 18100  # native passthrough, never derived
    assert u["cost_usd"] == pytest.approx(0.012)
    assert u["cost_source"] == "estimated"
    assert u["requests"] == 2
    assert u["raw_fields"]["sessions_observed"] == 1


def test_opencode_compatible_retransmit_merges():
    r = parse_generation_usage("opencode", fixture("opencode_duplicate_ids.jsonl"))
    u = r["usage"]
    assert u["requests"] == 2  # retransmit is not a new step
    assert u["input_tokens"] == 159  # (100+5+2) + (50+1+1)
    assert u["output_tokens"] == 15
    assert u["cost_usd"] == pytest.approx(0.003)
    assert u["cost_source"] == "estimated"
    assert u["raw_fields"]["duplicates_merged"] == 1
    assert r["parse_status"] == "partial"
    assert any("retransmit" in w for w in r["warnings"])


def test_opencode_conflicting_retransmit_is_error():
    r = parse_generation_usage("opencode", fixture("opencode_conflict_ids.jsonl"))
    assert r["parse_status"] == "error"
    assert r["usage"]["input_tokens"] is None
    assert any("conflicting" in w for w in r["warnings"])


def test_opencode_cross_session_same_part_is_error():
    r = parse_generation_usage("opencode", fixture("opencode_cross_session.jsonl"))
    assert r["parse_status"] == "error"
    assert any("session" in w for w in r["warnings"])


def test_opencode_later_completion_merges_deliberately():
    r = parse_generation_usage("opencode", fixture("opencode_completion.jsonl"))
    u = r["usage"]
    assert u["requests"] == 1
    # Later completion fills the earlier gap: not silent first-wins.
    assert u["output_tokens"] == 10
    assert u["input_tokens"] is None  # write still unknown everywhere
    assert r["parse_status"] == "partial"


def test_opencode_identical_without_ids_kept():
    r = parse_generation_usage("opencode", fixture("opencode_no_ids.jsonl"))
    assert r["usage"]["requests"] == 2  # no scope: no dedup, no collapse
    assert r["usage"]["output_tokens"] == 50
    assert r["usage"]["input_tokens"] is None  # write unknown
    assert r["usage"]["raw_fields"]["duplicates_merged"] == 0


def test_opencode_multi_session_is_error():
    r = parse_generation_usage("opencode", fixture("opencode_multi_session.jsonl"))
    assert r["parse_status"] == "error"
    assert r["usage"]["input_tokens"] is None  # not mixed, not attributed
    assert any("session" in w for w in r["warnings"])
    assert "s-A" not in dump(r) and "s-B" not in dump(r)


def test_opencode_envelope_part_mismatch_is_error():
    payload = json.dumps({"type": "step_finish", "sessionID": "ses-X",
                          "part": {"id": "prt-1", "messageID": "msg-1",
                                   "sessionID": "ses-Y",
                                   "tokens": {"input": 1, "output": 1}}})
    assert parse_generation_usage("opencode", payload)["parse_status"] == "error"


def test_opencode_partial_input17():
    r = parse_generation_usage("opencode", fixture("opencode_partial_input17.jsonl"))
    assert r["parse_status"] == "partial"
    assert r["usage"]["input_tokens"] is None  # write unknown
    assert r["usage"]["output_tokens"] is None
    assert r["usage"]["total_tokens"] is None


def test_opencode_no_steps_is_error():
    r = parse_generation_usage("opencode", '{"type": "turn.completed", "x": 1}')
    assert r["parse_status"] == "error"


def test_opencode_malformed_is_strict_error_with_recovery():
    r = parse_generation_usage("opencode", fixture("opencode_malformed.jsonl"))
    assert r["parse_status"] == "error"
    assert r["usage"]["output_tokens"] is None
    assert r["usage"]["requests"] is None
    raw = r["usage"]["raw_fields"]
    assert raw["malformed_lines"] == 1
    assert raw["recovered_step_finish"] == 2


def test_opencode_huge_integers_stay_exact():
    big = 9007199254740993
    stdout = "\n".join([
        json.dumps({"type": "step_finish", "part": {
            "tokens": {"input": big, "output": 1,
                       "cache": {"read": 0, "write": 0}}}}),
        json.dumps({"type": "step_finish", "part": {
            "tokens": {"input": 1, "output": 1,
                       "cache": {"read": 0, "write": 0}}}}),
    ])
    r = parse_generation_usage("opencode", stdout)
    assert r["usage"]["input_tokens"] == big + 1  # exact, not float-rounded
    assert r["usage"]["output_tokens"] == 2


# ---------------------------------------------------------------- Strictness

@pytest.mark.parametrize("cli", ["claude", "codex", "opencode"])
def test_empty_stdout_is_error(cli):
    for bad in ("", "   ", None):
        r = parse_generation_usage(cli, bad)
        assert r["parse_status"] == "error", (cli, bad)
        assert r["usage"]["input_tokens"] is None
        assert r["warnings"]


def test_claude_malformed_json_is_error():
    r = parse_generation_usage("claude", fixture("malformed.json"))
    assert r["parse_status"] == "error"
    assert r["warnings"]


def test_claude_unexpected_shape_is_error():
    r = parse_generation_usage("claude", fixture("unexpected.json"))
    assert r["parse_status"] == "error"


def test_unexpected_top_level_list_is_error():
    assert parse_generation_usage("codex", "[1, 2]")["parse_status"] == "error"
    assert parse_generation_usage("claude", "[1, 2]")["parse_status"] == "error"


def test_unexpected_input_type_is_error():
    assert parse_generation_usage("claude", 12345)["parse_status"] == "error"


@pytest.mark.parametrize("cli", ["copilot", "gemini", "unknown-cli", ""])
def test_unsupported_cli_is_error(cli):
    r = parse_generation_usage(cli, '{"usage": {}}')
    assert r["parse_status"] == "error"
    assert any("legacy" in w for w in r["warnings"])


# ---------------------------------------------------------------- Schema / privacy

def test_result_shape_and_defaults():
    r = parse_generation_usage("codex", fixture("codex_single_json.json"),
                               model="req", cli_version="9.9.9")
    assert set(r.keys()) == {"usage", "parse_status", "warnings"}
    assert set(r["usage"].keys()) == set(USAGE_FIELDS)
    assert isinstance(r["warnings"], list)
    assert r["usage"]["source"]["cli_version"] == "9.9.9"
    assert r["usage"]["source"]["provider"] is None  # never guessed
    assert r["usage"]["requested_model"] == "req"  # requested, not effective
    assert r["usage"]["process_duration_s"] is None
    assert r["usage"]["wall_duration_s"] is None
    assert r["usage"]["reasoning_tokens"] == 20


def test_warnings_never_leak_output_payload():
    payloads = [
        ("claude", json.dumps({"result": SENTINEL, "note": SENTINEL})),
        ("codex", json.dumps({"type": "turn.completed", "note": SENTINEL,
                              "usage": {"input_tokens": 5, "output_tokens": 2}})),
        ("opencode", json.dumps({"type": "step_finish", "text": SENTINEL,
                                 "part": {"id": "x", "tokens": {"input": 1,
                                                                "output": 1}}})),
    ]
    for cli, stdout in payloads:
        r = parse_generation_usage(cli, stdout)
        assert SENTINEL not in dump(r), cli


def test_raw_fields_hold_only_metrics():
    for name, cli in [("claude_full.json", "claude"),
                      ("codex_multi_turn.jsonl", "codex"),
                      ("opencode_full_metrics.jsonl", "opencode")]:
        r = parse_generation_usage(cli, fixture(name))
        blob = json.dumps(r["usage"]["raw_fields"]).lower()
        assert not any(d in blob for d in ("sessionid", "prompt", "sentinel")), name
        assert set(r["usage"]["raw_fields"]).isdisjoint({"text", "result", "prompt"}), name
    cov = parse_generation_usage(
        "opencode", fixture("opencode_full_metrics.jsonl"))["usage"]["raw_fields"]
    assert cov["coverage"]["input"] == {"valid": 2, "total": 2}


# ---------------------------------------------------------------- Legacy compatibility

def test_legacy_wrappers_keep_adapter_shapes():
    cl = legacy.claude_parse_stats({"num_turns": 2, "duration_api_ms": 1500,
                                    "usage": {"input_tokens": 100, "output_tokens": 20,
                                              "cache_read_input_tokens": 0,
                                              "cache_creation_input_tokens": 0}},
                                   "m")
    assert set(cl.keys()) == {"requests", "time_spent", "models_breakdown"}
    assert cl["requests"] == 2
    # No trusted CLI duration in phase 1: adapters own wall timing.
    assert cl["time_spent"] == ""

    for fn in (legacy.codex_parse_stats, legacy.opencode_parse_stats):
        assert set(fn("", "m").keys()) == {"changes_added", "changes_removed",
                                           "requests", "time_spent", "models_breakdown"}

    cx = legacy.codex_parse_stats(
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}, "m")
    assert cx["requests"] == 1 and "m" in cx["models_breakdown"]

    oc = legacy.opencode_parse_stats(
        fixture("opencode_multi_step.jsonl"), "m")
    assert oc["requests"] == 2 and "m" in oc["models_breakdown"]
    assert oc["time_spent"] == ""


def test_copilot_gemini_implementations_untouched():
    assert legacy.copilot_parse_stats("", "m") == {
        "changes_added": 0, "changes_removed": 0, "requests": 0, "time_spent": ""}
    sample = ("Changes    +13 -0\nsome work 6s\n"
              "Tokens     \u2191 544.8k (512.3k cached, 32.5k written) \u2022 "
              "\u2193 7.5k (4.9k reasoning)")
    out = legacy.copilot_parse_stats(sample, "m")
    assert out["changes_added"] == 13 and out["time_spent"] == "6s"
    assert "m" in out["models_breakdown"]

    g = legacy.gemini_parse_stats("", "m")
    assert g["requests"] == 0 and g["time_spent"] == "" and g["models_breakdown"] == {}
    g2 = legacy.gemini_parse_stats(json.dumps({
        "usage": {"input_tokens": 50, "output_tokens": 10, "cache_read_tokens": 5},
        "num_turns": 2, "duration_seconds": 1.5}), "m")
    assert g2["requests"] == 2 and g2["time_spent"] == "1.50s"


def test_math_finite_guard():
    assert not math.isfinite(float("nan"))
    r = parse_generation_usage("codex", json.dumps({
        "type": "turn.completed",
        "usage": {"input_tokens": 10, "output_tokens": float("inf")}}))
    assert r["usage"]["output_tokens"] is None
    assert r["parse_status"] == "partial"


# ---------------------------------------------------------------- Gate1 attempt2 regressions

def test_opencode_scoped_identity_not_borrowed():
    # Same message/part, session present on only one event: two steps,
    # never a proven retransmit.
    first = {"type": "step_finish", "sessionID": "ses-A",
             "part": {"id": "prt-1", "messageID": "msg-1",
                      "sessionID": "ses-A",
                      "tokens": {"input": 100, "output": 10}}}
    second = {"type": "step_finish",
              "part": {"id": "prt-1", "messageID": "msg-1",
                       "tokens": {"input": 100, "output": 10}}}
    for order in ((first, second), (second, first)):
        stdout = "\n".join(json.dumps(e) for e in order)
        r = parse_generation_usage("opencode", stdout)
        assert r["usage"]["requests"] == 2, order
        assert r["usage"]["raw_fields"]["duplicates_merged"] == 0, order


def test_opencode_cost_overflow_is_null_not_infinity():
    stdout = "\n".join(json.dumps({
        "type": "step_finish",
        "part": {"tokens": {"input": 1, "output": 1}, "cost": 1e308}})
        for _ in range(2))
    r = parse_generation_usage("opencode", stdout)
    assert r["usage"]["cost_usd"] is None
    assert r["usage"]["cost_source"] == "unavailable"
    assert "cost_sum" not in r["usage"]["raw_fields"]
    assert any("cost" in w for w in r["warnings"])
    json.dumps(r, allow_nan=False)  # strict JSON must succeed


def test_all_results_strict_json_serializable():
    cases = [("claude_full.json", "claude"),
             ("claude_zero_cost.json", "claude"),
             ("claude_missing_cost.json", "claude"),
             ("claude_partial_input17.json", "claude"),
             ("claude_invalid_numerics.json", "claude"),
             ("codex_multi_turn.jsonl", "codex"),
             ("codex_single_json.json", "codex"),
             ("codex_partial_input17.jsonl", "codex"),
             ("codex_duplicate_content.jsonl", "codex"),
             ("codex_invalid.jsonl", "codex"),
             ("codex_malformed.jsonl", "codex"),
             ("codex_new_thread.jsonl", "codex"),
             ("opencode_multi_step.jsonl", "opencode"),
             ("opencode_full_metrics.jsonl", "opencode"),
             ("opencode_duplicate_ids.jsonl", "opencode"),
             ("opencode_conflict_ids.jsonl", "opencode"),
             ("opencode_cross_session.jsonl", "opencode"),
             ("opencode_completion.jsonl", "opencode"),
             ("opencode_no_ids.jsonl", "opencode"),
             ("opencode_multi_session.jsonl", "opencode"),
             ("opencode_partial_input17.jsonl", "opencode"),
             ("opencode_malformed.jsonl", "opencode")]
    for name, cli in cases:
        r = parse_generation_usage(cli, fixture(name))
        json.dumps(r, allow_nan=False)  # no NaN/Infinity anywhere
    edge = [
        ("claude", {"num_turns": 1, "total_cost_usd": float("nan"),
                    "usage": {"input_tokens": 1}}),
        ("codex", {"type": "turn.completed",
                   "usage": {"input_tokens": float("inf")}}),
        ("opencode", {"type": "step_finish",
                      "part": {"tokens": {"input": 10 ** 400}}}),
    ]
    for cli, payload in edge:
        r = parse_generation_usage(cli, payload)
        json.dumps(r, allow_nan=False)


def test_codex_last_turn_without_usage_no_default_zero():
    stdout = "\n".join([
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 5, "output_tokens": 2,
            "cached_input_tokens": 1}}),
        json.dumps({"type": "turn.completed"}),
    ])
    r = parse_generation_usage("codex", stdout)
    assert r["parse_status"] == "partial"
    assert r["usage"]["cache_write_tokens"] is None  # unknown, not 0
    assert "cache_write_input_tokens" not in r["usage"]["raw_fields"]
    assert r["usage"]["requests"] == 2


def test_codex_explicit_null_cache_write_is_diagnosed():
    r = parse_generation_usage("codex", {
        "type": "turn.completed",
        "usage": {"input_tokens": 5, "output_tokens": 2,
                  "cached_input_tokens": 1,
                  "cache_write_input_tokens": None}})
    assert r["parse_status"] == "partial"  # not ok: explicit null diagnosed
    assert r["usage"]["cache_write_tokens"] is None
    assert any("cache_write_input_tokens" in w for w in r["warnings"])


def test_codex_absent_write_still_zero_with_provenance_label():
    r = parse_generation_usage("codex", {
        "type": "turn.completed",
        "usage": {"input_tokens": 10, "cached_input_tokens": 2,
                  "output_tokens": 5}})
    assert r["parse_status"] == "ok", r["warnings"]
    assert r["usage"]["cache_write_tokens"] == 0
    # Default labelled as default, never as an observed measurement.
    assert "cache_write_input_tokens" not in r["usage"]["raw_fields"]
    assert r["usage"]["raw_fields"]["cache_write_defaulted"] is True


def test_opencode_native_input_retained_when_inclusive_unknown():
    r = parse_generation_usage("opencode", fixture("opencode_multi_step.jsonl"))
    assert r["usage"]["input_tokens"] is None
    assert r["usage"]["raw_fields"]["input_incomplete_sum"] == 1200
    r = parse_generation_usage("opencode", fixture("opencode_no_ids.jsonl"))
    assert r["usage"]["raw_fields"]["input_incomplete_sum"] == 500
    r = parse_generation_usage("opencode", fixture("opencode_partial_input17.jsonl"))
    assert r["usage"]["raw_fields"]["input_incomplete_sum"] == 17
    # Complete inclusive input carries no incomplete subtotal.
    r = parse_generation_usage("opencode", fixture("opencode_full_metrics.jsonl"))
    assert "input_incomplete_sum" not in r["usage"]["raw_fields"]


# ---------------------------------------------------------------- Adapter imports / aliases

def test_adapter_imports_and_aliases():
    from adapters.claude import ClaudeCodeAdapter
    from adapters.codex import CodexAdapter
    from adapters.open_code import OpenCodeAdapter
    from adapters.copilot import CopilotAdapter
    from adapters.gemini import GeminiAdapter

    spec = importlib.util.spec_from_file_location(
        "gen01", str(ROOT / "pipeline" / "01_generate.py"))
    gen01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen01)

    assert set(gen01.ADAPTERS) == {
        "copilot", "gemini", "claude", "codex", "opencode",
        "claude-opus-4-6-low", "claude-opus-4-8-low", "claude-opus-5-5-low",
    }
    for alias in ("claude-opus-4-6-low", "claude-opus-4-8-low",
                  "claude-opus-5-5-low"):
        cls, kwargs = gen01.ADAPTERS[alias]
        assert cls is ClaudeCodeAdapter and kwargs.get("effort") == "low"
    assert gen01.ADAPTERS["claude"][0] is ClaudeCodeAdapter
    assert gen01.ADAPTERS["codex"][0] is CodexAdapter
    assert gen01.ADAPTERS["opencode"][0] is OpenCodeAdapter
    assert gen01.ADAPTERS["copilot"][0] is CopilotAdapter
    assert gen01.ADAPTERS["gemini"][0] is GeminiAdapter
