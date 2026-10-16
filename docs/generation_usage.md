# Generation usage (phase 1: numeric parser contract)

Parses Claude / Codex / OpenCode CLI stdout into shared numeric usage with
explicit status. Source: `pipeline/utils/generation_usage.py`, entrypoint
`parse_generation_usage(cli, stdout, model=None, cli_version=None)` returning
`{"usage", "parse_status", "warnings"}`.

## Pins and sources

Pinned versions come from `docker/Dockerfile.gen`: Claude 2.1.299, Codex
0.162.1, OpenCode 1.19.0. External semantics are closed from pinned source;
all examples below are synthetic source-grounded shapes, NOT runtime
captures (no real CLI fixture exists yet).

- Codex event schema and cumulative proof:
  `openai/codex@rust-v0.162.1` — `codex-rs/exec/src/exec_events.rs`
  (`ThreadEvent`, `TurnCompletedEvent {usage}`), `codex-rs/exec/src/
  event_processor_with_jsonl_output.rs` (`usage_from_last_total` maps
  `ThreadTokenUsage.total`, not `last`), `codex-rs/exec/src/cli.rs`
  (`--json`), `codex-rs/protocol/src/protocol.rs` (`TokenUsageInfo`),
  `sdk/typescript/src/thread.ts` (`cache_write ??= 0`).
- OpenCode run envelope, step emission and usage derivation:
  `anomalyco/opencode@v1.19.0` — `packages/opencode/src/cli/cmd/run.ts`
  (`--format json`, envelope `{type, timestamp, sessionID, ...data}`),
  `packages/opencode/src/session/processor.ts` (step-finish case),
  `packages/opencode/src/session/session.ts` (`Session.getUsage`),
  `packages/schema/src/v1/session.ts` (`StepFinishPart`, `prt_`/`msg_`),
  `packages/schema/src/session-id.ts` (`ses_`).
- Claude result shape (no public CLI source; official docs retrieved
  2026-10-16, gates ≤ v2.1.246 hold for 2.1.299): `code.claude.com/docs/en/`
  `headless.md` (json vs stream-json), `agent-sdk/agent-loop` (subtypes carry
  `total_cost_usd`/`usage`/`num_turns`/`session_id`), `agent-sdk/cost-tracking`
  (client-side estimate incl. subagents), `statusline`
  (`total_input_tokens = input + cache_creation + cache_read`), `prompt-caching`
  (per-TTL `cache_creation.{ephemeral_5m,ephemeral_1h}_input_tokens` detail).

## Schema

`usage` fields (all default null except `cost_source` / `source` /
`requested_model` / `raw_fields`):

| Field | Type | Rule |
| --- | --- | --- |
| `input_tokens` | int ≥ 0 or null | Inclusive of cache, complete components only |
| `output_tokens` | int ≥ 0 or null | Native per CLI (OpenCode excludes reasoning) |
| `cache_read_tokens` | int ≥ 0 or null | Subset of input; never re-added |
| `cache_write_tokens` | int ≥ 0 or null | Subset/component; never re-added |
| `reasoning_tokens` | int ≥ 0 or null | Only when reported (never Claude) |
| `total_tokens` | int ≥ 0 or null | Claude/Codex: input+output when complete; OpenCode: native passthrough only, never derived |
| `requests` | int ≥ 0 or null | Turns (Claude/Codex) / unique steps (OpenCode), not HTTP calls |
| `cost_usd` | float ≥ 0 or null | Only when reported; zero ≠ unknown |
| `cost_source` | `estimated` / `unavailable` | Both CLIs report estimates, never billed cost |
| `cli_duration_s` | null | No trusted CLI-duration semantic exists in phase 1 |
| `process_duration_s` | null | Parser level; harness owns timing (phase 2) |
| `wall_duration_s` | null | Parser level; harness owns timing (phase 2) |
| `source` | `{cli, cli_version, provider}` | `provider` always null: never guessed |
| `requested_model` | str or null | Requested identity echoed; never read from events |
| `raw_fields` | object | Allowlisted observed metrics + provenance (never sessions, prompts, text, credentials); holds `input_incomplete_sum` when the native OpenCode input is known but the inclusive total is not |

## Per-CLI semantics

- **Claude** (single `result` object): normalized input = `input_tokens` +
  `cache_read_input_tokens` + `cache_creation_input_tokens` (aggregate) iff
  all three known, else null. Per-TTL detail
  (`cache_creation.ephemeral_5m/1h`) and `modelUsage` numerics are kept in
  `raw_fields` under allowlist, never re-added, with no whole-tree
  normalization assumed. `total_cost_usd` → `cost_source "estimated"`.
  `duration_api_ms` is API wait and `duration_ms` is unverified: both retained
  raw, `cli_duration_s` stays null. Scope: `usage` excludes subagents while
  cost/`modelUsage` may include them — the two are not the same universe.
- **Codex** (JSONL with `thread.started {thread_id}`): usage is
  thread-cumulative — the LAST `turn.completed` of a single unambiguous
  thread is the total; earlier snapshots never backfill it. `cached ⊆ input`,
  `reasoning ⊆ output`; total = input + output when the snapshot is complete.
  `cache_write_input_tokens` absent means 0 (pinned SDK backing; the only
  absence-as-zero in the contract) — applied only to an absent key in valid
  usage and labelled `cache_write_defaulted` in provenance, never recorded
  as an observed measurement; missing/invalid usage or explicit null stays
  unknown with diagnosis. More than one thread, or a thread id
  mismatch, is `error`. Turn count is independent; identical turns are never
  collapsed. No cost/model/duration in the envelope.
- **OpenCode** (`step_finish` parts): sums with per-field completeness — a
  field total exists only when EVERY selected step carries it valid, else
  null with counts. Inclusive input = raw input + read + write, all complete.
  Output stays native reasoning-exclusive (reasoning preserved separately;
  comparison with reasoning-inclusive series is limited). `tokens.total` is
  optional native passthrough with full coverage, never `input+output`.
  `part.cost` sums to `cost_source "estimated"` with full coverage; finite
  aggregates only (an overflowing sum is null, never Infinity, and every
  result serializes under `json.dumps(allow_nan=False)`). Identity
  is event-local `(sessionID, messageID, part.id)` triple with exact typed
  values: the session must be present on that event itself and is never
  borrowed from the stream; compatible retransmits merge (later completions
  fill gaps deliberately), conflicting metrics are `error`, id-less steps
  are never collapsed, and multiple sessions without proven relation are
  `error` (envelope/part mismatches included).

## Strictness, null, status

- Null = unknown; zero preserved (incl. `cost_usd: 0` → `estimated`).
- Invalid numerics (bool, negative, non-finite, wrong type, unrepresentable
  huge) become null with a field-named warning, never raising, never leaking
  the value.
- Any malformed non-blank JSONL line is `parse_status "error"` (strict, no
  tolerant mode). Recovered observations survive in `raw_fields` counts only;
  normalized totals stay null.
- `parse_status`: `ok` (complete), `partial` (valid syntax, nulls with
  diagnosis), `error` (no usable normalized data).
- `warnings` carry field names, counts and step indexes only.
- Missing relevant events never vanish: uncovered steps break completeness
  instead of disappearing.

## Timing

No CLI duration is trusted in phase 1 (`cli_duration_s` always null):
Claude API-wait is not CLI time, Codex/OpenCode carry none. Adapter wall
measurements continue through the legacy `time_spent` string. Uniform
monotonic process/wall measurement is phase-2 work.

## Compatibility (phase 1)

- `claude_parse_stats` / `codex_parse_stats` / `opencode_parse_stats` keep
  signatures and legacy keys as thin display wrappers (Codex/OpenCode share
  one render helper; no blanket exception catch — parser defects fail loud).
- `copilot_parse_stats` / `gemini_parse_stats` byte-identical; numeric parser
  returns `error` for those CLIs.
- Docker producer, `01_generate.py`, shell, MVP tests untouched.

## Limitations

- Unknown local CLI versions: pins describe the Docker image; local runs
  record `cli_version` only when the caller provides it.
- Fixtures are synthetic, never real captures; no runtime/billing/throughput
  claims follow from them.
- Claude `stream-json` (multi-line) is rejected (single `result` object only).
- Claude resume/cumulative runs need per-run deltas outside the parser.
- Codex `turn.failed` / standalone `error` events are observed in counts but
  carry no usage contract here.
- OpenCode `reason` vocabulary, `snapshot` and provider-dependent `total`
  presence are unconsumed; `tokens.cost` fallback was removed for lack of
  evidence.
- Cost `estimated` taxonomy and usage-vs-cost scope mismatch are reconciled
  by the orchestrator at the report layer.

## Phase-2 harness notes (stage 01)

- `process_duration_s` / `wall_duration_s` are harness-measured around the
  full spawn-to-exit window and mirrored into the normalized usage dict;
  `cli_duration_s` stays null (no trusted CLI-duration semantic) and
  `source.cli_version` is recorded only when a caller provides it — stage
  01 never shells out to probe CLI versions.
- `parse_status "error"` is an operational failure for the canonical
  `claude` / `codex` / `opencode` adapters (nonzero exit, spawn failure and
  invalid artifacts fail the same way); `partial` only warns and keeps
  `status "ok"`. Copilot/Gemini legacy shapes bypass the numeric parser
  entirely, so missing numeric parse status alone does not fail their jobs.
- The report carries stable safe diagnostics only (`cli_exit_<N>`,
  `parse_error`, `spawn_failed: <prog>`, artifact tokens); raw CLI
  stdout/stderr are stored in the per-agent log files referenced by
  `log_paths`. Prompts and exception text are not included in the report.
