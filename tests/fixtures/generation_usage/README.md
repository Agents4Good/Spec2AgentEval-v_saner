# Synthetic fixtures (NOT real CLI captures)

All files in this directory are **synthetic fixtures hand-derived from the
shapes observed in `pipeline/utils/utils.py` and `pipeline/adapters/*.py`**
at baseline `30b4018`. They are NOT captured from real authenticated CLI
runs, carry no sessions, prompts, credentials or billing data, and must never
be presented as runtime evidence.

Shape reference (closed external semantics, source-grounded):

- Claude: single JSON `result` object with `usage.{input_tokens,
  output_tokens, cache_read_input_tokens, cache_creation_input_tokens
  (aggregate), cache_creation.{ephemeral_5m,ephemeral_1h}_input_tokens
  (detail, never re-added)}`, `num_turns`, `total_cost_usd` (client-side
  estimate), `duration_api_ms` / `duration_ms` (provenance only), `modelUsage`
  numerics, `session_id`. `input_tokens` excludes both cache buckets.
- Codex: JSONL with `thread.started {thread_id}`, then per turn
  `turn.started {}`, `item.*`, and `turn.completed {usage:
  {input_tokens, cached_input_tokens (subset), cache_write_input_tokens
  (absent means 0 per pinned SDK), output_tokens,
  reasoning_output_tokens (subset)}}`. Usage is thread-cumulative: last
  snapshot wins, never summed. No total/model/cost/duration in the envelope.
- OpenCode: JSONL envelopes `{type, timestamp, sessionID, ...}` with
  `step_finish {part: {id (prt_), messageID (msg_), sessionID (ses_),
  tokens: {input (cache-exclusive), output (reasoning-exclusive),
  reasoning, cache: {read, write}, total (optional native passthrough)},
  cost (pricing-table estimate)}}`. Identity is the scoped (session,
  message, part) triple with exact typed values.
- Pinned versions: Claude 2.1.299, Codex 0.162.1, OpenCode 1.19.0. Local CLI
  versions are unknown (fixtures are synthetic, not runtime evidence).

The `_synthetic` marker inside JSON files (and `SYNTHETIC` placeholders in
text) labels them as synthetic. Sentinel strings such as
`SENTINEL_SECRET_ABC123` exist to prove parser warnings never leak payload
content.
