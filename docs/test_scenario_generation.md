# Test Scenario Generation

The pipeline's behavioral checks (stage 07) and trace checks (stage 08) rely on two test files that live inside each specification folder:

- `test_deepeval.py` — black-box behavioral tests consumed by stage 07 via DeepEval metrics.
- `test_trace.py` — white-box trajectory tests consumed by stage 08 to verify tool-call order and reachable states.

Both files are produced from the same `spec.md` by the scripts in `test_generator/`.

## Black-box generator (DeepEval)

Reads the agent's observable contract (inputs, outputs, tools, constraints) from `spec.md` and emits DeepEval test cases.

```bash
python3 test_generator/blackbox_generator.py specs/001_list_issues_agent/spec.md
```

Output: `specs/001_list_issues_agent/test_deepeval.py`.

Run manually with:

```bash
pytest specs/001_list_issues_agent/test_deepeval.py -vv
```

## White-box generator (trace)

Reads the agent's expected trajectory from `spec.md` (nodes, tool calls, branch conditions) and emits trace assertions.

```bash
python3 test_generator/whitebox_generator.py specs/001_list_issues_agent/spec.md
```

Output: `specs/001_list_issues_agent/test_trace.py`.

Run manually with:

```bash
pytest specs/001_list_issues_agent/test_trace.py -vv
```

## When to regenerate

- Whenever `spec.md` changes in a way that affects contract or trajectory.
- Before adding a new agent to `pipe.sh` — stages 07 and 08 fail if either file is missing.

Both generators always overwrite the target file in the same folder as the spec, so regenerating is safe.
