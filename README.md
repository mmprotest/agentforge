# AgentForge

AgentForge is a Python 3.11+ agent framework that emphasizes deterministic search, verification, and explicit control loops for small language models.

## Features
- Typed agent state (no dict-based memory).
- Explicit propose → verify → select loop with deterministic selection.
- Best-of-N proposals with verifier-first scoring and explicit backtracking.
- Minimal benchmark harness for reproducible runs.
- CLI and FastAPI API built from a shared factory.

## Quickstart

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Run in mock mode (no API key)
```bash
python -m agentforge "Summarize 2+2"
```

### Use an OpenAI-compatible base URL
```bash
export OPENAI_API_KEY=sk-your-key
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4.1-mini
python -m agentforge "Summarize the weather" --base-url https://api.openai.com/v1
```

## Propose → verify → select
AgentForge generates N proposals (default 3) and runs each through verifiers. Failing proposals are strictly dominated. Passing proposals are selected deterministically, preferring ones that satisfy optional constraints or advance the current plan.

### Constraints
Constraints are strings with simple prefixes:
- `must:...` or plain text: required substring
- `must_not:...` / `avoid:...`: forbidden substring
- `prefer:...`: optional preference used for selection

### Format verification
If `require_json` or `output_schema` is provided, proposals must be valid JSON. `output_schema` may include `required` or `required_fields`.

## Backtracking
If all proposals fail, the controller explicitly backtracks one level, resets the branch ID, and retries with a fresh proposal set.

## Benchmark harness
Run a single-task benchmark:
```bash
python -m agentforge.benchmarks.runner --task-file examples/benchmark_simple.json --output results.json
```
The output JSON includes proposals, verifier results, and the selected branch.

## Configuration
Environment variables (CLI flags override env vars):
- `OPENAI_API_KEY` (optional; if missing, mock model is used)
- `OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- `OPENAI_MODEL` (default `gpt-4.1-mini`)
- `OPENAI_TIMEOUT_SECONDS` (default `30`)
- `OPENAI_EXTRA_HEADERS` (JSON string of headers)
- `OPENAI_DISABLE_TOOL_CHOICE` (default `false`)
- `OPENAI_FORCE_CHATCOMPLETIONS_PATH` (optional override)
- `PROPOSAL_COUNT` (default `3`)
- `MAX_BACKTRACKS` (default `1`)
- `MAX_ATTEMPTS` (default `10`)

## Development
Run tests:
```bash
pytest
```

Run lint:
```bash
ruff check .
```
