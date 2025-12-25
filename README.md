# AgentForge

AgentForge is a production-ready Python 3.11+ repository that implements an agent capable of using tools and creating tools with an OpenAI-compatible model backend.

## Features
- OpenAI-compatible REST client (`/chat/completions`) with configurable base URL.
- Deterministic controller that orchestrates microtasks, verification, and backtracking.
- Built-in tools: HTTP fetch, workspace filesystem, Python sandbox, deep thinking planner, calculator, regex extract, unit conversion, JSON repair, multi-file code runner.
- Tool registry and validation pipeline for safe tool creation.
- CLI and FastAPI server.
- Eval harness with scoring, traces, and replay support.

## Quickstart

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Run in mock mode (no API key)
```bash
python -m agentforge "Hello from mock mode"
```

### Benchmark-safe eval mode (answer-only)
```bash
python -m agentforge "2+2" --eval-mode
```
`--eval-mode` forces strict JSON contracts, verification, and prints only the final answer to stdout.

### Profiles (small-model friendly defaults)
Profiles tune budgets, routing thresholds, and verification defaults:
```bash
python -m agentforge "Write a script" --profile code
python -m agentforge "Solve 2+2" --profile math
```
Available profiles: `agent` (default), `code`, `math`, `qa`.

### Use an OpenAI-compatible base URL
```bash
export OPENAI_API_KEY=sk-your-key
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4.1-mini
python -m agentforge "Summarize the weather" --base-url https://api.openai.com/v1
```

#### Local OpenAI-compatible servers
The client accepts a base URL with or without `/v1`. If the base URL has no path, `/v1` is added automatically:
```bash
export OPENAI_BASE_URL=http://localhost:8000
python -m agentforge "hello"
```
You can also supply extra headers or override the path:
```bash
export OPENAI_EXTRA_HEADERS='{"X-Provider": "local"}'
export OPENAI_FORCE_CHATCOMPLETIONS_PATH=/v1/chat/completions
```

### Start the API
```bash
uvicorn agentforge.api:app --reload
```

## Configuration
Environment variables (CLI flags override env vars):
- `OPENAI_API_KEY` (optional; if missing, mock model is used)
- `OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- `OPENAI_MODEL` (default `gpt-4.1-mini`)
- `OPENAI_TIMEOUT_SECONDS` (default `30`)
- `OPENAI_EXTRA_HEADERS` (JSON string of headers)
- `OPENAI_DISABLE_TOOL_CHOICE` (default `false`)
- `OPENAI_FORCE_CHATCOMPLETIONS_PATH` (optional override)
- `AGENT_MODE` (default `direct`)
- `ALLOW_TOOL_CREATION` (default `false`)
- `WORKSPACE_DIR` (default `./workspace`)
- `MAX_TOOL_OUTPUT_CHARS` (default `4000`)
- `KEEP_RAW_TOOL_OUTPUT` (default `true`)
- `SUMMARY_LINES` (default `10`)
- `MAX_MODEL_CALLS` (default `20`)
- `MAX_MESSAGE_CHARS` (default `24000`)
- `MAX_SINGLE_MESSAGE_CHARS` (default `4000`)
- `MAX_MESSAGE_TOKENS_APPROX` (default `6000`)
- `TOKEN_CHAR_RATIO` (default `4`)
- `MAX_TURNS` (default `20`)
- `TRIM_STRATEGY` (default `drop_oldest`)
- `BRANCH_CANDIDATES` (default `1`)
- `STRICT_JSON_MODE` (default `false`)
- `CODE_CHECK` (default `false`)
- `CODE_CHECK_MAX_ITERS` (default `2`)
- `SANDBOX_PASSTHROUGH_ENV` (comma-separated allowlist of extra env vars)
- `EVAL_MODE` (default `false`)

Recommended small-model settings:
```bash
python -m agentforge "Question" --self-consistency 2 --verify --summary-lines 8 --profile qa
```

Additional CLI flags for small-model robustness:
```bash
python -m agentforge "Write code" --strict-json --code-check --max-message-chars 24000 --max-turns 20 --branch-candidates 2
```

Strict JSON mode enforces a single JSON object response and retries once on format errors.
Context trimming keeps recent requests and tool summaries within a hard budget, which helps smaller
local models stay focused. Single messages are hard-truncated to `MAX_SINGLE_MESSAGE_CHARS` before
budget trimming. Sandbox execution now sanitizes environment variables by default.

## Controller, microtasks, verification, backtracking
AgentForge now runs a deterministic controller that breaks each query into a small microtask graph.
The model proposes actions, but the controller decides when to route tools, verify outputs, and
backtrack after repeated failures or stalled progress. Verification runs local checks such as
schema validation, regex/predicate checks, tool recomputation, and code execution. Backtracking
rewinds the microtask graph and adds a brief hint for a new approach, which makes runs more robust
on smaller OpenAI-compatible models.

### Task graphs and checks
The planner emits a JSON task graph with strict schema validation:
- Top-level fields: `version`, `objective`, `tasks`, `final_task_id`
- Each task: `id`, `goal`, `inputs`, `tool_hint` (optional), `checks`, `max_attempts`, `budget_hint`
- Supported check types: `schema`, `regex`, `exact`, `numeric_tolerance`, `tool_recompute`, `unit_sanity`

If a plan fails validation (too many tasks, duplicate ids, cycles, unsupported checks), the controller
falls back to a safe default graph: extract constraints → compute/solve → format answer.

## Eval harness
Run the benchmark harness:
```bash
agentforge eval --dataset examples/eval_sample.jsonl --output results.jsonl --scorer exact --eval-mode
```
Each output record includes the input, prediction, expected value, score, failure tags, and trace path.

### Dataset format (JSONL)
Each line is a JSON object with:
- `id`: unique identifier
- `input` (or `query`): prompt text
- `expected`: expected answer (string/number)
- Optional fields for scorers: `pattern` (regex), `schema` (JSON schema)

## Tool creation gating
Tool creation is disabled by default. Enable it by setting `ALLOW_TOOL_CREATION=true` or using `--allow-tool-creation`. Generated tools are validated with AST-based checks and executed in a sandboxed test process before registration.

## Threat model
AgentForge assumes:
- Untrusted model output: tool code and arguments are validated and sandboxed.
- Tools are confined: filesystem access is limited to the workspace, Python sandbox blocks imports and enforces timeouts.
- Network exposure is explicit: only the `http_fetch` tool can perform outbound requests.
- `code_run_multi` executes code in a temporary workspace directory; do not provide secrets or untrusted system commands.
- Secrets are redacted in logs where possible.

## Development
Run tests:
```bash
pytest
```

Run lint:
```bash
ruff check .
```
