# AgentForge

AgentForge is a production-ready Python 3.11+ repository that implements an agent capable of using tools and creating tools with an OpenAI-compatible model backend.

## Features
- OpenAI-compatible REST client (`/chat/completions`) with configurable base URL.
- Built-in tools: HTTP fetch, workspace filesystem, Python sandbox, deep thinking planner.
- Tool registry and validation pipeline for safe tool creation.
- CLI and FastAPI server.

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

### Use an OpenAI-compatible base URL
```bash
export OPENAI_API_KEY=sk-your-key
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4.1-mini
python -m agentforge "Summarize the weather" --base-url https://api.openai.com/v1
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
- `AGENT_MODE` (default `direct`)
- `ALLOW_TOOL_CREATION` (default `false`)
- `WORKSPACE_DIR` (default `./workspace`)

## Tool creation gating
Tool creation is disabled by default. Enable it by setting `ALLOW_TOOL_CREATION=true` or using `--allow-tool-creation`. Generated tools are validated with AST-based checks and executed in a sandboxed test process before registration.

## Threat model
AgentForge assumes:
- Untrusted model output: tool code and arguments are validated and sandboxed.
- Tools are confined: filesystem access is limited to the workspace, Python sandbox blocks imports and enforces timeouts.
- Network exposure is explicit: only the `http_fetch` tool can perform outbound requests.
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
