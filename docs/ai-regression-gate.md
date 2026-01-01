# AI Regression Gate

This action runs an AgentForge eval pack, writes a report, and fails the job when scores regress.

## Setup (baseline required)

1. **Merge the baseline workflow to `main`.**
2. **Run the baseline workflow once** (via `workflow_dispatch`) to produce the artifact.
3. **Open pull requests**: the gate action downloads the latest baseline artifact
   from the default branch using `GITHUB_TOKEN`.

Example baseline workflow (push to main):

```yaml
name: AI Baseline

on:
  push:
    branches: [main]

jobs:
  baseline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python -m pip install .
      - run: agentforge --workspace default eval run --pack sample --report baseline_report.json
        env:
          OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_MODEL: ${{ secrets.OPENAI_MODEL }}
      - uses: actions/upload-artifact@v4
        with:
          name: ai-baseline-report
          path: baseline_report.json
```

Example PR workflow:

```yaml
- name: AI Regression Gate
  uses: ./
  with:
    workspace: default
    eval_pack: sample
    report_out: agentforge_eval_report.json
    allow_regression: "false"
  env:
    OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    OPENAI_MODEL: ${{ secrets.OPENAI_MODEL }}
```

## Example: OpenAI-compatible local endpoint

```yaml
env:
  OPENAI_BASE_URL: http://localhost:11434/v1
  OPENAI_API_KEY: ""
  OPENAI_MODEL: llama3
```

## Local usage

```bash
agentforge --workspace default gate run --pack <pack-name> --baseline baseline.json --report candidate.json --min-score 0.6
```
