# AI Regression Gate

This action runs an AgentForge eval pack, writes a report, and fails the job when scores regress.

## Quickstart (no baseline)

For forks or first-time setup, skip the baseline and only enforce a minimum score:

```yaml
- name: AI Regression Gate
  uses: ./
  with:
    workspace: default
    eval_pack: sample
    baseline_strategy: none
    min_score: "0.0"
    report_out: agentforge_eval_report.json
    allow_regression: "false"
    fail_on_missing_baseline: "false"
```

## Real gating (with baseline)

### Recommended baseline strategy

1. **On push to main**: run the eval and upload the report as an artifact named
   `ai-baseline-report`.
2. **On pull requests**: the gate action downloads the latest baseline artifact
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
      - uses: ./
        with:
          workspace: default
          eval_pack: sample
          baseline_strategy: none
          report_out: baseline_report.json
        env:
          OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          MODEL: ${{ secrets.OPENAI_MODEL }}
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
    baseline_strategy: artifact
    min_score: "0.6"
    report_out: agentforge_eval_report.json
    allow_regression: "false"
    fail_on_missing_baseline: "true"
  env:
    OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    MODEL: ${{ secrets.OPENAI_MODEL }}
```

## Example: OpenAI-compatible local endpoint

```yaml
env:
  OPENAI_BASE_URL: http://localhost:11434/v1
  OPENAI_API_KEY: ""
  MODEL: llama3
```

## Local usage

```bash
agentforge --workspace default gate run --pack <pack-name> --baseline baseline.json --report candidate.json --min-score 0.6
```
