# AI Regression Gate

This action runs an AgentForge eval pack, writes a report, and fails the job when scores regress.

## Quickstart (no baseline)

For forks or first-time setup, skip the baseline and only enforce a minimum score:

```yaml
- name: AI Regression Gate
  uses: ./.github/actions/ai-regression-gate
  with:
    workspace: default
    eval_pack: sample
    min_score: "0.0"
    report_out: agentforge_eval_report.json
    allow_regression: "false"
    fail_on_missing_baseline: "false"
```

## Real gating (with baseline)

### Recommended baseline strategy

1. **On push to main**: run the eval and upload the report as an artifact named
   `baseline-report`.
2. **On pull requests**: download the latest `baseline-report` artifact and pass
   it to the gate action via `baseline_report`.

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
      - run: pip install -e .
      - run: agentforge --workspace default eval run --pack sample --report evals/baseline_report.json
      - uses: actions/upload-artifact@v4
        with:
          name: baseline-report
          path: evals/baseline_report.json
```

Example PR workflow (download baseline):

```yaml
- uses: actions/download-artifact@v4
  with:
    name: baseline-report
    path: evals

- name: AI Regression Gate
  uses: ./.github/actions/ai-regression-gate
  with:
    workspace: default
    eval_pack: sample
    baseline_report: evals/baseline_report.json
    min_score: "0.6"
    report_out: agentforge_eval_report.json
    allow_regression: "false"
    fail_on_missing_baseline: "true"
```

## Example: OpenAI-compatible local endpoint

```yaml
env:
  OPENAI_BASE_URL: http://localhost:11434/v1
  OPENAI_API_KEY: ""
  MODEL: llama3
  AGENTFORGE_HOME: /data
```

## Local usage

```bash
agentforge --workspace default eval run --pack <pack-name> --report candidate.json
python - <<'PY'
import json
from agentforge.evals.gating import compare_reports, enforce_min_score

baseline = json.load(open("baseline.json"))
candidate = json.load(open("candidate.json"))
passed, summary = compare_reports(baseline, candidate, allow_regression=False)
min_passed, _ = enforce_min_score(candidate, 0.6)
print("pass", passed and min_passed, summary)
PY
```
