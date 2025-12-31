# AI Regression Gate

This action runs an AgentForge eval pack, writes a report, and fails the job when scores regress.

## Baseline reports

Generate and commit a baseline report:

```bash
agentforge eval run --pack <pack-name> --workspace default --report evals/baseline_report.json
git add evals/baseline_report.json
git commit -m "Add eval baseline"
```

Alternatively, store the baseline as a workflow artifact and pass the path from a download step.

## Example: OpenAI-compatible local endpoint

```yaml
env:
  OPENAI_BASE_URL: http://localhost:11434/v1
  OPENAI_API_KEY: ""
  MODEL: llama3
  AGENTFORGE_HOME: /data
```

## GitHub Actions usage

```yaml
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

## Local usage

```bash
agentforge eval run --pack <pack-name> --workspace default --report candidate.json
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
