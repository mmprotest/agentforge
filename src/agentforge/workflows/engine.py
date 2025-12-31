"""Workflow execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from agentforge.models.base import BaseChatModel
from agentforge.runtime.runtime import Runtime
from agentforge.tools.base import Tool
from agentforge.tools.registry import ToolRegistry
from agentforge.workflows.spec import WorkflowSpec


class WorkflowError(RuntimeError):
    pass


@dataclass
class WorkflowResult:
    outputs: dict[str, Any]
    state: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_workflow_spec(path: str | Path) -> WorkflowSpec:
    path_obj = Path(path)
    if path_obj.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise WorkflowError("Install agentforge[yaml] to load YAML workflows.") from exc
        payload = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    else:
        payload = _load_json(path_obj)
    if not isinstance(payload, dict):
        raise WorkflowError("Workflow spec must be a JSON object.")
    return WorkflowSpec.from_dict(payload)


def _validate_schema(data: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    schema_type = schema.get("type")
    if schema_type:
        if schema_type == "object" and not isinstance(data, dict):
            errors.append(f"{path} should be object")
            return errors
        if schema_type == "number" and not isinstance(data, (int, float)):
            errors.append(f"{path} should be number")
        if schema_type == "integer" and not isinstance(data, int):
            errors.append(f"{path} should be integer")
        if schema_type == "string" and not isinstance(data, str):
            errors.append(f"{path} should be string")
    if isinstance(data, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                errors.append(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in data and isinstance(subschema, dict):
                errors.extend(_validate_schema(data[key], subschema, f"{path}.{key}"))
    return errors


def _render_template(value: Any, state: dict[str, Any]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            expr = match.group(1)
            parts = expr.split(".")
            current: Any = state
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return match.group(0)
            return str(current)
        return re.sub(r"\{([^}]+)\}", replace, value)
    if isinstance(value, dict):
        return {key: _render_template(val, state) for key, val in value.items()}
    if isinstance(value, list):
        return [_render_template(item, state) for item in value]
    return value


def _check_acceptance(
    output: Any, acceptance: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if acceptance.get("require_schema"):
        if not isinstance(output, dict):
            errors.append("Output must be object for schema validation")
    regex = acceptance.get("regex")
    if regex and isinstance(output, str):
        if not re.search(regex, output):
            errors.append("Output did not match regex")
    if acceptance.get("require_citations"):
        if not (isinstance(output, str) and "Sources:" in output):
            errors.append("Output missing required citations")
    return errors


class WorkflowEngine:
    def __init__(
        self,
        model: BaseChatModel,
        registry: ToolRegistry,
        runtime: Runtime | None = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.runtime = runtime

    def validate_spec(self, spec: WorkflowSpec) -> list[str]:
        errors: list[str] = []
        if not spec.steps:
            errors.append("Workflow must include steps")
        return errors

    def run(
        self,
        spec: WorkflowSpec,
        inputs: dict[str, Any],
        run_context: Any | None = None,
    ) -> WorkflowResult:
        errors = _validate_schema(inputs, spec.inputs_schema or {})
        if errors:
            raise WorkflowError(f"Input schema validation failed: {errors}")
        state: dict[str, Any] = {"input": inputs}
        run_context = run_context or (self.runtime.new_run_context() if self.runtime else None)
        model_calls = 0
        tool_calls = 0
        retries = 0
        schema_violations = 0
        started_at = datetime.now(timezone.utc)
        for step in spec.steps:
            attempts = 0
            last_error: str | None = None
            while attempts < max(1, step.retry.max_attempts):
                attempts += 1
                if self.runtime:
                    self.runtime.metrics.inc("workflow_steps", 1)
                    self.runtime.emit_audit(
                        run_context,
                        "workflow_step_start",
                        {"step_id": step.id, "kind": step.kind, "attempt": attempts},
                    )
                try:
                    output = self._execute_step(step, state)
                    if step.kind == "tool":
                        tool_calls += 1
                        if self.runtime:
                            self.runtime.metrics.inc("tool_calls", 1)
                    if step.kind == "llm":
                        model_calls += 1
                        if self.runtime:
                            self.runtime.metrics.inc("model_calls", 1)
                    acceptance_errors = _check_acceptance(
                        output,
                        {
                            "require_schema": step.acceptance.require_schema,
                            "require_citations": step.acceptance.require_citations,
                            "regex": step.acceptance.regex,
                        },
                    )
                    if acceptance_errors:
                        schema_violations += 1
                        raise WorkflowError(
                            f"Acceptance failed for step {step.id}: {acceptance_errors}"
                        )
                    state[step.outputs_key] = output
                    if self.runtime:
                        self.runtime.emit_audit(
                            run_context,
                            "workflow_step_complete",
                            {"step_id": step.id, "outputs_key": step.outputs_key},
                        )
                    break
                except Exception as exc:
                    last_error = str(exc)
                    retries += 1
                    if self.runtime:
                        self.runtime.metrics.inc("workflow_retries", 1)
                        self.runtime.emit_audit(
                            run_context,
                            "workflow_step_error",
                            {"step_id": step.id, "error": last_error},
                        )
                    if step.retry.on_fail == "retry" and attempts < step.retry.max_attempts:
                        continue
                    if step.retry.on_fail == "fallback":
                        break
                    raise
            if last_error and step.retry.on_fail == "fallback":
                state[step.outputs_key] = {"error": last_error}
        outputs = state.get(spec.steps[-1].outputs_key) if spec.steps else {}
        errors = _validate_schema(outputs, spec.outputs_schema or {})
        if errors:
            schema_violations += 1
            raise WorkflowError(f"Output schema validation failed: {errors}")
        if self.runtime and run_context:
            self.runtime.emit_audit(
                run_context,
                "workflow_complete",
                {"outputs_key": spec.steps[-1].outputs_key},
            )
            duration_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
            summary = {
                "run_id": run_context.run_id,
                "trace_id": run_context.trace_id,
                "duration_ms": duration_ms,
                "model_calls": model_calls,
                "tool_calls": tool_calls,
                "retries": retries,
                "schema_violations": schema_violations,
            }
            self.runtime.metrics.write_run_summary(summary)
        return WorkflowResult(outputs=outputs or {}, state=state)

    def _execute_step(self, step: Any, state: dict[str, Any]) -> Any:
        if step.kind == "tool":
            if not step.tool_name:
                raise WorkflowError("Tool step missing tool_name")
            tool = self.registry.get(step.tool_name)
            if not isinstance(tool, Tool):
                raise WorkflowError(f"Tool '{step.tool_name}' not found")
            rendered_args = _render_template(step.tool_args_template or {}, state)
            validated = tool.input_schema.model_validate(rendered_args)
            result = tool.run(validated)
            return result.output
        if step.kind == "llm":
            prompt = _render_template(step.llm_prompt_template or "", state)
            response = self.model.chat(
                [{"role": "user", "content": prompt}], tools=None
            )
            if response.final_text is None:
                raise WorkflowError("LLM step returned no text")
            return response.final_text
        if step.kind == "template":
            template = step.llm_prompt_template or ""
            return _render_template(template, state)
        raise WorkflowError(f"Unknown step kind: {step.kind}")
