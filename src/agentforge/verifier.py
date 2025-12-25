"""Deterministic verification logic for model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Callable

from agentforge.tasks import CheckSpec, MicroTask


@dataclass
class FailureDetail:
    check_name: str
    reason: str
    expected: Any | None = None
    got: Any | None = None
    minimal_fix: str | None = None


@dataclass
class VerifierResult:
    passed: bool
    failures: list[FailureDetail]
    checks_run: list[str]


class Verifier:
    """Runs verification checks against a candidate output."""

    def __init__(self, tool_runner: Callable[[str, dict[str, Any]], tuple[Any, bool]]) -> None:
        self.tool_runner = tool_runner

    def verify(
        self, candidate_output: Any, task: MicroTask
    ) -> VerifierResult:
        failures, checks_run = self._evaluate_check(task.check, candidate_output, task)
        passed = not failures
        return VerifierResult(
            passed=passed,
            failures=failures,
            checks_run=checks_run,
        )

    def _evaluate_check(
        self, check: CheckSpec, candidate_output: Any, task: MicroTask
    ) -> tuple[list[FailureDetail], list[str]]:
        checks_run: list[str] = []
        failures: list[FailureDetail] = []
        if check.all_of:
            checks_run.append("all_of")
            for subcheck in check.all_of:
                sub_failures, sub_checks = self._evaluate_check(
                    subcheck, candidate_output, task
                )
                checks_run.extend(sub_checks)
                failures.extend(sub_failures)
            return failures, checks_run
        if check.any_of:
            checks_run.append("any_of")
            any_ok = False
            aggregated: list[FailureDetail] = []
            for subcheck in check.any_of:
                sub_failures, sub_checks = self._evaluate_check(
                    subcheck, candidate_output, task
                )
                checks_run.extend(sub_checks)
                if not sub_failures:
                    any_ok = True
                aggregated.extend(sub_failures)
            if any_ok:
                return [], checks_run
            return aggregated, checks_run
        if check.type == "none":
            return [], ["none"]
        if check.type == "schema":
            checks_run.append("schema")
            payload = self._load_json(candidate_output)
            if payload is None:
                failures.append(
                    FailureDetail(
                        check_name="schema",
                        reason="Output is not valid JSON",
                        expected="JSON object matching schema",
                        got=candidate_output,
                        minimal_fix="Return valid JSON that matches the required schema.",
                    )
                )
            else:
                schema = check.params.get("schema") or task.expected_schema or {}
                schema_issues = self._validate_schema(schema, payload)
                for issue in schema_issues:
                    failures.append(
                        FailureDetail(
                            check_name="schema",
                            reason=issue,
                            expected=schema,
                            got=payload,
                            minimal_fix="Ensure the JSON schema requirements are satisfied.",
                        )
                    )
        elif check.type == "regex":
            checks_run.append("regex")
            pattern = str(check.params.get("pattern") or "")
            if not pattern:
                failures.append(
                    FailureDetail(
                        check_name="regex",
                        reason="Missing regex pattern",
                        expected="pattern",
                        got=None,
                        minimal_fix="Provide a regex pattern to match.",
                    )
                )
            else:
                if not re.search(pattern, str(candidate_output), re.DOTALL):
                    failures.append(
                        FailureDetail(
                            check_name="regex",
                            reason=f"Output did not match /{pattern}/",
                            expected=pattern,
                            got=str(candidate_output),
                            minimal_fix=f"Ensure the output matches /{pattern}/.",
                        )
                    )
        elif check.type == "exact":
            checks_run.append("exact")
            expected = check.params.get("expected")
            if str(candidate_output) != str(expected):
                failures.append(
                    FailureDetail(
                        check_name="exact",
                        reason="Output did not match expected value",
                        expected=expected,
                        got=candidate_output,
                        minimal_fix=f"Return exactly: {expected}",
                    )
                )
        elif check.type == "numeric_tolerance":
            checks_run.append("numeric_tolerance")
            expected = check.params.get("expected")
            tolerance = float(check.params.get("tolerance", 1e-6))
            if not numeric_close(candidate_output, expected, rel=tolerance, abs=tolerance):
                failures.append(
                    FailureDetail(
                        check_name="numeric_tolerance",
                        reason="Output not within numeric tolerance",
                        expected=expected,
                        got=candidate_output,
                        minimal_fix=(
                            f"Return a number within ±{tolerance} of {expected}."
                        ),
                    )
                )
        elif check.type == "unit_sanity":
            checks_run.append("unit_sanity")
            unit = check.params.get("unit")
            if unit and str(unit) not in str(candidate_output):
                failures.append(
                    FailureDetail(
                        check_name="unit_sanity",
                        reason=f"Missing unit '{unit}'",
                        expected=unit,
                        got=str(candidate_output),
                        minimal_fix=f"Include the unit '{unit}' in the answer.",
                    )
                )
        elif check.type == "predicate":
            checks_run.append("predicate")
            issues = self._run_predicate(check, candidate_output)
            failures.extend(self._predicate_failures(check, issues, candidate_output))
        elif check.type == "contains_fields":
            checks_run.append("contains_fields")
            fields = check.params.get("required_fields") or []
            payload = self._load_json(candidate_output)
            if not isinstance(payload, dict):
                failures.append(
                    FailureDetail(
                        check_name="contains_fields",
                        reason="Output is not a JSON object",
                        expected=fields,
                        got=payload,
                        minimal_fix="Return a JSON object with required fields.",
                    )
                )
            else:
                for field in fields:
                    if field not in payload:
                        failures.append(
                            FailureDetail(
                                check_name="contains_fields",
                                reason=f"Missing field '{field}'",
                                expected=fields,
                                got=payload,
                                minimal_fix=f"Include field '{field}' in the JSON output.",
                            )
                        )
        elif check.type == "json_protocol":
            checks_run.append("json_protocol")
            required_keys = check.params.get("required_keys") or ["answer"]
            payload = self._load_json(candidate_output)
            if not isinstance(payload, dict):
                failures.append(
                    FailureDetail(
                        check_name="json_protocol",
                        reason="Output is not a JSON object",
                        expected=required_keys,
                        got=payload,
                        minimal_fix="Return a JSON object with the required keys.",
                    )
                )
            else:
                for field in required_keys:
                    if field not in payload:
                        failures.append(
                            FailureDetail(
                                check_name="json_protocol",
                                reason=f"Missing protocol key '{field}'",
                                expected=required_keys,
                                got=payload,
                                minimal_fix=f"Include key '{field}' in the JSON output.",
                            )
                        )
        elif check.type == "tool_error_absent":
            checks_run.append("tool_error_absent")
            tool_output = check.params.get("tool_output", candidate_output)
            if isinstance(tool_output, dict) and tool_output.get("ok") is False:
                failures.append(
                    FailureDetail(
                        check_name="tool_error_absent",
                        reason="Tool output indicates failure",
                        expected="Tool output ok",
                        got=tool_output,
                        minimal_fix="Fix tool arguments and rerun the tool.",
                    )
                )
        elif check.type == "tool_recompute":
            checks_run.append("tool_recompute")
            issues = self._run_tool_recompute(check, candidate_output)
            for issue in issues:
                failures.append(
                    FailureDetail(
                        check_name="tool_recompute",
                        reason=issue,
                        expected="Match tool output",
                        got=candidate_output,
                        minimal_fix="Match the tool recompute output exactly.",
                    )
                )
        elif check.type == "code_run":
            checks_run.append("code_run")
            issues = self._run_code_check(check, candidate_output)
            for issue in issues:
                failures.append(
                    FailureDetail(
                        check_name="code_run",
                        reason=issue,
                        expected="Python code executes without errors",
                        got=candidate_output,
                        minimal_fix="Fix the code so it runs without errors.",
                    )
                )
        else:
            failures.append(
                FailureDetail(
                    check_name=check.type,
                    reason=f"Unknown check type: {check.type}",
                    expected=None,
                    got=None,
                    minimal_fix="Update the check type to a supported value.",
                )
            )
        return failures, checks_run

    def _predicate_failures(
        self, check: CheckSpec, issues: list[str], candidate_output: Any
    ) -> list[FailureDetail]:
        failures: list[FailureDetail] = []
        name = str(check.params.get("name") or "non_empty")
        for issue in issues:
            minimal_fix = "Adjust the output to satisfy the predicate."
            if name == "non_empty":
                minimal_fix = "Return a non-empty response."
            elif name == "looks_numeric":
                minimal_fix = "Return a numeric value."
            elif name == "numeric_range":
                minimal_fix = "Return a value within the specified range."
            failures.append(
                FailureDetail(
                    check_name=f"predicate:{name}",
                    reason=issue,
                    expected=check.params,
                    got=candidate_output,
                    minimal_fix=minimal_fix,
                )
            )
        return failures

    def _load_json(self, candidate_output: Any) -> Any | None:
        if isinstance(candidate_output, (dict, list)):
            return candidate_output
        if not isinstance(candidate_output, str):
            return None
        try:
            return json.loads(candidate_output)
        except json.JSONDecodeError:
            return None

    def _validate_schema(self, schema: dict[str, Any], payload: Any, path: str = "$") -> list[str]:
        issues: list[str] = []
        expected_type = schema.get("type")
        if expected_type:
            type_ok = self._matches_type(payload, expected_type)
            if not type_ok:
                issues.append(f"{path} expected type {expected_type}")
                return issues
        if expected_type == "object" or isinstance(payload, dict):
            required = schema.get("required") or []
            for field in required:
                if field not in payload:
                    issues.append(f"{path} missing required field '{field}'")
            properties = schema.get("properties") or {}
            if isinstance(payload, dict):
                for field, field_schema in properties.items():
                    if field in payload and isinstance(field_schema, dict):
                        issues.extend(
                            self._validate_schema(
                                field_schema,
                                payload[field],
                                path=f"{path}.{field}",
                            )
                        )
        if expected_type == "array" and isinstance(payload, list):
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for idx, item in enumerate(payload):
                    issues.extend(
                        self._validate_schema(item_schema, item, path=f"{path}[{idx}]")
                    )
        return issues

    def _matches_type(self, payload: Any, expected_type: str) -> bool:
        if expected_type == "object":
            return isinstance(payload, dict)
        if expected_type == "array":
            return isinstance(payload, list)
        if expected_type == "string":
            return isinstance(payload, str)
        if expected_type == "number":
            return isinstance(payload, (int, float)) and not isinstance(payload, bool)
        if expected_type == "integer":
            return isinstance(payload, int) and not isinstance(payload, bool)
        if expected_type == "boolean":
            return isinstance(payload, bool)
        return False

    def _run_predicate(self, check: CheckSpec, candidate_output: Any) -> list[str]:
        issues: list[str] = []
        name = str(check.params.get("name") or "non_empty")
        if name == "non_empty":
            if not str(candidate_output).strip():
                issues.append("Output is empty")
        elif name == "contains_fields":
            fields = check.params.get("fields") or []
            payload = self._load_json(candidate_output)
            if not isinstance(payload, dict):
                issues.append("Output is not a JSON object")
            else:
                for field in fields:
                    if field not in payload:
                        issues.append(f"Missing field '{field}'")
        elif name == "numeric_range":
            minimum = check.params.get("min")
            maximum = check.params.get("max")
            try:
                value = float(candidate_output)
            except (TypeError, ValueError):
                issues.append("Output is not numeric")
            else:
                if minimum is not None and value < float(minimum):
                    issues.append("Output below minimum")
                if maximum is not None and value > float(maximum):
                    issues.append("Output above maximum")
        elif name == "looks_numeric":
            try:
                float(str(candidate_output).strip())
            except (TypeError, ValueError):
                issues.append("Output is not numeric")
        elif name == "unit_present":
            unit = check.params.get("unit")
            text = str(candidate_output)
            if unit and str(unit) not in text:
                issues.append(f"Missing unit '{unit}'")
        elif name == "mcq_choice":
            if not re.search(r"\b[A-E]\b", str(candidate_output), re.IGNORECASE):
                issues.append("Output is not a valid MCQ choice (A-E)")
        else:
            issues.append(f"Unknown predicate '{name}'")
        return issues

    def _run_tool_recompute(self, check: CheckSpec, candidate_output: Any) -> list[str]:
        tool_name = str(check.params.get("tool_name") or "")
        tool_args = check.params.get("tool_args")
        if not tool_name or not isinstance(tool_args, dict):
            return ["Missing tool recompute parameters"]
        output, ok = self.tool_runner(tool_name, tool_args)
        if not ok:
            return [f"Tool recompute failed for {tool_name}"]
        compare = str(check.params.get("compare") or "smart").lower()
        direction = str(check.params.get("subset_direction") or "expected_in_actual").lower()
        if compare == "exact":
            if candidate_output != output:
                return ["Tool recompute output did not match"]
            return []
        if compare == "numeric":
            return [] if numeric_close(candidate_output, output) else ["Tool recompute output did not match"]
        if compare == "text":
            return [] if normalize_text(candidate_output) == normalize_text(output) else [
                "Tool recompute output did not match"
            ]
        if compare == "subset":
            return [] if dict_subset_compare(candidate_output, output, direction) else [
                "Tool recompute output did not match"
            ]
        if numeric_close(candidate_output, output):
            return []
        if isinstance(candidate_output, str) and isinstance(output, str):
            if normalize_text(candidate_output) == normalize_text(output):
                return []
        if isinstance(candidate_output, dict) and isinstance(output, dict):
            if candidate_output == output:
                return []
        if str(candidate_output).strip() != str(output).strip():
            return ["Tool recompute output did not match"]
        return []

    def _run_code_check(self, check: CheckSpec, candidate_output: Any) -> list[str]:
        text = str(candidate_output)
        source_key = check.params.get("source_key")
        if source_key:
            text = str(check.params.get("source") or text)
        code_blocks = self._extract_python_blocks(text)
        if not code_blocks:
            return ["No Python code blocks to run"]
        if len(code_blocks) > 1:
            files = self._blocks_to_files(code_blocks)
            output, ok = self.tool_runner(
                "code_run_multi",
                {
                    "files": files,
                    "command": "python main.py" if "main.py" in files else "python snippet_0.py",
                    "timeout_seconds": 10,
                },
            )
        else:
            output, ok = self.tool_runner(
                "python_sandbox",
                {"code": code_blocks[0], "timeout_seconds": 2},
            )
        if not ok:
            return ["Code execution failed"]
        if isinstance(output, dict):
            error = output.get("error") or output.get("stderr")
            exit_code = output.get("exit_code")
            if error:
                return [str(error)]
            if exit_code not in (None, 0):
                return ["Non-zero exit code"]
        return []

    def _extract_python_blocks(self, text: str) -> list[str]:
        blocks: list[str] = []
        for match in re.finditer(
            r"```(?P<lang>[a-zA-Z0-9_-]*)\s*\n(?P<code>.*?)```",
            text,
            re.DOTALL,
        ):
            lang = (match.group("lang") or "").strip().lower()
            if lang and not lang.startswith("python"):
                continue
            blocks.append(match.group("code").strip())
        return blocks

    def _blocks_to_files(self, code_blocks: list[str]) -> dict[str, str]:
        files: dict[str, str] = {}
        for idx, block in enumerate(code_blocks):
            lines = block.splitlines()
            file_path = None
            if lines:
                match = re.match(r"#\s*file(name)?:\s*(\S+)", lines[0].strip(), re.IGNORECASE)
                if match:
                    file_path = match.group(2)
                    lines = lines[1:]
            if not file_path:
                file_path = f"snippet_{idx}.py"
            files[file_path] = "\n".join(lines).strip() + "\n"
        return files


def normalize_text(value: Any) -> str:
    text = str(value or "")
    return " ".join(text.strip().split())


def try_parse_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Fraction):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+/\d+", text):
        try:
            return float(Fraction(text))
        except (ZeroDivisionError, ValueError):
            return None
    try:
        return float(text)
    except ValueError:
        return None


def numeric_close(a: Any, b: Any, rel: float = 1e-6, abs: float = 1e-9) -> bool:
    first = try_parse_number(a)
    second = try_parse_number(b)
    if first is None or second is None:
        return False
    return math.isclose(first, second, rel_tol=rel, abs_tol=abs)


def dict_subset(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(value, dict) and isinstance(actual_value, dict):
            if not dict_subset(value, actual_value):
                return False
        else:
            if value != actual_value:
                return False
    return True


def dict_subset_compare(
    candidate_output: Any, output: Any, direction: str
) -> bool:
    if not isinstance(candidate_output, dict) or not isinstance(output, dict):
        return False
    if direction == "actual_in_expected":
        return dict_subset(output, candidate_output)
    return dict_subset(candidate_output, output)
