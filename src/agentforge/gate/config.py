"""Gate configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class GateConfigError(ValueError):
    """Raised when gate configuration is invalid."""


def _import_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in integration
        raise GateConfigError("Install agentforge[yaml] to use gate.yml.") from exc
    return yaml


def _ensure_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateConfigError(f"{context} must be a mapping.")
    return value


def _ensure_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GateConfigError(f"{context} must be a list.")
    return value


def _require_keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(data.keys()) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise GateConfigError(f"{context} has unknown fields: {names}.")


def _get_bool(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    raise GateConfigError(f"{context} must be a boolean.")


def _get_str(value: Any, context: str) -> str:
    if isinstance(value, str):
        return value
    raise GateConfigError(f"{context} must be a string.")


def _get_float(value: Any, context: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise GateConfigError(f"{context} must be a number.")


def _get_int(value: Any, context: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    raise GateConfigError(f"{context} must be an integer.")


@dataclass(frozen=True)
class EvalPackConfig:
    id: str
    path: str
    mode: str = "auto"


@dataclass(frozen=True)
class BaselineConfig:
    workflow: str = "ai-baseline"
    artifact_name: str = "ai-baseline-report"
    artifact_path: str = "baseline_report.json"


@dataclass(frozen=True)
class GateSettings:
    min_score: float = 0.0
    allow_regression: bool = False


@dataclass(frozen=True)
class OutputSettings:
    report_path: str = "agentforge_eval_report.json"
    junit_path: str = "agentforge_junit.xml"
    summary_max_failures: int = 20


@dataclass(frozen=True)
class PrivacySettings:
    redact_env_patterns: list[str] = field(
        default_factory=lambda: ["*KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*"]
    )
    never_print_payloads: bool = True


@dataclass(frozen=True)
class StrictSettings:
    require_baseline: bool = True
    no_fallbacks: bool = True


@dataclass(frozen=True)
class GateConfig:
    version: str
    eval_packs: list[EvalPackConfig]
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    gate: GateSettings = field(default_factory=GateSettings)
    outputs: OutputSettings = field(default_factory=OutputSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    strict: StrictSettings = field(default_factory=StrictSettings)


def _parse_eval_pack(item: Any) -> EvalPackConfig:
    data = _ensure_dict(item, "eval_packs entry")
    _require_keys(data, {"id", "path", "mode"}, "eval_packs entry")
    if "id" not in data:
        raise GateConfigError("eval_packs entry missing required id.")
    if "path" not in data:
        raise GateConfigError("eval_packs entry missing required path.")
    pack_id = _get_str(data["id"], "eval_packs.id")
    path = _get_str(data["path"], "eval_packs.path")
    mode = _get_str(data.get("mode", "auto"), "eval_packs.mode")
    if mode not in {"agent", "workflow", "auto"}:
        raise GateConfigError("eval_packs.mode must be agent, workflow, or auto.")
    return EvalPackConfig(id=pack_id, path=path, mode=mode)


def _parse_baseline(data: Any) -> BaselineConfig:
    if data is None:
        return BaselineConfig()
    payload = _ensure_dict(data, "baseline")
    _require_keys(payload, {"workflow", "artifact_name", "artifact_path"}, "baseline")
    return BaselineConfig(
        workflow=_get_str(payload.get("workflow", "ai-baseline"), "baseline.workflow"),
        artifact_name=_get_str(
            payload.get("artifact_name", "ai-baseline-report"),
            "baseline.artifact_name",
        ),
        artifact_path=_get_str(
            payload.get("artifact_path", "baseline_report.json"),
            "baseline.artifact_path",
        ),
    )


def _parse_gate(data: Any) -> GateSettings:
    if data is None:
        return GateSettings()
    payload = _ensure_dict(data, "gate")
    _require_keys(payload, {"min_score", "allow_regression"}, "gate")
    min_score = _get_float(payload.get("min_score", 0.0), "gate.min_score")
    if not 0.0 <= min_score <= 1.0:
        raise GateConfigError("gate.min_score must be between 0 and 1.")
    allow_regression = _get_bool(
        payload.get("allow_regression", False), "gate.allow_regression"
    )
    return GateSettings(min_score=min_score, allow_regression=allow_regression)


def _parse_outputs(data: Any) -> OutputSettings:
    if data is None:
        return OutputSettings()
    payload = _ensure_dict(data, "outputs")
    _require_keys(payload, {"report_path", "junit_path", "summary_max_failures"}, "outputs")
    summary_max = _get_int(
        payload.get("summary_max_failures", 20), "outputs.summary_max_failures"
    )
    if summary_max < 1:
        raise GateConfigError("outputs.summary_max_failures must be at least 1.")
    return OutputSettings(
        report_path=_get_str(
            payload.get("report_path", "agentforge_eval_report.json"),
            "outputs.report_path",
        ),
        junit_path=_get_str(
            payload.get("junit_path", "agentforge_junit.xml"), "outputs.junit_path"
        ),
        summary_max_failures=summary_max,
    )


def _parse_privacy(data: Any) -> PrivacySettings:
    if data is None:
        settings = PrivacySettings()
    else:
        payload = _ensure_dict(data, "privacy")
        _require_keys(payload, {"redact_env_patterns", "never_print_payloads"}, "privacy")
        patterns = payload.get(
            "redact_env_patterns", ["*KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*"]
        )
        patterns_list = _ensure_list(patterns, "privacy.redact_env_patterns")
        patterns_value = [_get_str(item, "privacy.redact_env_patterns entry") for item in patterns_list]
        settings = PrivacySettings(
            redact_env_patterns=patterns_value,
            never_print_payloads=_get_bool(
                payload.get("never_print_payloads", True), "privacy.never_print_payloads"
            ),
        )
    if settings.never_print_payloads is not True:
        raise GateConfigError("privacy.never_print_payloads must be true.")
    return settings


def _parse_strict(data: Any) -> StrictSettings:
    if data is None:
        settings = StrictSettings()
    else:
        payload = _ensure_dict(data, "strict")
        _require_keys(payload, {"require_baseline", "no_fallbacks"}, "strict")
        settings = StrictSettings(
            require_baseline=_get_bool(
                payload.get("require_baseline", True), "strict.require_baseline"
            ),
            no_fallbacks=_get_bool(payload.get("no_fallbacks", True), "strict.no_fallbacks"),
        )
    if settings.require_baseline is not True:
        raise GateConfigError("strict.require_baseline must be true.")
    if settings.no_fallbacks is not True:
        raise GateConfigError("strict.no_fallbacks must be true.")
    return settings


def load_gate_config(path: Path) -> GateConfig:
    if not path.exists():
        raise GateConfigError(f"gate config not found at {path}.")
    yaml = _import_yaml()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        raise GateConfigError("gate config is empty.")
    payload = _ensure_dict(data, "gate config")
    _require_keys(
        payload,
        {"version", "eval_packs", "baseline", "gate", "outputs", "privacy", "strict"},
        "gate config",
    )
    if "version" not in payload:
        raise GateConfigError("gate config missing required version.")
    version = _get_str(payload["version"], "version")
    if version != "v1":
        raise GateConfigError(f"Unsupported gate config version {version!r}.")
    eval_packs_raw = payload.get("eval_packs")
    if eval_packs_raw is None:
        raise GateConfigError("gate config missing required eval_packs.")
    eval_list = _ensure_list(eval_packs_raw, "eval_packs")
    eval_packs = [_parse_eval_pack(item) for item in eval_list]
    if not eval_packs:
        raise GateConfigError("eval_packs must include at least one entry.")
    return GateConfig(
        version=version,
        eval_packs=eval_packs,
        baseline=_parse_baseline(payload.get("baseline")),
        gate=_parse_gate(payload.get("gate")),
        outputs=_parse_outputs(payload.get("outputs")),
        privacy=_parse_privacy(payload.get("privacy")),
        strict=_parse_strict(payload.get("strict")),
    )
