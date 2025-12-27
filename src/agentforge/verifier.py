"""Verifier interfaces and implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from agentforge.state import AgentState, Proposal, VerifierResult


class Verifier(ABC):
    @abstractmethod
    def verify(self, proposal: Proposal, state: AgentState) -> VerifierResult:
        raise NotImplementedError


class FormatVerifier(Verifier):
    def verify(self, proposal: Proposal, state: AgentState) -> VerifierResult:
        require_json = bool(state.artifacts.get("require_json"))
        schema = state.artifacts.get("output_schema")
        if not require_json and not schema:
            return VerifierResult(
                pass_fail=bool(proposal.action.strip()),
                reason="non-empty action",
                verifier=self.__class__.__name__,
            )
        try:
            payload = json.loads(proposal.action)
        except json.JSONDecodeError as exc:
            return VerifierResult(
                pass_fail=False,
                reason=f"invalid JSON: {exc}",
                verifier=self.__class__.__name__,
            )
        if schema:
            required = schema.get("required") if isinstance(schema, dict) else None
            if required is None and isinstance(schema, dict):
                required = schema.get("required_fields", [])
            if required:
                if not isinstance(payload, dict):
                    return VerifierResult(
                        pass_fail=False,
                        reason="schema requires JSON object",
                        verifier=self.__class__.__name__,
                    )
                missing = [field for field in required if field not in payload]
                if missing:
                    return VerifierResult(
                        pass_fail=False,
                        reason=f"missing fields: {', '.join(missing)}",
                        verifier=self.__class__.__name__,
                    )
        return VerifierResult(
            pass_fail=True,
            reason="format ok",
            verifier=self.__class__.__name__,
        )


class ConstraintVerifier(Verifier):
    def verify(self, proposal: Proposal, state: AgentState) -> VerifierResult:
        failures: list[str] = []
        for constraint in state.constraints:
            rule, detail = _split_constraint(constraint)
            action_text = proposal.action
            if rule in {"must", "include"}:
                if detail not in action_text:
                    failures.append(f"missing '{detail}'")
            elif rule in {"must_not", "avoid"}:
                if detail and detail in action_text:
                    failures.append(f"contains forbidden '{detail}'")
        if failures:
            return VerifierResult(
                pass_fail=False,
                reason="; ".join(failures),
                verifier=self.__class__.__name__,
            )
        return VerifierResult(
            pass_fail=True,
            reason="constraints satisfied",
            verifier=self.__class__.__name__,
        )


def _split_constraint(constraint: str) -> tuple[str, str]:
    if ":" in constraint:
        prefix, detail = constraint.split(":", 1)
        return prefix.strip(), detail.strip()
    return "must", constraint.strip()
