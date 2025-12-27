"""Controller implementing propose -> verify -> select with backtracking."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentforge.models.base import BaseChatModel
from agentforge.state import AgentState, Proposal, VerifierResult
from agentforge.verifier import Verifier


@dataclass
class ProposalAssessment:
    proposal: Proposal
    results: list[VerifierResult]

    @property
    def passed(self) -> bool:
        return all(result.pass_fail for result in self.results)


class Controller:
    def __init__(
        self,
        model: BaseChatModel,
        verifiers: list[Verifier],
        proposal_count: int = 3,
        max_backtracks: int = 1,
        max_attempts: int = 10,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.verifiers = verifiers
        self.proposal_count = max(1, proposal_count)
        self.max_backtracks = max(0, max_backtracks)
        self.max_attempts = max(1, max_attempts)
        self.seed = seed

    def run(self, state: AgentState) -> str:
        backtracks_remaining = self.max_backtracks
        while state.attempts < self.max_attempts:
            snapshot = state.model_copy(deep=True)
            assessments = self._assess_proposals(state)
            state.artifacts["proposals"] = [
                {
                    "action": item.proposal.action,
                    "rationale": item.proposal.rationale,
                    "results": [result.model_dump() for result in item.results],
                }
                for item in assessments
            ]
            state.verifier_results.extend(
                result for item in assessments for result in item.results
            )
            ordered = self._rank_assessments(assessments, state)
            selected = None
            for item in ordered:
                if not item.passed:
                    state.attempts += 1
                    continue
                selected = item.proposal
                break
            if selected is not None:
                state.history.append(selected.action)
                state.artifacts["selected_branch"] = state.branch_id
                state.artifacts["selected_action"] = selected.action
                return selected.action
            if backtracks_remaining <= 0:
                state.artifacts["selected_branch"] = state.branch_id
                state.artifacts["selected_action"] = ""
                return ""
            self._restore_state(state, snapshot)
            backtracks_remaining -= 1
            state.branch_id = f"branch-{self.max_backtracks - backtracks_remaining}"
            state.history.append("backtrack")
            state.artifacts["backtracks"] = self.max_backtracks - backtracks_remaining
        return ""

    def _restore_state(self, target: AgentState, snapshot: AgentState) -> None:
        preserved_attempts = target.attempts
        preserved_results = list(target.verifier_results)
        for field in AgentState.model_fields:
            setattr(target, field, getattr(snapshot, field))
        target.attempts = preserved_attempts
        target.verifier_results = preserved_results

    def _assess_proposals(self, state: AgentState) -> list[ProposalAssessment]:
        proposals = self._propose(state)
        assessments: list[ProposalAssessment] = []
        for proposal in proposals:
            results = [verifier.verify(proposal, state) for verifier in self.verifiers]
            assessments.append(ProposalAssessment(proposal=proposal, results=results))
        return assessments

    def _propose(self, state: AgentState) -> list[Proposal]:
        proposals: list[Proposal] = []
        for index in range(self.proposal_count):
            prompt = self._proposal_prompt(state, index)
            response = self.model.chat([{"role": "user", "content": prompt}], tools=None)
            raw = response.final_text or ""
            proposals.append(self._parse_proposal(raw, index))
        return proposals

    def _proposal_prompt(self, state: AgentState, index: int) -> str:
        constraints = "\n".join(state.constraints) if state.constraints else "none"
        plan = state.current_plan or "none"
        history = "\n".join(state.history) if state.history else "none"
        payload = {
            "task": state.task,
            "constraints": constraints,
            "plan": plan,
            "history": history,
            "proposal_index": index,
            "seed": self.seed,
            "instruction": "Respond with JSON: {\"action\": str, \"rationale\": str}.",
        }
        return json.dumps(payload, ensure_ascii=False)

    def _parse_proposal(self, text: str, index: int) -> Proposal:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return Proposal(action=text, rationale="model output not JSON")
        action = data.get("action") if isinstance(data, dict) else None
        rationale = data.get("rationale") if isinstance(data, dict) else None
        if not action:
            action = text
        if not rationale:
            rationale = f"proposal {index}"
        return Proposal(action=str(action), rationale=str(rationale))

    def _rank_assessments(
        self, assessments: list[ProposalAssessment], state: AgentState
    ) -> list[ProposalAssessment]:
        scored = [
            (self._score(item.proposal, state), idx, item)
            for idx, item in enumerate(assessments)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored]

    def _score(self, proposal: Proposal, state: AgentState) -> int:
        score = 0
        score += self._preference_score(proposal, state)
        score += self._plan_score(proposal, state)
        return score

    def _preference_score(self, proposal: Proposal, state: AgentState) -> int:
        score = 0
        for constraint in state.constraints:
            rule, detail = _split_constraint(constraint)
            if rule == "prefer" and detail and detail in proposal.action:
                score += 1
        return score

    def _plan_score(self, proposal: Proposal, state: AgentState) -> int:
        if not state.current_plan:
            return 0
        steps = [step.strip() for step in state.current_plan.split(";") if step.strip()]
        completed = set()
        for entry in state.history:
            for step in steps:
                if step and step in entry:
                    completed.add(step)
        for step in steps:
            if step not in completed and step in proposal.action:
                return 1
        return 0


def _split_constraint(constraint: str) -> tuple[str, str]:
    if ":" in constraint:
        prefix, detail = constraint.split(":", 1)
        return prefix.strip(), detail.strip()
    return "must", constraint.strip()
