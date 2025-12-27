"""Core agent loop built around propose-verify-select."""

from __future__ import annotations

from dataclasses import dataclass

from agentforge.controller import Controller
from agentforge.state import AgentState


@dataclass
class AgentResult:
    answer: str
    state: AgentState


class Agent:
    def __init__(self, controller: Controller) -> None:
        self.controller = controller

    def run(self, state: AgentState) -> AgentResult:
        answer = self.controller.run(state)
        return AgentResult(answer=answer, state=state)
