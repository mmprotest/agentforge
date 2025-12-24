"""Deep thinking planner tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult


class DeepThinkInput(BaseModel):
    problem: str
    constraints: list[str] = Field(default_factory=list)


class DeepThinkOutput(BaseModel):
    plan: list[str]
    checks: list[str]


class DeepThinkTool(Tool):
    name = "deep_think"
    description = "Generate a structured plan and checks for a problem."
    input_schema = DeepThinkInput
    output_schema = DeepThinkOutput

    def run(self, data: BaseModel) -> ToolResult:
        input_data = DeepThinkInput.model_validate(data)
        plan = [
            "Identify key requirements",
            "Select tools and data sources",
            "Execute tasks with checkpoints",
            "Summarize outcome",
        ]
        if input_data.constraints:
            plan.append("Validate against constraints")
        checks = ["Validate outputs", "Ensure constraints met"]
        output = DeepThinkOutput(plan=plan, checks=checks)
        return ToolResult(output=output.model_dump())
