"""Tool generation helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentforge.models.base import BaseChatModel
from agentforge.safety.sandbox import run_pytest
from agentforge.tools.base import Tool, ToolResult
from agentforge.tools.registry import ToolRegistry
from agentforge.tools.validation import validate_source


class ToolSpec(BaseModel):
    purpose: str
    desired_inputs: dict[str, str]
    desired_outputs: dict[str, str]
    examples: list[str]


class ToolMakerInput(BaseModel):
    spec: ToolSpec


@dataclass
class ToolCreationResult:
    success: bool
    tool_name: str | None = None
    tool: Tool | None = None
    errors: list[str] | None = None


class ToolMaker:
    """Create tools from specs using a chat model."""

    def __init__(self, model: BaseChatModel, workspace_dir: str) -> None:
        self.model = model
        self.workspace_dir = Path(workspace_dir).resolve()
        self.generated_dir = self.workspace_dir / "generated_tools"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    def _prompt(self, spec: ToolSpec) -> list[dict[str, Any]]:
        content = (
            "You are generating a new tool for AgentForge. "
            "Return JSON with keys: tool_name, tool_code, schemas_code, test_code. "
            "tool_code should define a Tool subclass. schemas_code should define Pydantic models. "
            "test_code should be a pytest file. Spec: "
            f"{spec.model_dump_json()}"
        )
        return [
            {"role": "system", "content": "You output only JSON."},
            {"role": "user", "content": content},
        ]

    def _write_files(self, tool_name: str, bundle: dict[str, str]) -> Path:
        tool_dir = self.generated_dir / tool_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        (tool_dir / "__init__.py").write_text("", encoding="utf-8")
        (tool_dir / "tool.py").write_text(bundle["tool_code"], encoding="utf-8")
        (tool_dir / "schemas.py").write_text(bundle["schemas_code"], encoding="utf-8")
        (tool_dir / "test_tool.py").write_text(bundle["test_code"], encoding="utf-8")
        return tool_dir

    def _load_tool(self, tool_dir: Path) -> Tool:
        sys.path.insert(0, str(self.generated_dir))
        module_path = tool_dir / "tool.py"
        spec = importlib.util.spec_from_file_location(f"{tool_dir.name}.tool", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load tool module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for obj in module.__dict__.values():
            if isinstance(obj, type) and issubclass(obj, Tool) and obj is not Tool:
                return obj()  # type: ignore[return-value]
        raise RuntimeError("No Tool subclass found in generated tool")

    def create_tool(self, spec: ToolSpec) -> ToolCreationResult:
        response = self.model.chat(self._prompt(spec), tools=None)
        if not response.final_text:
            return ToolCreationResult(success=False, errors=["Model returned no content"])
        try:
            bundle = json.loads(response.final_text)
        except json.JSONDecodeError as exc:
            return ToolCreationResult(success=False, errors=[f"Invalid JSON: {exc}"])
        required_keys = {"tool_name", "tool_code", "schemas_code", "test_code"}
        if not required_keys.issubset(bundle):
            return ToolCreationResult(success=False, errors=["Missing keys in tool bundle"])
        errors = []
        errors.extend(validate_source(bundle["tool_code"]))
        errors.extend(validate_source(bundle["schemas_code"]))
        errors.extend(validate_source(bundle["test_code"], allow_network=True))
        if errors:
            return ToolCreationResult(success=False, errors=errors)
        tool_dir = self._write_files(bundle["tool_name"], bundle)
        test_result = run_pytest(tool_dir / "test_tool.py")
        if not test_result.success:
            return ToolCreationResult(success=False, errors=[test_result.output])
        tool = self._load_tool(tool_dir)
        return ToolCreationResult(success=True, tool_name=bundle["tool_name"], tool=tool)


class ToolMakerTool(Tool):
    name = "tool_maker"
    description = "Generate a new tool from a spec and register it."
    input_schema = ToolMakerInput

    def __init__(self, maker: ToolMaker, registry: "ToolRegistry") -> None:
        self.maker = maker
        self.registry = registry

    def run(self, data: BaseModel) -> ToolResult:
        input_data = ToolMakerInput.model_validate(data)
        result = self.maker.create_tool(input_data.spec)
        output = {
            "success": result.success,
            "tool_name": result.tool_name,
            "errors": result.errors,
        }
        if result.success and result.tool is not None:
            self.registry.register(result.tool)
            output["tool"] = result.tool.name
        return ToolResult(output=output)
