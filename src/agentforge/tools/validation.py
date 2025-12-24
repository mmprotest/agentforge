"""Tool validation pipeline."""

from __future__ import annotations

import ast

FORBIDDEN_IMPORTS = {"subprocess", "socket", "requests", "httpx"}
FORBIDDEN_CALLS = {"eval", "exec"}
FORBIDDEN_ATTR_CALLS = {"system", "popen", "rmtree"}


def validate_source(source: str, allow_network: bool = False) -> list[str]:
    """Return list of validation errors for tool source code."""
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names]
            for module in modules:
                if module in FORBIDDEN_IMPORTS and not allow_network:
                    errors.append(f"Forbidden import: {module}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                errors.append(f"Forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTR_CALLS:
                errors.append(f"Forbidden call: {node.func.attr}")
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                errors.append("Direct open() is not allowed")
    if "shutil.rmtree" in source:
        errors.append("Forbidden use of shutil.rmtree")
    if "os.system" in source:
        errors.append("Forbidden use of os.system")
    return errors
