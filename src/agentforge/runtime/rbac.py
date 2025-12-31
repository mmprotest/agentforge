"""Role-based access control."""

from __future__ import annotations

from dataclasses import dataclass, field


ROLES = ("admin", "operator", "viewer")


@dataclass
class RBACConfig:
    role_tools: dict[str, list[str]] = field(default_factory=dict)
    role_connectors: dict[str, list[str]] = field(default_factory=dict)
    role_workflow: dict[str, list[str]] = field(default_factory=dict)

    def tools_for(self, role: str) -> list[str] | None:
        return self.role_tools.get(role)

    def connectors_for(self, role: str) -> list[str] | None:
        return self.role_connectors.get(role)

    def workflows_for(self, role: str) -> list[str] | None:
        return self.role_workflow.get(role)


class AuthorizationError(RuntimeError):
    pass


def authorize_tool(role: str, tool_name: str, config: RBACConfig) -> None:
    allowed = config.tools_for(role)
    if allowed is not None and tool_name not in allowed:
        raise AuthorizationError(f"Role '{role}' cannot use tool '{tool_name}'.")


def authorize_connector(role: str, connector_name: str, config: RBACConfig) -> None:
    allowed = config.connectors_for(role)
    if allowed is not None and connector_name not in allowed:
        raise AuthorizationError(
            f"Role '{role}' cannot use connector '{connector_name}'."
        )


def authorize_workflow(role: str, workflow_action: str, config: RBACConfig) -> None:
    allowed = config.workflows_for(role)
    if allowed is not None and workflow_action not in allowed:
        raise AuthorizationError(
            f"Role '{role}' cannot perform workflow '{workflow_action}'."
        )
