import pytest

from agentforge.runtime.rbac import AuthorizationError, RBACConfig, authorize_tool


def test_rbac_blocks_viewer_tool() -> None:
    config = RBACConfig(role_tools={"viewer": ["calculator"]})
    with pytest.raises(AuthorizationError):
        authorize_tool("viewer", "python_sandbox", config)


def test_rbac_allows_viewer_tool() -> None:
    config = RBACConfig(role_tools={"viewer": ["calculator"]})
    authorize_tool("viewer", "calculator", config)
