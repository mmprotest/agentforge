"""Configuration settings for AgentForge."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or overrides."""

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_MODEL")
    openai_timeout_seconds: int = Field(
        default=30, validation_alias="OPENAI_TIMEOUT_SECONDS"
    )
    agent_mode: Literal["direct", "deep"] = Field(
        default="direct", validation_alias="AGENT_MODE"
    )
    allow_tool_creation: bool = Field(
        default=False, validation_alias="ALLOW_TOOL_CREATION"
    )
    workspace_dir: str = Field(default="./workspace", validation_alias="WORKSPACE_DIR")


DEFAULT_SETTINGS = Settings()
