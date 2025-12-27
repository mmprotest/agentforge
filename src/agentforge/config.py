"""Configuration settings for AgentForge."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or overrides."""

    model_config = SettingsConfigDict(
        env_prefix="", case_sensitive=False, populate_by_name=True
    )

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_MODEL")
    openai_timeout_seconds: int = Field(
        default=30, validation_alias="OPENAI_TIMEOUT_SECONDS"
    )
    openai_extra_headers: str | None = Field(
        default=None, validation_alias="OPENAI_EXTRA_HEADERS"
    )
    openai_disable_tool_choice: bool = Field(
        default=False, validation_alias="OPENAI_DISABLE_TOOL_CHOICE"
    )
    openai_force_chatcompletions_path: str | None = Field(
        default=None, validation_alias="OPENAI_FORCE_CHATCOMPLETIONS_PATH"
    )
    proposal_count: int = Field(default=3, validation_alias="PROPOSAL_COUNT")
    max_backtracks: int = Field(default=1, validation_alias="MAX_BACKTRACKS")
    max_attempts: int = Field(default=10, validation_alias="MAX_ATTEMPTS")


DEFAULT_SETTINGS = Settings()
