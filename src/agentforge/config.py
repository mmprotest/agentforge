"""Configuration settings for AgentForge."""

from __future__ import annotations

from typing import Literal

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
    agent_mode: Literal["direct", "deep"] = Field(
        default="direct", validation_alias="AGENT_MODE"
    )
    allow_tool_creation: bool = Field(
        default=False, validation_alias="ALLOW_TOOL_CREATION"
    )
    workspace_dir: str = Field(default="./workspace", validation_alias="WORKSPACE_DIR")
    max_tool_output_chars: int = Field(
        default=4000, validation_alias="MAX_TOOL_OUTPUT_CHARS"
    )
    keep_raw_tool_output: bool = Field(
        default=True, validation_alias="KEEP_RAW_TOOL_OUTPUT"
    )
    summary_lines: int = Field(default=10, validation_alias="SUMMARY_LINES")
    max_model_calls: int = Field(default=20, validation_alias="MAX_MODEL_CALLS")
    max_message_chars: int = Field(default=24000, validation_alias="MAX_MESSAGE_CHARS")
    max_message_tokens_approx: int = Field(
        default=6000, validation_alias="MAX_MESSAGE_TOKENS_APPROX"
    )
    token_char_ratio: int = Field(default=4, validation_alias="TOKEN_CHAR_RATIO")
    max_single_message_chars: int = Field(
        default=4000, validation_alias="MAX_SINGLE_MESSAGE_CHARS"
    )
    max_turns: int = Field(default=20, validation_alias="MAX_TURNS")
    trim_strategy: Literal["drop_oldest"] = Field(
        default="drop_oldest", validation_alias="TRIM_STRATEGY"
    )
    strict_json_mode: bool = Field(default=False, validation_alias="STRICT_JSON_MODE")
    code_check: bool = Field(default=False, validation_alias="CODE_CHECK")
    code_check_max_iters: int = Field(
        default=2, validation_alias="CODE_CHECK_MAX_ITERS"
    )
    sandbox_passthrough_env: str | None = Field(
        default=None, validation_alias="SANDBOX_PASSTHROUGH_ENV"
    )


DEFAULT_SETTINGS = Settings()
