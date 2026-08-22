"""Runtime configuration from environment variables. No secrets are hardcoded."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Prefix HABITAT_ except for conventional OpenAI vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    serial_port: str | None = Field(default=None, validation_alias="HABITAT_SERIAL_PORT")
    baud: int = Field(default=115200, validation_alias="HABITAT_BAUD")
    log_level: str = Field(default="INFO", validation_alias="HABITAT_LOG_LEVEL")
    device_id: str = Field(default="hab-01", validation_alias="HABITAT_DEVICE_ID")

    genai_mode: Literal["mock", "openai"] = Field(
        default="mock",
        validation_alias=AliasChoices("GENAI_MODE", "HABITAT_GENAI_MODE"),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "HABITAT_OPENAI_API_KEY"),
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "HABITAT_OPENAI_BASE_URL"),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("OPENAI_MODEL", "HABITAT_OPENAI_MODEL"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
