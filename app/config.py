from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SamplingConfig(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload


class PoolCandidate(BaseModel):
    model: str
    count: int | None = Field(default=None, ge=1)
    weight: float | None = Field(default=None, gt=0)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


class PoolConfig(BaseModel):
    candidates: list[PoolCandidate]


class SyntheticModelConfig(BaseModel):
    base_model: str | None = None
    pool: str | None = None
    n: int | None = Field(default=None, ge=1)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    @model_validator(mode="after")
    def validate_source(self) -> "SyntheticModelConfig":
        if bool(self.base_model) == bool(self.pool):
            raise ValueError("exactly one of base_model or pool must be set")
        return self


class FileConfig(BaseModel):
    judge_model: str = "qwen3.5:35b"
    ollama_base_url: str = "http://ollama:11434"
    default_n: int = Field(default=3, ge=1)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    judge_retry_count: int = Field(default=1, ge=0)
    passthrough_models: bool = False
    models: dict[str, SyntheticModelConfig] = Field(default_factory=dict)
    pools: dict[str, PoolConfig] = Field(default_factory=dict)


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BON_", extra="ignore")

    config_path: str | None = None
    judge_model: str | None = None
    ollama_base_url: str | None = None
    default_n: int | None = None
    request_timeout_seconds: float | None = None
    judge_retry_count: int | None = None
    passthrough_models: bool | None = None
    models_json: str | None = None
    pools_json: str | None = None


def load_config() -> FileConfig:
    settings = RuntimeSettings()
    base = FileConfig()

    if settings.config_path:
        raw = _read_config_file(Path(settings.config_path))
        if isinstance(raw.get("proxy"), dict):
            raw = raw["proxy"]
        base = FileConfig.model_validate(raw)

    overrides = settings.model_dump(exclude_none=True)
    overrides.pop("config_path", None)
    models_json = overrides.pop("models_json", None)
    pools_json = overrides.pop("pools_json", None)
    if models_json is not None:
        overrides["models"] = json.loads(models_json)
    if pools_json is not None:
        overrides["pools"] = json.loads(pools_json)
    if overrides:
        merged = base.model_dump()
        merged.update(overrides)
        base = FileConfig.model_validate(merged)
    return base


def _read_config_file(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(content) or {}
    elif path.suffix == ".json":
        data = json.loads(content)
    else:
        raise ValueError(f"unsupported config file format: {path}")
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data
