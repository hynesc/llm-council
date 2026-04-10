from __future__ import annotations

import json

from app.config import load_config


def test_load_config_from_proxy_subkey(tmp_path, monkeypatch):
    config_path = tmp_path / "stack-config.json"
    config_path.write_text(
        json.dumps(
            {
                "stack": {"host_proxy_port": 8000},
                "proxy": {
                    "judge_model": "judge-model",
                    "models": {"bon/test": {"base_model": "llama3.1:8b", "n": 2}},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BON_CONFIG_PATH", str(config_path))

    config = load_config()

    assert config.judge_model == "judge-model"
    assert config.models["bon/test"].base_model == "llama3.1:8b"


def test_load_config_from_json_env(monkeypatch):
    monkeypatch.delenv("BON_CONFIG_PATH", raising=False)
    monkeypatch.setenv("BON_MODELS_JSON", json.dumps({"bon/test": {"base_model": "llama3.1:8b", "n": 2}}))
    monkeypatch.setenv("BON_POOLS_JSON", json.dumps({"coding": {"candidates": [{"model": "qwen", "count": 1}]}}))

    config = load_config()

    assert config.models["bon/test"].n == 2
    assert config.pools["coding"].candidates[0].model == "qwen"
