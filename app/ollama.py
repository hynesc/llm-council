from __future__ import annotations

from typing import Any

import httpx

from app.schemas import ChatCompletionRequest


class OllamaGateway:
    def __init__(self, base_url: str, timeout_seconds: float):
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_seconds)

    async def list_models(self) -> list[str]:
        response = await self._client.get("/v1/models")
        response.raise_for_status()
        payload = response.json()
        return [item["id"] for item in payload.get("data", []) if "id" in item]

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()


def request_to_payload(request: ChatCompletionRequest) -> dict[str, Any]:
    return request.model_dump(by_alias=True, exclude_none=True)
