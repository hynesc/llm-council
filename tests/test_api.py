from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


class StubService:
    async def list_models(self):
        return {"object": "list", "data": [{"id": "bon/test", "object": "model", "created": 0, "owned_by": "x"}]}

    async def handle_chat(self, request):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "winner"}, "finish_reason": "stop"}],
        }


def test_models_endpoint_and_streaming():
    app.state.service = StubService()
    client = TestClient(app)

    models_response = client.get("/v1/models")
    assert models_response.status_code == 200
    assert models_response.json()["data"][0]["id"] == "bon/test"

    stream_response = client.post(
        "/v1/chat/completions",
        json={"model": "bon/test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert stream_response.status_code == 200
    assert "data: [DONE]" in stream_response.text
