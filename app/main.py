from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import load_config
from app.logging_utils import configure_logging
from app.ollama import OllamaGateway
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, ErrorPayload, ErrorResponse
from app.service import BestOfNService


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    config = load_config()
    gateway = OllamaGateway(config.ollama_base_url, config.request_timeout_seconds)
    app.state.service = BestOfNService(config, gateway)
    yield
    await gateway.close()


app = FastAPI(title="Ollama Best-of-N Proxy", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return await app.state.service.list_models()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        response = await app.state.service.handle_chat(request)
    except HTTPException as exc:
        payload = ErrorResponse(error=ErrorPayload(message=str(exc.detail), code=str(exc.status_code)))
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())
    response = ChatCompletionResponse.model_validate(response)

    if not request.stream:
        return response
    return StreamingResponse(_stream_single_response(response), media_type="text/event-stream")


async def _stream_single_response(response: ChatCompletionResponse):
    chunk = {
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": response.choices[0].message.content},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
