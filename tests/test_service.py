from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import FileConfig, PoolConfig, PoolCandidate, SamplingConfig, SyntheticModelConfig
from app.schemas import ChatCompletionRequest
from app.service import (
    BestOfNService,
    CandidateResult,
    build_judge_prompt,
    expand_pool,
    heuristic_select,
    parse_winner_index,
)


class FakeGateway:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    async def list_models(self):
        return ["llama3.1:8b"]

    async def chat_completion(self, payload):
        self.calls.append(payload)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_config() -> FileConfig:
    return FileConfig(
        judge_model="judge",
        default_n=3,
        passthrough_models=True,
        models={
            "bon/llama3.1:8b": SyntheticModelConfig(
                base_model="llama3.1:8b",
                n=3,
                sampling=SamplingConfig(temperature=0.7),
            ),
            "bon-pool/coding": SyntheticModelConfig(pool="coding", n=3),
        },
        pools={
            "coding": PoolConfig(
                candidates=[
                    PoolCandidate(model="qwen2.5-coder:14b", count=2),
                    PoolCandidate(model="deepseek-coder:6.7b", weight=1),
                ]
            )
        },
    )


def test_expand_pool_counts_and_weights():
    pool = [
        PoolCandidate(model="a", count=1),
        PoolCandidate(model="b", weight=2),
        PoolCandidate(model="c", weight=1),
    ]
    expanded = expand_pool(pool, 5)
    assert [candidate.model for candidate in expanded].count("a") == 1
    assert [candidate.model for candidate in expanded].count("b") == 3
    assert [candidate.model for candidate in expanded].count("c") == 1


def test_parse_winner_index_valid_and_invalid():
    assert parse_winner_index('{"winner_index": 2, "reason": "best"}') == 2
    assert parse_winner_index("winner is 2") is None


def test_heuristic_select_prefers_non_refusal_content():
    candidates = [
        CandidateResult(0, "a", {}, {"choices": [{"message": {"content": "I'm sorry, I can't help."}}]}, None, 0.1),
        CandidateResult(1, "a", {}, {"choices": [{"message": {"content": "Concrete answer with details"}}]}, None, 0.1),
    ]
    assert heuristic_select(candidates).index == 1


@pytest.mark.asyncio
async def test_single_model_expansion_and_judged_winner():
    gateway = FakeGateway(
        responses=[
            {"choices": [{"message": {"content": "candidate zero"}}], "usage": {"total_tokens": 10}},
            {"choices": [{"message": {"content": "candidate one"}}], "usage": {"total_tokens": 11}},
            {"choices": [{"message": {"content": "candidate two"}}], "usage": {"total_tokens": 12}},
            {"choices": [{"message": {"content": '{"winner_index": 1, "reason": "best"}'}}]},
        ]
    )
    service = BestOfNService(make_config(), gateway)
    request = ChatCompletionRequest(
        model="bon/llama3.1:8b",
        messages=[{"role": "user", "content": "say hi"}],
        seed=41,
        temperature=0.2,
    )

    response = await service.handle_chat(request)

    assert response.choices[0].message.content == "candidate one"
    assert gateway.calls[0]["model"] == "llama3.1:8b"
    assert gateway.calls[0]["temperature"] == 0.7
    assert gateway.calls[0]["seed"] == 41
    assert gateway.calls[1]["seed"] == 42
    assert gateway.calls[2]["seed"] == 43


@pytest.mark.asyncio
async def test_judge_retry_then_fallback():
    gateway = FakeGateway(
        responses=[
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": "useful answer"}}]},
            {"choices": [{"message": {"content": "not json"}}]},
            {"choices": [{"message": {"content": "still not json"}}]},
        ]
    )
    config = make_config().model_copy(update={"judge_retry_count": 1})
    service = BestOfNService(config, gateway)
    request = ChatCompletionRequest(model="bon/llama3.1:8b", messages=[{"role": "user", "content": "x"}], seed=1)

    response = await service.handle_chat(request)

    assert response.choices[0].message.content == "useful answer"


@pytest.mark.asyncio
async def test_passthrough_non_synthetic_model():
    gateway = FakeGateway(
        responses=[{"id": "x", "object": "chat.completion", "created": 1, "model": "llama3.1:8b", "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}]}]
    )
    service = BestOfNService(make_config(), gateway)
    request = ChatCompletionRequest(model="llama3.1:8b", messages=[{"role": "user", "content": "ping"}])

    response = await service.handle_chat(request)

    assert response.choices[0].message.content == "pong"


def test_build_judge_prompt_contains_candidates():
    candidates = [
        CandidateResult(0, "model-a", {}, {"choices": [{"message": {"content": "alpha"}}]}, None, 0.1),
        CandidateResult(1, "model-b", {}, {"choices": [{"message": {"content": "beta"}}]}, None, 0.1),
    ]
    request_messages = [SimpleNamespace(model_dump=lambda: {"role": "user", "content": "pick"})]
    prompt = build_judge_prompt(request_messages, candidates)
    assert "Candidate 0 (model-a)" in prompt
    assert "Candidate 1 (model-b)" in prompt
