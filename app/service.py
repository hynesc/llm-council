from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.config import FileConfig, PoolCandidate, SamplingConfig, SyntheticModelConfig
from app.ollama import OllamaGateway, request_to_payload
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ModelCard, ModelListResponse

logger = logging.getLogger(__name__)


@dataclass
class CandidateSpec:
    index: int
    model: str
    payload: dict[str, Any]


@dataclass
class CandidateResult:
    index: int
    model: str
    payload: dict[str, Any]
    response: dict[str, Any] | None
    error: str | None
    latency_seconds: float

    @property
    def content(self) -> str:
        if not self.response:
            return ""
        choices = self.response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""


class BestOfNService:
    def __init__(self, config: FileConfig, gateway: OllamaGateway):
        self.config = config
        self.gateway = gateway

    async def list_models(self) -> ModelListResponse:
        synthetic_cards = [ModelCard(id=model_id) for model_id in sorted(self.config.models)]
        if not self.config.passthrough_models:
            return ModelListResponse(data=synthetic_cards)

        upstream = await self.gateway.list_models()
        upstream_cards = [ModelCard(id=model_id, owned_by="ollama") for model_id in sorted(upstream)]
        return ModelListResponse(data=synthetic_cards + upstream_cards)

    async def handle_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.model not in self.config.models:
            if self.config.passthrough_models:
                payload = request_to_payload(request)
                upstream = await self.gateway.chat_completion(payload)
                return ChatCompletionResponse.model_validate(upstream)
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")

        model_cfg = self.config.models[request.model]
        candidate_specs = self._expand_candidates(request, model_cfg)
        candidate_results = await self._generate_candidates(candidate_specs)
        winner = await self._select_winner(request, candidate_results)
        if winner.response is None:
            raise HTTPException(status_code=502, detail="all candidate generations failed")
        return self._build_response(request.model, winner.response, winner.content)

    def _expand_candidates(
        self, request: ChatCompletionRequest, model_cfg: SyntheticModelConfig
    ) -> list[CandidateSpec]:
        requested_n = model_cfg.n or self.config.default_n
        base_payload = request_to_payload(request)
        base_payload["stream"] = False

        if model_cfg.base_model:
            specs = []
            for idx in range(requested_n):
                payload = dict(base_payload)
                payload["model"] = model_cfg.base_model
                merged_sampling = self._merge_sampling(payload, model_cfg.sampling)
                payload = merged_sampling
                payload["seed"] = self._seed_for_index(request.seed, idx)
                specs.append(CandidateSpec(index=idx, model=model_cfg.base_model, payload=payload))
            return specs

        pool = self.config.pools.get(model_cfg.pool or "")
        if pool is None:
            raise HTTPException(status_code=500, detail=f"unknown pool: {model_cfg.pool}")

        expanded_candidates = expand_pool(pool.candidates, requested_n)
        specs = []
        for idx, pool_candidate in enumerate(expanded_candidates):
            payload = dict(base_payload)
            payload["model"] = pool_candidate.model
            payload = self._merge_sampling(payload, model_cfg.sampling)
            payload = self._merge_sampling(payload, pool_candidate.sampling)
            payload["seed"] = self._seed_for_index(request.seed, idx)
            specs.append(CandidateSpec(index=idx, model=pool_candidate.model, payload=payload))
        return specs

    def _merge_sampling(self, payload: dict[str, Any], sampling: SamplingConfig) -> dict[str, Any]:
        merged = dict(payload)
        return sampling.apply(merged)

    def _seed_for_index(self, seed: int | None, index: int) -> int:
        if seed is not None:
            return seed + index
        return random.randint(0, 2**31 - 1)

    async def _generate_candidates(self, specs: list[CandidateSpec]) -> list[CandidateResult]:
        tasks = [self._run_candidate(spec) for spec in specs]
        return await asyncio.gather(*tasks)

    async def _run_candidate(self, spec: CandidateSpec) -> CandidateResult:
        started = time.perf_counter()
        try:
            response = await self.gateway.chat_completion(spec.payload)
            latency = time.perf_counter() - started
            logger.info(
                "candidate_completed",
                extra={
                    "event_data": {
                        "candidate_index": spec.index,
                        "candidate_model": spec.model,
                        "latency_seconds": round(latency, 3),
                    }
                },
            )
            return CandidateResult(
                index=spec.index,
                model=spec.model,
                payload=spec.payload,
                response=response,
                error=None,
                latency_seconds=latency,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            logger.warning(
                "candidate_failed",
                extra={
                    "event_data": {
                        "candidate_index": spec.index,
                        "candidate_model": spec.model,
                        "latency_seconds": round(latency, 3),
                        "error": str(exc),
                    }
                },
            )
            return CandidateResult(
                index=spec.index,
                model=spec.model,
                payload=spec.payload,
                response=None,
                error=str(exc),
                latency_seconds=latency,
            )

    async def _select_winner(
        self, request: ChatCompletionRequest, candidates: list[CandidateResult]
    ) -> CandidateResult:
        valid_candidates = [candidate for candidate in candidates if candidate.content.strip()]
        if len(valid_candidates) == 1:
            return valid_candidates[0]
        if not valid_candidates:
            fallback = heuristic_select(candidates)
            logger.info("fallback_selected", extra={"event_data": {"reason": "no_valid_candidates"}})
            return fallback

        for attempt in range(self.config.judge_retry_count + 1):
            judge_result = await self._judge(request, valid_candidates)
            if judge_result is not None:
                logger.info(
                    "judge_selected",
                    extra={
                        "event_data": {
                            "winner_index": judge_result.index,
                            "attempt": attempt,
                        }
                    },
                )
                return judge_result

        fallback = heuristic_select(candidates)
        logger.info("fallback_selected", extra={"event_data": {"reason": "judge_failed"}})
        return fallback

    async def _judge(
        self, request: ChatCompletionRequest, candidates: list[CandidateResult]
    ) -> CandidateResult | None:
        judge_payload = {
            "model": self.config.judge_model,
            "stream": False,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict response judge. Return JSON only with keys "
                        "winner_index, confidence, and reason. winner_index must be an integer "
                        "matching one of the candidate indices."
                    ),
                },
                {
                    "role": "user",
                    "content": build_judge_prompt(request.messages, candidates),
                },
            ],
        }
        try:
            response = await self.gateway.chat_completion(judge_payload)
        except Exception:
            return None

        choices = response.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        winner_index = parse_winner_index(content)
        if winner_index is None:
            return None
        return next((candidate for candidate in candidates if candidate.index == winner_index), None)

    def _build_response(
        self, model_name: str, upstream_response: dict[str, Any], content: str
    ) -> ChatCompletionResponse:
        usage = upstream_response.get("usage")
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model_name,
            choices=[ChatChoice(message={"role": "assistant", "content": content})],
            usage=usage,
        )


def expand_pool(candidates: list[PoolCandidate], total_n: int) -> list[PoolCandidate]:
    fixed = [candidate for candidate in candidates if candidate.count is not None]
    weighted = [candidate for candidate in candidates if candidate.count is None]
    expanded: list[PoolCandidate] = []

    fixed_count = sum(candidate.count or 0 for candidate in fixed)
    if fixed_count > total_n:
        raise HTTPException(status_code=500, detail="pool fixed counts exceed requested n")

    for candidate in fixed:
        expanded.extend([candidate] * (candidate.count or 0))

    remaining = total_n - fixed_count
    if remaining == 0:
        return expanded
    if not weighted:
        raise HTTPException(status_code=500, detail="pool does not have enough candidates to satisfy n")

    total_weight = sum(candidate.weight or 1.0 for candidate in weighted)
    allocations = []
    remainders = []
    assigned = 0
    for candidate in weighted:
        exact = remaining * ((candidate.weight or 1.0) / total_weight)
        count = int(exact)
        allocations.append((candidate, count))
        remainders.append((exact - count, candidate))
        assigned += count

    for candidate, count in allocations:
        expanded.extend([candidate] * count)

    leftover = remaining - assigned
    for _, candidate in sorted(remainders, key=lambda item: item[0], reverse=True)[:leftover]:
        expanded.append(candidate)

    return expanded


def build_judge_prompt(messages: list[Any], candidates: list[CandidateResult]) -> str:
    conversation = json.dumps([message.model_dump() for message in messages], ensure_ascii=True)
    candidate_blocks = [
        f"Candidate {candidate.index} ({candidate.model}):\n{candidate.content}" for candidate in candidates
    ]
    return (
        "Choose the best assistant reply for the conversation below. Favor correctness, relevance, "
        "instruction-following, and completeness. Penalize refusals and empty answers.\n\n"
        f"Conversation:\n{conversation}\n\nCandidates:\n" + "\n\n".join(candidate_blocks)
    )


def parse_winner_index(content: Any) -> int | None:
    if not isinstance(content, str):
        return None
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    winner = payload.get("winner_index")
    return winner if isinstance(winner, int) else None


def heuristic_select(candidates: list[CandidateResult]) -> CandidateResult:
    ranked = sorted(candidates, key=_heuristic_score, reverse=True)
    return ranked[0]


def _heuristic_score(candidate: CandidateResult) -> tuple[int, int, int]:
    content = candidate.content.strip()
    completed = 1 if candidate.response is not None else 0
    non_empty = 1 if content else 0
    refusal_penalty = 0
    lowered = content.lower()
    refusal_markers = [
        "i can't",
        "i cannot",
        "i'm sorry",
        "as an ai",
        "unable to",
    ]
    if any(marker in lowered for marker in refusal_markers):
        refusal_penalty = -1
    bounded_length = min(len(content), 4000)
    return (completed + non_empty + refusal_penalty, non_empty, bounded_length)
