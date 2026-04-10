# Best-of-N Ollama Proxy for Open WebUI

## Summary
Build a standalone OpenAI-compatible proxy service in this directory that sits between Open WebUI and a local Ollama instance. Open WebUI will connect to the proxy as an OpenAI-compatible provider; the proxy will expose synthetic chat models that run `N` parallel Ollama generations, score them with a local judge model, and return only the selected winner.

Defaults chosen:
- Integration: standalone OpenAI-compatible proxy
- Packaging: Docker Compose
- Judge: local LLM-as-judge via Ollama
- Candidate generation: support both repeated samples of one base model and optional multi-model pools
- UX: final-answer-only, not candidate streaming to the UI

## Key Changes
- Create a small Python service, preferably `FastAPI`, exposing:
  - `GET /v1/models` to list synthetic best-of-N models plus passthrough/base models if enabled
  - `POST /v1/chat/completions` to accept OpenAI-style requests from Open WebUI
  - Optional `GET /healthz` for local health checks
- Route requests by synthetic model name, for example:
  - `bon/llama3.1:8b`
  - `bon/qwen2.5:14b`
  - `bon-pool/coding`
- Add config for:
  - Ollama base URL
  - Judge model name
  - Default `N`
  - Per-model overrides for `N`, temperature, top_p, max_tokens, timeout
  - Candidate strategy:
    - single base model with `N` varied samples/seeds
    - named candidate pools with explicit model lists and optional per-model weights/counts
- Candidate generation behavior:
  - Fan out `N` parallel calls to Ollama's OpenAI-compatible `POST /v1/chat/completions`
  - For single-model mode, vary `seed` and keep user-provided generation settings unless overridden by config defaults
  - For model-pool mode, expand the configured pool into `N` concrete calls
- Judge behavior:
  - After all candidates complete, call a local Ollama judge model with a strict comparison prompt
  - Judge output must be machine-parseable, returning selected candidate index and brief rationale
  - Proxy returns only the winning candidate as the assistant response to Open WebUI
  - If judge output is invalid, fall back to deterministic retry once, then to a configured heuristic fallback
- Fallback heuristic:
  - Prefer completed non-empty candidates
  - Then prefer candidates with lower refusal rate / higher content length within sane bounds
  - If only one candidate succeeds, return it
  - If all fail, return an OpenAI-style error payload
- Open WebUI integration:
  - Connect Open WebUI to the proxy as an OpenAI-compatible provider
  - Configure the proxy to advertise the synthetic best-of-N models via `/v1/models`
  - No Open WebUI plugin/function dependency in v1
- Docker Compose setup:
  - Service for the proxy
  - Optional Open WebUI service wired to the proxy
  - Ollama assumed to be reachable locally or via compose network; document both modes
- Basic project assets:
  - `README.md` with local startup, Open WebUI connection steps, and model config examples
  - `.env.example` for runtime configuration
  - Structured logging for candidate latencies, judge outcome, and fallback usage

## Public Interfaces / Types
- OpenAI-compatible API surface:
  - Accept standard chat completion request fields needed by Open WebUI: `model`, `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `seed`
  - `stream=true` is accepted but handled as final-answer-only in v1; the proxy should return the chosen answer as a single completion stream or non-streamed completion depending on implementation simplicity
- Internal config schema:
  - `judge_model`
  - `ollama_base_url`
  - `default_n`
  - `models.<synthetic_model>.base_model`
  - `models.<synthetic_model>.n`
  - `models.<synthetic_model>.sampling`
  - `pools.<pool_name>.candidates[]`
- Judge contract:
  - Input: original user/system conversation plus labeled candidate responses
  - Output: strict JSON with `winner_index`, optional `confidence`, optional `reason`

## Test Plan
- Unit tests:
  - Expand single-model `N` sampling into correct parallel Ollama requests
  - Expand model-pool definitions into correct candidate call set
  - Parse valid and invalid judge outputs
  - Heuristic fallback selection under partial failure
  - Synthetic model name resolution and config override precedence
- Integration tests with mocked Ollama endpoints:
  - `/v1/models` exposes configured synthetic models
  - `/v1/chat/completions` returns the judged winner in OpenAI-compatible shape
  - One or more candidate failures still produce a valid winner when possible
  - Judge failure triggers retry, then fallback heuristic
- Manual scenarios:
  - Open WebUI can add the proxy as an OpenAI-compatible provider and chat with a synthetic model
  - Single-model best-of-N improves over one-shot sampling on at least a few fixed prompts
  - Model-pool routing works for a named pool
  - Latency remains acceptable with the chosen local models and `N`

## Assumptions
- v1 targets chat completions first; embeddings, tools, image inputs, and full Responses API support are out of scope
- v1 does not expose candidate-by-candidate streaming to Open WebUI
- Ollama's OpenAI-compatible API remains the upstream interface for both generation and judging
- Python is the implementation language, with `FastAPI` plus async HTTP client for fan-out
- The proxy may expose passthrough models later, but the primary deliverable is synthetic best-of-N models for Open WebUI
