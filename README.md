# Ollama Best-of-N Proxy

This project provides a standalone OpenAI-compatible proxy that sits between Open WebUI and Ollama. It exposes synthetic models that generate `N` parallel candidates, judges them with a local Ollama judge model, and returns only the selected winner.

## Features

- `GET /v1/models` exposes configured synthetic best-of-N models and optional passthrough upstream models.
- `POST /v1/chat/completions` accepts OpenAI-style chat requests from Open WebUI.
- Single-model best-of-N and named multi-model pools.
- Judge retry plus deterministic heuristic fallback.
- Docker Compose setup for the proxy and optional Open WebUI.
- Structured JSON logging for candidate latency, judge selections, and fallback use.

## Project Layout

- `app/main.py`: FastAPI app and OpenAI-compatible endpoints.
- `app/service.py`: best-of-N orchestration, judging, and fallback logic.
- `app/config.py`: YAML and environment-driven runtime configuration.
- `config/proxy-config.yaml`: sample synthetic model and pool definitions.
- `tests/`: unit and API tests.

## Local Run

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

2. Copy the environment file and adjust values as needed:

```bash
cp .env.example .env
```

3. Start the API:

```bash
uvicorn app.main:app --reload
```

The proxy listens on `http://localhost:8000`.

## Docker Workflow

The Docker-first path is driven by one file: `config/stack-config.json`.

1. Create your local config:

```bash
cp config/stack-config.json.example config/stack-config.json
```

2. Edit `config/stack-config.json` to tune ports, judge model, best-of-N settings, and model pools.

3. Start the full stack:

```bash
bash stack.sh start
```

This starts:

- `ollama`
- `bon-proxy`
- `open-webui`

Useful commands:

```bash
bash stack.sh status
bash stack.sh logs
bash stack.sh pull-models
bash stack.sh stop
```

Open WebUI will be available on `http://localhost:3000` by default.

## Open WebUI Setup

If you use the included Docker stack, Open WebUI is already wired to the proxy through:

- Base URL: `http://bon-proxy:8000/v1`
- API key: value from `stack.open_webui_api_key`

If you run Open WebUI separately, add the proxy as an OpenAI-compatible provider with:

- Base URL: `http://localhost:8000/v1`
- API key: the same value from `stack.open_webui_api_key`

Once connected, Open WebUI will see the synthetic model IDs from `/v1/models`, such as `bon/llama3.1:8b` and `bon-pool/coding`.

## Configuration

For Docker, use `config/stack-config.json` as the single source of truth. It has two sections:

- `stack`: host ports and the Open WebUI API key
- `proxy`: all proxy behavior, including judge model, default `N`, synthetic model definitions, and pools

The launcher script renders `.docker/generated.env` and `.docker/proxy-config.json` from that file, and the proxy loads the generated config automatically.

You can still run the proxy directly with `BON_` environment variables or a standalone JSON/YAML config file, but the managed Docker path uses `config/stack-config.json`.

## Notes

- `stream=true` is accepted, but v1 returns the chosen answer as a single streamed event sequence rather than candidate-by-candidate streaming.
- The proxy currently targets chat completions only. Embeddings, tools, images, and the Responses API are out of scope.
- Candidate pools support explicit `count` and proportional `weight` expansion.
