#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${STACK_CONFIG_PATH:-$ROOT_DIR/config/stack-config.json}"
GENERATED_DIR="$ROOT_DIR/.docker"
GENERATED_ENV="$GENERATED_DIR/generated.env"
GENERATED_PROXY_CONFIG="$GENERATED_DIR/proxy-config.json"

usage() {
  cat <<'EOF'
Usage: bash stack.sh <command>

Commands:
  start         Build and start Ollama, the proxy, and Open WebUI
  stop          Stop and remove the stack
  restart       Restart the stack
  status        Show container status
  logs          Tail compose logs
  config        Regenerate derived Docker files from config/stack-config.json
  pull-models   Pull the base and judge models configured in the stack config
EOF
}

ensure_config() {
  if [[ ! -f "$CONFIG_PATH" ]]; then
    cp "$ROOT_DIR/config/stack-config.json.example" "$CONFIG_PATH"
    echo "Created $CONFIG_PATH from example. Edit it, then rerun the command."
    exit 0
  fi
}

render_files() {
  mkdir -p "$GENERATED_DIR"
  python3 - "$CONFIG_PATH" "$GENERATED_ENV" "$GENERATED_PROXY_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
env_path = Path(sys.argv[2])
proxy_path = Path(sys.argv[3])

data = json.loads(config_path.read_text(encoding="utf-8"))
stack = data.get("stack", {})
proxy = data.get("proxy", {})

proxy.setdefault("ollama_base_url", "http://ollama:11434")

env_lines = [
    f"HOST_OLLAMA_PORT={stack.get('host_ollama_port', 11434)}",
    f"HOST_PROXY_PORT={stack.get('host_proxy_port', 8000)}",
    f"HOST_OPEN_WEBUI_PORT={stack.get('host_open_webui_port', 3000)}",
    f"OPEN_WEBUI_API_KEY={stack.get('open_webui_api_key', 'dummy')}",
    "BON_CONFIG_PATH=/app/.docker/proxy-config.json",
]

env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
proxy_path.write_text(json.dumps(proxy, indent=2) + "\n", encoding="utf-8")
PY
}

compose() {
  docker compose "$@"
}

pull_models() {
  local container_id
  local models_output
  container_id="$(compose ps -q ollama)"
  if [[ -z "$container_id" ]]; then
    echo "Ollama container is not running."
    exit 1
  fi

  models_output="$(python3 - "$CONFIG_PATH" <<'PY'
import json
import sys
from collections import OrderedDict

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

proxy = data.get("proxy", {})
models = OrderedDict()
judge_model = proxy.get("judge_model")
if judge_model:
    models[judge_model] = True
for model_cfg in (proxy.get("models") or {}).values():
    base_model = model_cfg.get("base_model")
    if base_model:
        models[base_model] = True
for pool_cfg in (proxy.get("pools") or {}).values():
    for candidate in pool_cfg.get("candidates", []):
        model = candidate.get("model")
        if model:
            models[model] = True
for model in models:
    print(model)
PY
)"

  while IFS= read -r model; do
    [[ -z "$model" ]] && continue
    docker exec "$container_id" ollama pull "$model"
  done <<< "$models_output"
}

main() {
  local command="${1:-}"
  if [[ -z "$command" ]]; then
    usage
    exit 1
  fi

  ensure_config
  render_files

  case "$command" in
    start)
      compose up -d --build
      ;;
    stop)
      compose down
      ;;
    restart)
      compose down
      compose up -d --build
      ;;
    status)
      compose ps
      ;;
    logs)
      compose logs -f
      ;;
    config)
      echo "Generated $GENERATED_ENV and $GENERATED_PROXY_CONFIG"
      ;;
    pull-models)
      compose up -d ollama
      pull_models
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
