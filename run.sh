#!/usr/bin/env bash
# Run the full Transformer-GAT MedRec pipeline (setup → train → eval → test → UI).
# Clinical decision support only — not for autonomous prescribing.
#
# Usage:
#   ./run.sh              # full pipeline + start dashboard
#   ./run.sh all          # same as above
#   ./run.sh setup        # venv + dependencies only
#   ./run.sh train        # training only
#   ./run.sh eval         # evaluation only (needs checkpoint)
#   ./run.sh test         # unit tests
#   ./run.sh explain      # SHAP demo for patient 0
#   ./run.sh ui           # dashboard only (needs checkpoint for full features)
#   ./run.sh stop         # stop UI server on port 8080 (or --port)
#   ./run.sh all --skip-train   # skip training if checkpoint exists
#   ./run.sh all --force-train  # always retrain
#   ./run.sh ui --port 8081     # custom UI port (fails if busy unless you use stop)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  elif command -v python3 &>/dev/null; then
    PYTHON="python3"
  else
    echo "Error: python3 not found. Install Python 3.10+ or set PYTHON=..." >&2
    exit 1
  fi
fi

PIP="${PIP:-}"
if [[ -z "$PIP" ]]; then
  if [[ -x "$ROOT/.venv/bin/pip" ]]; then
    PIP="$ROOT/.venv/bin/pip"
  else
    PIP="$PYTHON -m pip"
  fi
fi

CONFIG="${CONFIG:-config.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/best.pt}"
USE_SYNTHETIC="${USE_SYNTHETIC:-false}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8080}"
UI_PORT_EXPLICIT=0
SKIP_TRAIN=0
FORCE_TRAIN=0
CMD="${1:-all}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-train) SKIP_TRAIN=1 ;;
    --force-train) FORCE_TRAIN=1 ;;
    --port) UI_PORT="$2"; UI_PORT_EXPLICIT=1; shift ;;
    --host) UI_HOST="$2"; shift ;;
    --config) CONFIG="$2"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

log() { echo "==> $*"; }

port_in_use() {
  local port="$1"
  "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('127.0.0.1', int(sys.argv[1])))
    sys.exit(1)
except OSError:
    sys.exit(0)
finally:
    s.close()
" "$port"
}

find_free_port() {
  local start="${1:-8080}" max="${2:-8099}" p
  for ((p = start; p <= max; p++)); do
    if ! port_in_use "$p"; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

pid_on_port() {
  local port="$1"
  if command -v ss &>/dev/null; then
    ss -tlnp 2>/dev/null | grep -E ":${port}([^0-9]|$)" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
  elif command -v lsof &>/dev/null; then
    lsof -ti :"$port" 2>/dev/null | head -1
  fi
}

step_stop_ui() {
  local pid
  pid="$(pid_on_port "$UI_PORT" || true)"
  if [[ -z "$pid" ]]; then
    log "No process listening on port ${UI_PORT}"
    return 0
  fi
  log "Stopping UI server (PID ${pid}) on port ${UI_PORT}"
  kill "$pid" 2>/dev/null || true
  sleep 1
  if port_in_use "$UI_PORT"; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  if port_in_use "$UI_PORT"; then
    echo "Error: could not free port ${UI_PORT}" >&2
    exit 1
  fi
  log "Port ${UI_PORT} is free"
}

resolve_ui_port() {
  if port_in_use "$UI_PORT"; then
    if [[ "$UI_PORT_EXPLICIT" -eq 1 ]]; then
      echo "Error: port ${UI_PORT} is already in use." >&2
      echo "  Stop the old server:  ./run.sh stop --port ${UI_PORT}" >&2
      echo "  Or pick another port: ./run.sh ui --port 8081" >&2
      pid="$(pid_on_port "$UI_PORT" || true)"
      [[ -n "$pid" ]] && echo "  Process on port: PID ${pid}" >&2
      exit 1
    fi
    local new_port
    new_port="$(find_free_port "$UI_PORT" 8099)" || {
      echo "Error: no free port between ${UI_PORT} and 8099" >&2
      echo "  Run: ./run.sh stop" >&2
      exit 1
    }
    log "Port ${UI_PORT} is in use — using ${new_port} instead (or run: ./run.sh stop)"
    UI_PORT="$new_port"
  fi
}

step_setup() {
  log "Creating virtual environment (if needed)"
  if [[ ! -d "$ROOT/.venv" ]]; then
    "$PYTHON" -m venv "$ROOT/.venv"
    PYTHON="$ROOT/.venv/bin/python"
    PIP="$ROOT/.venv/bin/pip"
  else
    PYTHON="$ROOT/.venv/bin/python"
    PIP="$ROOT/.venv/bin/pip"
  fi

  log "Installing dependencies"
  $PIP install -q --upgrade pip
  $PIP install -q -r "$ROOT/requirements.txt"
  echo "Setup complete. Python: $PYTHON"
}

step_train() {
  if [[ "$FORCE_TRAIN" -eq 0 && -f "$ROOT/$CHECKPOINT" ]]; then
    if [[ "$SKIP_TRAIN" -eq 1 ]]; then
      log "Skipping training (checkpoint exists: $CHECKPOINT). Use --force-train to retrain."
      return 0
    fi
  elif [[ "$SKIP_TRAIN" -eq 1 ]]; then
    log "No checkpoint found — training anyway."
  fi
  log "Training model (synthetic data)"
  "$PYTHON" src/main.py --config "$CONFIG" --mode train --use_synthetic "$USE_SYNTHETIC"
}

step_eval() {
  if [[ ! -f "$ROOT/$CHECKPOINT" ]]; then
    echo "Error: checkpoint not found at $CHECKPOINT. Run: ./run.sh train" >&2
    exit 1
  fi
  log "Evaluating on test set"
  "$PYTHON" src/evaluate.py --checkpoint "$CHECKPOINT" --use_synthetic "$USE_SYNTHETIC"
}

step_test() {
  log "Running unit tests"
  if [[ -x "$ROOT/.venv/bin/pytest" ]]; then
    "$ROOT/.venv/bin/pytest" tests/ -v
  else
    "$PYTHON" -m pytest tests/ -v
  fi
}

step_explain() {
  if [[ ! -f "$ROOT/$CHECKPOINT" ]]; then
    echo "Error: checkpoint not found at $CHECKPOINT. Run: ./run.sh train" >&2
    exit 1
  fi
  log "Running SHAP explanation (patient 0)"
  "$PYTHON" src/explain.py --checkpoint "$CHECKPOINT" --use_synthetic "$USE_SYNTHETIC" --patient_idx 0
}

step_ui() {
  if [[ ! -f "$ROOT/$CHECKPOINT" ]]; then
    echo "Warning: no checkpoint at $CHECKPOINT — UI will load but predictions may be untrained." >&2
  fi
  resolve_ui_port
  log "Starting dashboard at http://${UI_HOST}:${UI_PORT}"
  log "Press Ctrl+C to stop the server."
  export UI_HOST UI_PORT
  "$PYTHON" -c "
import os, uvicorn
from pathlib import Path
root = Path('$ROOT')
os.chdir(root)
import sys
sys.path.insert(0, str(root))
uvicorn.run('ui.server:app', host=os.environ.get('UI_HOST', '127.0.0.1'), port=int(os.environ.get('UI_PORT', '8080')), app_dir=str(root))
"
}

step_all() {
  step_setup
  step_train
  step_eval
  step_test
  echo ""
  log "Pipeline finished. Launching web dashboard..."
  echo "    Open http://${UI_HOST}:${UI_PORT} in your browser."
  echo "    Clinical decision support only — physician review required."
  echo ""
  step_ui
}

case "$CMD" in
  setup)   step_setup ;;
  train)   step_setup; step_train ;;
  eval)    step_setup; step_eval ;;
  test)    step_setup; step_test ;;
  explain) step_setup; step_explain ;;
  stop)    step_stop_ui ;;
  ui)      step_setup; step_ui ;;
  all)     step_all ;;
  *)
    echo "Unknown command: $CMD" >&2
    echo "Commands: all, setup, train, eval, test, explain, ui, stop" >&2
    exit 1
    ;;
esac
