#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_PROJECT_DIR="${ROOT_DIR}/python"
VENV_DIR="${SGLANG_VENV_DIR:-${ROOT_DIR}/.venv}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
HOST="${HOST:-127.0.0.1}"
REQUEST_LENGTH="${REQUEST_LENGTH:-2048}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-256}"
NUM_CLIENTS="${NUM_CLIENTS:-80}"
NUM_ROUNDS="${NUM_ROUNDS:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
REQUEST_RATE="${REQUEST_RATE:-16}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-3600}"
RUN_TAG="${RUN_TAG:-tp8_multiturn_baseline_no_hicache}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/root/Documents/myelon-launch/ai-chat-exports/.0_agentic_engineering/1_sglang/2_myelon_mq/2_artifacts/$(date -u +%Y-%m-%dT%H%M%SZ)_${RUN_TAG}}"

mkdir -p "${ARTIFACT_DIR}"

SERVER_LOG="${ARTIFACT_DIR}/server.log"
BENCH_LOG="${ARTIFACT_DIR}/bench_multiturn.jsonl"
IPC_JSONL="${ARTIFACT_DIR}/ipc_stats.jsonl"
IPC_SUMMARY_TXT="${ARTIFACT_DIR}/ipc_summary.txt"
IPC_SUMMARY_JSON="${ARTIFACT_DIR}/ipc_summary.json"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv "${VENV_DIR}"
fi

export PATH="${VENV_DIR}/bin:${PATH}"

if ! "${VENV_DIR}/bin/python" -c "import sglang" >/dev/null 2>&1; then
  uv pip install --python "${VENV_DIR}/bin/python" -e "${PY_PROJECT_DIR}"
fi

SERVER_PID=""
stop_server() {
  if [[ -z "${SERVER_PID}" ]] || ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    return 0
  fi

  kill -INT "${SERVER_PID}" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      SERVER_PID=""
      return 0
    fi
    sleep 1
  done

  kill -TERM "${SERVER_PID}" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      SERVER_PID=""
      return 0
    fi
    sleep 1
  done

  kill -KILL "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" || true
  SERVER_PID=""
}

cleanup() {
  stop_server
}
trap cleanup EXIT

echo "Artifacts: ${ARTIFACT_DIR}"

MYELON_INSTRUMENT=1 \
MYELON_INSTRUMENT_JSONL="${IPC_JSONL}" \
MYELON_INSTRUMENT_TAG="${RUN_TAG}" \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER="${SGLANG_USE_MESSAGE_QUEUE_BROADCASTER:-true}" \
"${VENV_DIR}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp "${TP}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Server PID: ${SERVER_PID}"

"${VENV_DIR}/bin/python" - <<'PY' "${HOST}" "${PORT}" "${SERVER_TIMEOUT_SEC}" "${SERVER_PID}"
import sys
import time
import requests
import os

host, port, timeout_s, server_pid = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
)
url = f"http://{host}:{port}/health"
deadline = time.time() + timeout_s
last_error = None
while time.time() < deadline:
    try:
        os.kill(server_pid, 0)
    except ProcessLookupError:
        raise SystemExit(f"server exited before reaching health: {last_error}")
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            raise SystemExit(0)
        last_error = f"unexpected status {response.status_code}"
    except Exception as exc:
        last_error = str(exc)
    time.sleep(5)
raise SystemExit(f"server did not become healthy before timeout: {last_error}")
PY

"${VENV_DIR}/bin/python" "${ROOT_DIR}/benchmark/hicache/bench_multiturn.py" \
  --disable-auto-run \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --log-file "${BENCH_LOG}" \
  --tag "${RUN_TAG}" \
  --request-length "${REQUEST_LENGTH}" \
  --output-length "${OUTPUT_LENGTH}" \
  --num-clients "${NUM_CLIENTS}" \
  --num-rounds "${NUM_ROUNDS}" \
  --max-parallel "${MAX_PARALLEL}" \
  --request-rate "${REQUEST_RATE}"

stop_server

"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/myelon/analyze_ipc_jsonl.py" \
  --input "${IPC_JSONL}" \
  --tag "${RUN_TAG}" \
  --json-out "${IPC_SUMMARY_JSON}" \
  | tee "${IPC_SUMMARY_TXT}"

echo "Completed baseline run: ${ARTIFACT_DIR}"
