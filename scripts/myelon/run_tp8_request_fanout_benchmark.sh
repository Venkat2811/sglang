#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_PROJECT_DIR="${ROOT_DIR}/python"
VENV_DIR="${SGLANG_VENV_DIR:-${ROOT_DIR}/.venv}"
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30120}"
TP="${TP:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-3600}"
SERVER_EXTRA_ARGS_STR="${SERVER_EXTRA_ARGS:---log-level warning}"

GLOO_REPEATS="${GLOO_REPEATS:-64}"
GLOO_WARMUP_REPEATS="${GLOO_WARMUP_REPEATS:-2}"
INLINE_REPEATS="${INLINE_REPEATS:-64}"
INLINE_WARMUP_REPEATS="${INLINE_WARMUP_REPEATS:-2}"
OVERFLOW_REPEATS="${OVERFLOW_REPEATS:-16}"
OVERFLOW_WARMUP_REPEATS="${OVERFLOW_WARMUP_REPEATS:-1}"

INLINE_BATCH_SIZE="${INLINE_BATCH_SIZE:-8}"
INLINE_INPUT_LEN="${INLINE_INPUT_LEN:-2048}"
INLINE_OUTPUT_LEN="${INLINE_OUTPUT_LEN:-128}"

OVERFLOW_BATCH_SIZE="${OVERFLOW_BATCH_SIZE:-96}"
OVERFLOW_INPUT_LEN="${OVERFLOW_INPUT_LEN:-16384}"
OVERFLOW_OUTPUT_LEN="${OVERFLOW_OUTPUT_LEN:-1}"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-tp8_request_fanout}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/Documents/myelon-launch/ai-chat-exports/.0_agentic_engineering/1_sglang/2_myelon_mq/2_artifacts/$(date -u +%Y-%m-%dT%H%M%SZ)_${RUN_TAG_PREFIX}}"

mkdir -p "${ARTIFACT_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv "${VENV_DIR}"
fi

export PATH="${VENV_DIR}/bin:${PATH}"

if ! "${VENV_DIR}/bin/python" -c "import sglang" >/dev/null 2>&1; then
  uv pip install --python "${VENV_DIR}/bin/python" -e "${PY_PROJECT_DIR}"
fi

SERVER_EXTRA_ARGS=()
if [[ -n "${SERVER_EXTRA_ARGS_STR}" ]]; then
  # shellcheck disable=SC2206
  SERVER_EXTRA_ARGS=( ${SERVER_EXTRA_ARGS_STR} )
fi

SERVER_PID=""
CURRENT_ARTIFACT_DIR=""

stop_server() {
  if [[ -z "${SERVER_PID}" ]] || ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    SERVER_PID=""
    return 0
  fi

  kill -INT "${SERVER_PID}" >/dev/null 2>&1 || true
  for _ in $(seq 1 45); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      SERVER_PID=""
      return 0
    fi
    sleep 1
  done

  kill -TERM "${SERVER_PID}" >/dev/null 2>&1 || true
  for _ in $(seq 1 15); do
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

wait_for_server() {
  "${VENV_DIR}/bin/python" - <<'PY' "${HOST}" "${PORT}" "${SERVER_TIMEOUT_SEC}" "${SERVER_PID}"
import os
import sys
import time

import requests

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
}

start_server() {
  local artifact_dir="$1"
  local mq_enabled="$2"
  local scheduler_group_broadcast="$3"
  local instrument_mq="$4"

  CURRENT_ARTIFACT_DIR="${artifact_dir}"
  mkdir -p "${artifact_dir}"

  local server_log="${artifact_dir}/server.log"
  local pyobj_jsonl="${artifact_dir}/pyobj_broadcast.jsonl"
  local parallel_jsonl="${artifact_dir}/parallel_state.jsonl"
  local ipc_jsonl="${artifact_dir}/ipc_stats.jsonl"
  local ipc_event_jsonl="${artifact_dir}/ipc_events.jsonl"
  local tag
  tag="$(basename "${artifact_dir}")"

  env \
    SGLANG_LOG_MS=1 \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER="${mq_enabled}" \
    SGLANG_SCHEDULER_USE_GROUP_BROADCAST_OBJECT="${scheduler_group_broadcast}" \
    MYELON_PYOBJ_BCAST_JSONL="${pyobj_jsonl}" \
    MYELON_PYOBJ_BCAST_TAG="${tag}" \
    MYELON_PYOBJ_BCAST_NONEMPTY_ONLY=1 \
    MYELON_PARALLEL_STATE_JSONL="${parallel_jsonl}" \
    MYELON_PARALLEL_STATE_TAG="${tag}" \
    MYELON_PARALLEL_STATE_NONEMPTY_ONLY=1 \
    MYELON_INSTRUMENT="${instrument_mq}" \
    MYELON_INSTRUMENT_JSONL="${ipc_jsonl}" \
    MYELON_INSTRUMENT_EVENT_JSONL="${ipc_event_jsonl}" \
    MYELON_INSTRUMENT_EVENT_NONEMPTY_ONLY=1 \
    MYELON_INSTRUMENT_TAG="${tag}" \
    "${VENV_DIR}/bin/python" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --tp "${TP}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      "${SERVER_EXTRA_ARGS[@]}" \
      >"${server_log}" 2>&1 &
  SERVER_PID=$!
  echo "Server PID (${tag}): ${SERVER_PID}"
  wait_for_server
}

run_repeat_bench() {
  local artifact_dir="$1"
  local run_name="$2"
  local repeats="$3"
  local warmup_repeats="$4"
  local batch_size="$5"
  local input_len="$6"
  local output_len="$7"

  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/myelon/repeat_bench_one_batch_server.py" \
    --base-url "http://${HOST}:${PORT}" \
    --run-name "${run_name}" \
    --batch-size "${batch_size}" \
    --input-len "${input_len}" \
    --output-len "${output_len}" \
    --repeats "${repeats}" \
    --warmup-repeats "${warmup_repeats}" \
    --result-jsonl "${artifact_dir}/bench_repeat.jsonl" \
    --summary-json "${artifact_dir}/bench_repeat_summary.json" \
    --summary-txt "${artifact_dir}/bench_repeat_summary.txt"
}

GLOO_ARTIFACT="${ARTIFACT_ROOT}/gloo_tcp"
MQ_INLINE_ARTIFACT="${ARTIFACT_ROOT}/mq_shm_inline"
MQ_OVERFLOW_ARTIFACT="${ARTIFACT_ROOT}/mq_zmq_overflow"
COMPARISON_MD="${ARTIFACT_ROOT}/comparison.md"
COMPARISON_JSON="${ARTIFACT_ROOT}/comparison.json"

echo "Artifact root: ${ARTIFACT_ROOT}"

start_server "${GLOO_ARTIFACT}" 0 0 0
run_repeat_bench "${GLOO_ARTIFACT}" "gloo_tcp" "${GLOO_REPEATS}" "${GLOO_WARMUP_REPEATS}" "${INLINE_BATCH_SIZE}" "${INLINE_INPUT_LEN}" "${INLINE_OUTPUT_LEN}"
stop_server

start_server "${MQ_INLINE_ARTIFACT}" 1 1 1
run_repeat_bench "${MQ_INLINE_ARTIFACT}" "mq_shm_inline" "${INLINE_REPEATS}" "${INLINE_WARMUP_REPEATS}" "${INLINE_BATCH_SIZE}" "${INLINE_INPUT_LEN}" "${INLINE_OUTPUT_LEN}"
stop_server

start_server "${MQ_OVERFLOW_ARTIFACT}" 1 1 1
run_repeat_bench "${MQ_OVERFLOW_ARTIFACT}" "mq_zmq_overflow" "${OVERFLOW_REPEATS}" "${OVERFLOW_WARMUP_REPEATS}" "${OVERFLOW_BATCH_SIZE}" "${OVERFLOW_INPUT_LEN}" "${OVERFLOW_OUTPUT_LEN}"
stop_server

"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/myelon/summarize_transport_comparison.py" \
  --gloo-artifact "${GLOO_ARTIFACT}" \
  --mq-inline-artifact "${MQ_INLINE_ARTIFACT}" \
  --mq-overflow-artifact "${MQ_OVERFLOW_ARTIFACT}" \
  --output-md "${COMPARISON_MD}" \
  --output-json "${COMPARISON_JSON}"

echo "Completed transport benchmark: ${ARTIFACT_ROOT}"
