# SGL Model Gateway E2E Local Loop

This test harness supports heavyweight multi-model coverage, but the fastest
local iteration path on a single GPU should stay small and reproducible.

## Recommended Local WS / Responses Smoke Setup

Hardware target:

- 1x RTX 3060 with 12 GB VRAM

Environment:

```bash
export HF_HOME=/home/venkat/.cache/huggingface
export SHOW_WORKER_LOGS=1
export SHOW_ROUTER_LOGS=1
```

Install local editable packages with `uv` so the router subprocesses see current
gateway and `sglang` code:

```bash
cd sgl-model-gateway/e2e_test
uv pip install -p .venv/bin/python -e ../../python -e ../bindings/python
```

Optional:

```bash
# Use local unpacked model paths when available.
export ROUTER_LOCAL_MODEL_PATH=/path/to/local/model/root
```

## Small-Model Smoke Tests

Preferred local smoke model:

- `qwen-0.5b` -> `Qwen/Qwen2.5-0.5B-Instruct`
- `qwen-3b` -> `Qwen/Qwen2.5-3B-Instruct` for local function-calling work

Run the local Responses smoke tier:

```bash
cd sgl-model-gateway/e2e_test
uv run --python .venv/bin/python \
  pytest --workers 1 --tests-per-worker 1 responses/test_local_smoke.py
```

Run a single test first if startup or model readiness is uncertain:

```bash
cd sgl-model-gateway/e2e_test
uv run --python .venv/bin/python \
  pytest --workers 1 --tests-per-worker 1 \
  responses/test_local_smoke.py -k basic_response_creation
```

Run the real WebSocket smoke path:

```bash
cd sgl-model-gateway/e2e_test
uv run --python .venv/bin/python \
  pytest --workers 1 --tests-per-worker 1 \
  responses/test_local_smoke.py -k websocket_response_create
```

## WS Microbenchmark

Run the lightweight local WS benchmark harness:

```bash
cd sgl-model-gateway/e2e_test
uv run --python .venv/bin/python \
  pytest --workers 1 --tests-per-worker 1 benchmarks/test_ws_microbench.py -q
```

Optional profile overrides:

```bash
export SGLANG_WS_BENCH_CONCURRENCY=1,2,4
export SGLANG_WS_BENCH_SAMPLES_PER_CONCURRENCY=2
```

## HTTP vs WS Comparison

Run the apples-to-apples HTTP SSE versus WebSocket comparison on the same local
prompt shape and model:

```bash
cd sgl-model-gateway/e2e_test
export SGLANG_HTTP_WS_COMPARE_SAMPLES=1
uv run --python .venv/bin/python \
  pytest --workers 1 --tests-per-worker 1 \
  benchmarks/test_ws_microbench.py -k http_vs_ws_transport_compare -q
```

## Notes

- The larger default e2e models such as `llama-8b`, `qwen-14b`, and `gpt-oss`
  are not intended to be the first green path on a 12 GB developer GPU.
- `qwen-3b` is the best current local candidate for tool-call semantics on a
  single RTX 3060-class machine.
- Keep the small-model smoke path green before expanding to heavier semantic
  or compatibility coverage.
- Normal pytest exit now tears down the pooled local worker automatically.
- If a run is hard-killed, kill stale `sglang.launch_server` processes before
  rerunning if the 3060 appears unexpectedly full.
