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

Optional:

```bash
# Use local unpacked model paths when available.
export ROUTER_LOCAL_MODEL_PATH=/path/to/local/model/root
```

## Small-Model Smoke Tests

Preferred local smoke model:

- `qwen-0.5b` -> `Qwen/Qwen2.5-0.5B-Instruct`

Run the local Responses smoke tier:

```bash
cd sgl-model-gateway/e2e_test
pytest --workers 1 --tests-per-worker 1 responses/test_local_smoke.py
```

Run a single test first if startup or model readiness is uncertain:

```bash
cd sgl-model-gateway/e2e_test
pytest --workers 1 --tests-per-worker 1 \
  responses/test_local_smoke.py -k basic_response_creation
```

## Notes

- The larger default e2e models such as `llama-8b`, `qwen-14b`, and `gpt-oss`
  are not intended to be the first green path on a 12 GB developer GPU.
- Keep the small-model smoke path green before expanding to heavier semantic
  or compatibility coverage.
