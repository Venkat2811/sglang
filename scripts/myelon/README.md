# Myelon Measurement Scripts

This folder is the reproducible baseline surface for the `myelon_mq` SGLang
work.

Current scope:

- baseline the existing `shm_broadcast` path on real hardware
- capture machine-readable payload and overflow stats from `shm_broadcast`
- run a TP=8 multi-turn benchmark without HiCache
- leave the actual transport swap for the next slice

## Scripts

- `run_tp8_multiturn_baseline.sh`
  - creates a `uv`-managed venv if needed
  - installs editable `sglang` if needed
  - launches the server with `MYELON_INSTRUMENT=1`
  - runs `benchmark/hicache/bench_multiturn.py`
  - writes artifacts under `ai-chat-exports/.../2_artifacts/...`
- `analyze_ipc_jsonl.py`
  - summarizes the JSONL stats emitted by `shm_broadcast`

## Important Env Overrides

- `MODEL_PATH`
- `PORT`
- `TP`
- `MEM_FRACTION_STATIC`
- `REQUEST_LENGTH`
- `OUTPUT_LENGTH`
- `NUM_CLIENTS`
- `NUM_ROUNDS`
- `MAX_PARALLEL`
- `REQUEST_RATE`
- `ARTIFACT_DIR`
- `RUN_TAG`
- `SGLANG_USE_MESSAGE_QUEUE_BROADCASTER`

## Output Files

- `server.log`
- `bench_multiturn.jsonl`
- `ipc_stats.jsonl`
- `ipc_summary.txt`
- `ipc_summary.json`
