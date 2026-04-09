# Myelon Measurement Scripts

This folder is the reproducible baseline surface for the `myelon_mq` SGLang
work.

Current scope:

- baseline the existing `shm_broadcast` path on real hardware
- capture machine-readable payload and overflow stats from `shm_broadcast`
- run a TP=8 multi-turn benchmark without HiCache
- run repeated actual-inference TP=8 request-fanout benchmarks on Gloo and MQ
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
- `repeat_bench_one_batch_server.py`
  - drives repeated actual-inference batches against a live server
  - writes per-iteration JSONL plus an aligned summary table
- `summarize_transport_comparison.py`
  - compares Gloo, MQ inline SHM, and MQ overflow/ZMQ artifact directories
  - renders aligned markdown tables for end-to-end and transport metrics
- `run_tp8_request_fanout_benchmark.sh`
  - creates a `uv`-managed venv if needed
  - installs editable `sglang` if needed
  - runs three sequential actual-inference lanes:
    - Gloo TP scheduler fanout
    - MQ inline SHM TP scheduler fanout
    - MQ overflow via ZeroMQ on oversized real requests
  - writes a combined comparison report under `ai-chat-exports/.../2_artifacts/...`

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
- `GLOO_REPEATS`
- `INLINE_REPEATS`
- `OVERFLOW_REPEATS`
- `INLINE_BATCH_SIZE`
- `INLINE_INPUT_LEN`
- `INLINE_OUTPUT_LEN`
- `OVERFLOW_BATCH_SIZE`
- `OVERFLOW_INPUT_LEN`
- `OVERFLOW_OUTPUT_LEN`

## Output Files

- `server.log`
- `bench_multiturn.jsonl`
- `ipc_stats.jsonl`
- `ipc_summary.txt`
- `ipc_summary.json`
- `bench_repeat.jsonl`
- `bench_repeat_summary.txt`
- `bench_repeat_summary.json`
- `ipc_events.jsonl`
- `pyobj_broadcast.jsonl`
- `parallel_state.jsonl`
- `comparison.md`
- `comparison.json`
