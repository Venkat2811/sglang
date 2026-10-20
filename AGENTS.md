# Agent instructions for branch `codex/k3-rust-frontend-router`

This branch builds a Rust frontend + cache-aware, tier-aware router for prefill/decode and aggregated SGLang serving of
hybrid models (Kimi-K3: MLA + KDA linear attention, MoE), on 8-GPU Blackwell nodes. The full task plan, cluster runbook
and milestone list are given to you out of band (local folder named in your starting prompt). Read them before coding.

## Ground rules
- Work only on this branch. Small commits, each one building and passing tests. Push often. No pull requests.
- This repository is PUBLIC. Commit only code, tests and generic engineering docs. Never commit plans, run logs,
  benchmark results, hostnames, cluster or namespace names, registry paths, company or customer names. Commit messages
  describe the change and, where useful, the hardware ("8x B300").
- Test first: every behaviour change comes with a test that fails before the change and passes after. Run the new test
  on the old code once and keep that evidence in your records.
- Do not trust a health endpoint by itself: readiness means one real request through the full path succeeded.
- Prefer reading code over guessing; cite file:line in your records when you rely on behaviour.

## Where things are
- Router: `experimental/sgl-router` (identical to upstream main as of this branch's start, plus the base below).
- K3 chat rendering / tokenization: `rust/sglang-processor` (Dynamo frontend crates as dependencies; a fork branch for
  them is named in your plan).
- Rust HTTP server inside the engine: `rust/sglang-server` (built into
  `python/sglang/srt/rust_extensions/_server.cpython-312-x86_64-linux-gnu.so`).
- Engine KV events: `python/sglang/srt/disaggregation/kv_events.py` and the radix/HiCache emitters.
- Base of the branch: upstream main 3c560089e1 + per-rank host-pool budget, DCP pack device, DCP-aware HiCache L3
  (#41693 port, #41694, #41695), K3 chat for the Rust server; then five upstream router commits (#44402, #45120, #45100,
  #45102, #45103). Engines you test against run this same tree.

## Build and test
- Router (pure Rust, toolchain pinned by `rust-toolchain.toml`):
  - `cd experimental/sgl-router && cargo build --locked --release`
  - `cargo test --locked --release --workspace -- --skip parity_matrix`
  - `cargo clippy --locked --all-targets -- -D warnings` and `cargo +nightly fmt --check`
  - Existing harnesses to extend: `tests/proxy/common/mock_worker.rs` (fake HTTP workers: hanging, slow stream, errors),
    `tests/component/policies/zmq_helpers.rs` (fake KV-event publisher), `tests/fixtures/kimi_k3`.
- Rust server extension: inside an engine container of the matching image,
  `cd /sgl-workspace/sglang/rust && PYO3_PYTHON=$(command -v python3) cargo build --release --locked -p sglang-server`,
  then stop the engine, copy `target/release/libsglang_server.so` over the `_server` extension file, restart the engine
  from `/` (not from `/sgl-workspace`).
- Python engine edits: copy the changed files into the running container's `/sgl-workspace/sglang` and restart.
- Never `pip install` / `cmake --install` into a shared container's Python environment.
