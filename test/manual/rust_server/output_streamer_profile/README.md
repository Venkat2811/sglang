# Decode output-stage measurements

This opt-in import hook measures the real scheduler output producer and its
Python-to-Rust bridge at a chosen batch size. It does not replace the scheduler,
construct synthetic output frames, synchronize the GPU, or alter token counts.

Start the engine with this directory on `PYTHONPATH`, set
`OUTPUT_STREAMER_BENCH_DIR` to a new output directory and optionally set
`OUTPUT_STREAMER_BENCH_BATCH_SIZE` (default 256). Retain the same instrument,
engine arguments, prompt and output workload for before/after measurements.

Each scheduler PID writes one JSONL file. `streamer_ns` measures
`_stream_output_generation`, including serialization and Rust queue backpressure.
`bridge_ns` measures `RustServer.push_generation`. Request classification and
file writes sit outside the outer timer; the nested timer bookkeeping remains
inside it. `first`, `finished`, `cached`, `detail_entries` and
`emitted_requests` allow first-output, middle and terminal batches to be reported
separately. Differences between successive `at_ns` values also include the rest
of the decode loop, overlap and any intervening batches; they are not isolated
GPU kernel timings. Rust consumer execution is asynchronous and is not part of
the enqueue timer.

Use only runs that actually produce enough observations at the requested batch
size. Keep errors, rejected requests and smaller batches visible in the client
and engine logs; an empty profile is not a successful measurement. Verify normal
HTTP generation and cache metadata independently. Stop the engine before
installing a different Python/extension pair, then restart with a new output
directory. Remove this directory from `PYTHONPATH` for normal serving.
