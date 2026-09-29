# SQD performance check (2026-09-29)

**No repeatable throughput, latency or energy saving is established.** The
first SQD launch beat the eager control; the repeat did not. This is a
correctness prototype with substantial execution overhead, not an optimized
serving implementation.

## Matched eager comparison

Eight B300 SXM6 AC GPUs in one NVSwitch node, unchanged 1100 W limits,
Kimi-Linear-48B-A3B-Instruct revision
`e1df551a447157d4658b573f9a695d57658590e9`, BF16, FlashInfer MLA, Triton KDA.
See [README.md](README.md) for the pinned image, dependencies, case generator
and full/split/replica commands. Eager harness revision:
`2fafcbbaf48203bf8d1c6d7a2c148929dcbba6a2`.

Every request generates 128 tokens before EOS. Decode timing covers the last
127 tokens; prefill and recurrent-state handoff are separate. Two warmups
precede five measured trials in launch A and three in reverse-order launch B.
Cells show the two launch medians separately. Tokens/s is batch throughput;
TPOT is per request. Energy counts all eight GPUs, including four idle GPUs
in the native TP4 control, with no idle subtraction or host energy.

| Batch / input tokens | Native TP4/DCP4 TPOT, ms (A / B) | SQD 2×TP4/DCP4 TPOT, ms (A / B) | Native tokens/s (A / B) | SQD tokens/s (A / B) | Native J/token (A / B) | SQD J/token (A / B) |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 29 | 21.50 / 21.56 | 18.22 / 26.50 | 46.51 / 46.38 | 54.89 / 37.74 | 39.19 / 39.07 | 37.37 / 53.42 |
| 4 / 7832–7834 | 22.05 / 21.67 | 18.45 / 26.29 | 181.45 / 184.56 | 216.81 / 152.16 | 10.43 / 10.28 | 9.76 / 13.69 |
| 8 / 982–983 | 21.77 / 21.37 | 18.15 / 25.92 | 367.40 / 374.43 | 440.84 / 308.61 | 5.36 / 5.26 | 5.03 / 7.03 |

SQD's B8 average GPU power was 2.21 / 2.17 kW, versus native's 1.97 / 1.97 kW.
It did not reduce watts. Faster completion gave the first run a small
energy/token gain, which reversed in the repeat. Counter granularity limits
precision; these are not whole-system energy measurements.

Native TP8/DCP8 controls measured 22.01–23.38 ms TPOT across the two launches.
Two native TP4/DCP4 replicas, dividing the same batch, measured 23.32 ms at
B4 and 22.99 ms at B8. Different TP/batch layouts change continuations, so
only same-layout comparisons provide a strict numerical oracle.

## Correctness and control qualification

Both eager SQD launches match all 13,312 measured generated token IDs against
the native TP4 reference. Saved final full-vocabulary logits are exactly
equal. Earlier functional runs separately checked every step's full logits
in four TP/DCP layouts. All main runs have finite saved logits and no EOS;
each SQD rank executes seven remote MLA calls per decode step.

Native graph controls at harness revision
`b41c578785ab035d284ed7ba52e34ed60cea63af` require replay, but are **provisional**.
They measured 2.5–3.4 ms TPOT and produce coherent text, yet repeated
continuations vary. An untimed audit with identical prescribed prefixes
finds large graph/eager logit differences, also at DCP1 and with custom
all-reduce disabled. No SQD transport runs in that audit. Root cause is
unresolved; do not use those timings to claim a qualified SQD slowdown or
attribute the discrepancy to an upstream bug without further isolation.

## Interpretation and next gate

A diagnostic B1 trace showed 735 kernels per native eager step, with about
3.06 ms of kernel intervals in a 27.27 ms instrumented step. The split MLA
role spent most of its time in NCCL waiting for its peer. Profiling perturbs
execution and does not explain all cross-launch variation.

A first-order TP4 parameter/state/KV traffic model estimates MLA at only
2–7% of traffic for these workloads. Moving it to four additional identical
GPUs preserves serial dependencies and leaves the KDA/FFN pool dominant.
Seven paired NCCL round trips plus token IDs take roughly 0.14–0.20 ms in
an isolated four-pair calibration; raw NVLink bandwidth is not the main
constraint at these payload sizes. This does not rule out gains with longer
contexts, better pipeline balance, capacity pressure or asymmetric hardware.

First isolate the graph-control discrepancy, then reduce host submission
and synchronization overhead and evaluate independent-microbatch overlap.
Require repeatable gains against validated graph/replica controls before
native HTTP integration and SGLang serving benchmarks. Explicit Python/Rust
TreeCore lanes belong to that later stage: this runner disables radix caching
and provides no Rust TreeCore coverage. No Kimi-K3, multi-node, capacity or
HTTP goodput claim follows from these measurements.
