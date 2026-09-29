# SQD performance check (2026-09-29)

**No repeatable throughput, latency or energy saving is established.**
Coordinated CUDA graphs remove most prototype launch overhead, but native
execution benefits too. At the tested batches and contexts, splitting MLA
across another identical GPU pool provides no meaningful speedup.

## Qualified graph comparison

B300 SXM6 AC GPUs in one NVSwitch node, unchanged 1100 W limits,
Kimi-Linear-48B-A3B-Instruct revision
`e1df551a447157d4658b573f9a695d57658590e9`, BF16, FlashInfer MLA, Triton KDA.
Runtime: `f8e9773f2fa471fa4bfa5f4566f32994961e0e0a`.
The tested FlashInfer configuration uses its fa2 MLA fallback on B300;
this does not establish the strongest result across all native backends.
[README.md](README.md) pins the image, installation, prompts and commands.

Each corrected-source launch has two warmups and three measured trials.
Every request generates 128 tokens before EOS; decode timing covers the last
127. Prefill and recurrent-state handoff are separate. Cells show measured
medians, not service-latency percentiles. Positive TPOT change means SQD is
slower. These are fixed-batch model-runner results, not HTTP goodput.

For TP4/DCP4, both modes count eight GPUs for energy: native uses four plus
four idle; SQD uses 4+4.

| Batch / input tokens | Native TPOT, ms | SQD TPOT, ms | SQD TPOT change | Native J/token | SQD J/token |
|---|---:|---:|---:|---:|---:|
| 1 / 29 | 2.674 | 2.645 | −1.1% | 6.140 | 6.802 |
| 4 / 7832–7834 | 3.015 | 3.122 | +3.6% | 2.008 | 2.122 |
| 8 / 982–983 | 3.440 | 3.541 | +2.9% | 1.196 | 1.310 |

The 1% B1 observation is too small to call a win: independent corrected
native launches varied by about 1–2%. These are one launch per mode after
both fixes, not three independent launches.

For TP2/DCP2, both modes count four GPUs: native uses two plus two idle;
SQD uses 2+2. Longer prompts are repetitive retrieval/explanation cases.

| Batch / input tokens | Native TPOT, ms | SQD TPOT, ms | SQD TPOT change | Native J/token | SQD J/token |
|---|---:|---:|---:|---:|---:|
| 1 / 29 | 3.047 | 3.085 | +1.2% | 3.550 | 4.063 |
| 1 / 32752 | 3.279 | 3.349 | +2.1% | 4.519 | 4.526 |
| 4 / 32752–32753 | 3.822 | 3.969 | +3.9% | 1.456 | 1.532 |
| 1 / 131059 | 3.631 | 3.740 | +3.0% | 4.694 | 5.100 |

An additional native TP4/DCP4 control uses all four budgeted GPUs:

| Batch / input tokens | Native TP4 TPOT, ms | SQD 2+2 TPOT, ms | SQD TPOT change |
|---|---:|---:|---:|
| 1 / 29 | 2.659 | 3.085 | +16.0% |
| 1 / 32752 | 2.837 | 3.349 | +18.0% |
| 4 / 32752–32753 | 3.143 | 3.969 | +26.3% |
| 1 / 131059 | 3.127 | 3.740 | +19.6% |

At B4/~32K, two native TP2 replicas split the same four requests and take
3.485 ms TPOT (1,148 tokens/s). Native TP4 takes 3.143 ms (1,273 tokens/s);
SQD 2+2 takes 3.969 ms (1,008 tokens/s). The single TP4 model is the fastest
of these measured four-GPU controls. Replica energy is 1.783 J/token, versus
1.532 for native TP4 and 1.532 for SQD, subject to counter granularity.

Different TP/batch layouts can change continuations, so the TP4 control is
not the strict numerical oracle for TP2 SQD. All accepted runs retain finite
final logits and repeat-identical token sequences within their own layout.

Energy integrates whole-GPU NVML counters, including idle GPUs, without idle
subtraction; host CPU energy is excluded. Graph decode intervals are only
about 0.3–0.5 seconds, so counter granularity makes J/token coarse. Keep the
raw samples; no precise energy percentage or repeatable power saving follows.
Do not compare energy values across the four- and eight-GPU budgets directly.

## Correctness and fixes

All five trials in the accepted TP4 split launch match **8,320 generated IDs**
against native TP4. The TP2 split launch matches **4,480 IDs**, including the
128K case. Saved final full-vocabulary logit differences are exactly zero.
Measured-only ID counts are 4,992 and 2,688; the larger counts include warmups.
Native TP2 eager and graph runs also match IDs and final logits exactly on
all four bounded cases. Replay is required and counted on every rank.
This timing harness saves only final logits; earlier functional tests
separately compare every step's full logits in four TP/DCP layouts.

Two focused one-GPU regressions fail before their fixes and pass after:

- MLA planning must wait before reusing pinned host staging memory still
  consumed by an asynchronous plan copy (`9801189d7f`).
- KimiMoE must copy the shared-expert input before its routed side stream can
  overwrite the original (`f8e9773f2f`).

The benchmark releases captured graphs before NCCL communicator destruction
and records actual replay counts. An early teardown failure and a numerically
mismatching pre-fix split run are retained but excluded from these tables.
Successful replay or coherent text alone is not numerical qualification.
The five manual CPU tests and the applicable pre-commit/registry checks pass.
GitHub CI has not been run.

## Earlier eager results and stopping decision

The original same-layout eager comparison was numerically exact, but its
apparent gain did not repeat: SQD moved from about 18 ms to 26 ms TPOT across
launches while native stayed around 21–23 ms. Both are now superseded for the
performance decision by the corrected graph-against-graph comparison above.
Comparing new SQD graphs with old native eager execution would be misleading.

Seven remote MLA round trips retain within-request dependencies. A first-order
TP4 traffic model puts MLA at about 2–7% for the short cases and 9–20% for the
new long cases, under uniform-routing and ideal-reuse assumptions. Those are
modeled byte fractions, not measured kernel time. Independent-microbatch
pipelining, capacity pressure and heterogeneous hardware could change the
tradeoff; they have not been implemented or qualified.

Stop expanding this homogeneous small-batch optimization path. No tested
configuration approaches the chosen approximate 10% repeatable gain target.
A smaller checkpoint with the same active experts and seven MLA boundaries
does not, by itself, address this result. HTTP integration and explicit
Python/Rust TreeCore benchmarks remain conditional on a justified workload
and a repeatable win. No Kimi-K3, speculation, multi-node, concurrent-prefill
or memory-capacity benefit is claimed.
