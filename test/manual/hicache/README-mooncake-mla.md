# Manual Mooncake MLA inference witness

`test_mooncake_mla_inference.py` verifies fixed-topology MLA cache reuse using
real inference and a private native Mooncake TCP store. It requires the
Mooncake Python binding, `mooncake_master` on PATH, the source-compatible
SGLang runtime, and enough visible GPUs for the selected TP size. It downloads
the pinned DeepSeek-V2-Lite-Chat checkpoint on first use.

Run from the repository root, using a **new output directory** each time:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_SMOKE_TP=8 MOONCAKE_SMOKE_DCP=8 \
MOONCAKE_SMOKE_OUTPUT_DIR=/tmp/mooncake-mla-tp8-dcp8 \
python -m pytest test/manual/hicache/test_mooncake_mla_inference.py -q -s
```

For the replicated MLA shard case, use four visible GPUs and set TP=4, DCP=2.
TP8/DCP8 uses eight GPUs total. The engines run sequentially on the same
allocation; no second node or existing storage service is required. Leave
TP/DCP unset for the original TP1/DCP1 witness.

## What must pass

1. Two fresh cache-disabled engines establish the numerical reference.
2. A cold writer produces matching outputs and publishes all required shard
   objects. Native GET checks each physical payload size. A separate 2 GiB
   donor retains the objects after the writer exits.
3. A fresh reader must reproduce the outputs and attribute exactly the expected
   complete logical pages to storage. Every TP rank must report the same
   positive restore-prefix histogram.
4. For DCP>1, remove one shard of the second logical page of the longest C1
   prompt. Verify the other objects remain, then start another fresh reader.
   All ranks must stop at the first logical page and recompute correctly.

The physical page is 64 tokens; the logical page is `64 * DCP`. Twelve prompt
lengths at C1/C4 straddle physical, logical and chunk boundaries. The four main
phases contain 96 scored requests; the DCP fault adds one request. Raw requests,
responses, logs, phase metrics, storage sizes, runner hash, prompt hash and
summary are saved in the output directory. Server warmup is additional.

## Numerical controls and scope

DCP1 defaults to deterministic Triton and eight generated tokens. DCP>1 defaults
to FlashInfer, matching the existing file-backed DCP gate, and scores the
complete six-token access-code answer. Each answer must equal the independently
specified expected text. Output IDs must match the same-topology cold reference
exactly; each finite output-token logprob must be within **0.05** absolute error.
The test never changes that tolerance after observing restored outputs.

The September 28 cold controls exposed two limitations before any L3 test:

- Deterministic Triton MLA at DCP8 failed during warmup: its write path supplied
  `dcp_kv_mask` to an MLA pool API that rejects it.
- Ordinary FlashInfer C4 could diverge in the unconstrained continuation after
  the answer. All six answer tokens matched, but an eight-token oracle failed.
  The six-token answer contract was selected before any L3 observations and
  then validated with two new cold runs. This is not a claim of general batch
  invariance or arbitrary-length continuation parity.

`MOONCAKE_SMOKE_MODE=cold_repeat` runs only the two cold phases for diagnosing
an oracle failure. `MOONCAKE_SMOKE_PROFILE` can explicitly select `deterministic`
or `flashinfer`; an override does not imply that backend supports every topology.

This fixture covers aggregated pure MLA, BF16, eager execution, `ag_rs`, TCP,
page-first/kernel copies and wait-complete prefetch. It does not qualify hybrid
KDA/Kimi, live P/D, runtime attachment, graph replay, RDMA, speculation, or
cross-topology persisted-object reuse.

## Observed GPU checks

At source `8405aced7a`, both bounded checks passed on B300 GPUs with the pinned
model revision in the runner, Torch 2.13.0+cu130, SGLang kernel 0.4.7, FlashInfer
0.6.18, Transformers 5.12.1 and Mooncake CUDA13 binding 0.3.13:

| Topology | Scored requests | Required native objects | Fresh-reader logical tokens | Fault prefix, every rank | Maximum reader logprob delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP8/DCP8 | 97 | 240 | 15,360 | 512 | 0.027040 |
| TP4/DCP2 | 97 | 60 | 3,840 | 128 | 0.031756 |

Every required object contained 1,990,656 bytes. All answers and compared IDs
matched. All eight/four ranks agreed, including equivalent replica readers in
TP4/DCP2. Object cardinality alone does not measure duplicate PUT traffic.
The runs took 371.00s and 282.53s respectively, including sequential engine
startups with cached model weights. These are correctness witnesses, not
throughput measurements or a substitute for the broader component test suite.
