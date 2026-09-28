# L3 backend coverage, SSD verification, and hybrid/P-D gates

Updated 2026-09-28. File and Mooncake are required backends for this feature. Both file-backed SSD and Mooncake SSD offload are mandatory acceptance lanes.
Storage identity, logical/physical page geometry, rank agreement, checkpoint
selection and buffer lifetime belong to the shared HiCache contract. Each
adapter must demonstrate that contract; enabling one does not qualify every
registered L3 backend.

## Current evidence

| Path | Implemented / available | Verification still required |
| --- | --- | --- |
| Pure MLA + DCP + Mooncake DRAM | Enabled; TP8/DCP8 and TP4/DCP2 inference passed | Other explicitly selected layouts/policies and runtime attachment |
| Pure MLA + DCP + file | Enabled; TP8/DCP8 and TP4/DCP2 SSD inference passed | Other explicitly selected layouts/policies and runtime attachment |
| File directory on SSD | TP1/DCP1, TP8/DCP8 and TP4/DCP2 pressure, physical SSD reads and missing-shard fallback passed | Independent durability/crash recovery qualification |
| Mooncake SSD offload | TP1/DCP1, TP8/DCP8 and TP4/DCP2 pressure, SSD-only retrieval and missing-shard fallback passed | Independent owner/master restart recovery |
| Hybrid KDA + MLA + DCP + L3 | Component contracts implemented; public enablement still guarded | Composed checkpoint capture/publication/restore and real model continuation, for each backend |
| Live P/D with role-local L3 | Existing P/D fixtures and implementation paths | Explicit cache-off baselines, role-local restores and their composition with live transfer |

The earlier 194 passing requests were **Mooncake DRAM** tests. The separate
SSD results below now exercise both backends. See also
[the DRAM results](README-mooncake-mla.md).

## File backend

The branch accepts `--hicache-storage-backend file` under DCP for a supported
single materialized MLA pool. The path is selected by
`SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR`; the default is `/tmp/hicache`.
Point it at a dedicated directory on the intended SSD filesystem. A path name
or Kubernetes ephemeral volume alone is not evidence of the underlying medium.
Different nodes can reuse these files only when they access the same backing
filesystem and compatible namespace.

The existing narrow GPU entry point is:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DCP_L3_TOPOLOGIES=8:8,4:2 DCP_L3_SKIP_CONCURRENCY=1 \
DCP_L3_OUTPUT_DIR=/tmp/file-dcp-results \
python -m pytest test/manual/hicache/test_dcp_l3.py -q -s
```

That inherited fixture uses an internally selected temporary storage directory,
overriding the storage-directory environment variable for its engines. It is
a file correctness gate, **not an SSD-specific launch recipe**. It uses one
prompt and injects a missing shard only for TP4/DCP2. The shared
`test_hicache_ssd_pressure.py` runner below supplies pinned revisions, paired
cold controls, explicit owned storage roots, known answers and missing-shard
checks for both backends at TP8/DCP8 and TP4/DCP2.

Required file cases: fresh reader after writer exit, exact physical page bytes,
all-rank prefix agreement, TP>DCP equivalent readers, missing middle shard,
short/truncated object, and write failure without publishing a complete hit.
Use owned files only. File-backed cache correctness does not establish
power-loss durability or uncached physical-SSD throughput: filesystem page
cache can satisfy reads, and atomic rename is not an fsync guarantee.

SGLang documents the file backend as a simple reference implementation and
explains its node-local/shared-mount scope in the
[HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design).

## Native SSD offload

HiCache's hierarchy remains GPU HBM → instance-local host DRAM → L3 backend.
The L3 backend may use files on SSD, or Mooncake's distributed DRAM and SSD
tiers. `--hicache-io-backend direct/kernel` controls the host/GPU copy path;
it is not a switch for direct SSD I/O or GPUDirect Storage.

The [SGLang Mooncake connector documentation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/README.md)
exposes `enable_ssd_offload` and `ssd_offload_path` through backend extra config.
The [Mooncake deployment guide](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html)
describes master `--enable_offload=true`, optional eviction-triggered offload
and promotion on read. SSD-owning clients/services must initialize their
FileStorage and use dedicated writable paths on the desired medium. Enabling
offload only on a zero-contribution inference client does not establish an
SSD tier for the independent memory donor that owns the objects.

Use deliberately small configured cache pools, not node-wide memory exhaustion. Bound GPU KV and HiCache host capacity so a handful of distinct prompts evict the target from both L1 and L2; independently bound the Mooncake donor DRAM segment so the same working set forces SSD offload. Record effective capacities and tier evidence rather than inferring eviction from request counts. Keep enough host staging capacity for forward progress.

The SSD witness must:

1. Pin the actual wheel/build, enable and verify master and storage-owner
   offload capabilities, and fail if configuration falls back to memory-only.
2. Publish known objects and exceed the configured Mooncake DRAM capacity with
   a bounded working set. Wait for completed disk replicas and memory eviction
   while preserving the disk replicas and owner. Use explicit memory-replica
   removal only as a separate diagnostic, not as proof of pressure-triggered
   offload.
3. Prove disk residency and retrieve exact bytes; a successful GET alone does
   not establish which replica served it.
4. Restore through a fresh inference process and compare against a matched
   cold reference. Check a missing disk shard/checkpoint and safe fallback.

Cache-spill correctness is separate from recovering the store after its owner
or master restarts. The first witness keeps the independent SSD owner alive.
No shared node-wide page-cache flush or destructive device operation is needed.

### Bounded SSD inference witness

`test_hicache_ssd_pressure.py` runs the same answer and cache-tier assertions
against either backend. First independently map the selected filesystem to its
physical SSD device and corresponding container cgroup `io.stat` entry. Supply
that device's major:minor ID; do not assume that `/tmp`, an emptyDir, or a path
containing "ssd" is backed by SSD. The fixture uses Linux `findmnt`, cgroup v2
I/O accounting, and `posix_fadvise` on its own files.

```sh
CUDA_VISIBLE_DEVICES=0 \
HICACHE_SSD_BACKEND=file \
HICACHE_SSD_OUTPUT_DIR=/path/to/new/file-results \
HICACHE_SSD_STORAGE_ROOT=/verified/ssd/new-file-cache \
HICACHE_SSD_BLOCK_DEVICE=259:0 \
python -m pytest test/manual/hicache/test_hicache_ssd_pressure.py -q -s
```

For Mooncake, set `HICACHE_SSD_BACKEND=mooncake` and use different new output
and storage directories. `mooncake_master` and the native Python binding must
support SSD offload and replica inspection. The fixture starts a private
master and a 128 MiB memory donor with its own SSD directory. Inference clients
contribute no store memory. The SSD owner stays alive across engine restarts;
the test does not claim crash recovery.

The fixture also scales SSD buckets down: one key per bucket, an 8 MiB size
limit, and a 1 GiB total bucket quota. With native Mooncake 0.3.13, the default
256 MiB/500-key flush threshold exceeded the entire 128 MiB donor and prevented
spill: the first pressure attempt exhausted DRAM with no disk replicas. A
native reproduction confirmed that smaller buckets allowed pressure offload.
The master starts eviction at 50% occupancy, and writer requests are spaced
by two seconds to allow the one-second offload heartbeat to make progress.
Small-pool tests must scale asynchronous batching as well as DRAM capacity.

Size SSD staging for all concurrent TP readers, including equivalent MLA
readers that share a stored DCP shard. The runner rounds
`target KV bytes * (TP / DCP)` up to a power of two, with a 64 MiB floor.
TP4/DCP2 therefore uses 128 MiB of staging, independently of its unchanged
128 MiB persistent DRAM segment. The initial 64 MiB staging attempt returned
native `BUFFER_OVERFLOW` on two ranks; all ranks safely fell back to zero
storage tokens and recomputed the correct answer. That attempt failed the
cache-hit acceptance check and is preserved as a diagnostic, not a pass.
The distinction between staging and cache capacity is also documented in the
[Mooncake SSD settings](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)
and [vLLM Ascend KV pool guidance](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_pool.html).

The pinned DeepSeek-V2-Lite-Chat run uses TP1/DCP1, BF16, deterministic Triton,
64-token pages, a 2048-token GPU KV cap and HiCache ratio 1.5. Six distinct
1537-token prompts exceed L1/L2 capacity. Two cache-disabled controls precede
the writer. Same-engine replay must restore 1536 storage tokens with zero
host/device hits; a fresh engine must do the same. Removing the second stored
page must limit another fresh reader to 64 storage tokens. Each response must
match the expected answer, exact six cold-reference token IDs, and finite
logprobs within 0.05. These are 21 scored requests per backend.

Mooncake additionally requires complete disk replicas and no memory replicas
for every restored target page. Eviction pressure can include at most 32
private 8 MiB native objects; explicit memory-replica deletion cannot qualify
the pressure test. Target payload sizes and hashes must survive offload. Before
each restore, only owned files are fsynced and advised out of the OS page cache.
The selected SSD's cgroup read-byte increase must cover all restored KV bytes.
Preserve `summary.json`, per-request responses, server arguments/logs, metrics,
disk metadata, runner hash, and the independent mount/device audit.

The default TP1/DCP1 mode verifies tiering. Distributed modes below add shard
agreement; neither mode establishes hybrid state restore or live P/D behavior.

The same runner also accepts `HICACHE_SSD_TP` and `HICACHE_SSD_DCP` (default
1/1). Use `8`/`8` with eight visible GPUs, or `4`/`2` with four, to exercise
distributed SSD restores. DCP runs use FlashInfer with paired cold controls,
the same exact six-token answer, and the unchanged 0.05 logprob tolerance.
Each rank must report the same restored logical prefix. The fault removes only
the last DCP rank's second-page object; all ranks must fall back to one logical
page (512 tokens at DCP8, 128 at DCP2). Equivalent MLA readers in TP>DCP share
the same stored shards.

`--max-total-tokens` limits physical rows per rank. The runner scales that cap
to keep logical capacity small, while retaining room for the rounded prompt
and the allocator's extra decode page. TP8/DCP8 needs 384 physical rows per
rank (3072 logical tokens); its effective L2 is 640 physical rows per rank
(5120 logical tokens). The initial 256-row attempt generated zero tokens in
the cold control because its decode reserve exhausted the tiny pool; it did
not reach storage qualification. Six 1537-token prompts still exceed both
corrected cache capacities. DCP1 retains its original 2048-row cap.

### Measured SSD results, 2026-09-28

Both lanes passed on one B300 with the pinned model above, physical NVMe
storage, kernel 0.4.7, Torch 2.13.0+cu130, and native Mooncake 0.3.13. The
effective HiCache host pool was 3136 tokens (about 93 MiB). Each lane completed
21 scored requests; all answer text and token IDs matched, and the largest
observed output-logprob delta was zero (assertion tolerance remains 0.05).

| Backend | Runtime | Same-engine / fresh-reader storage hits | NVMe bytes per full restore | Missing-page storage hit / NVMe bytes |
| --- | --- | --- | --- | --- |
| File | 287.82 s | 1536 / 1536 tokens | 47,775,744 | 64 tokens / 1,990,656 |
| Mooncake SSD | 258.98 s | 1536 / 1536 tokens | 47,874,048 | 64 tokens / 1,994,752 |

All three restores in each lane reported zero device and host hits. Each target
contained 24 physical KV pages of 1,990,656 bytes. Mooncake initially reported
24 complete memory replicas, then 24 complete local-disk-only replicas after
the six writer prompts. No extra native pressure objects were needed. Exact
payload hashes survived offload. The larger physical read count includes
Mooncake's bucket storage overhead.

File runner SHA256: `1ff23bdf568ae8e23bd08464683521c0e4fba9879b6291e39f03caaadb16c196`.
Mooncake runner SHA256: `bd59be805d39eb0c4463abb417f3e5ba90aa52bc2f319a94b5eb131a56d89406`.
The latter adds the Mooncake bucket/eviction settings described above; the file
restore path is unchanged. Production source was identical for both runs.
This verifies bounded SSD spill/reuse with `write_through` and `wait_complete`;
it does not qualify every policy, DCP topology, hybrid model or P/D role.

### Distributed SSD results, 2026-09-28

The same pinned model and runtime passed 21 scored requests in each lane below
on B300 GPUs: 84 requests total. Every rank agreed on both 1536-token full
restores and the listed missing-shard prefix. All restores reported zero
device/host hits. Known answers and all six output token IDs matched the cold
reference; the output-logprob tolerance remained 0.05.

| Backend / topology | Runtime | Same-engine / fresh-reader NVMe bytes | Missing-shard prefix / NVMe bytes | Largest logprob delta |
| --- | --- | --- | --- | --- |
| File TP8/DCP8 | 358.47 s | 47,775,744 / 47,779,840 | 512 / 15,925,248 | 0.008035 |
| Mooncake TP8/DCP8 | 357.85 s | 47,874,048 / 47,874,048 | 512 / 15,958,016 | 0.008035 |
| File TP4/DCP2 | 271.94 s | 47,775,744 / 47,775,744 | 128 / 3,981,312 | 0.006237 |
| Mooncake TP4/DCP2 | 285.16 s | 47,874,048 / 47,874,048 | 128 / 3,989,504 | 0.006237 |

All 24 target objects in each Mooncake lane became complete disk-only replicas
under the six inference writer prompts; no extra native pressure objects were
needed. Payload sizes and hashes survived offload. The owner/master remained
alive across inference restarts. TP4/DCP2 required the staging adjustment
described above; its failed undersized-buffer run is retained separately.

TP8 production source was `e7fd249387`; TP4 used `730dc17ba2`, including the
file v2 hybrid identity fix. The primary MLA file path was unchanged by that
fix. File TP8/TP4 and Mooncake TP8 used runner SHA256
`d99d6a67d33da4b2e14109b0d3d8de7a8ea9c4f731201c6da7a27f13bbfad901`;
Mooncake TP4 used `d8eca3f5db763a00ce5a352e7a2a5c71348bd5773df0752040144ce74f9c9f6e`.

Related upstream reports reinforce the need for these separate oracles:
[Mooncake #3465](https://github.com/kvcache-ai/Mooncake/issues/3465) reports
listed SSD keys returning empty payloads, while
[#2632](https://github.com/kvcache-ai/Mooncake/issues/2632) reports expired
offload tasks despite apparently working offload. Both were open when reviewed
on 2026-09-28. Our bounded evidence does not resolve those reports: it requires
completed disk replicas, exact bytes, actual storage hits, and model parity,
rather than accepting key discovery or an offload log message alone.

## Next hybrid and live P/D work

### Small live P/D baseline

`test_pd_dcp_baseline.py` launches a monolithic reference, then independent
prefill/decode engines and a router. `PD_DCP_MODEL=mla` uses the pinned
DeepSeek-V2-Lite-Chat checkpoint with P2/DCP1 and D2/DCP2 on four GPUs.
`PD_DCP_MODEL=hybrid` selects pinned Kimi-Linear with P4/EP4 and D4/DCP4 on
eight GPUs, following the existing Kimi P/D fixture's tokenspeed MLA / FP8 KV
configuration. Both fit on one eight-GPU node. A local pinned weight snapshot
can be supplied through `PD_DCP_MODEL_PATH`.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MC_FORCE_TCP=1 MOONCAKE_PROTOCOL=tcp \
PD_DCP_MODEL=mla PD_DCP_OUTPUT_DIR=/path/to/new/pd-results \
python -m pytest test/manual/hicache/test_pd_dcp_baseline.py -q -s
```

The MLA TCP lane passed 30 scored requests on four B200s, with exact six-token
known answers and maximum output-logprob difference 0.031334 against the fixed
0.05 limit. It covered physical-page, DCP logical-page and 1024-token chunk
boundaries, repeated requests and a three-request batch. This is live transfer
evidence with HiCache/L3 disabled. The subsequent runner adds a matched batch
reference rather than comparing batched and sequential numerical schedules.

The initial automatic RDMA lane failed GPU memory registration with
`Bad address [14]`. A native 1 MiB CUDA registration probe succeeded with
`WITH_NVIDIA_PEERMEM=0`, which selects DMA-BUF instead of the default legacy
peer-memory path. The full RDMA follow-up is pending. Select and record the
actual transport; a TCP pass does not qualify RDMA. See Mooncake's
[GPU registration troubleshooting](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/troubleshooting/troubleshooting.md).

Feature enablement still requires the composed hybrid and role-local cache
gates below.

### Kimi DCP4 cold/warm baseline

`test_kimi_linear_dcp_baseline.py` passed 56 requests and 39 numerical checks
in 137.71 seconds on four B300 GPUs. Set `KIMI_DCP_OUTPUT_DIR` to a new
directory and optionally `KIMI_DCP_MODEL_PATH` to a local snapshot of
`moonshotai/Kimi-Linear-48B-A3B-Instruct` revision
`e1df551a447157d4658b573f9a695d57658590e9`.

The BF16/cutedsl MLA baseline follows the registered DCP4 HiCache fixture:
64-token physical pages, 256-token logical pages, a2a and replicated Q
projection. It covers 63/64/65, 255/256/257 and 1023/1024/1025-token prompts,
two short known-answer needles, multi-turn continuation, forced prefill replay
and a three-request batch. The candidate enables a 10 GiB HiCache host pool;
the reference has HiCache disabled. No `max-total-tokens` override is used.

Controls must match the work being compared. Append the reference's first
generated token to make a terminal KDA checkpoint reusable: prefix matching
leaves one input token for logits. Compare that warm request against a cold
request with the same extended input. Seed the same checkpoint-producing
sequence for the multi-turn reference, and compare batches against batches.
The initial witnesses violated these requirements; all failed attempts and
the isolating diagnostics are retained.

All generated IDs matched. The largest absolute logprob difference in the
matched cache comparisons was 0.109716, below the predeclared 0.20 bound.
Multi-turn HiCache and ordinary radix reuse matched exactly. Cold full prefill
versus warm recurrent-state reuse had a 0.271841 maximum logprob difference
even with HiCache disabled; its KL was 0.008006. Forced replay KL was 0.008390.
Those different numerical schedules use the registered Kimi fixture's 0.01
KL criterion. They are not reported as satisfying the absolute 0.20 bound.

Runner SHA256: `621d667d85e6bc95e1160cfeb338a8a282c5db48c579bb1db1441c01c4f94076`.
This qualifies the bounded model/prefix-continuation baseline. Positive hits
were device-cache hits; it does not establish forced L2 load-back, hybrid L3
restore, or live P/D. The hybrid L3 guard stays enabled.

1. **Compose the hybrid restore path.** Drive actual KDA checkpoint capture and
   publication, native storage, common legal boundary selection, radix host
   insertion, GPU load-back and continued generation. Existing byte-copy and
   mocked-forward publication tests cover pieces, not this end-to-end path.
   Prove missing state on one rank falls back to a checkpoint shared by all
   ranks; never relabel a later active state as an earlier prefix checkpoint.
2. **Compose the tested file sidecar contracts.** CPU regressions reproduced
   TP4/DCP2 ranks 0 and 2 restoring the same KDA state, equal-byte incompatible
   schemas hitting, logical MLA v2 indices being rejected, and independent
   checkpoint maxima selecting an absent state. File KDA keys now include TP
   ownership and a capacity-independent tensor schema; v2 uses each pool's
   logical page geometry and returns complete legal checkpoint sets. Tests
   cover both supported state layouts and preserve unselected destination
   bytes. Run `test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py`
   for these contracts. The current hybrid DCP guard remains in place until
   composed publication/restore and actual model continuation are qualified.
3. **Run a small Kimi-Linear baseline and L3 continuation gate.** Use the
   existing `moonshotai/Kimi-Linear-48B-A3B-Instruct` TP4/DCP4 Blackwell fixture
   as the source for model/pool/kernel settings. Pin its model revision and
   first validate the cache-off/L2 controls. Then test a cold writer and fresh
   L3 reader after the composed contract passes. Preserve the established
   numerical oracle; do not reuse the small MLA answer tolerance blindly.
4. **Run a bounded live P4+D4 baseline on one eight-GPU node.** Reuse the
   existing Kimi-Linear P/D topology and boundary prompts with speculation off.
   Its full GSM8K/long-context campaign is not needed for the initial smoke.
   Preserve the native TCP admission settings where applicable and use the
   current graceful-cleanup fixture pattern.
5. **Add role-local L3 to live handoff.** Test P-only first, then D-only and
   both roles, with matched controls. Decode needs the actual decode radix,
   preallocation and restore path enabled; a transfer-only success is not a
   decode L3 pass. Cover equal DCP4 roles and P-DCP1→D-DCP4, plus a fresh role,
   partial prefix and cancellation. Role-local objects remain same-topology;
   live P→D relayout is a separate contract.

One eight-GPU Blackwell node is sufficient for these initial lanes: sequential
four-GPU aggregated engines, then simultaneous P4+D4. CPU and one-GPU state
tests can precede that allocation. Actual Kimi-K3 recipes remain later
qualification: the reduced P8+D16 shape needs three eight-GPU nodes and the
supplied full P/D layouts need four. A Kimi-Linear pass is not a Kimi-K3 pass.
Both SSD lanes are required; the hybrid/P-D baseline can proceed in the same development cycle.
