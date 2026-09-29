# SQD inference prototype

This opt-in runner tests subquadratic decode disaggregation with real
Kimi-Linear weights. One GPU pool runs full prefill and retains MLA KV. A
second pool receives request-owned KDA state and runs KDA, FFNs and sampling.
Each MLA decode layer executes on the first pool through paired NCCL calls.
The second pool's MLA modules are replaced, so they cannot silently run locally.

Both pools use the same TP/DCP layout and a fixed batch in eager mode. The
runner does not implement an HTTP service, dynamic admission, cancellation,
speculation, CUDA graphs, cross-layout transfer, multi-node launch or Kimi-K3
AttnRes. These are later integration steps, not supported configurations.
It allocates normal SGLang pools on both sides; unused decoder KV allocation
has not been removed. Do not use this prototype to claim memory savings.

## Run

Use at least `2 * TP` visible GPUs. The tested hardware is B300 with NVLink.
The public model downloads on first use; allow about 100 GB of model disk
space, plus space for full-vocabulary logits in the result directories.

Start a fresh container on a host with NVIDIA Container Toolkit and a CUDA
13-compatible driver:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -it \
  lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9 bash
git init sglang-sqd
cd sglang-sqd
git remote add origin https://github.com/Venkat2811/sglang.git
git fetch --depth 1 origin 25f9a07a4e548817cdbd353ea0ffe52d6e3fc6ae
git checkout --detach FETCH_HEAD
python -m pip install 'sglang-kernel==0.4.7'
python -m pip install --no-deps --no-build-isolation -e python
```

This gives PyTorch 2.13.0, FlashInfer 0.6.18 and Transformers 5.12.1 in the
tested image. Then run:

```bash
python test/manual/sqd/test_contract.py
python test/manual/sqd/test_execution.py
python -m sglang.benchmark.sqd \
  --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --revision e1df551a447157d4658b573f9a695d57658590e9 \
  --tp-size 1 --dcp-size 1 --output-tokens 16 \
  --output-dir /tmp/sqd-tp1-dcp1
```

Increase TP/DCP only within the visible GPU count. Output directories must be
new. `--cases cases.json` accepts a JSON list of prompt batches, for example:

```json
[["Calculate 17 plus 25."], ["Explain photosynthesis briefly.", "Name two prime numbers."]]
```

For each batch, the runner first generates an ordinary full-model baseline,
then reruns prefill and performs split decode. It independently generates the
split tokens: baseline tokens/logits are only read by the comparison after
all workers finish. Generation uses a fixed length, including past EOS.

Success requires finite logits, identical greedy token IDs and maximum
full-vocabulary logit error at most 0.05. `comparison.json` records the exact
error; `.pt` files retain both logits and token IDs. Per-rank JSON records
remote call counts and payload bytes, and text files show generated output.
The expected call count is `MLA layers * (output tokens - 1)` on every rank.
No MLA KV tensors are sent in the prefill-to-decode handoff.

The timings include correctness instrumentation and cold starts. They are
not a performance benchmark or evidence of an energy-efficiency gain.
The launcher terminates its workers on failure or deadline; reconnects are
not supported. Keep failed attempts alongside successful results.

## Verified prototype results

On 2026-09-29, using the pinned model above, BF16 KV, FlashInfer MLA and
Triton KDA, every tested prompt produced identical greedy IDs and **zero
full-vocabulary logit difference** against its ordinary full-model baseline.
TP/DCP below applies to each of the two pools.

| TP / DCP | Total GPUs | Prompt runs | Fixed output steps per prompt | Runtime revision |
|---|---:|---:|---:|---|
| 1 / 1 | 2 | 3 | 16 | `5a1a9727b6` |
| 2 / 2 | 4 | 5 | 64 | `5a1a9727b6` |
| 4 / 4 | 8 | 5 | 64 | `25f9a07a4e` |
| 4 / 2 | 8 | 5 | 64 | `25f9a07a4e` |

The later revision adds input validation, model revision pinning and explicit
communicator teardown. Its tested source files were verified by hash. Batches
include unequal prompt lengths, successive slot reuse and a 1067-token prompt.
Each rank executed seven remote MLA calls per decode step. A separate negative
control zeroed received recurrent state: token IDs diverged and the oracle
rejected both batches (maximum logit errors 22.46 and 38.94). Five CPU tests and
the applicable pre-commit checks also passed. These are functional results;
throughput, latency and energy gains have not been qualified.

The five-prompt batches used for TP2/TP4 can be recreated as follows:

```bash
python - <<'PY'
import json
from pathlib import Path
cases = [
    ["What is the capital of France? Answer briefly."],
    ["Calculate 17 plus 25.", "Explain in one sentence why the sky looks blue."],
    ["The secret code is BLUEBIRD.\n"
     + "The library holds books about mountains, rivers, astronomy and music.\n" * 80
     + "What is the secret code? Answer with only the code.",
     "Write a Python function that returns the square of an integer."],
]
Path("/tmp/sqd-cases.json").write_text(json.dumps(cases))
PY
python -m sglang.benchmark.sqd \
  --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --revision e1df551a447157d4658b573f9a695d57658590e9 \
  --tp-size 4 --dcp-size 4 --output-tokens 64 \
  --cases /tmp/sqd-cases.json --output-dir /tmp/sqd-tp4-dcp4
```

For the other layouts, change TP/DCP and use a new output directory. The
TP1/TP2 observations above preceded the validation/teardown changes; the pinned
revision is the later runtime exercised by both TP4 layouts.
