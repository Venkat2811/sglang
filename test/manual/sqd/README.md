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

Use an installed checkout of this branch in an SGLang CUDA environment, with
at least `2 * TP` visible GPUs. The initial tested environment is CUDA 13,
PyTorch 2.13.0, FlashInfer 0.6.18, Transformers 5.12.1 and sglang-kernel 0.4.7.
The public model downloads on first use; allow about 100 GB of model disk space.

```bash
python test/manual/sqd/test_contract.py
python test/manual/sqd/test_execution.py
python -m sglang.benchmark.sqd \
  --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --revision e1df551a447157d4658b573f9a695d57658590e9 \
  --tp-size 1 --dcp-size 1 --output-tokens 32 \
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
