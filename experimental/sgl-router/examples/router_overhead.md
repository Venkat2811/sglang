# HTTP frontend overhead benchmark

Build with `cargo build --locked --release --example router_overhead`.

Run `router_overhead serve --model MODEL` to start one prefill and two decode
HTTP mocks on ports 18200, 18201 and 18202. Point the frontends being compared at
those workers. Each decode stream has a 10 ms initial delay and emits 40 output
events per second. The prefill returns immediately; this measures the HTTP
frontend and P/D dispatch paths without GPU execution or KV transfer.

Supply local JSONL requests with one object per line:

```json
{"band":"small","body":{"model":"MODEL","messages":[{"role":"user","content":"Hello"}]}}
```

Keep workload data outside the source tree. Each request receives an independent
request ID, and the mock correlates its service timestamps through that header.
The client forces streaming and the requested output limit, retaining the other
body fields. Both OpenAI output-limit spellings and native generate are supported.
Each active slot leases its own reusable HTTP client through response EOF, so a
shared HTTP/1 pool cannot queue a slot behind another active stream. Pool setup
is outside the timed interval; keep-alive connections remain enabled.

```sh
router_overhead run --router http://127.0.0.1:8000 \
  --workers http://127.0.0.1:18200 --workload /tmp/requests.jsonl \
  --out /tmp/overhead --router-pid ROUTER_PID \
  --requests 512 --concurrency 48 --output-tokens 128
```

The run emits client and worker timing records plus a per-band summary. Added
TTFT subtracts the actual mock service interval from client TTFT. Added chunk
gap subtracts the corresponding mock emission gap from the client arrival gap.
Timestamps use Unix nanoseconds so independent router-metric samples can be
joined to requests. Check for clock discontinuities when comparing processes.
Worker records include exact active counts at arrival and emission, and the
stream guard's completion timestamp. The observation includes transport,
dispatch and body preparation.
CPU/request uses the frontend process's Linux user and system CPU counters;
isolate a band when attributing CPU to that band, and include the relevant server
process when a launcher spawns children. Keep the CPU allocation fixed across modes.

Use `--metrics-url URL` for the candidate admission gauge, and `--unpaced` for a
large completion/permit check. A run fails on HTTP/SSE errors, missing correlation,
an output count different from the requested limit, active mock streams or a
nonzero candidate admission gauge. Input-ID and native-path counts expose which
forwarding path was exercised. Run separate long-output cases up to 8192 events
to observe sustained streaming; a short-output table alone does not cover that case.
