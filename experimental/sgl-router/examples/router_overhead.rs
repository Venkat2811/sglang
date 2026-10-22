// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Fixed-rate HTTP workers and an SSE client for measuring frontend overhead.

use anyhow::{bail, ensure, Context};
use axum::serve::ListenerExt;
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use bytes::Bytes;
use clap::{Parser, Subcommand};
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Serve {
        #[arg(long, value_delimiter = ',', default_value = "18200,18201,18202")]
        ports: Vec<u16>,
        #[arg(long, default_value = "tiny")]
        model: String,
    },
    Run {
        #[arg(long)]
        router: String,
        #[arg(long)]
        workers: String,
        #[arg(long)]
        workload: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        router_pid: u32,
        #[arg(long, default_value_t = 48)]
        concurrency: usize,
        #[arg(long, default_value_t = 512)]
        requests: usize,
        #[arg(long, default_value_t = 128)]
        output_tokens: usize,
        #[arg(long)]
        metrics_url: Option<String>,
        #[arg(long)]
        unpaced: bool,
    },
}

fn now_ns() -> u64 {
    // Share a time base with independent process and router-metric samples.
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[derive(Clone, Deserialize, Serialize)]
struct Timing {
    delay_ms: f64,
    tokens_per_second: f64,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            delay_ms: 10.0,
            tokens_per_second: 40.0,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct WorkerRecord {
    arrived_ns: u64,
    emitted_ns: Vec<u64>,
    active_on_arrival: usize,
    active_at_emission: Vec<usize>,
    finished_ns: Option<u64>,
    path: String,
    input_ids: bool,
}

#[derive(Clone, Default)]
struct Workers {
    records: Arc<Mutex<HashMap<String, WorkerRecord>>>,
    active: Arc<AtomicUsize>,
    timing: Arc<Mutex<Timing>>,
}

#[derive(Clone)]
struct WorkerState {
    common: Workers,
    prefill: bool,
    model: String,
    bootstrap: u16,
}

struct Active {
    common: Workers,
    id: String,
}
impl Drop for Active {
    fn drop(&mut self) {
        self.common
            .records
            .lock()
            .unwrap()
            .get_mut(&self.id)
            .unwrap()
            .finished_ns = Some(now_ns());
        self.common.active.fetch_sub(1, Ordering::SeqCst);
    }
}

fn output(native: bool, index: usize, total: usize, model: &str) -> Value {
    if native {
        json!({"text":" x".repeat(index + 1),"output_ids":vec![249_u32; index + 1],"meta_info":{"prompt_tokens":1,"completion_tokens":index + 1,"cached_tokens":0,"finish_reason":(index + 1 == total).then(|| json!({"type":"length","length":total}))}})
    } else {
        json!({"id":"mock","object":"chat.completion.chunk","model":model,"choices":[{"index":0,"delta":{"content":" x"},"finish_reason":(index + 1 == total).then_some("length")}]})
    }
}

fn requested_output_tokens(body: &Value) -> usize {
    body["sampling_params"]["max_new_tokens"]
        .as_u64()
        .or_else(|| body["max_completion_tokens"].as_u64())
        .or_else(|| body["max_tokens"].as_u64())
        .unwrap_or(128) as usize
}

async fn generation(
    State(state): State<WorkerState>,
    uri: Uri,
    headers: HeaderMap,
    bytes: Bytes,
) -> Response {
    let arrived_ns = now_ns();
    let Ok(body) = serde_json::from_slice::<Value>(&bytes) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let native = uri.path() == "/generate";
    if state.prefill {
        return Json(json!({"text":"","meta_info":{"prompt_tokens":1,"completion_tokens":1},"output_ids":[249]})).into_response();
    }
    let id = headers
        .get("x-request-id")
        .and_then(|x| x.to_str().ok())
        .map(str::to_owned)
        .or_else(|| body["rid"].as_str().map(str::to_owned))
        .unwrap_or_default();
    let total = requested_output_tokens(&body);
    if total == 0 || total > 8192 {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let active_on_arrival = state.common.active.fetch_add(1, Ordering::SeqCst) + 1;
    state.common.records.lock().unwrap().insert(
        id.clone(),
        WorkerRecord {
            arrived_ns,
            emitted_ns: Vec::with_capacity(total),
            active_on_arrival,
            active_at_emission: Vec::with_capacity(total),
            finished_ns: None,
            path: uri.path().to_owned(),
            input_ids: body.get("input_ids").is_some(),
        },
    );
    let active = Active {
        common: state.common.clone(),
        id: id.clone(),
    };
    let timing = state.common.timing.lock().unwrap().clone();
    let first = tokio::time::Instant::now() + Duration::from_secs_f64(timing.delay_ms / 1000.0);
    let chunks = stream::unfold(
        (0, state, id, active),
        move |(index, state, id, active)| async move {
            if index >= total {
                return None;
            }
            if timing.tokens_per_second > 0.0 {
                tokio::time::sleep_until(
                    first + Duration::from_secs_f64(index as f64 / timing.tokens_per_second),
                )
                .await;
            }
            let data = output(native, index, total, &state.model);
            let wire = Bytes::from(format!(
                "data: {data}\n\n{}",
                if index + 1 == total {
                    "data: [DONE]\n\n"
                } else {
                    ""
                }
            ));
            {
                let mut records = state.common.records.lock().unwrap();
                let record = records.get_mut(&id).unwrap();
                record.emitted_ns.push(now_ns());
                record
                    .active_at_emission
                    .push(state.common.active.load(Ordering::SeqCst));
            }
            Some((
                Ok::<_, std::io::Error>(wire),
                (index + 1, state, id, active),
            ))
        },
    );
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(chunks))
        .unwrap()
}

async fn serve(ports: Vec<u16>, model: String) -> anyhow::Result<()> {
    ensure!(ports.len() == 3, "need one prefill and two decode ports");
    let common = Workers::default();
    for (index, port) in ports.iter().enumerate() {
        let state = WorkerState {
            common: common.clone(),
            prefill: index == 0,
            model: model.clone(),
            bootstrap: ports[0],
        };
        let app = axum::Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route("/health_generate", get(|| async { StatusCode::OK }))
            .route("/server_info", get(|State(s): State<WorkerState>| async move { Json(json!({"served_model_name":s.model,"model_path":s.model,"disaggregation_mode":if s.prefill { "prefill" } else { "decode" },"disaggregation_bootstrap_port":s.prefill.then_some(s.bootstrap)})) }))
            .route("/get_model_info", get(|State(s): State<WorkerState>| async move { Json(json!({"model_path":s.model,"is_generation":true})) }))
            .route("/generate", post(generation))
            .route("/v1/chat/completions", post(generation))
            .route("/abort_request", post(|| async { StatusCode::OK }))
            .route("/bench/stats", get(|State(s): State<WorkerState>| async move { Json(json!({"records":*s.common.records.lock().unwrap(),"active":s.common.active.load(Ordering::SeqCst)})) }).delete(|State(s): State<WorkerState>| async move {
                if s.common.active.load(Ordering::SeqCst) != 0 { return StatusCode::CONFLICT; }
                s.common.records.lock().unwrap().clear(); StatusCode::NO_CONTENT
            }))
            .route("/bench/config", axum::routing::put(|State(s): State<WorkerState>, Json(t): Json<Timing>| async move {
                if !t.delay_ms.is_finite() || t.delay_ms < 0.0 || !t.tokens_per_second.is_finite() || t.tokens_per_second < 0.0 { return StatusCode::BAD_REQUEST; }
                *s.common.timing.lock().unwrap() = t; StatusCode::OK
            }))
            .layer(DefaultBodyLimit::max(512 << 20)).with_state(state);
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", *port)).await?;
        tokio::spawn(async move {
            axum::serve(
                listener.tap_io(|socket| socket.set_nodelay(true).expect("TCP_NODELAY")),
                app,
            )
            .await
            .unwrap();
        });
    }
    println!("ready: {ports:?}");
    tokio::signal::ctrl_c().await?;
    Ok(())
}

#[derive(Default)]
struct TokenEvents {
    pending: Vec<u8>,
    native_text_bytes: usize,
    times: Vec<u64>,
}

impl TokenEvents {
    fn feed(&mut self, bytes: &[u8], at_ns: u64) -> anyhow::Result<()> {
        self.pending.extend_from_slice(bytes);
        ensure!(self.pending.len() < 8 << 20, "oversized SSE event");
        while let Some(end) = self.pending.iter().position(|x| *x == b'\n') {
            let line: Vec<_> = self.pending.drain(..=end).collect();
            let Some(data) = line.strip_prefix(b"data:") else {
                continue;
            };
            let data = std::str::from_utf8(data)?.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let value: Value = serde_json::from_str(data)?;
            if value.get("error").is_some() {
                bail!("upstream SSE error: {value}");
            }
            let delta = &value["choices"][0]["delta"];
            let openai = ["content", "reasoning_content"]
                .iter()
                .any(|key| delta[key].as_str().is_some_and(|x| !x.is_empty()))
                || delta["tool_calls"].as_array().is_some_and(|calls| {
                    calls.iter().any(|call| {
                        call["function"]["arguments"]
                            .as_str()
                            .is_some_and(|x| !x.is_empty())
                    })
                });
            let native = value["text"].as_str().is_some_and(|text| {
                let longer = text.len() > self.native_text_bytes;
                self.native_text_bytes = text.len();
                longer
            });
            if openai || native {
                self.times.push(at_ns);
            }
        }
        Ok(())
    }
}

fn overhead_ms(start: u64, received: u64, arrived: u64, emitted: u64) -> f64 {
    (received as f64 - start as f64 - (emitted as f64 - arrived as f64)) / 1e6
}

#[derive(Deserialize)]
struct Case {
    band: String,
    body: Value,
}

#[derive(Serialize)]
struct ClientRecord {
    id: String,
    band: String,
    start_ns: u64,
    tokens_ns: Vec<u64>,
    error: Option<String>,
}

async fn request(client: reqwest::Client, url: String, band: String, body: Bytes) -> ClientRecord {
    let id = uuid::Uuid::new_v4().to_string();
    let mut record = ClientRecord {
        id: id.clone(),
        band,
        start_ns: 0,
        tokens_ns: Vec::new(),
        error: None,
    };
    let operation = async {
        let request = client
            .post(url)
            .header("content-type", "application/json")
            .header("x-request-id", &id)
            .body(body);
        record.start_ns = now_ns();
        let response = request.send().await?.error_for_status()?;
        let mut chunks = response.bytes_stream();
        let mut events = TokenEvents::default();
        while let Some(chunk) = chunks.next().await {
            events.feed(&chunk?, now_ns())?;
        }
        ensure!(events.pending.is_empty(), "incomplete SSE event");
        ensure!(!events.times.is_empty(), "no generated output");
        record.tokens_ns = events.times;
        Ok::<_, anyhow::Error>(())
    };
    if let Err(error) = operation.await {
        record.error = Some(error.to_string());
    }
    record
}

fn cpu_ticks(pid: u32) -> anyhow::Result<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let fields: Vec<_> = stat
        .rsplit_once(')')
        .context("invalid process stat")?
        .1
        .split_whitespace()
        .collect();
    Ok(fields[11].parse::<u64>()? + fields[12].parse::<u64>()?)
}

fn percentile(mut values: Vec<f64>, p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[((values.len() - 1) as f64 * p).round() as usize])
}

#[derive(Default)]
struct Band {
    ttft: Vec<f64>,
    added: Vec<f64>,
    gaps: Vec<f64>,
    emitted_gaps: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
async fn run(
    router: String,
    workers: String,
    workload: PathBuf,
    out: PathBuf,
    router_pid: u32,
    concurrency: usize,
    requests: usize,
    output_tokens: usize,
    metrics_url: Option<String>,
    unpaced: bool,
) -> anyhow::Result<()> {
    ensure!(
        concurrency > 0 && requests > 0 && output_tokens > 0 && output_tokens <= 8192,
        "invalid run dimensions"
    );
    ensure!(!out.exists(), "output directory already exists");
    std::fs::create_dir_all(&out)?;
    let cases: Vec<Case> = std::fs::read_to_string(workload)?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    ensure!(!cases.is_empty(), "empty workload");
    let prepared: Vec<_> = cases
        .into_iter()
        .map(|mut case| {
            case.body
                .as_object_mut()
                .context("request must be an object")?
                .remove("max_completion_tokens");
            case.body["stream"] = json!(true);
            case.body["max_tokens"] = json!(output_tokens);
            case.body["stream_options"] = json!({"include_usage":true});
            Ok((case.band, Bytes::from(serde_json::to_vec(&case.body)?)))
        })
        .collect::<anyhow::Result<_>>()?;
    let client = reqwest::Client::builder()
        .tcp_nodelay(true)
        .timeout(Duration::from_secs(900))
        .build()?;
    // Keep one connection pool exclusive to each active request through EOF.
    // Concurrent checkouts from a shared HTTP/1 pool can queue a request
    // behind a stream that already has response headers.
    let lane_clients = Arc::new(Mutex::new(
        (0..concurrency)
            .map(|_| {
                reqwest::Client::builder()
                    .tcp_nodelay(true)
                    .timeout(Duration::from_secs(900))
                    .build()
            })
            .collect::<Result<Vec<_>, _>>()?,
    ));
    client
        .delete(format!("{workers}/bench/stats"))
        .send()
        .await?
        .error_for_status()?;
    let timing = if unpaced {
        Timing {
            delay_ms: 0.0,
            tokens_per_second: 0.0,
        }
    } else {
        Timing::default()
    };
    client
        .put(format!("{workers}/bench/config"))
        .json(&timing)
        .send()
        .await?
        .error_for_status()?;
    let ticks_per_second: f64 = String::from_utf8(
        std::process::Command::new("getconf")
            .arg("CLK_TCK")
            .output()?
            .stdout,
    )?
    .trim()
    .parse()?;
    let cpu_start = cpu_ticks(router_pid)?;
    let wall_start = now_ns();
    let records: Vec<_> = stream::iter((0..requests).map(|index| {
        let (band, body) = &prepared[index % prepared.len()];
        let clients = Arc::clone(&lane_clients);
        let url = format!("{router}/v1/chat/completions");
        let band = band.clone();
        let body = body.clone();
        async move {
            let client = clients.lock().unwrap().pop().expect("one client per lane");
            let record = request(client.clone(), url, band, body).await;
            clients.lock().unwrap().push(client);
            record
        }
    }))
    .buffer_unordered(concurrency)
    .collect()
    .await;
    ensure!(
        lane_clients.lock().unwrap().len() == concurrency,
        "client lease leaked"
    );
    let cpu_seconds = (cpu_ticks(router_pid)? - cpu_start) as f64 / ticks_per_second;
    let wall_seconds = (now_ns() - wall_start) as f64 / 1e9;
    let snapshot: Value = client
        .get(format!("{workers}/bench/stats"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let worker_records: HashMap<String, WorkerRecord> =
        serde_json::from_value(snapshot["records"].clone())?;
    let mut errors = Vec::new();
    let mut bands = BTreeMap::<String, Band>::new();
    for record in &records {
        if let Some(error) = &record.error {
            errors.push(json!({"id":record.id,"error":error}));
            continue;
        }
        let Some(worker) = worker_records.get(&record.id) else {
            errors.push(json!({"id":record.id,"error":"missing worker correlation"}));
            continue;
        };
        if record.tokens_ns.len() != worker.emitted_ns.len()
            || worker.emitted_ns.len() != output_tokens
        {
            errors.push(json!({"id":record.id,"error":"output event count mismatch","received":record.tokens_ns.len(),"emitted":worker.emitted_ns.len(),"expected":output_tokens}));
            continue;
        }
        let band = bands.entry(record.band.clone()).or_default();
        band.ttft
            .push((record.tokens_ns[0] - record.start_ns) as f64 / 1e6);
        band.added.push(overhead_ms(
            record.start_ns,
            record.tokens_ns[0],
            worker.arrived_ns,
            worker.emitted_ns[0],
        ));
        for (received, emitted) in record
            .tokens_ns
            .windows(2)
            .zip(worker.emitted_ns.windows(2))
        {
            band.gaps.push(
                (received[1] as f64 - received[0] as f64 - (emitted[1] as f64 - emitted[0] as f64))
                    / 1e6,
            );
            band.emitted_gaps
                .push((emitted[1] - emitted[0]) as f64 / 1e6);
        }
    }
    let bands: BTreeMap<_, _> = bands.into_iter().map(|(key, band)| (key, json!({"samples":band.added.len(),"chunk_gaps":band.gaps.len(),"ttft_p50_ms":percentile(band.ttft.clone(),0.5),"ttft_p99_ms":percentile(band.ttft,0.99),"added_ttft_p50_ms":percentile(band.added.clone(),0.5),"added_ttft_p99_ms":percentile(band.added,0.99),"added_chunk_gap_p99_ms":percentile(band.gaps,0.99),"worker_chunk_gap_p99_ms":percentile(band.emitted_gaps,0.99)}))).collect();
    let admission = if let Some(url) = metrics_url {
        let metrics = client
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        Some(
            metrics
                .lines()
                .find_map(|line| {
                    line.strip_prefix("sgl_router_admission_inflight ")
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .context("missing admission gauge")?,
        )
    } else {
        None
    };
    let report = json!({"router":router,"router_pid":router_pid,"concurrency":concurrency,"requests":requests,"output_tokens":output_tokens,"timing":timing,"cpu_seconds":cpu_seconds,"cpu_ms_per_request":1000.0*cpu_seconds/requests as f64,"wall_seconds":wall_seconds,"bands":bands,"errors":errors,"mock_active":snapshot["active"],"admission_inflight":admission,"native_worker_requests":worker_records.values().filter(|r| r.path=="/generate").count(),"input_ids_worker_requests":worker_records.values().filter(|r| r.input_ids).count()});
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let mut client_rows = String::new();
    for record in &records {
        client_rows += &serde_json::to_string(record)?;
        client_rows.push('\n');
    }
    std::fs::write(out.join("client.jsonl"), client_rows)?;
    std::fs::write(out.join("worker.json"), serde_json::to_vec(&snapshot)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        report["errors"].as_array().unwrap().is_empty(),
        "request errors or missing/mismatched correlation"
    );
    ensure!(
        snapshot["active"] == 0 && admission.is_none_or(|n| n == 0),
        "active streams or admission permits leaked"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Args::parse().command {
        Command::Serve { ports, model } => serve(ports, model).await,
        Command::Run {
            router,
            workers,
            workload,
            out,
            router_pid,
            concurrency,
            requests,
            output_tokens,
            metrics_url,
            unpaced,
        } => {
            run(
                router,
                workers,
                workload,
                out,
                router_pid,
                concurrency,
                requests,
                output_tokens,
                metrics_url,
                unpaced,
            )
            .await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragmented_sse_counts_output_instead_of_network_chunks() {
        let mut events = TokenEvents::default();
        events
            .feed(
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
                10,
            )
            .unwrap();
        events
            .feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"a", 20)
            .unwrap();
        events.feed(b"b\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"c\"}}]}\n\ndata: [DONE]\n\n", 30).unwrap();
        assert_eq!(events.times, [30, 30]);
        assert!(events.pending.is_empty());
    }

    #[test]
    fn native_cumulative_text_does_not_count_terminal_usage_as_output() {
        let mut events = TokenEvents::default();
        for (data, at) in [("a", 1), ("ab", 2), ("ab", 3)] {
            let wire = format!(
                "data: {{\"text\":\"{data}\",\"meta_info\":{{\"completion_tokens\":2}}}}\n\n"
            );
            events.feed(wire.as_bytes(), at).unwrap();
        }
        assert_eq!(events.times, [1, 2]);
    }

    #[test]
    fn worker_service_is_subtracted_instead_of_assuming_ideal_timers() {
        assert_eq!(
            overhead_ms(100_000_000, 160_000_000, 110_000_000, 150_000_000),
            20.0
        );
    }

    #[test]
    fn worker_honors_both_openai_output_limit_spellings() {
        for body in [
            json!({"max_tokens":8}),
            json!({"max_completion_tokens":8}),
            json!({"sampling_params":{"max_new_tokens":8}}),
        ] {
            assert_eq!(requested_output_tokens(&body), 8);
        }
    }
}
