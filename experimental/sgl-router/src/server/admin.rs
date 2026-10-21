// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bounded inspection and live log filtering on the separate loopback listener.
use super::app_context::AppContext;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

pub fn router(ctx: Arc<AppContext>) -> Router {
    Router::new()
        .route("/admin/workers", get(workers))
        .route("/admin/config", get(config))
        .route("/admin/index", get(index))
        .route("/admin/policy", get(index))
        .route("/admin/cap", axum::routing::put(cap))
        .route("/admin/log-level", post(log_level).put(log_level))
        .layer(axum::extract::DefaultBodyLimit::max(4096))
        .with_state(ctx)
}
async fn workers(State(ctx): State<Arc<AppContext>>) -> Json<Value> {
    let all = ctx.registry.all();
    let truncated = all.len() > 256;
    let workers:Vec<_>=all.into_iter().take(256).map(|w| {let cb=w.breaker.snapshot();json!({"url":redact_url(&w.url),"mode":format!("{:?}",w.mode()),"is_healthy":cb.admit&&w.active_health_passes(),"breaker_state":cb.state_code,"router_inflight":w.router_inflight_load()})}).collect();
    Json(
        json!({"workers":workers,"truncated":truncated,"kv_bootstrap_ready":ctx.kv_bootstrap_admit_ready()}),
    )
}
async fn config(State(ctx): State<Arc<AppContext>>) -> Json<Value> {
    let c = &ctx.config;
    Json(
        json!({"model_id":c.model.id,"host":c.server.host,"port":c.server.port,"prometheus_port":c.server.prometheus_port,"admin_port":c.server.admin_port,"worker_auth_configured":c.server.worker_auth.is_some(),"max_payload_size":c.server.max_payload_size,"max_concurrent_requests":ctx.admission.capacity(),"refill_per_second":c.server.admission.refill_per_second,"queue_size":0,"request_timeout_secs":c.proxy.request_timeout_secs,"health_check_interval_secs":c.server.health_check.interval_secs,"health_check_timeout_secs":c.server.health_check.timeout_secs}),
    )
}
#[derive(Deserialize)]
struct IndexQuery {
    limit: Option<usize>,
}
async fn index(State(ctx): State<Arc<AppContext>>, Query(query): Query<IndexQuery>) -> Json<Value> {
    let limit = query.limit.unwrap_or(64).clamp(1, 256);
    Json(match ctx.kv_index.as_ref() {
        Some(index) => index.tree().debug_nodes(limit),
        None => {
            json!({"maintained":false,"limit":limit,"nodes":[],"truncated":false,"busy_shards":0})
        }
    })
}
#[derive(Deserialize)]
struct LogFilter {
    filter: String,
}
async fn log_level(Json(body): Json<LogFilter>) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    super::logging::reload_filter(&body.filter)
        .map_err(|message| (StatusCode::BAD_REQUEST, Json(json!({"error":message}))))?;
    Ok(Json(json!({"status":"success","filter":body.filter})))
}
fn redact_url(url: &str) -> String {
    let Ok(mut parsed) = url::Url::parse(url) else {
        return "<invalid-url>".into();
    };
    let _ = parsed.set_username("");
    let _ = parsed.set_password(None);
    parsed.set_query(None);
    parsed.set_fragment(None);
    parsed.to_string()
}

#[derive(Deserialize)]
struct Cap {
    max_concurrent_requests: usize,
}
async fn cap(State(ctx): State<Arc<AppContext>>, Json(body): Json<Cap>) -> Json<Value> {
    let previous = ctx.admission.capacity();
    ctx.admission
        .set_capacity(Some(body.max_concurrent_requests));
    tracing::info!(previous_capacity=?previous,capacity=?ctx.admission.capacity(),inflight=ctx.admission.active(),"admission cap changed");
    Json(
        json!({"max_concurrent_requests":ctx.admission.capacity(),"inflight":ctx.admission.active()}),
    )
}
