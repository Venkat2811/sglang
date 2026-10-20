// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Active HTTP health, proved by bounded generation through each usable role pair.

use crate::{
    config::HealthCheckConfig,
    discovery::{ModelId, WorkerMode},
    server::app_context::AppContext,
    workers::Worker,
};
use futures::{stream, StreamExt};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

pub struct ActiveHealth {
    config: HealthCheckConfig,
    state: Mutex<HealthState>,
}
#[derive(Default)]
struct HealthState {
    topology: Vec<String>,
    proven: bool,
    streaks: HashMap<String, Streak>,
}
#[derive(Default)]
struct Streak {
    successes: u32,
    failures: u32,
}

fn topology(workers: &[Arc<Worker>]) -> Vec<String> {
    let mut ids: Vec<_> = workers
        .iter()
        .map(|w| {
            format!(
                "{}|{}|{:?}|{:?}|{:?}|{}",
                w.id.0,
                w.url,
                w.mode(),
                w.bootstrap_port(),
                w.version_group(),
                w.health_instance()
            )
        })
        .collect();
    ids.sort();
    ids
}

impl ActiveHealth {
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            config,
            state: Mutex::new(HealthState::default()),
        }
    }

    pub fn ready(&self, ctx: &AppContext) -> bool {
        let workers = ctx
            .registry
            .workers_for(&ModelId(ctx.config.model.id.clone()));
        let state = self.state.lock().unwrap();
        state.proven && state.topology == topology(&workers)
    }

    pub async fn run(ctx: Arc<AppContext>) {
        let health = Arc::clone(
            ctx.active_health
                .as_ref()
                .expect("active health configured"),
        );
        let mut interval = tokio::time::interval(Duration::from_secs(health.config.interval_secs));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            health.check(&ctx).await;
        }
    }

    async fn check(&self, ctx: &AppContext) {
        let workers = ctx
            .registry
            .workers_for(&ModelId(ctx.config.model.id.clone()));
        let current = topology(&workers);
        let timeout = Duration::from_secs(self.config.timeout_secs);
        // A cheap GET is necessary but cannot prove that generation works.
        let live: HashSet<_> = stream::iter(workers.iter().cloned())
            .map(|worker| async move {
                let mut request = ctx
                    .proxy
                    .client_for(worker.protocol())
                    .get(format!("{}/health", worker.url.trim_end_matches('/')))
                    .timeout(timeout);
                if let Some(auth) = &ctx.config.server.worker_auth {
                    request = request.header(reqwest::header::AUTHORIZATION, auth);
                }
                let ok = request.send().await.is_ok_and(|r| r.status().is_success());
                (worker.id.0.clone(), ok)
            })
            .buffer_unordered(16)
            .filter_map(|(id, ok)| async move { ok.then_some(id) })
            .collect()
            .await;
        let mut pairs = Vec::new();
        for worker in &workers {
            if !live.contains(&worker.id.0) || worker.awaiting_bootstrap_port() {
                continue;
            }
            match worker.mode() {
                WorkerMode::Plain => pairs.push((Arc::clone(worker), None)),
                WorkerMode::Prefill => {
                    for decode in &workers {
                        if decode.mode() == WorkerMode::Decode
                            && live.contains(&decode.id.0)
                            && decode.version_group() == worker.version_group()
                        {
                            pairs.push((Arc::clone(worker), Some(Arc::clone(decode))));
                        }
                    }
                }
                WorkerMode::Decode => {}
            }
        }
        let mut proven = HashSet::new();
        let mut probes = stream::iter(pairs)
            .map(|(prefill, decode)| async move {
                let ok = tokio::time::timeout(
                    timeout,
                    crate::server::routes::chat::generation_probe(
                        ctx,
                        Arc::clone(&prefill),
                        decode.clone(),
                    ),
                )
                .await
                .unwrap_or(false);
                (prefill, decode, ok)
            })
            .buffer_unordered(8);
        while let Some((prefill, decode, ok)) = probes.next().await {
            if ok {
                proven.insert(prefill.id.0.clone());
                if let Some(decode) = decode {
                    proven.insert(decode.id.0.clone());
                }
            }
        }
        // Discovery changed under the requests: none of that evidence proves the new topology.
        if current
            != topology(
                &ctx.registry
                    .workers_for(&ModelId(ctx.config.model.id.clone())),
            )
        {
            self.state.lock().unwrap().proven = false;
            return;
        }
        let mut state = self.state.lock().unwrap();
        if state.topology != current {
            state.streaks.clear();
        }
        state.topology = current;
        state.proven = !proven.is_empty();
        for worker in workers {
            let streak = state.streaks.entry(worker.id.0.clone()).or_default();
            if proven.contains(&worker.id.0) {
                streak.failures = 0;
                streak.successes = streak.successes.saturating_add(1);
                if streak.successes >= self.config.success_threshold {
                    worker.set_active_healthy(true);
                }
            } else {
                streak.successes = 0;
                streak.failures = streak.failures.saturating_add(1);
                if streak.failures >= self.config.failure_threshold {
                    worker.set_active_healthy(false);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{WorkerId, WorkerSpec};
    #[test]
    fn topology_proof_does_not_survive_removal_and_readdition_of_the_same_endpoint() {
        let ctx = AppContext::stub();
        let spec = || WorkerSpec {
            id: WorkerId("worker".into()),
            url: "http://127.0.0.1:1".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(ctx.config.model.id.clone())],
            ..Default::default()
        };
        ctx.registry.add(spec()).unwrap();
        let health = ActiveHealth::new(HealthCheckConfig::default());
        {
            let mut state = health.state.lock().unwrap();
            state.proven = true;
            state.topology = topology(
                &ctx.registry
                    .workers_for(&ModelId(ctx.config.model.id.clone())),
            );
        }
        assert!(health.ready(&ctx));
        ctx.registry.add(spec()).unwrap();
        assert!(
            health.ready(&ctx),
            "unchanged metadata refresh preserves live proof"
        );
        ctx.registry.remove(&WorkerId("worker".into()));
        assert!(!health.ready(&ctx));
        ctx.registry.add(spec()).unwrap();
        assert!(
            !health.ready(&ctx),
            "recreated endpoints need fresh generation evidence"
        );
    }
}
