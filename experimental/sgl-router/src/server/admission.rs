// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Global admission and the lifetime shared by a client's outstanding worker legs.

use crate::{config::AdmissionConfig, server::error::ApiError};
use axum::body::{Body, Bytes, HttpBody};
use http_body::{Frame, SizeHint};
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
pub struct Admission {
    limit: Option<Arc<Limit>>,
    stats: Arc<AdmissionStats>,
}
#[derive(Default)]
struct AdmissionStats {
    active: std::sync::atomic::AtomicUsize,
    accepted: std::sync::atomic::AtomicU64,
    rejected: std::sync::atomic::AtomicU64,
}
struct Limit {
    capacity: usize,
    refill: f64,
    state: Mutex<State>,
}
struct State {
    active: usize,
    credits: f64,
    updated: Instant,
}
impl Admission {
    pub fn new(config: &AdmissionConfig) -> Self {
        Self {
            limit: config.max_concurrent_requests.map(|capacity| {
                Arc::new(Limit {
                    capacity,
                    refill: f64::from(config.refill_per_second),
                    state: Mutex::new(State {
                        active: 0,
                        credits: capacity as f64,
                        updated: Instant::now(),
                    }),
                })
            }),
            stats: Arc::default(),
        }
    }
    fn enter(&self) -> Result<Option<Permit>, ApiError> {
        use std::sync::atomic::Ordering;
        if let Some(limit) = &self.limit {
            let mut state = limit.state.lock().unwrap();
            let now = Instant::now();
            state.credits = (state.credits
                + now.duration_since(state.updated).as_secs_f64() * limit.refill)
                .min(limit.capacity as f64);
            state.updated = now;
            if state.active >= limit.capacity || state.credits < 1.0 {
                self.stats.rejected.fetch_add(1, Ordering::Relaxed);
                return Err(ApiError::AdmissionRejected);
            }
            state.active += 1;
            state.credits -= 1.0;
        }
        self.stats.active.fetch_add(1, Ordering::Relaxed);
        self.stats.accepted.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Permit {
            limit: self.limit.clone(),
            stats: Arc::clone(&self.stats),
        }))
    }
    pub fn active(&self) -> usize {
        self.stats.active.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn render_metrics(&self) -> String {
        use std::sync::atomic::Ordering;
        format!("# HELP sgl_router_admission_inflight Requests retaining admission through response and worker cleanup.\n# TYPE sgl_router_admission_inflight gauge\nsgl_router_admission_inflight {}\n# HELP sgl_router_admission_capacity Concurrent ceiling; -1 means disabled.\n# TYPE sgl_router_admission_capacity gauge\nsgl_router_admission_capacity {}\n# HELP sgl_router_admission_total Generation admission outcomes.\n# TYPE sgl_router_admission_total counter\nsgl_router_admission_total{{outcome=\"accepted\"}} {}\nsgl_router_admission_total{{outcome=\"rejected\"}} {}\n# HELP sgl_router_admission_queue_length Requests waiting for admission (queueing disabled).\n# TYPE sgl_router_admission_queue_length gauge\nsgl_router_admission_queue_length 0\n",self.active(),self.limit.as_ref().map_or(-1,|l|i64::try_from(l.capacity).unwrap_or(i64::MAX)),self.stats.accepted.load(Ordering::Relaxed),self.stats.rejected.load(Ordering::Relaxed))
    }
}
struct Permit {
    limit: Option<Arc<Limit>>,
    stats: Arc<AdmissionStats>,
}
impl Drop for Permit {
    fn drop(&mut self) {
        if let Some(limit) = &self.limit {
            let mut state = limit.state.lock().unwrap();
            state.active -= 1;
            state.credits = (state.credits + 1.0).min(limit.capacity as f64);
        }
        self.stats
            .active
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Retained by each dispatched worker until its work or abort has completed.
/// Client correlation ids are deliberately unrelated to engine abort identities.
pub(crate) struct RequestScope {
    _permit: Option<Permit>,
    pub cancelled: CancellationToken,
    pub life: crate::server::request_log::Lifecycle,
}
impl RequestScope {
    pub fn enter(admission: &Admission) -> Result<Arc<Self>, ApiError> {
        Ok(Arc::new(Self {
            _permit: admission.enter()?,
            cancelled: CancellationToken::new(),
            life: Default::default(),
        }))
    }
    pub fn enter_request(
        ctx: &crate::server::app_context::AppContext,
        endpoint: &'static str,
        headers: &axum::http::HeaderMap,
    ) -> Result<Arc<Self>, ApiError> {
        let id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        match Self::enter(&ctx.admission) {
            Ok(scope) => {
                scope.life.configure(
                    id,
                    ctx.config.model.id.clone(),
                    endpoint,
                    Arc::clone(&ctx.metrics),
                    ctx.admission.active(),
                    ctx.config.server.admission.max_concurrent_requests,
                );
                Ok(scope)
            }
            Err(error) => {
                tracing::info!(event="reject",request_id=%id,endpoint,reason=error.error_code(),"request_lifecycle");
                Err(error)
            }
        }
    }
    pub fn client(self: &Arc<Self>) -> ClientLifetime {
        ClientLifetime {
            scope: Arc::clone(self),
            cancel_on_drop: true,
        }
    }
}

pub(crate) struct ClientLifetime {
    scope: Arc<RequestScope>,
    cancel_on_drop: bool,
}
impl Drop for ClientLifetime {
    fn drop(&mut self) {
        if self.cancel_on_drop {
            self.scope.life.abort();
            self.scope.cancelled.cancel();
        }
    }
}
impl ClientLifetime {
    pub fn complete(&mut self) {
        self.scope.life.complete();
        self.cancel_on_drop = false;
    }

    pub fn replies<T: Send + 'static>(
        self,
        inner: crate::proxy::grpc::Replies<T>,
        successful: bool,
    ) -> crate::proxy::grpc::Replies<T> {
        if !successful {
            self.scope.life.fail("grpc_status");
        }
        Box::pin(ClientReplies {
            inner,
            lifetime: Some(self),
            successful,
        })
    }
    pub fn body(mut self, inner: Body, successful: bool) -> Body {
        if !successful {
            self.scope.life.fail("http_status");
        }
        if successful && inner.is_end_stream() {
            self.complete();
        }
        Body::new(ClientBody {
            inner,
            lifetime: Some(self),
            successful,
        })
    }
}
struct ClientReplies<T> {
    inner: crate::proxy::grpc::Replies<T>,
    lifetime: Option<ClientLifetime>,
    successful: bool,
}
impl<T> futures::Stream for ClientReplies<T> {
    type Item = Result<T, tonic::Status>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let result = this.inner.as_mut().poll_next(cx);
        if matches!(result, Poll::Ready(Some(Err(_)))) {
            if let Some(lifetime) = &this.lifetime {
                lifetime.scope.life.fail("grpc_stream_error");
            }
        }
        if matches!(result, Poll::Ready(None)) && this.successful {
            if let Some(lifetime) = &mut this.lifetime {
                lifetime.complete();
            }
        }
        if matches!(result, Poll::Ready(None | Some(Err(_)))) {
            this.lifetime.take();
        }
        result
    }
}
struct ClientBody {
    inner: Body,
    lifetime: Option<ClientLifetime>,
    successful: bool,
}
impl HttpBody for ClientBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_frame(cx);
        if let Some(lifetime) = &this.lifetime {
            if let Poll::Ready(Some(Ok(frame))) = &result {
                if let Some(data) = frame.data_ref() {
                    lifetime.scope.life.observe(data);
                }
            }
            if matches!(result, Poll::Ready(Some(Err(_)))) {
                lifetime.scope.life.fail("body_error");
            }
        }
        let completed = matches!(result, Poll::Ready(None))
            || (matches!(result, Poll::Ready(Some(Ok(_)))) && this.inner.is_end_stream());
        if completed && this.successful {
            if let Some(lifetime) = &mut this.lifetime {
                lifetime.complete();
            }
        }
        if completed || matches!(result, Poll::Ready(Some(Err(_)))) {
            this.lifetime.take();
        }
        result
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test(start_paused = true)]
    async fn refill_obeys_the_exact_boundary_and_never_bypasses_active_capacity() {
        let admission = Admission::new(&AdmissionConfig {
            max_concurrent_requests: Some(1),
            refill_per_second: 1,
        });
        let permit = admission.enter().unwrap();
        tokio::time::advance(std::time::Duration::from_millis(999)).await;
        assert!(admission.enter().is_err());
        assert!(
            admission
                .limit
                .as_ref()
                .unwrap()
                .state
                .lock()
                .unwrap()
                .credits
                < 1.0
        );
        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        assert!(
            admission.enter().is_err(),
            "refilled credits cannot increase concurrency"
        );
        assert_eq!(
            admission
                .limit
                .as_ref()
                .unwrap()
                .state
                .lock()
                .unwrap()
                .credits,
            1.0
        );
        drop(permit);
        assert!(admission.enter().is_ok());
    }
}
