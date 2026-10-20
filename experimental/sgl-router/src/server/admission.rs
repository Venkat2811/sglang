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
pub struct Admission(Option<Arc<Limit>>);
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
        Self(config.max_concurrent_requests.map(|capacity| {
            Arc::new(Limit {
                capacity,
                refill: f64::from(config.refill_per_second),
                state: Mutex::new(State {
                    active: 0,
                    credits: capacity as f64,
                    updated: Instant::now(),
                }),
            })
        }))
    }

    fn enter(&self) -> Result<Option<Permit>, ApiError> {
        let Some(limit) = &self.0 else {
            return Ok(None);
        };
        let mut state = limit.state.lock().unwrap();
        let now = Instant::now();
        state.credits = (state.credits
            + now.duration_since(state.updated).as_secs_f64() * limit.refill)
            .min(limit.capacity as f64);
        state.updated = now;
        if state.active >= limit.capacity || state.credits < 1.0 {
            return Err(ApiError::AdmissionRejected);
        }
        state.active += 1;
        state.credits -= 1.0;
        Ok(Some(Permit(Arc::clone(limit))))
    }

    pub fn active(&self) -> usize {
        self.0
            .as_ref()
            .map_or(0, |l| l.state.lock().unwrap().active)
    }
}

struct Permit(Arc<Limit>);
impl Drop for Permit {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap();
        state.active -= 1;
        state.credits = (state.credits + 1.0).min(self.0.capacity as f64);
    }
}

/// Retained by each dispatched worker until its work or abort has completed.
/// Client correlation ids are deliberately unrelated to engine abort identities.
pub(crate) struct RequestScope {
    _permit: Option<Permit>,
    pub cancelled: CancellationToken,
}
impl RequestScope {
    pub fn enter(admission: &Admission) -> Result<Arc<Self>, ApiError> {
        Ok(Arc::new(Self {
            _permit: admission.enter()?,
            cancelled: CancellationToken::new(),
        }))
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
            self.scope.cancelled.cancel();
        }
    }
}
impl ClientLifetime {
    pub fn complete(&mut self) {
        self.cancel_on_drop = false;
    }

    pub fn replies<T: Send + 'static>(
        self,
        inner: crate::proxy::grpc::Replies<T>,
        successful: bool,
    ) -> crate::proxy::grpc::Replies<T> {
        Box::pin(ClientReplies {
            inner,
            lifetime: Some(self),
            successful,
        })
    }
    pub fn body(mut self, inner: Body, successful: bool) -> Body {
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
        assert!(admission.0.as_ref().unwrap().state.lock().unwrap().credits < 1.0);
        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        assert!(
            admission.enter().is_err(),
            "refilled credits cannot increase concurrency"
        );
        assert_eq!(
            admission.0.as_ref().unwrap().state.lock().unwrap().credits,
            1.0
        );
        drop(permit);
        assert!(admission.enter().is_ok());
    }
}
