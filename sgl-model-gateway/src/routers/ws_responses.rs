use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    extract::ws::{Message, WebSocket},
    http::HeaderMap,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, warn};

use crate::protocols::responses::{ResponseInputOutputItem, ResponsesRequest, ResponsesResponse};

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct CachedWsResponse {
    pub response: ResponsesResponse,
    pub input_items: Vec<ResponseInputOutputItem>,
}

impl CachedWsResponse {
    pub fn to_conversation_items(&self) -> Vec<ResponseInputOutputItem> {
        let mut items = self.input_items.clone();

        for output_item in &self.response.output {
            let Ok(value) = serde_json::to_value(output_item) else {
                continue;
            };
            let Ok(item) = serde_json::from_value::<ResponseInputOutputItem>(value) else {
                continue;
            };
            items.push(item);
        }

        items
    }
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct WsClientError {
    pub code: String,
    pub message: String,
}

impl WsClientError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

#[async_trait]
#[doc(hidden)]
pub trait WsResponsesExecutor: Send + Sync {
    async fn execute_response_create(
        &self,
        headers: HeaderMap,
        request: ResponsesRequest,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::UnboundedSender<Message>,
    ) -> Result<CachedWsResponse, WsClientError>;
}

#[derive(Default)]
struct WsSessionState {
    active_request: bool,
    cached_response: Option<CachedWsResponse>,
}

#[derive(Debug, Deserialize)]
struct RawClientEvent {
    #[serde(rename = "type")]
    event_type: String,
    response: Option<ResponsesRequest>,
    event_id: Option<String>,
}

#[doc(hidden)]
pub async fn serve_responses_ws(
    socket: WebSocket,
    headers: HeaderMap,
    executor: Arc<dyn WsResponsesExecutor>,
) {
    let (mut sink, mut stream) = socket.split();
    let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<Message>();
    let session = Arc::new(Mutex::new(WsSessionState::default()));

    let writer = tokio::spawn(async move {
        while let Some(message) = outbound_rx.recv().await {
            if sink.send(message).await.is_err() {
                break;
            }
        }
    });

    while let Some(message_result) = stream.next().await {
        let message = match message_result {
            Ok(message) => message,
            Err(err) => {
                debug!("responses websocket receive error: {}", err);
                break;
            }
        };

        match message {
            Message::Text(text) => {
                handle_text_event(
                    text.as_ref(),
                    headers.clone(),
                    executor.clone(),
                    session.clone(),
                    outbound_tx.clone(),
                )
                .await;
            }
            Message::Binary(_) => {
                send_error_json(
                    &outbound_tx,
                    "unsupported_message_type",
                    "Binary WebSocket messages are not supported on /v1/responses.",
                    None,
                );
            }
            Message::Ping(payload) => {
                let _ = outbound_tx.send(Message::Pong(payload));
            }
            Message::Pong(_) => {}
            Message::Close(_) => break,
        }
    }

    drop(outbound_tx);
    let _ = writer.await;
}

async fn handle_text_event(
    payload: &str,
    headers: HeaderMap,
    executor: Arc<dyn WsResponsesExecutor>,
    session: Arc<Mutex<WsSessionState>>,
    outbound_tx: mpsc::UnboundedSender<Message>,
) {
    let raw_event = match serde_json::from_str::<RawClientEvent>(payload) {
        Ok(raw_event) => raw_event,
        Err(err) => {
            send_error_json(
                &outbound_tx,
                "invalid_json",
                format!("Failed to parse WebSocket client event JSON: {}", err),
                None,
            );
            return;
        }
    };

    match raw_event.event_type.as_str() {
        "response.create" => {
            let Some(request) = raw_event.response else {
                send_error_json(
                    &outbound_tx,
                    "missing_response",
                    "The `response.create` event requires a `response` object.",
                    raw_event.event_id.as_deref(),
                );
                return;
            };

            let mut session_guard = session.lock().await;
            if session_guard.active_request {
                drop(session_guard);
                send_error_json(
                    &outbound_tx,
                    "concurrent_response_create",
                    "Only one in-flight `response.create` is allowed per connection.",
                    raw_event.event_id.as_deref(),
                );
                return;
            }

            session_guard.active_request = true;
            let cached_response = session_guard.cached_response.clone();
            drop(session_guard);

            let event_id = raw_event.event_id.clone();
            let session_clone = session.clone();
            let outbound_clone = outbound_tx.clone();
            tokio::spawn(async move {
                let result = executor
                    .execute_response_create(
                        headers,
                        request,
                        cached_response,
                        outbound_clone.clone(),
                    )
                    .await;

                let mut session_guard = session_clone.lock().await;
                session_guard.active_request = false;

                match result {
                    Ok(cached_response) => {
                        session_guard.cached_response = Some(cached_response);
                    }
                    Err(err) => {
                        drop(session_guard);
                        send_error_json(
                            &outbound_clone,
                            &err.code,
                            err.message,
                            event_id.as_deref(),
                        );
                    }
                }
            });
        }
        other => {
            send_error_json(
                &outbound_tx,
                "unsupported_event",
                format!("Unsupported WebSocket client event type: {}", other),
                raw_event.event_id.as_deref(),
            );
        }
    }
}

pub(crate) fn send_error_json(
    outbound_tx: &mpsc::UnboundedSender<Message>,
    code: &str,
    message: impl Into<String>,
    event_id: Option<&str>,
) {
    let message = message.into();
    let mut error = json!({
        "type": "error",
        "code": code,
        "message": message,
    });

    if let Some(event_id) = event_id {
        error["event_id"] = Value::String(event_id.to_string());
    }

    if outbound_tx
        .send(Message::Text(error.to_string().into()))
        .is_err()
    {
        warn!("responses websocket client disconnected before error delivery");
    }
}
