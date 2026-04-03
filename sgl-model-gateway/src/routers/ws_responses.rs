use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    extract::ws::{CloseFrame, Message, WebSocket},
    http::HeaderMap,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, warn};

use crate::protocols::responses::{ResponseInputOutputItem, ResponsesRequest, ResponsesResponse};

const DEFAULT_WS_SESSION_LIFETIME: Duration = Duration::from_secs(60 * 60);

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
    pub status: u16,
    pub error_type: String,
    pub param: Option<String>,
}

impl WsClientError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            status: 400,
            error_type: "invalid_request_error".to_string(),
            param: None,
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = status;
        self
    }

    pub fn with_type(mut self, error_type: impl Into<String>) -> Self {
        self.error_type = error_type.into();
        self
    }

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
}

#[derive(Clone, Debug, Default)]
#[doc(hidden)]
pub struct WsResponseCreateOptions {
    pub generate: Option<bool>,
}

#[async_trait]
#[doc(hidden)]
pub trait WsResponsesExecutor: Send + Sync {
    async fn execute_response_create(
        &self,
        headers: HeaderMap,
        request: ResponsesRequest,
        options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::UnboundedSender<Message>,
    ) -> Result<CachedWsResponse, WsClientError>;
}

#[derive(Default)]
struct WsSessionState {
    active_request: bool,
    cached_response: Option<CachedWsResponse>,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct WsRuntimeConfig {
    pub max_session_lifetime: Duration,
}

impl Default for WsRuntimeConfig {
    fn default() -> Self {
        Self {
            max_session_lifetime: DEFAULT_WS_SESSION_LIFETIME,
        }
    }
}

#[derive(Debug)]
struct ParsedClientEvent {
    event_type: String,
    event_id: Option<String>,
    request: Option<ResponsesRequest>,
    options: WsResponseCreateOptions,
}

#[doc(hidden)]
pub async fn serve_responses_ws(
    socket: WebSocket,
    headers: HeaderMap,
    executor: Arc<dyn WsResponsesExecutor>,
) {
    serve_responses_ws_with_config(socket, headers, executor, WsRuntimeConfig::default()).await;
}

#[doc(hidden)]
pub async fn serve_responses_ws_with_config(
    socket: WebSocket,
    headers: HeaderMap,
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
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

    let session_timeout = {
        let outbound_tx = outbound_tx.clone();
        let max_session_lifetime = runtime_config.max_session_lifetime;
        tokio::spawn(async move {
            if max_session_lifetime.is_zero() {
                return;
            }

            tokio::time::sleep(max_session_lifetime).await;
            send_client_error_json(
                &outbound_tx,
                &WsClientError::new(
                    "websocket_connection_limit_reached",
                    "Responses websocket connection limit reached (60 minutes). Create a new websocket connection to continue.",
                )
                .with_type("invalid_request_error"),
                None,
            );
            let _ = outbound_tx.send(Message::Close(Some(CloseFrame {
                code: axum::extract::ws::close_code::NORMAL,
                reason:
                    "Responses websocket connection limit reached (60 minutes). Create a new websocket connection to continue."
                        .into(),
            })));
        })
    };

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

    session_timeout.abort();
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
    let raw_event = match parse_client_event(payload) {
        Ok(raw_event) => raw_event,
        Err(err) => {
            send_client_error_json(&outbound_tx, &err, None);
            return;
        }
    };

    match raw_event.event_type.as_str() {
        "response.create" => {
            let Some(request) = raw_event.request else {
                send_error_json(
                    &outbound_tx,
                    "missing_response",
                    "The `response.create` event requires a valid Responses request body.",
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
            let options = raw_event.options.clone();
            let referenced_previous_response_id = request.previous_response_id.clone();
            let session_clone = session.clone();
            let outbound_clone = outbound_tx.clone();
            tokio::spawn(async move {
                let result = executor
                    .execute_response_create(
                        headers,
                        request,
                        options,
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
                        let should_evict_cached_response = referenced_previous_response_id
                            .as_deref()
                            .is_some_and(|previous_id| {
                                session_guard
                                    .cached_response
                                    .as_ref()
                                    .map(|cached| cached.response.id == previous_id)
                                    .unwrap_or(false)
                            });
                        if should_evict_cached_response {
                            session_guard.cached_response = None;
                        }
                        drop(session_guard);
                        send_client_error_json(&outbound_clone, &err, event_id.as_deref());
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

fn parse_client_event(payload: &str) -> Result<ParsedClientEvent, WsClientError> {
    let event_value = serde_json::from_str::<Value>(payload).map_err(|err| {
        WsClientError::new(
            "invalid_json",
            format!("Failed to parse WebSocket client event JSON: {}", err),
        )
    })?;

    let mut event_object = match event_value {
        Value::Object(event_object) => event_object,
        _ => {
            return Err(WsClientError::new(
                "invalid_json",
                "WebSocket client event JSON must be an object.",
            ))
        }
    };

    let event_type = take_string_field(&mut event_object, "type").ok_or_else(|| {
        WsClientError::new(
            "invalid_json",
            "WebSocket client event JSON must include a string `type` field.",
        )
    })?;
    let event_id = take_string_field(&mut event_object, "event_id");

    let (request, options) = match event_type.as_str() {
        "response.create" => {
            if let Some(mut request_value) = event_object.remove("response") {
                let options = extract_request_options(&mut request_value)?;
                let request =
                    serde_json::from_value::<ResponsesRequest>(request_value).map_err(|err| {
                        WsClientError::new(
                            "invalid_request",
                            format!("Failed to parse `response.create` payload: {}", err),
                        )
                    })?;
                (Some(request), options)
            } else {
                if event_object.is_empty() {
                    return Ok(ParsedClientEvent {
                        event_type,
                        event_id,
                        request: None,
                        options: WsResponseCreateOptions::default(),
                    });
                }

                let mut request_value = Value::Object(event_object);
                let options = extract_request_options(&mut request_value)?;
                let request =
                    serde_json::from_value::<ResponsesRequest>(request_value).map_err(|err| {
                        WsClientError::new(
                            "invalid_request",
                            format!("Failed to parse `response.create` payload: {}", err),
                        )
                    })?;
                (Some(request), options)
            }
        }
        _ => (None, WsResponseCreateOptions::default()),
    };

    Ok(ParsedClientEvent {
        event_type,
        event_id,
        request,
        options,
    })
}

fn extract_request_options(
    request_value: &mut Value,
) -> Result<WsResponseCreateOptions, WsClientError> {
    let request_object = request_value.as_object_mut().ok_or_else(|| {
        WsClientError::new(
            "invalid_request",
            "The `response.create` payload must be a JSON object.",
        )
    })?;

    Ok(WsResponseCreateOptions {
        generate: take_bool_field(request_object, "generate"),
    })
}

fn take_string_field(object: &mut serde_json::Map<String, Value>, key: &str) -> Option<String> {
    object
        .remove(key)
        .and_then(|value| value.as_str().map(str::to_owned))
}

fn take_bool_field(object: &mut serde_json::Map<String, Value>, key: &str) -> Option<bool> {
    object.remove(key).and_then(|value| value.as_bool())
}

pub(crate) fn send_error_json(
    outbound_tx: &mpsc::UnboundedSender<Message>,
    code: &str,
    message: impl Into<String>,
    event_id: Option<&str>,
) {
    send_client_error_json(outbound_tx, &WsClientError::new(code, message), event_id);
}

pub(crate) fn send_client_error_json(
    outbound_tx: &mpsc::UnboundedSender<Message>,
    error: &WsClientError,
    event_id: Option<&str>,
) {
    let mut error_json = json!({
        "type": "error",
        "status": error.status,
        "error": {
            "type": error.error_type,
            "code": error.code,
            "message": error.message,
        },
        "code": error.code,
        "message": error.message,
    });

    if let Some(param) = &error.param {
        error_json["error"]["param"] = Value::String(param.clone());
    }

    if let Some(event_id) = event_id {
        error_json["event_id"] = Value::String(event_id.to_string());
    }

    if outbound_tx
        .send(Message::Text(error_json.to_string().into()))
        .is_err()
    {
        warn!("responses websocket client disconnected before error delivery");
    }
}
