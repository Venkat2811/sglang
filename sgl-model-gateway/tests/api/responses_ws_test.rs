use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::ws::{Message, WebSocket},
    http::{HeaderMap, Request, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::{SinkExt, StreamExt};
use smg::{
    core::WorkerRegistry,
    protocols::responses::{
        ResponseContentPart, ResponseInputOutputItem, ResponseOutputItem, ResponseStatus,
        ResponsesRequest, ResponsesResponse,
    },
    routers::{
        router_manager::{router_ids, RouterManager},
        ws_responses::{
            serve_responses_ws_with_config, CachedWsResponse, WsClientError, WsResponsesExecutor,
            WsRuntimeConfig,
        },
        RouterTrait,
    },
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, Notify},
};
use tokio_tungstenite::connect_async;
use tower::ServiceExt;

use crate::common::test_app::{create_test_app_context, create_test_app_with_context};

#[derive(Clone)]
struct StubWsExecutor {
    gate: Option<Arc<Notify>>,
}

impl StubWsExecutor {
    fn immediate() -> Self {
        Self { gate: None }
    }

    fn gated(gate: Arc<Notify>) -> Self {
        Self { gate: Some(gate) }
    }
}

#[async_trait]
impl WsResponsesExecutor for StubWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::UnboundedSender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let model = request.model.clone();
        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_ws_test",
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": []
            }
        });
        let _ = outbound_tx.send(Message::Text(created.to_string().into()));

        if let Some(gate) = &self.gate {
            gate.notified().await;
        }

        let output_text = "stub websocket output";
        let response = ResponsesResponse::builder("resp_ws_test", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_test".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: output_text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response,
        });
        let _ = outbound_tx.send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response: ResponsesResponse::builder("resp_ws_test", request.model.clone())
                .copy_from_request(&request)
                .status(ResponseStatus::Completed)
                .output(vec![ResponseOutputItem::Message {
                    id: "msg_ws_test".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: output_text.to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: "completed".to_string(),
                }])
                .build(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_test".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello websocket".to_string(),
                }],
                status: Some("completed".to_string()),
            }],
        })
    }
}

#[derive(Clone)]
struct StubWsRouter {
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
}

impl fmt::Debug for StubWsRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StubWsRouter")
    }
}

#[async_trait]
impl RouterTrait for StubWsRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &smg::protocols::chat::ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        StatusCode::NOT_IMPLEMENTED.into_response()
    }

    fn supports_responses_ws(&self) -> bool {
        true
    }

    async fn route_responses_ws(&self, headers: HeaderMap, socket: WebSocket) {
        serve_responses_ws_with_config(
            socket,
            headers,
            self.executor.clone(),
            self.runtime_config.clone(),
        )
        .await;
    }

    fn router_type(&self) -> &'static str {
        "stub-ws"
    }
}

async fn build_stub_app(executor: Arc<dyn WsResponsesExecutor>) -> axum::Router {
    build_stub_app_with_runtime_config(executor, WsRuntimeConfig::default()).await
}

async fn build_stub_app_with_runtime_config(
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
) -> axum::Router {
    let ctx = create_test_app_context().await;
    let router = Arc::new(StubWsRouter {
        executor,
        runtime_config,
    });
    create_test_app_with_context(router, ctx)
}

async fn serve_app(app: axum::Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("ws://{}", addr)
}

async fn recv_json(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> serde_json::Value {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(3), socket.next())
            .await
            .expect("timed out waiting for websocket message")
            .expect("websocket stream ended")
            .expect("websocket receive failed");

        match message {
            tokio_tungstenite::tungstenite::Message::Text(text) => {
                return serde_json::from_str(text.as_ref()).expect("message should be valid JSON");
            }
            tokio_tungstenite::tungstenite::Message::Ping(_) => continue,
            tokio_tungstenite::tungstenite::Message::Pong(_) => continue,
            tokio_tungstenite::tungstenite::Message::Close(frame) => {
                panic!("unexpected websocket close frame: {:?}", frame)
            }
            other => panic!("unexpected websocket message: {:?}", other),
        }
    }
}

async fn send_ws_request_and_collect(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    request: serde_json::Value,
) -> Vec<serde_json::Value> {
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let mut events = Vec::new();
    loop {
        let event = recv_json(socket).await;
        let is_terminal = matches!(
            event["type"].as_str(),
            Some("response.completed") | Some("error")
        );
        events.push(event);
        if is_terminal {
            break;
        }
    }

    events
}

#[derive(Clone, Default)]
struct SemanticWsExecutor {
    durable_store: Arc<std::sync::Mutex<HashMap<String, CachedWsResponse>>>,
}

impl SemanticWsExecutor {
    fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl WsResponsesExecutor for SemanticWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::UnboundedSender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        if request.background.unwrap_or(false) {
            return Err(WsClientError::new(
                "unsupported_parameter",
                "Background mode is not supported in WebSocket Responses V1.",
            ));
        }

        if request.conversation.is_some() {
            return Err(WsClientError::new(
                "unsupported_parameter",
                "The `conversation` field is not supported in WebSocket Responses V1.",
            ));
        }

        let previous_response = if let Some(previous_id) = request.previous_response_id.as_deref() {
            if let Some(cached) = cached_response.filter(|cached| cached.response.id == previous_id)
            {
                Some(cached)
            } else {
                self.durable_store
                    .lock()
                    .unwrap()
                    .get(previous_id)
                    .cloned()
                    .ok_or_else(|| {
                        WsClientError::new(
                            "previous_response_not_found",
                            format!(
                                "Previous response '{}' was not found in the current session or durable storage.",
                                previous_id
                            ),
                        )
                    })?
                    .into()
            }
        } else {
            None
        };

        let response_id = format!("resp_ws_{}", uuid::Uuid::new_v4().simple());
        let output_text = if previous_response.is_some() {
            "stub websocket continuation output"
        } else {
            "stub websocket output"
        };

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": request.model.clone(),
                "output": []
            }
        });
        let _ = outbound_tx.send(Message::Text(created.to_string().into()));

        let response = ResponsesResponse::builder(response_id.clone(), request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_semantic".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: output_text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response,
        });
        let _ = outbound_tx.send(Message::Text(completed.to_string().into()));

        let cached = CachedWsResponse {
            response: response.clone(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_semantic".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello websocket".to_string(),
                }],
                status: Some("completed".to_string()),
            }],
        };

        if request.store.unwrap_or(true) {
            self.durable_store
                .lock()
                .unwrap()
                .insert(cached.response.id.clone(), cached.clone());
        }

        Ok(cached)
    }
}

#[tokio::test]
async fn test_v1_responses_get_requires_websocket_upgrade() {
    let app = build_stub_app(Arc::new(StubWsExecutor::immediate())).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/responses")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        smg::routers::error::extract_error_code_from_response(&response),
        "websocket_upgrade_required"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_unknown_event_type() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({ "type": "response.delete" })
                .to_string()
                .into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(event["code"], "unsupported_event");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_invalid_json() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            "{\"type\":\"response.create\"".into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(event["code"], "invalid_json");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_binary_messages() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Binary(
            vec![0xde, 0xad, 0xbe, 0xef].into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(event["code"], "unsupported_message_type");
}

#[tokio::test]
async fn test_v1_responses_ws_replies_to_ping_and_keeps_session_healthy() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let ping_payload = vec![0x1, 0x2, 0x3, 0x4];
    socket
        .send(tokio_tungstenite::tungstenite::Message::Ping(
            ping_payload.clone().into(),
        ))
        .await
        .unwrap();

    let pong = tokio::time::timeout(Duration::from_secs(3), socket.next())
        .await
        .expect("timed out waiting for pong")
        .expect("websocket stream ended")
        .expect("websocket receive failed");

    match pong {
        tokio_tungstenite::tungstenite::Message::Pong(payload) => {
            assert_eq!(payload.as_ref(), ping_payload.as_slice());
        }
        other => panic!("expected pong after ping, got {:?}", other),
    }

    let events = send_ws_request_and_collect(
        &mut socket,
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "Hello websocket after ping",
                "store": false
            }
        }),
    )
    .await;

    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_closes_when_session_lifetime_expires() {
    let url = serve_app(
        build_stub_app_with_runtime_config(
            Arc::new(StubWsExecutor::immediate()),
            WsRuntimeConfig {
                max_session_lifetime: Duration::from_millis(50),
            },
        )
        .await,
    )
    .await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let close_message = tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("timed out waiting for websocket close")
        .expect("websocket stream ended without close frame")
        .expect("websocket receive failed");

    match close_message {
        tokio_tungstenite::tungstenite::Message::Close(frame) => {
            let frame = frame.expect("expected server close frame");
            assert_eq!(
                frame.reason.to_string(),
                "WebSocket Responses session lifetime expired."
            );
        }
        other => panic!("expected websocket close frame, got {:?}", other),
    }
}

#[tokio::test]
async fn test_v1_responses_ws_response_create_streams_events() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({
                "type": "response.create",
                "response": {
                    "model": "mock-model",
                    "input": "Hello websocket",
                    "store": false
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    let completed = recv_json(&mut socket).await;

    assert_eq!(created["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "stub websocket output"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_accepts_transcript_fixture_request() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let request = serde_json::from_str::<serde_json::Value>(include_str!(
        "../fixtures/responses_ws/transcript_request.json"
    ))
    .expect("transcript fixture should be valid JSON");

    let events = send_ws_request_and_collect(&mut socket, request).await;
    let completed = events.last().unwrap();

    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_second_inflight_request() {
    let gate = Arc::new(Notify::new());
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::gated(gate.clone()))).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let request = serde_json::json!({
        "type": "response.create",
        "response": {
            "model": "mock-model",
            "input": "Hello websocket",
            "store": false
        }
    });

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    assert_eq!(created["type"], "response.created");

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let error = recv_json(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["code"], "concurrent_response_create");

    gate.notify_waiters();
    let completed = recv_json(&mut socket).await;
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_via_router_manager_streams_events() {
    let ctx = create_test_app_context().await;
    let manager = Arc::new(RouterManager::new(Arc::new(WorkerRegistry::new())));
    manager.register_router(
        router_ids::GRPC_REGULAR,
        Arc::new(StubWsRouter {
            executor: Arc::new(StubWsExecutor::immediate()),
            runtime_config: WsRuntimeConfig::default(),
        }),
    );

    let app = create_test_app_with_context(manager as Arc<dyn RouterTrait>, ctx);
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({
                "type": "response.create",
                "response": {
                    "model": "mock-model",
                    "input": "Hello websocket",
                    "store": false
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    let completed = recv_json(&mut socket).await;

    assert_eq!(created["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "stub websocket output"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_same_connection_store_false_continuation_completes() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "First websocket turn",
                "store": false
            }
        }),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");

    let response_id = first_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();

    let second_events = send_ws_request_and_collect(
        &mut socket,
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "Follow up websocket turn",
                "previous_response_id": response_id,
                "store": false
            }
        }),
    )
    .await;

    let second_completed = second_events.last().unwrap();
    assert_eq!(second_completed["type"], "response.completed");

    let second_response_id = second_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();

    let third_events = send_ws_request_and_collect(
        &mut socket,
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "Third websocket turn",
                "previous_response_id": second_response_id,
                "store": false
            }
        }),
    )
    .await;

    let third_completed = third_events.last().unwrap();
    assert_eq!(third_completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_store_true_continuation_survives_reconnect() {
    let executor = Arc::new(SemanticWsExecutor::new());
    let url = serve_app(build_stub_app(executor).await).await;

    let (mut first_socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();
    let first_events = send_ws_request_and_collect(
        &mut first_socket,
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "Persist this websocket turn",
                "store": true
            }
        }),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");
    let response_id = first_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();
    drop(first_socket);

    let (mut second_socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();
    let second_events = send_ws_request_and_collect(
        &mut second_socket,
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "Reconnect follow up websocket turn",
                "previous_response_id": response_id,
                "store": false
            }
        }),
    )
    .await;

    let second_completed = second_events.last().unwrap();
    assert_eq!(second_completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_missing_previous_response_errors() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "Missing previous response id",
                "previous_response_id": "resp_missing_ws",
                "store": false
            }
        }),
    )
    .await;

    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["code"], "previous_response_not_found");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_unsupported_parameters() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    for request in [
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "Background websocket request",
                "background": true
            }
        }),
        serde_json::json!({
            "type": "response.create",
            "response": {
                "model": "mock-model",
                "input": "Conversation websocket request",
                "conversation": "conv_test_123"
            }
        }),
    ] {
        let events = send_ws_request_and_collect(&mut socket, request).await;
        let error = events.last().unwrap();
        assert_eq!(error["type"], "error");
        assert_eq!(error["code"], "unsupported_parameter");
    }
}

#[tokio::test]
async fn test_v1_responses_ws_errors_echo_event_id() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        serde_json::json!({
            "type": "response.create",
            "event_id": "evt_ws_123",
            "response": {
                "model": "mock-model",
                "input": "Conversation websocket request",
                "conversation": "conv_test_123"
            }
        }),
    )
    .await;

    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["code"], "unsupported_parameter");
    assert_eq!(error["event_id"], "evt_ws_123");
    assert!(error["message"].is_string());
}
