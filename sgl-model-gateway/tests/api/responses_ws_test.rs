use std::{fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::ws::{Message, WebSocket},
    http::{HeaderMap, Request, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::{SinkExt, StreamExt};
use smg::{
    protocols::responses::{
        ResponseContentPart, ResponseInputOutputItem, ResponseOutputItem, ResponseStatus,
        ResponsesRequest, ResponsesResponse,
    },
    routers::{
        ws_responses::{serve_responses_ws, CachedWsResponse, WsClientError, WsResponsesExecutor},
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
        serve_responses_ws(socket, headers, self.executor.clone()).await;
    }

    fn router_type(&self) -> &'static str {
        "stub-ws"
    }
}

async fn build_stub_app(executor: Arc<dyn WsResponsesExecutor>) -> axum::Router {
    let ctx = create_test_app_context().await;
    let router = Arc::new(StubWsRouter { executor });
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
    let message = tokio::time::timeout(Duration::from_secs(3), socket.next())
        .await
        .expect("timed out waiting for websocket message")
        .expect("websocket stream ended")
        .expect("websocket receive failed");

    match message {
        tokio_tungstenite::tungstenite::Message::Text(text) => {
            serde_json::from_str(text.as_ref()).expect("message should be valid JSON")
        }
        other => panic!("unexpected websocket message: {:?}", other),
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
