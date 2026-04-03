use std::sync::Arc;

use async_trait::async_trait;
use axum::{extract::ws::Message, http::HeaderMap};
use serde_json::json;
use tokio::sync::mpsc;
use validator::Validate;

use super::{
    common::{load_conversation_history_with_cache, normalize_request_input_items},
    conversions,
    streaming::{execute_non_mcp_stream_with_sink, execute_tool_loop_streaming_with_sink},
};
use crate::{
    core::WorkerRegistry,
    protocols::{
        responses::{generate_id, ResponseStatus, ResponsesRequest, ResponsesResponse},
        validated::Normalizable,
    },
    routers::{
        error,
        grpc::{
            common::responses::{
                ensure_mcp_connection,
                utils::{persist_response_if_needed, validate_worker_availability},
                ResponsesContext,
            },
            harmony::HarmonyDetector,
        },
        ws_responses::{
            CachedWsResponse, WsClientError, WsResponseCreateOptions, WsResponsesExecutor,
        },
    },
};

#[derive(Clone)]
pub(crate) struct GrpcWsResponsesExecutor {
    worker_registry: Arc<WorkerRegistry>,
    responses_context: ResponsesContext,
}

impl GrpcWsResponsesExecutor {
    pub fn new(worker_registry: Arc<WorkerRegistry>, responses_context: ResponsesContext) -> Self {
        Self {
            worker_registry,
            responses_context,
        }
    }
}

#[async_trait]
impl WsResponsesExecutor for GrpcWsResponsesExecutor {
    async fn execute_response_create(
        &self,
        headers: HeaderMap,
        mut request: ResponsesRequest,
        options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::UnboundedSender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        request.normalize();
        // WebSocket Responses is inherently event-streamed, so force streaming
        // on the downstream chat pipeline regardless of the client payload.
        request.stream = Some(true);
        request.background = Some(false);
        request
            .validate()
            .map_err(|err| WsClientError::new("invalid_request", err.to_string()))?;

        if request.conversation.is_some() {
            return Err(WsClientError::new(
                "unsupported_parameter",
                "The `conversation` field is not supported in WebSocket Responses V1.",
            ));
        }

        if let Some(error_response) =
            validate_worker_availability(&self.worker_registry, request.model.as_str())
        {
            return Err(response_to_ws_error(error_response));
        }

        if HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &request.model) {
            return Err(WsClientError::new(
                "unsupported_model",
                "Harmony-backed Responses are not supported on the WebSocket path in V1.",
            ));
        }

        let ctx = self.responses_context.clone_for_request();
        let modified_request =
            load_conversation_history_with_cache(&ctx, &request, cached_response.as_ref(), true)
                .await
                .map_err(response_to_ws_error)?;

        if options.generate == Some(false) {
            return warmup_response_create(&ctx, &request, &modified_request, outbound_tx).await;
        }

        let (has_mcp_tools, server_keys) =
            ensure_mcp_connection(&ctx.mcp_manager, request.tools.as_deref())
                .await
                .map_err(response_to_ws_error)?;

        {
            let mut servers = ctx.requested_servers.write().unwrap();
            *servers = server_keys;
        }

        let final_response = if has_mcp_tools {
            execute_tool_loop_streaming_with_sink(
                &ctx,
                modified_request.clone(),
                &request,
                Some(headers),
                Some(request.model.clone()),
                outbound_tx,
            )
            .await
            .map_err(|err| WsClientError::new("stream_execution_failed", err))?
        } else {
            let chat_request = conversions::responses_to_chat(&modified_request)
                .map_err(|err| WsClientError::new("convert_request_failed", err.to_string()))?;

            execute_non_mcp_stream_with_sink(
                &ctx,
                Arc::new(chat_request),
                request.clone(),
                Some(headers),
                Some(request.model.clone()),
                &crate::routers::grpc::common::responses::streaming::WsResponseEventSink::new(
                    outbound_tx,
                ),
            )
            .await
            .map_err(|err| WsClientError::new("stream_execution_failed", err))?
        };

        Ok(CachedWsResponse {
            response: final_response,
            input_items: normalize_request_input_items(&modified_request),
        })
    }
}

fn response_to_ws_error(response: axum::response::Response) -> WsClientError {
    let status = response.status();
    let header_code = response
        .headers()
        .get(error::HEADER_X_SMG_ERROR_CODE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("responses_ws_error")
        .to_string();

    WsClientError::new(
        header_code,
        format!("WebSocket Responses request failed with status {}", status),
    )
    .with_status(status.as_u16())
    .with_param_if_previous_response_not_found()
}

async fn warmup_response_create(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
    modified_request: &ResponsesRequest,
    outbound_tx: mpsc::UnboundedSender<Message>,
) -> Result<CachedWsResponse, WsClientError> {
    let response = ResponsesResponse::builder(generate_id("resp"), &request.model)
        .copy_from_request(request)
        .status(ResponseStatus::Completed)
        .output(vec![])
        .build();

    let created = json!({
        "type": "response.created",
        "response": {
            "id": response.id,
            "object": "response",
            "status": "in_progress",
            "model": response.model,
            "output": []
        }
    });
    send_ws_message(&outbound_tx, created)?;

    let completed = json!({
        "type": "response.completed",
        "response": response.clone(),
    });
    send_ws_message(&outbound_tx, completed)?;

    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &response,
        request,
    )
    .await;

    Ok(CachedWsResponse {
        response,
        input_items: normalize_request_input_items(modified_request),
    })
}

fn send_ws_message(
    outbound_tx: &mpsc::UnboundedSender<Message>,
    payload: serde_json::Value,
) -> Result<(), WsClientError> {
    outbound_tx
        .send(Message::Text(payload.to_string().into()))
        .map_err(|_| {
            WsClientError::new(
                "client_disconnected",
                "WebSocket client disconnected before response delivery.",
            )
            .with_status(499)
        })
}

trait WsClientErrorExt {
    fn with_param_if_previous_response_not_found(self) -> Self;
}

impl WsClientErrorExt for WsClientError {
    fn with_param_if_previous_response_not_found(self) -> Self {
        if self.code == "previous_response_not_found" {
            self.with_param("previous_response_id")
        } else {
            self
        }
    }
}
