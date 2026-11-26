//! OpenAI-compatible routes.
//!
//! `POST /v1/chat/completions` and `POST /v1/completions` share one scheduler.
//! Streaming responses are `text/event-stream` frames terminated by
//! `data: [DONE]`. `n` other than 1 is rejected: concurrent HTTP requests are
//! how a client batches.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use serde::Deserialize;
use serde_json::{json, Value};

use flint_core::{ChatTurn, SamplingParams};

use crate::engine::{EngineEvent, EngineHandle, FinishReason, Submission, SubmitError, TextCodec};

/// Shared server state.
#[derive(Clone)]
pub(crate) struct AppState {
    /// Engine command channel.
    pub engine: EngineHandle,
    /// Tokenizer or the dummy byte codec.
    pub codec: Arc<dyn TextCodec>,
    /// Name reported by `/v1/models`.
    pub model_name: String,
    /// Hard cap on prompt tokens.
    pub max_seq_len: usize,
    /// `max_tokens` when the request omits it.
    pub default_max_tokens: usize,
    /// Largest `max_tokens` a request may ask for.
    pub max_new_tokens: usize,
    /// `num_blocks * block_size`. A single request cannot exceed this.
    pub max_cache_tokens: usize,
}

/// Routes served by [`crate::serve`].
pub(crate) fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": unix_secs(),
            "owned_by": "flint",
        }],
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<ChatRequest>,
) -> Result<axum::response::Response, ApiFailure> {
    if let Some(n) = body.n {
        if n != 1 {
            return Err(failure(StatusCode::BAD_REQUEST, "only n = 1 is supported"));
        }
    }
    let sampling = sampling_params(body.temperature, body.top_p, body.top_k, body.seed)?;
    let max_tokens = clamp_max_tokens(body.max_completion_tokens.or(body.max_tokens), &state)?;
    let turns = chat_turns(&body.messages)?;
    let prompt = state
        .codec
        .apply_chat_template(&turns)
        .map_err(|err| failure(StatusCode::BAD_REQUEST, err.to_string()))?;
    let tokens = state
        .codec
        .encode(&prompt)
        .map_err(|err| failure(StatusCode::BAD_REQUEST, err.to_string()))?;
    let prepared = fit_tokens(tokens, max_tokens, &state)?;
    let submission = submit(&state, prepared, sampling, stop_list(body.stop))?;
    let id = new_id("chatcmpl");
    let created = unix_secs();
    if body.stream {
        Ok(
            stream_response(submission, id, created, state.model_name, StreamKind::Chat)
                .into_response(),
        )
    } else {
        Ok(chat_json(submission, id, created, state.model_name)
            .await
            .into_response())
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(body): Json<CompletionRequest>,
) -> Result<axum::response::Response, ApiFailure> {
    if let Some(n) = body.n {
        if n != 1 {
            return Err(failure(StatusCode::BAD_REQUEST, "only n = 1 is supported"));
        }
    }
    let sampling = sampling_params(body.temperature, body.top_p, body.top_k, body.seed)?;
    let max_tokens = clamp_max_tokens(body.max_tokens, &state)?;
    let tokens = prompt_tokens(&state.codec, body.prompt)?;
    let prepared = fit_tokens(tokens, max_tokens, &state)?;
    let submission = submit(&state, prepared, sampling, stop_list(body.stop))?;
    let id = new_id("cmpl");
    let created = unix_secs();
    if body.stream {
        Ok(
            stream_response(submission, id, created, state.model_name, StreamKind::Text)
                .into_response(),
        )
    } else {
        Ok(text_json(submission, id, created, state.model_name)
            .await
            .into_response())
    }
}

fn submit(
    state: &AppState,
    prepared: Prepared,
    sampling: SamplingParams,
    stop: Vec<String>,
) -> Result<Submission, ApiFailure> {
    state
        .engine
        .submit(prepared.tokens, sampling, prepared.max_tokens, stop)
        .map_err(|err| match err {
            SubmitError::Overloaded => {
                failure(StatusCode::SERVICE_UNAVAILABLE, "server is overloaded")
            }
            SubmitError::Dead => failure(StatusCode::INTERNAL_SERVER_ERROR, "engine has stopped"),
        })
}

struct Prepared {
    tokens: Vec<u32>,
    max_tokens: usize,
}

fn fit_tokens(
    tokens: Vec<u32>,
    max_tokens: usize,
    state: &AppState,
) -> Result<Prepared, ApiFailure> {
    if tokens.is_empty() {
        return Err(failure(StatusCode::BAD_REQUEST, "prompt is empty"));
    }
    if tokens.len() > state.max_seq_len {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt has {} tokens, which is above max_seq_len {}",
                tokens.len(),
                state.max_seq_len
            ),
        ));
    }
    if tokens.len() > state.max_cache_tokens {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt has {} tokens, which is above the KV cache capacity of {}",
                tokens.len(),
                state.max_cache_tokens
            ),
        ));
    }
    let room = state
        .max_cache_tokens
        .saturating_sub(tokens.len())
        .min(state.max_seq_len.saturating_sub(tokens.len()));
    if room == 0 {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            "no room left in the context window for generated tokens",
        ));
    }
    Ok(Prepared {
        tokens,
        max_tokens: max_tokens.min(room),
    })
}

fn clamp_max_tokens(requested: Option<usize>, state: &AppState) -> Result<usize, ApiFailure> {
    let max_tokens = requested.unwrap_or(state.default_max_tokens);
    if max_tokens == 0 {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            "max_tokens must be at least 1",
        ));
    }
    if max_tokens > state.max_new_tokens {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            format!(
                "max_tokens {max_tokens} exceeds the server cap of {}",
                state.max_new_tokens
            ),
        ));
    }
    Ok(max_tokens)
}

fn sampling_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
) -> Result<SamplingParams, ApiFailure> {
    let params = SamplingParams {
        temperature: temperature.unwrap_or(1.0),
        top_k,
        top_p,
        seed,
    };
    params
        .validate()
        .map_err(|err| failure(StatusCode::BAD_REQUEST, err.to_string()))?;
    Ok(params)
}

fn chat_turns(messages: &[IncomingMessage]) -> Result<Vec<ChatTurn>, ApiFailure> {
    if messages.is_empty() {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
        ));
    }
    let mut turns = Vec::with_capacity(messages.len());
    for message in messages {
        turns.push(ChatTurn {
            role: message.role.clone(),
            content: message_text(message)?,
        });
    }
    Ok(turns)
}

fn message_text(message: &IncomingMessage) -> Result<String, ApiFailure> {
    let Some(content) = &message.content else {
        return Ok(String::new());
    };
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut text = String::new();
            for part in parts {
                if matches!(part.kind.as_deref(), Some("image_url" | "input_audio")) {
                    return Err(failure(
                        StatusCode::BAD_REQUEST,
                        "phase 1 accepts text content only",
                    ));
                }
                if let Some(piece) = &part.text {
                    text.push_str(piece);
                }
            }
            Ok(text)
        }
    }
}

fn prompt_tokens(codec: &Arc<dyn TextCodec>, prompt: PromptInput) -> Result<Vec<u32>, ApiFailure> {
    match prompt {
        PromptInput::Text(text) => codec
            .encode(&text)
            .map_err(|err| failure(StatusCode::BAD_REQUEST, err.to_string())),
        PromptInput::Ids(ids) => {
            if ids.is_empty() {
                Err(failure(StatusCode::BAD_REQUEST, "prompt is empty"))
            } else {
                Ok(ids)
            }
        }
        PromptInput::Texts(texts) => {
            if texts.len() != 1 {
                return Err(failure(
                    StatusCode::BAD_REQUEST,
                    "prompt arrays must contain exactly one string; issue concurrent requests to batch",
                ));
            }
            codec
                .encode(&texts[0])
                .map_err(|err| failure(StatusCode::BAD_REQUEST, err.to_string()))
        }
    }
}

fn stop_list(stop: Option<StopField>) -> Vec<String> {
    match stop {
        None => Vec::new(),
        Some(StopField::One(text)) => vec![text],
        Some(StopField::Many(texts)) => texts,
    }
}

struct Collected {
    text: String,
    reason: FinishReason,
    prompt_tokens: usize,
    completion_tokens: usize,
}

async fn collect(mut submission: Submission) -> Result<Collected, ApiFailure> {
    let mut text = String::new();
    while let Some(event) = submission.events.recv().await {
        match event {
            EngineEvent::Delta(delta) => text.push_str(&delta),
            EngineEvent::Done {
                reason,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                return Ok(Collected {
                    text,
                    reason,
                    prompt_tokens,
                    completion_tokens,
                });
            }
            EngineEvent::Error(message) => return Err(error_status(message)),
        }
    }
    Err(failure(
        StatusCode::INTERNAL_SERVER_ERROR,
        "engine disconnected",
    ))
}

fn error_status(message: String) -> ApiFailure {
    let status = if message.contains("exceeds KV")
        || message.contains("prompt is empty")
        || message.contains("max_tokens")
    {
        StatusCode::BAD_REQUEST
    } else if message.contains("too many waiting") {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    failure(status, message)
}

async fn chat_json(
    submission: Submission,
    id: String,
    created: u64,
    model: String,
) -> impl IntoResponse {
    match collect(submission).await {
        Ok(done) => (
            StatusCode::OK,
            Json(json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": done.text},
                    "finish_reason": done.reason.as_str(),
                }],
                "usage": usage(done.prompt_tokens, done.completion_tokens),
            })),
        )
            .into_response(),
        Err(err) => err.into_response(),
    }
}

async fn text_json(
    submission: Submission,
    id: String,
    created: u64,
    model: String,
) -> impl IntoResponse {
    match collect(submission).await {
        Ok(done) => (
            StatusCode::OK,
            Json(json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": done.text,
                    "finish_reason": done.reason.as_str(),
                }],
                "usage": usage(done.prompt_tokens, done.completion_tokens),
            })),
        )
            .into_response(),
        Err(err) => err.into_response(),
    }
}

#[derive(Clone, Copy)]
enum StreamKind {
    Chat,
    Text,
}

fn stream_response(
    submission: Submission,
    id: String,
    created: u64,
    model: String,
    kind: StreamKind,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream = async_stream::stream! {
        let mut submission = submission;
        let mut role_sent = false;
        while let Some(event) = submission.events.recv().await {
            match event {
                EngineEvent::Delta(text) => {
                    let payload = chunk_json(
                        ChunkHeader { id: &id, created, model: &model, kind },
                        Some(&text),
                        None,
                        !role_sent,
                        None,
                    );
                    role_sent = true;
                    yield Ok(Event::default().data(payload));
                }
                EngineEvent::Done { reason, prompt_tokens, completion_tokens, .. } => {
                    let payload = chunk_json(
                        ChunkHeader { id: &id, created, model: &model, kind },
                        None,
                        Some(reason.as_str()),
                        !role_sent,
                        Some((prompt_tokens, completion_tokens)),
                    );
                    yield Ok(Event::default().data(payload));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineEvent::Error(message) => {
                    let body = json!({"error": {"message": message, "type": "server_error"}});
                    yield Ok(Event::default().data(body.to_string()));
                    break;
                }
            }
        }
    };
    Sse::new(stream)
}

struct ChunkHeader<'a> {
    id: &'a str,
    created: u64,
    model: &'a str,
    kind: StreamKind,
}

fn chunk_json(
    header: ChunkHeader<'_>,
    text: Option<&str>,
    finish: Option<&str>,
    include_role: bool,
    usage_tokens: Option<(usize, usize)>,
) -> String {
    let ChunkHeader {
        id,
        created,
        model,
        kind,
    } = header;
    let choice = match kind {
        StreamKind::Chat => {
            let mut delta = serde_json::Map::new();
            if include_role {
                delta.insert("role".into(), json!("assistant"));
            }
            if let Some(text) = text {
                delta.insert("content".into(), json!(text));
            }
            json!({
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            })
        }
        StreamKind::Text => json!({
            "index": 0,
            "text": text.unwrap_or(""),
            "finish_reason": finish,
        }),
    };
    let object = match kind {
        StreamKind::Chat => "chat.completion.chunk",
        StreamKind::Text => "text_completion",
    };
    let mut body = json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [choice],
    });
    if let Some((prompt, completion)) = usage_tokens {
        body["usage"] = usage(prompt, completion);
    }
    body.to_string()
}

fn usage(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

fn failure(status: StatusCode, message: impl Into<String>) -> ApiFailure {
    ApiFailure {
        status,
        message: message.into(),
    }
}

fn new_id(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}-{n:x}", unix_secs())
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// JSON error body plus an HTTP status.
struct ApiFailure {
    status: StatusCode,
    message: String,
}

impl IntoResponse for ApiFailure {
    fn into_response(self) -> axum::response::Response {
        let kind = if self.status == StatusCode::SERVICE_UNAVAILABLE {
            "overloaded"
        } else if self.status.is_server_error() {
            "server_error"
        } else {
            "invalid_request_error"
        };
        (
            self.status,
            Json(json!({"error": {"message": self.message, "type": kind}})),
        )
            .into_response()
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<IncomingMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(default)]
    n: Option<u32>,
}

#[derive(Deserialize)]
struct IncomingMessage {
    role: String,
    #[serde(default)]
    content: Option<MessageContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: PromptInput,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(default)]
    n: Option<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PromptInput {
    Text(String),
    Ids(Vec<u32>),
    Texts(Vec<String>),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopField {
    One(String),
    Many(Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{spawn, LanguageModel};
    use flint_core::Error;
    use flint_kv::PagedKvCache;
    use flint_schedule::SchedulerConfig;
    use tokio::net::TcpListener;

    struct CharCodec;

    impl TextCodec for CharCodec {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.chars().map(|ch| ch as u32).collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String, Error> {
            Ok(ids.iter().filter_map(|id| char::from_u32(*id)).collect())
        }

        fn apply_chat_template(&self, messages: &[ChatTurn]) -> Result<String, Error> {
            Ok(messages
                .iter()
                .map(|message| message.content.clone())
                .collect::<Vec<_>>()
                .join(""))
        }
    }

    /// Always samples `token`. Writes zeros into the cache so the real paged
    /// attention path is exercised by the HTTP tests.
    struct ForcedModel {
        token: u32,
        eos: Vec<u32>,
    }

    impl LanguageModel for ForcedModel {
        fn forward_chunk(
            &mut self,
            tokens: &[u32],
            position: usize,
            cache: &mut PagedKvCache,
            block_table: &[u32],
            want_logits: bool,
        ) -> Result<Option<Vec<f32>>, Error> {
            let width = cache.num_kv_heads() * cache.head_dim();
            let zeros = vec![0.0; width];
            let block_size = cache.block_size();
            for (offset, _) in tokens.iter().enumerate() {
                let pos = position + offset;
                let block = *block_table
                    .get(pos / block_size)
                    .ok_or_else(|| Error::Inference(format!("block table does not cover {pos}")))?;
                cache
                    .write_token(0, block, pos % block_size, &zeros, &zeros)
                    .map_err(|err| Error::Inference(err.to_string()))?;
            }
            if !want_logits {
                return Ok(None);
            }
            let mut logits = vec![0.0; 128];
            logits[self.token as usize] = 10.0;
            Ok(Some(logits))
        }

        fn vocab_size(&self) -> usize {
            128
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn num_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            4
        }
        fn eos_token_ids(&self) -> &[u32] {
            &self.eos
        }
    }

    async fn spawn_server(num_blocks: usize) -> String {
        let model = Box::new(ForcedModel {
            token: b'A' as u32,
            eos: Vec::new(),
        });
        let codec: Arc<dyn TextCodec> = Arc::new(CharCodec);
        let (engine, _thread) = spawn(
            model,
            None,
            Arc::clone(&codec),
            num_blocks,
            4,
            SchedulerConfig {
                max_num_seqs: 4,
                max_batched_tokens: 64,
                prefill_chunk_size: 4,
                max_waiting: 8,
            },
        )
        .unwrap();
        let state = AppState {
            engine,
            codec,
            model_name: "test-model".into(),
            max_seq_len: 64,
            default_max_tokens: 8,
            max_new_tokens: 32,
            max_cache_tokens: num_blocks * 4,
        };
        let app = router(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn client() -> reqwest::Client {
        reqwest::Client::new()
    }

    #[tokio::test]
    async fn health_and_models() {
        let base = spawn_server(8).await;
        let client = client();
        let health: Value = client
            .get(format!("{base}/health"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(health["status"], "ok");
        let models: Value = client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(models["data"][0]["id"], "test-model");
    }

    #[tokio::test]
    async fn chat_completion_is_the_forced_token() {
        let base = spawn_server(8).await;
        let body = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "temperature": 0
        });
        let response: Value = client()
            .post(format!("{base}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(response["choices"][0]["message"]["content"], "AAAA");
        assert_eq!(response["choices"][0]["finish_reason"], "length");
        assert_eq!(response["usage"]["completion_tokens"], 4);
        assert_eq!(response["usage"]["prompt_tokens"], 2);
    }

    #[tokio::test]
    async fn chat_stream_ends_with_done() {
        let base = spawn_server(8).await;
        let body = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Yo"}]}],
            "max_tokens": 3,
            "temperature": 0,
            "stream": true
        });
        let response = client()
            .post(format!("{base}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        let content_type = response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(content_type.contains("text/event-stream"), "{content_type}");
        let text = response.text().await.unwrap();
        assert!(text.contains("data: [DONE]"), "{text}");
        assert_eq!(text.matches("\"content\":\"A\"").count(), 3, "{text}");
        assert!(text.contains("\"finish_reason\":\"length\""), "{text}");
    }

    #[tokio::test]
    async fn completions_and_stop_string() {
        let base = spawn_server(8).await;
        let body = json!({
            "model": "test-model",
            "prompt": "Hi",
            "max_tokens": 8,
            "temperature": 0,
            "stop": "AA"
        });
        let response: Value = client()
            .post(format!("{base}/v1/completions"))
            .json(&body)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(response["choices"][0]["text"], "");
        assert_eq!(response["choices"][0]["finish_reason"], "stop");
        assert_eq!(response["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn rejects_n_and_an_oversized_prompt() {
        let base = spawn_server(4).await;
        let client = client();
        let n = client
            .post(format!("{base}/v1/completions"))
            .json(&json!({"model": "test-model", "prompt": "Hi", "n": 2}))
            .send()
            .await
            .unwrap();
        assert_eq!(n.status(), 400);
        let long = "x".repeat(20);
        let too_long = client
            .post(format!("{base}/v1/completions"))
            .json(&json!({"model": "test-model", "prompt": long, "max_tokens": 1}))
            .send()
            .await
            .unwrap();
        assert_eq!(too_long.status(), 400);
    }

    #[tokio::test]
    async fn two_requests_share_a_tiny_cache() {
        let base = spawn_server(4).await;
        let client = client();
        let body = json!({
            "model": "test-model",
            "prompt": "12345678",
            "max_tokens": 2,
            "temperature": 0
        });
        let first = client
            .post(format!("{base}/v1/completions"))
            .json(&body)
            .send();
        let second = client
            .post(format!("{base}/v1/completions"))
            .json(&body)
            .send();
        let (left, right) = tokio::join!(first, second);
        for response in [left.unwrap(), right.unwrap()] {
            assert_eq!(response.status(), 200);
            let value: Value = response.json().await.unwrap();
            assert_eq!(value["choices"][0]["text"], "AA", "{value}");
            assert_eq!(value["choices"][0]["finish_reason"], "length");
        }
    }
}
