//! Single-threaded engine that turns scheduler steps into tokens.
//!
//! HTTP handlers submit prompts over a bounded channel and read token events
//! back. The engine thread is the only thing that touches the model and the
//! KV cache, so the forward pass stays synchronous and the scheduler stays
//! race-free. On CPU each sequence in an iteration is executed one after
//! another; the iteration is still planned as a batch so the same scheduler
//! can drive a fused GPU kernel later.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{
    sync_channel, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError,
};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use flint_core::{sample, ChatTurn, Error, Model, SamplingParams};
use flint_kv::PagedKvCache;
use flint_schedule::{PushOutcome, Scheduler, SchedulerConfig};
use rand::rngs::StdRng;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::filter::{text_delta, StreamFilter};

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// Events a client reads, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EngineEvent {
    /// Text that is safe to append. Stop strings have already been removed.
    Delta(String),
    /// Generation finished. `prompt_tokens` is the encoded prompt.
    Done {
        /// Stop string, eos, or the max-token cap.
        reason: FinishReason,
        /// Encoded prompt length.
        prompt_tokens: usize,
        /// Sampled tokens, including the one that hit eos or a stop string.
        completion_tokens: usize,
    },
    /// The request will produce no further events.
    Error(String),
}

struct Command {
    tokens: Vec<u32>,
    sampling: SamplingParams,
    max_tokens: usize,
    stop: Vec<String>,
    cancel: Arc<AtomicBool>,
    events: UnboundedSender<EngineEvent>,
}

struct Live {
    sampling: SamplingParams,
    rng: StdRng,
    stop: StreamFilter,
    generated_ids: Vec<u32>,
    decoded: String,
    events: UnboundedSender<EngineEvent>,
    cancel: Arc<AtomicBool>,
    prompt_len: usize,
    completion_tokens: usize,
}

/// Handle used by HTTP handlers. Cloning it shares the command channel.
#[derive(Clone)]
pub(crate) struct EngineHandle {
    tx: SyncSender<Command>,
}

/// Receiver plus a cancel flag that trips when this value is dropped.
pub(crate) struct Submission {
    pub events: UnboundedReceiver<EngineEvent>,
    cancel: Arc<AtomicBool>,
}

impl Drop for Submission {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}

/// Why [`EngineHandle::submit`] refused a request before the scheduler saw it.
#[derive(Debug)]
pub(crate) enum SubmitError {
    /// The command queue is full.
    Overloaded,
    /// The engine thread has exited.
    Dead,
}

/// Something the engine can run a forward chunk through.
pub(crate) trait LanguageModel: Send {
    /// Run one chunk. `Ok(None)` when logits were not requested.
    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        position: usize,
        cache: &mut PagedKvCache,
        block_table: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, Error>;

    /// Vocabulary width of the logit vector.
    fn vocab_size(&self) -> usize;
    /// Transformer layers.
    fn num_layers(&self) -> usize;
    /// KV heads stored in the cache.
    fn num_kv_heads(&self) -> usize;
    /// Channels per head.
    fn head_dim(&self) -> usize;
    /// Token ids that end generation.
    fn eos_token_ids(&self) -> &[u32];
}

impl LanguageModel for Model {
    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        position: usize,
        cache: &mut PagedKvCache,
        block_table: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, Error> {
        Ok(self
            .forward(tokens, position, cache, block_table, want_logits)?
            .logits)
    }

    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    fn num_layers(&self) -> usize {
        self.config().num_hidden_layers
    }

    fn num_kv_heads(&self) -> usize {
        self.config().num_key_value_heads
    }

    fn head_dim(&self) -> usize {
        self.config().head_dim()
    }

    fn eos_token_ids(&self) -> &[u32] {
        Model::eos_token_ids(self)
    }
}

/// Encode prompts and decode generated ids.
pub(crate) trait TextCodec: Send + Sync {
    /// Encode raw text. Special tokens inside the string stay atomic.
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error>;
    /// Decode ids, skipping special tokens.
    fn decode(&self, ids: &[u32]) -> Result<String, Error>;
    /// Render a chat conversation into the string that [`Self::encode`] sees.
    fn apply_chat_template(&self, messages: &[ChatTurn]) -> Result<String, Error>;
}

/// Start the engine thread.
pub(crate) fn spawn(
    model: Box<dyn LanguageModel>,
    codec: Arc<dyn TextCodec>,
    num_blocks: usize,
    block_size: usize,
    sched: SchedulerConfig,
) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    tracing::debug!(
        vocab = model.vocab_size(),
        layers = model.num_layers(),
        "starting engine"
    );
    let cache = PagedKvCache::with_dims(
        num_blocks,
        block_size,
        model.num_layers(),
        model.num_kv_heads(),
        model.head_dim(),
    )
    .map_err(|err| Error::Inference(err.to_string()))?;
    let scheduler = Scheduler::new(sched, num_blocks, block_size)
        .map_err(|err| Error::Inference(err.to_string()))?;
    let (tx, rx) = sync_channel(256);
    let handle = thread::Builder::new()
        .name("flint-engine".into())
        .spawn(move || engine_loop(model, codec, scheduler, cache, rx))
        .map_err(|err| Error::Inference(format!("spawn engine thread: {err}")))?;
    Ok((EngineHandle { tx }, handle))
}

impl EngineHandle {
    pub(crate) fn submit(
        &self,
        tokens: Vec<u32>,
        sampling: SamplingParams,
        max_tokens: usize,
        stop: Vec<String>,
    ) -> Result<Submission, SubmitError> {
        let (events_tx, events) = unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let command = Command {
            tokens,
            sampling,
            max_tokens,
            stop,
            cancel: Arc::clone(&cancel),
            events: events_tx,
        };
        match self.tx.try_send(command) {
            Ok(()) => Ok(Submission { events, cancel }),
            Err(TrySendError::Full(_)) => Err(SubmitError::Overloaded),
            Err(TrySendError::Disconnected(_)) => Err(SubmitError::Dead),
        }
    }
}

fn engine_loop(
    mut model: Box<dyn LanguageModel>,
    codec: Arc<dyn TextCodec>,
    mut scheduler: Scheduler,
    mut cache: PagedKvCache,
    rx: Receiver<Command>,
) {
    let eos: Vec<u32> = model.eos_token_ids().to_vec();
    let mut live: HashMap<u64, Live> = HashMap::new();
    let mut next_id = 1_u64;

    loop {
        loop {
            match rx.try_recv() {
                Ok(command) => accept(&mut scheduler, &mut live, &mut next_id, command),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        reap_cancelled(&mut scheduler, &mut live);

        let step = scheduler.schedule();
        let idle = step.items.is_empty();
        for failure in step.failed {
            if let Some(req) = live.remove(&failure.request_id) {
                let _ = req.events.send(EngineEvent::Error(failure.reason));
            }
        }
        if !step.preempted.is_empty() {
            tracing::debug!(ids = ?step.preempted, "preempted sequences");
        }

        for item in step.items {
            if cancelled(&live, item.request_id) {
                drop_request(&mut scheduler, &mut live, item.request_id, None);
                continue;
            }
            let logits = match model.forward_chunk(
                &item.tokens,
                item.position,
                &mut cache,
                &item.block_table,
                item.sample_after,
            ) {
                Ok(logits) => logits,
                Err(err) => {
                    drop_request(
                        &mut scheduler,
                        &mut live,
                        item.request_id,
                        Some(err.to_string()),
                    );
                    continue;
                }
            };
            if let Err(err) = scheduler.mark_computed(item.request_id, item.tokens.len()) {
                drop_request(
                    &mut scheduler,
                    &mut live,
                    item.request_id,
                    Some(err.to_string()),
                );
                continue;
            }
            if item.sample_after {
                let Some(logits) = logits else {
                    drop_request(
                        &mut scheduler,
                        &mut live,
                        item.request_id,
                        Some("forward pass returned no logits".into()),
                    );
                    continue;
                };
                sample_and_emit(
                    &mut scheduler,
                    &mut live,
                    codec.as_ref(),
                    &eos,
                    item.request_id,
                    &logits,
                );
            }
        }

        if idle {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(command) => accept(&mut scheduler, &mut live, &mut next_id, command),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

fn accept(
    scheduler: &mut Scheduler,
    live: &mut HashMap<u64, Live>,
    next_id: &mut u64,
    command: Command,
) {
    let id = *next_id;
    *next_id = next_id.wrapping_add(1);
    let prompt_len = command.tokens.len();
    match scheduler.add_request(id, command.tokens, command.max_tokens) {
        Ok(()) => {
            let rng = command.sampling.rng();
            live.insert(
                id,
                Live {
                    sampling: command.sampling,
                    rng,
                    stop: StreamFilter::new(command.stop),
                    generated_ids: Vec::new(),
                    decoded: String::new(),
                    events: command.events,
                    cancel: command.cancel,
                    prompt_len,
                    completion_tokens: 0,
                },
            );
        }
        Err(err) => {
            let _ = command.events.send(EngineEvent::Error(err.to_string()));
        }
    }
}

fn reap_cancelled(scheduler: &mut Scheduler, live: &mut HashMap<u64, Live>) {
    let gone: Vec<u64> = live
        .iter()
        .filter(|(_, req)| req.cancel.load(Ordering::Acquire))
        .map(|(id, _)| *id)
        .collect();
    for id in gone {
        drop_request(scheduler, live, id, None);
    }
}

fn cancelled(live: &HashMap<u64, Live>, id: u64) -> bool {
    live.get(&id)
        .is_none_or(|req| req.cancel.load(Ordering::Acquire))
}

fn drop_request(
    scheduler: &mut Scheduler,
    live: &mut HashMap<u64, Live>,
    id: u64,
    message: Option<String>,
) {
    let _ = scheduler.finish(id);
    if let Some(req) = live.remove(&id) {
        if let Some(message) = message {
            let _ = req.events.send(EngineEvent::Error(message));
        }
    }
}

enum SampleStep {
    Missing,
    Cancelled,
    Failed(String),
    Stopped { tail: String },
    Token { token: u32, visible: String },
}

fn sample_and_emit(
    scheduler: &mut Scheduler,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    id: u64,
    logits: &[f32],
) {
    let step = sample_step(live, codec, eos, id, logits);
    match step {
        SampleStep::Missing => {
            let _ = scheduler.finish(id);
        }
        SampleStep::Cancelled => drop_request(scheduler, live, id, None),
        SampleStep::Failed(message) => drop_request(scheduler, live, id, Some(message)),
        SampleStep::Stopped { tail } => finish_stop(scheduler, live, id, tail),
        SampleStep::Token { token, visible } => push_sampled(scheduler, live, id, token, visible),
    }
}

fn sample_step(
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    id: u64,
    logits: &[f32],
) -> SampleStep {
    let Some(req) = live.get_mut(&id) else {
        return SampleStep::Missing;
    };
    if req.cancel.load(Ordering::Acquire) {
        return SampleStep::Cancelled;
    }
    let token = match sample(logits, &req.sampling, &mut req.rng) {
        Ok(token) => token,
        Err(err) => return SampleStep::Failed(err.to_string()),
    };
    req.completion_tokens += 1;
    if eos.contains(&token) {
        let tail = req.stop.flush();
        return SampleStep::Stopped { tail };
    }
    req.generated_ids.push(token);
    let new_text = match codec.decode(&req.generated_ids) {
        Ok(text) => text,
        Err(err) => return SampleStep::Failed(err.to_string()),
    };
    let piece = text_delta(&req.decoded, &new_text);
    req.decoded = new_text;
    let (visible, hit) = req.stop.push_delta(&piece);
    if hit {
        SampleStep::Stopped { tail: visible }
    } else {
        SampleStep::Token { token, visible }
    }
}

fn finish_stop(scheduler: &mut Scheduler, live: &mut HashMap<u64, Live>, id: u64, tail: String) {
    let Some((events, prompt_tokens, completion_tokens)) = live
        .get(&id)
        .map(|req| (req.events.clone(), req.prompt_len, req.completion_tokens))
    else {
        let _ = scheduler.finish(id);
        return;
    };
    if !tail.is_empty() {
        let _ = events.send(EngineEvent::Delta(tail));
    }
    let _ = events.send(EngineEvent::Done {
        reason: FinishReason::Stop,
        prompt_tokens,
        completion_tokens,
    });
    drop_request(scheduler, live, id, None);
}

fn push_sampled(
    scheduler: &mut Scheduler,
    live: &mut HashMap<u64, Live>,
    id: u64,
    token: u32,
    visible: String,
) {
    let sent = match live.get(&id) {
        Some(_) if visible.is_empty() => true,
        Some(req) => req.events.send(EngineEvent::Delta(visible)).is_ok(),
        None => false,
    };
    if !sent {
        drop_request(scheduler, live, id, None);
        return;
    }
    match scheduler.push_token(id, token) {
        Ok(PushOutcome::Continue) => {}
        Ok(PushOutcome::Finished(_)) => {
            let Some((events, tail, prompt_tokens, completion_tokens)) =
                live.get_mut(&id).map(|req| {
                    (
                        req.events.clone(),
                        req.stop.flush(),
                        req.prompt_len,
                        req.completion_tokens,
                    )
                })
            else {
                return;
            };
            if !tail.is_empty() {
                let _ = events.send(EngineEvent::Delta(tail));
            }
            let _ = events.send(EngineEvent::Done {
                reason: FinishReason::Length,
                prompt_tokens,
                completion_tokens,
            });
            live.remove(&id);
        }
        Err(err) => drop_request(scheduler, live, id, Some(err.to_string())),
    }
}
