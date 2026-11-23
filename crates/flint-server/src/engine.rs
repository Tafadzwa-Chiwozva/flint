//! Single-threaded engine that turns scheduler steps into tokens.
//!
//! HTTP handlers submit prompts over a bounded channel and read token events
//! back. The engine thread is the only thing that touches the model and the
//! KV cache, so the forward pass stays synchronous and the scheduler stays
//! race-free. The CPU model runs each sequence in an iteration one after
//! another. A model that opts into [`LanguageModel::uses_batched_forward`]
//! receives the whole iteration in one call.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{
    sync_channel, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError,
};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use flint_core::{sample, ChatTurn, Error, ForwardChunk, Model, SamplingParams};
use flint_kv::PagedKvCache;
use flint_schedule::{PushOutcome, Scheduler, SchedulerConfig, WorkItem};
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

    /// When true, the engine calls [`Self::forward_batch`] once per scheduler step.
    fn uses_batched_forward(&self) -> bool {
        false
    }

    /// Run every chunk in this scheduler step.
    ///
    /// The default walks [`Self::forward_chunk`]. The CUDA model replaces it
    /// with one batched GEMM and one paged-attention launch per layer.
    fn forward_batch(
        &mut self,
        items: &[ForwardChunk<'_>],
        cache: &mut PagedKvCache,
    ) -> Result<Vec<Option<Vec<f32>>>, Error> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            out.push(self.forward_chunk(
                item.tokens,
                item.position,
                cache,
                item.block_table,
                item.want_logits,
            )?);
        }
        Ok(out)
    }

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

/// Where the forward pass runs. `Cuda` requires the `cuda` crate feature and a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecDevice {
    /// Host reference forward pass.
    Cpu,
    /// Device weights, cuBLAS, and the paged CUDA kernels.
    Cuda,
}

/// Box the loaded model for the engine thread.
pub(crate) fn into_engine_model(
    model: Model,
    device: ExecDevice,
    num_blocks: usize,
    block_size: usize,
) -> Result<Box<dyn LanguageModel>, Error> {
    match device {
        ExecDevice::Cpu => Ok(Box::new(model)),
        ExecDevice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let gpu = model.into_gpu(num_blocks, block_size)?;
                Ok(Box::new(CudaModel { inner: gpu }))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model, num_blocks, block_size);
                Err(Error::Inference(
                    "--device cuda requires rebuilding flint with `--features cuda`".into(),
                ))
            }
        }
    }
}

#[cfg(feature = "cuda")]
struct CudaModel {
    inner: flint_core::GpuModel,
}

#[cfg(feature = "cuda")]
impl LanguageModel for CudaModel {
    fn uses_batched_forward(&self) -> bool {
        true
    }

    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        position: usize,
        cache: &mut PagedKvCache,
        block_table: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, Error> {
        let item = [ForwardChunk {
            tokens,
            position,
            block_table,
            want_logits,
        }];
        let mut rows = self.forward_batch(&item, cache)?;
        Ok(rows.pop().flatten())
    }

    fn forward_batch(
        &mut self,
        items: &[ForwardChunk<'_>],
        _cache: &mut PagedKvCache,
    ) -> Result<Vec<Option<Vec<f32>>>, Error> {
        self.inner.forward_batch(items)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn num_kv_heads(&self) -> usize {
        self.inner.num_kv_heads()
    }

    fn head_dim(&self) -> usize {
        self.inner.head_dim()
    }

    fn eos_token_ids(&self) -> &[u32] {
        self.inner.eos_token_ids()
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

        if model.uses_batched_forward() {
            execute_batch(
                &mut *model,
                &mut scheduler,
                &mut cache,
                &mut live,
                codec.as_ref(),
                &eos,
                step.items,
            );
        } else {
            for item in step.items {
                execute_one(
                    &mut *model,
                    &mut scheduler,
                    &mut cache,
                    &mut live,
                    codec.as_ref(),
                    &eos,
                    item,
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

fn execute_one(
    model: &mut dyn LanguageModel,
    scheduler: &mut Scheduler,
    cache: &mut PagedKvCache,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    item: WorkItem,
) {
    if cancelled(live, item.request_id) {
        drop_request(scheduler, live, item.request_id, None);
        return;
    }
    let logits = match model.forward_chunk(
        &item.tokens,
        item.position,
        cache,
        &item.block_table,
        item.sample_after,
    ) {
        Ok(logits) => logits,
        Err(err) => {
            drop_request(scheduler, live, item.request_id, Some(err.to_string()));
            return;
        }
    };
    apply_step(scheduler, live, codec, eos, item, logits);
}

fn execute_batch(
    model: &mut dyn LanguageModel,
    scheduler: &mut Scheduler,
    cache: &mut PagedKvCache,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    items: Vec<WorkItem>,
) {
    let mut kept = Vec::new();
    for item in items {
        if cancelled(live, item.request_id) {
            drop_request(scheduler, live, item.request_id, None);
            continue;
        }
        kept.push(item);
    }
    if kept.is_empty() {
        return;
    }
    let logits = {
        let reqs: Vec<ForwardChunk<'_>> = kept
            .iter()
            .map(|item| ForwardChunk {
                tokens: &item.tokens,
                position: item.position,
                block_table: &item.block_table,
                want_logits: item.sample_after,
            })
            .collect();
        match model.forward_batch(&reqs, cache) {
            Ok(rows) => rows,
            Err(err) => {
                let message = err.to_string();
                for item in kept {
                    drop_request(scheduler, live, item.request_id, Some(message.clone()));
                }
                return;
            }
        }
    };
    if logits.len() != kept.len() {
        for item in kept {
            drop_request(
                scheduler,
                live,
                item.request_id,
                Some("batched forward returned the wrong number of rows".into()),
            );
        }
        return;
    }
    for (item, row) in kept.into_iter().zip(logits) {
        apply_step(scheduler, live, codec, eos, item, row);
    }
}

fn apply_step(
    scheduler: &mut Scheduler,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    item: WorkItem,
    logits: Option<Vec<f32>>,
) {
    if let Err(err) = scheduler.mark_computed(item.request_id, item.tokens.len()) {
        drop_request(scheduler, live, item.request_id, Some(err.to_string()));
        return;
    }
    if item.sample_after {
        let Some(logits) = logits else {
            drop_request(
                scheduler,
                live,
                item.request_id,
                Some("forward pass returned no logits".into()),
            );
            return;
        };
        sample_and_emit(scheduler, live, codec, eos, item.request_id, &logits);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    struct IdCodec;

    impl TextCodec for IdCodec {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.bytes().map(|byte| byte as u32).collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String, Error> {
            Ok(ids
                .iter()
                .map(|id| char::from_u32(*id).unwrap_or('?'))
                .collect())
        }

        fn apply_chat_template(&self, messages: &[ChatTurn]) -> Result<String, Error> {
            Ok(messages
                .iter()
                .map(|message| message.content.clone())
                .collect::<Vec<_>>()
                .join(""))
        }
    }

    struct Gate {
        hold_first: AtomicBool,
        entered: AtomicBool,
        release: AtomicBool,
        sizes: Mutex<Vec<usize>>,
    }

    struct Probe {
        gate: Arc<Gate>,
    }

    impl LanguageModel for Probe {
        fn uses_batched_forward(&self) -> bool {
            true
        }

        fn forward_chunk(
            &mut self,
            _tokens: &[u32],
            _position: usize,
            _cache: &mut PagedKvCache,
            _block_table: &[u32],
            _want_logits: bool,
        ) -> Result<Option<Vec<f32>>, Error> {
            panic!("batched model was invoked one sequence at a time");
        }

        fn forward_batch(
            &mut self,
            items: &[ForwardChunk<'_>],
            _cache: &mut PagedKvCache,
        ) -> Result<Vec<Option<Vec<f32>>>, Error> {
            if self
                .gate
                .hold_first
                .swap(false, std::sync::atomic::Ordering::AcqRel)
            {
                self.gate
                    .entered
                    .store(true, std::sync::atomic::Ordering::Release);
                while !self.gate.release.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
            self.gate.sizes.lock().expect("sizes").push(items.len());
            Ok(items
                .iter()
                .map(|item| {
                    if item.want_logits {
                        Some(vec![0.0; 32])
                    } else {
                        None
                    }
                })
                .collect())
        }

        fn vocab_size(&self) -> usize {
            32
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
            &[]
        }
    }

    #[test]
    fn batched_forward_covers_every_sequence_in_the_step() {
        let gate = Arc::new(Gate {
            hold_first: AtomicBool::new(true),
            entered: AtomicBool::new(false),
            release: AtomicBool::new(false),
            sizes: Mutex::new(Vec::new()),
        });
        let model = Box::new(Probe {
            gate: Arc::clone(&gate),
        });
        let (engine, thread) = spawn(
            model,
            Arc::new(IdCodec),
            16,
            4,
            SchedulerConfig {
                max_num_seqs: 4,
                max_batched_tokens: 64,
                prefill_chunk_size: 16,
                max_waiting: 8,
            },
        )
        .unwrap();
        let _first = engine
            .submit(vec![1, 2], SamplingParams::greedy(), 4, Vec::new())
            .unwrap();
        let start = Instant::now();
        while !gate.entered.load(std::sync::atomic::Ordering::Acquire) {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "engine did not enter the first batched forward"
            );
            std::thread::yield_now();
        }
        let _second = engine
            .submit(vec![3, 4], SamplingParams::greedy(), 4, Vec::new())
            .unwrap();
        gate.release
            .store(true, std::sync::atomic::Ordering::Release);
        while !gate
            .sizes
            .lock()
            .expect("sizes")
            .iter()
            .any(|size| *size >= 2)
        {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "scheduler step never forwarded two sequences together: {:?}",
                gate.sizes.lock().unwrap()
            );
            std::thread::yield_now();
        }
        drop(_first);
        drop(_second);
        drop(engine);
        thread.join().unwrap();
    }
}
