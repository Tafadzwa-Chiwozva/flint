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

use flint_core::{
    accept_greedy, accept_stochastic, draft_propose, keep_kv_len, plan_verify, sample, ChatTurn,
    DraftStart, Error, ForwardChunk, Model, SamplingParams, SpecModel, SpeculativeStats,
};
use flint_kv::{BlockSpaceManager, CacheGeometry, PagedKvCache};
use flint_schedule::{PushOutcome, ScheduleError, Scheduler, SchedulerConfig, WorkItem};
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
        /// Draft-model counters. `None` when speculative decoding is off.
        spec: Option<SpeculativeStats>,
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
    max_tokens: usize,
    /// Present when the engine was started with a draft model.
    spec: Option<SpeculativeStats>,
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

    /// Logits for one chunk.
    ///
    /// `all_positions` returns one vector per token. The default walks
    /// [`Self::forward_chunk`] one token at a time. The CPU and CUDA models
    /// replace it with one multi-token forward, which is the speculative
    /// verification pass.
    fn forward_logits(
        &mut self,
        tokens: &[u32],
        position: usize,
        cache: &mut PagedKvCache,
        block_table: &[u32],
        all_positions: bool,
    ) -> Result<Vec<Vec<f32>>, Error> {
        if tokens.is_empty() {
            return Err(Error::Inference("forward chunk is empty".into()));
        }
        if !all_positions {
            let logits = self
                .forward_chunk(tokens, position, cache, block_table, true)?
                .ok_or_else(|| Error::Inference("forward pass returned no logits".into()))?;
            return Ok(vec![logits]);
        }
        let mut rows = Vec::with_capacity(tokens.len());
        for (offset, token) in tokens.iter().enumerate() {
            let logits = self
                .forward_chunk(&[*token], position + offset, cache, block_table, true)?
                .ok_or_else(|| Error::Inference("forward pass returned no logits".into()))?;
            rows.push(logits);
        }
        Ok(rows)
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

    fn forward_logits(
        &mut self,
        tokens: &[u32],
        position: usize,
        cache: &mut PagedKvCache,
        block_table: &[u32],
        all_positions: bool,
    ) -> Result<Vec<Vec<f32>>, Error> {
        self.forward_token_logits(tokens, position, cache, block_table, all_positions)
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

/// Storage for a CUDA model. The CPU path is always f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecDType {
    /// Full precision. This is the GPU parity path.
    F32,
    /// bf16 weights and activations, f32 accumulation.
    Bf16,
}

/// Box the loaded model for the engine thread.
pub(crate) fn into_engine_model(
    model: Model,
    device: ExecDevice,
    dtype: ExecDType,
    num_blocks: usize,
    block_size: usize,
) -> Result<Box<dyn LanguageModel>, Error> {
    match device {
        ExecDevice::Cpu => {
            if dtype == ExecDType::Bf16 {
                return Err(Error::Inference(
                    "--dtype bf16 requires --device cuda".into(),
                ));
            }
            Ok(Box::new(model))
        }
        ExecDevice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let gpu_dtype = match dtype {
                    ExecDType::F32 => flint_core::GpuDType::F32,
                    ExecDType::Bf16 => flint_core::GpuDType::Bf16,
                };
                let gpu = model.into_gpu_with(num_blocks, block_size, gpu_dtype)?;
                Ok(Box::new(CudaModel { inner: gpu }))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model, dtype, num_blocks, block_size);
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

    fn forward_logits(
        &mut self,
        tokens: &[u32],
        position: usize,
        _cache: &mut PagedKvCache,
        block_table: &[u32],
        all_positions: bool,
    ) -> Result<Vec<Vec<f32>>, Error> {
        self.inner
            .forward_token_logits(tokens, position, block_table, all_positions)
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

/// Draft model that proposes tokens for the target to verify.
pub(crate) struct DraftSpec {
    /// Same device and dtype as the target. Quantization may differ.
    pub model: Box<dyn LanguageModel>,
    /// Draft tokens proposed on each speculative step.
    pub k: usize,
}

struct DraftSeq {
    table: Vec<u32>,
    /// Tokens written into the draft cache. The recovered token stays pending.
    len: usize,
}

struct DraftRuntime {
    model: Box<dyn LanguageModel>,
    cache: PagedKvCache,
    blocks: BlockSpaceManager,
    seqs: HashMap<u64, DraftSeq>,
    k: usize,
    totals: SpeculativeStats,
}

type DraftSlot = Option<DraftRuntime>;

impl DraftRuntime {
    fn new(spec: DraftSpec, num_blocks: usize, block_size: usize) -> Result<Self, Error> {
        let cache = PagedKvCache::with_dims(
            num_blocks,
            block_size,
            spec.model.num_layers(),
            spec.model.num_kv_heads(),
            spec.model.head_dim(),
        )
        .map_err(|err| Error::Inference(err.to_string()))?;
        let blocks = BlockSpaceManager::new(num_blocks, block_size)
            .map_err(|err| Error::Inference(err.to_string()))?;
        Ok(Self {
            model: spec.model,
            cache,
            blocks,
            seqs: HashMap::new(),
            k: spec.k,
            totals: SpeculativeStats::default(),
        })
    }
}

/// Adapts the engine's model trait to the speculative proposer.
struct AsSpec<'a>(&'a mut dyn LanguageModel);

impl SpecModel for AsSpec<'_> {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }

    fn cache_geometry(&self, num_blocks: usize, block_size: usize) -> CacheGeometry {
        CacheGeometry {
            num_blocks,
            num_layers: self.0.num_layers(),
            num_kv_heads: self.0.num_kv_heads(),
            block_size,
            head_dim: self.0.head_dim(),
        }
    }

    fn forward_spec(
        &mut self,
        cache: &mut PagedKvCache,
        tokens: &[u32],
        position: usize,
        block_table: &[u32],
        all_positions: bool,
    ) -> Result<Vec<Vec<f32>>, Error> {
        self.0
            .forward_logits(tokens, position, cache, block_table, all_positions)
    }
}

/// Start the engine thread.
///
/// `draft` enables lossless speculative decoding. `None` is the ordinary
/// one-token path, including CUDA-graph decode.
pub(crate) fn spawn(
    model: Box<dyn LanguageModel>,
    draft: Option<DraftSpec>,
    codec: Arc<dyn TextCodec>,
    num_blocks: usize,
    block_size: usize,
    sched: SchedulerConfig,
) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    if let Some(spec) = &draft {
        if spec.k == 0 {
            return Err(Error::Config(
                "--speculative-tokens must be at least 1".into(),
            ));
        }
        if spec.model.vocab_size() != model.vocab_size() {
            return Err(Error::Config(format!(
                "draft vocab size {} is not compatible with target vocab size {}",
                spec.model.vocab_size(),
                model.vocab_size()
            )));
        }
        tracing::info!(k = spec.k, "speculative decoding enabled");
    }
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
    let draft = match draft {
        Some(spec) => Some(DraftRuntime::new(spec, num_blocks, block_size)?),
        None => None,
    };
    let (tx, rx) = sync_channel(256);
    let handle = thread::Builder::new()
        .name("flint-engine".into())
        .spawn(move || engine_loop(model, draft, codec, scheduler, cache, rx))
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
    mut draft: DraftSlot,
    codec: Arc<dyn TextCodec>,
    mut scheduler: Scheduler,
    mut cache: PagedKvCache,
    rx: Receiver<Command>,
) {
    let eos: Vec<u32> = model.eos_token_ids().to_vec();
    let mut live: HashMap<u64, Live> = HashMap::new();
    let mut next_id = 1_u64;
    let track_spec = draft.is_some();

    loop {
        loop {
            match rx.try_recv() {
                Ok(command) => {
                    accept(&mut scheduler, &mut live, &mut next_id, track_spec, command);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        reap_cancelled(&mut scheduler, &mut draft, &mut live);

        let step = scheduler.schedule();
        let idle = step.items.is_empty();
        for failure in step.failed {
            release_draft(&mut draft, failure.request_id);
            if let Some(req) = live.remove(&failure.request_id) {
                let _ = req.events.send(EngineEvent::Error(failure.reason));
            }
        }
        if !step.preempted.is_empty() {
            tracing::debug!(ids = ?step.preempted, "preempted sequences");
            for id in &step.preempted {
                release_draft(&mut draft, *id);
            }
        }

        execute_step(
            &mut *model,
            &mut draft,
            &mut scheduler,
            &mut cache,
            &mut live,
            codec.as_ref(),
            &eos,
            step.items,
        );

        if idle {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(command) => {
                    accept(&mut scheduler, &mut live, &mut next_id, track_spec, command);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_step(
    model: &mut dyn LanguageModel,
    draft: &mut DraftSlot,
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
            drop_request(scheduler, draft, live, item.request_id, None);
            continue;
        }
        kept.push(item);
    }
    if kept.is_empty() {
        return;
    }
    if kept.len() == 1
        && matches!(
            speculate(model, draft, scheduler, cache, live, codec, eos, &kept[0]),
            SpecOutcome::Handled
        )
    {
        return;
    }
    if model.uses_batched_forward() {
        execute_batch(
            model,
            draft,
            scheduler,
            cache,
            live,
            codec,
            eos,
            kept.clone(),
        );
    } else {
        for item in kept.iter().cloned() {
            execute_one(model, draft, scheduler, cache, live, codec, eos, item);
        }
    }
    if draft.is_none() {
        return;
    }
    let alive: Vec<WorkItem> = kept
        .into_iter()
        .filter(|item| live.contains_key(&item.request_id))
        .collect();
    if let Err(err) = sync_draft(draft, &alive) {
        let message = err.to_string();
        for item in alive {
            drop_request(
                scheduler,
                draft,
                live,
                item.request_id,
                Some(message.clone()),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_one(
    model: &mut dyn LanguageModel,
    draft: &mut DraftSlot,
    scheduler: &mut Scheduler,
    cache: &mut PagedKvCache,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    item: WorkItem,
) {
    if cancelled(live, item.request_id) {
        drop_request(scheduler, draft, live, item.request_id, None);
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
            drop_request(
                scheduler,
                draft,
                live,
                item.request_id,
                Some(err.to_string()),
            );
            return;
        }
    };
    apply_step(scheduler, draft, live, codec, eos, item, logits);
}

#[allow(clippy::too_many_arguments)]
fn execute_batch(
    model: &mut dyn LanguageModel,
    draft: &mut DraftSlot,
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
            drop_request(scheduler, draft, live, item.request_id, None);
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
                    drop_request(
                        scheduler,
                        draft,
                        live,
                        item.request_id,
                        Some(message.clone()),
                    );
                }
                return;
            }
        }
    };
    if logits.len() != kept.len() {
        for item in kept {
            drop_request(
                scheduler,
                draft,
                live,
                item.request_id,
                Some("batched forward returned the wrong number of rows".into()),
            );
        }
        return;
    }
    for (item, row) in kept.into_iter().zip(logits) {
        apply_step(scheduler, draft, live, codec, eos, item, row);
    }
}

fn apply_step(
    scheduler: &mut Scheduler,
    draft: &mut DraftSlot,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    item: WorkItem,
    logits: Option<Vec<f32>>,
) {
    if let Err(err) = scheduler.mark_computed(item.request_id, item.tokens.len()) {
        drop_request(
            scheduler,
            draft,
            live,
            item.request_id,
            Some(err.to_string()),
        );
        return;
    }
    if item.sample_after {
        let Some(logits) = logits else {
            drop_request(
                scheduler,
                draft,
                live,
                item.request_id,
                Some("forward pass returned no logits".into()),
            );
            return;
        };
        sample_and_emit(scheduler, draft, live, codec, eos, item.request_id, &logits);
    }
}

enum SpecOutcome {
    /// The step wrote KV, sampled, and rolled the caches back.
    Handled,
    /// Run the ordinary one-token forward for this step.
    Fallback,
}

enum SpecFail {
    Fallback,
    Fatal(String),
}

#[allow(clippy::too_many_arguments)]
fn speculate(
    model: &mut dyn LanguageModel,
    draft: &mut DraftSlot,
    scheduler: &mut Scheduler,
    cache: &mut PagedKvCache,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    item: &WorkItem,
) -> SpecOutcome {
    if draft.is_none() || !item.sample_after {
        return SpecOutcome::Fallback;
    }
    let id = item.request_id;
    let k = draft.as_ref().map(|runtime| runtime.k).unwrap_or(0);
    let snapshot = live.get(&id).map(|req| {
        (
            req.cancel.load(Ordering::Acquire),
            req.prompt_len,
            req.sampling.clone(),
            req.max_tokens.saturating_sub(req.completion_tokens),
        )
    });
    let Some((cancelled, prompt_len, params, generated)) = snapshot else {
        return SpecOutcome::Handled;
    };
    if cancelled {
        drop_request(scheduler, draft, live, id, None);
        return SpecOutcome::Handled;
    }
    let k_step = k.min(generated.saturating_sub(1));
    if k_step == 0 {
        tracing::debug!(
            request = id,
            "speculative width is zero; decoding one token"
        );
        return SpecOutcome::Fallback;
    }
    let Some(token_len) = scheduler.token_len(id) else {
        drop_request(
            scheduler,
            draft,
            live,
            id,
            Some("request left the scheduler".into()),
        );
        return SpecOutcome::Handled;
    };
    let max_seq = token_len + k_step + 1;
    match scheduler.ensure_blocks(id, max_seq) {
        Ok(()) => {}
        Err(ScheduleError::OutOfBlocks { .. }) => {
            tracing::debug!(
                request = id,
                "speculative lookahead did not fit; decoding one token"
            );
            return SpecOutcome::Fallback;
        }
        Err(err) => {
            drop_request(scheduler, draft, live, id, Some(err.to_string()));
            return SpecOutcome::Handled;
        }
    }
    if let Err(err) = align_draft(draft, id, item.position, max_seq) {
        if is_out_of_blocks(&err) {
            let _ = scheduler.truncate_blocks(id, token_len);
            tracing::debug!(
                request = id,
                "draft lookahead did not fit; decoding one token"
            );
            return SpecOutcome::Fallback;
        }
        let _ = scheduler.truncate_blocks(id, token_len);
        drop_request(scheduler, draft, live, id, Some(err.to_string()));
        return SpecOutcome::Handled;
    }
    let is_decode = item.tokens.len() == 1 && item.position >= prompt_len;
    let outcome = if is_decode {
        speculate_decode(
            model, draft, scheduler, cache, live, codec, eos, item, &params, k_step,
        )
    } else {
        speculate_prefill(
            model, draft, scheduler, cache, live, codec, eos, item, &params, k_step,
        )
    };
    match outcome {
        Ok(()) => SpecOutcome::Handled,
        Err(SpecFail::Fallback) => {
            let _ = scheduler.truncate_blocks(id, token_len);
            rollback_draft(draft, id, item.position, token_len);
            tracing::debug!(request = id, "speculative step fell back to one token");
            SpecOutcome::Fallback
        }
        Err(SpecFail::Fatal(message)) => {
            drop_request(scheduler, draft, live, id, Some(message));
            SpecOutcome::Handled
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn speculate_decode(
    model: &mut dyn LanguageModel,
    draft: &mut DraftSlot,
    scheduler: &mut Scheduler,
    cache: &mut PagedKvCache,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    item: &WorkItem,
    params: &SamplingParams,
    k_step: usize,
) -> Result<(), SpecFail> {
    let id = item.request_id;
    let pending = *item
        .tokens
        .first()
        .ok_or_else(|| SpecFail::Fatal("decode step has no token".into()))?;
    let proposal = propose(
        draft,
        live,
        id,
        DraftStart::Pending(pending),
        k_step,
        params,
    )?;
    let plan = plan_verify(item.position, Some(pending), None, &proposal.tokens)
        .map_err(|err| SpecFail::Fatal(err.to_string()))?;
    verify_and_commit(
        model,
        draft,
        scheduler,
        cache,
        live,
        codec,
        eos,
        id,
        plan,
        proposal,
        k_step,
        params,
        item.tokens.len(),
    )
}

#[allow(clippy::too_many_arguments)]
fn speculate_prefill(
    model: &mut dyn LanguageModel,
    draft: &mut DraftSlot,
    scheduler: &mut Scheduler,
    cache: &mut PagedKvCache,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    item: &WorkItem,
    params: &SamplingParams,
    k_step: usize,
) -> Result<(), SpecFail> {
    let id = item.request_id;
    let table = target_table(scheduler, id)?;
    let prefix_logits = model
        .forward_logits(&item.tokens, item.position, cache, &table, false)
        .map_err(|err| SpecFail::Fatal(err.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| SpecFail::Fatal("prefill returned no logits".into()))?;
    let draft_logits = forward_draft_chunk(draft, id, &item.tokens, item.position)?;
    let proposal = propose(
        draft,
        live,
        id,
        DraftStart::Logits(draft_logits),
        k_step,
        params,
    )?;
    let kv_len = item.position + item.tokens.len();
    let plan = plan_verify(kv_len, None, Some(prefix_logits), &proposal.tokens)
        .map_err(|err| SpecFail::Fatal(err.to_string()))?;
    verify_and_commit(
        model,
        draft,
        scheduler,
        cache,
        live,
        codec,
        eos,
        id,
        plan,
        proposal,
        k_step,
        params,
        item.tokens.len(),
    )
}

fn propose(
    draft: &mut DraftSlot,
    live: &mut HashMap<u64, Live>,
    id: u64,
    start: DraftStart,
    k_step: usize,
    params: &SamplingParams,
) -> Result<flint_core::Proposal, SpecFail> {
    let runtime = draft
        .as_mut()
        .ok_or_else(|| SpecFail::Fatal("draft model disappeared".into()))?;
    let flint_core::Proposal { tokens, probs } = {
        let DraftRuntime {
            model,
            cache,
            blocks,
            seqs,
            ..
        } = runtime;
        let seq = seqs
            .get_mut(&id)
            .ok_or_else(|| SpecFail::Fatal("draft sequence disappeared".into()))?;
        let req = live
            .get_mut(&id)
            .ok_or_else(|| SpecFail::Fatal("request disappeared".into()))?;
        draft_propose(
            &mut AsSpec(&mut **model),
            cache,
            blocks,
            &mut seq.table,
            &mut seq.len,
            start,
            k_step,
            params,
            &mut req.rng,
        )
        .map_err(spec_error)?
    };
    Ok(flint_core::Proposal { tokens, probs })
}

#[allow(clippy::too_many_arguments)]
fn verify_and_commit(
    model: &mut dyn LanguageModel,
    draft: &mut DraftSlot,
    scheduler: &mut Scheduler,
    cache: &mut PagedKvCache,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    id: u64,
    mut plan: flint_core::VerifyPlan,
    proposal: flint_core::Proposal,
    k_step: usize,
    params: &SamplingParams,
    mark_prefix: usize,
) -> Result<(), SpecFail> {
    let table = target_table(scheduler, id)?;
    let mut rows = model
        .forward_logits(&plan.tokens, plan.position, cache, &table, true)
        .map_err(|err| SpecFail::Fatal(err.to_string()))?;
    if let Some(prefix) = plan.prefix_logits.take() {
        rows.insert(0, prefix);
    }
    let decision = {
        let req = live
            .get_mut(&id)
            .ok_or_else(|| SpecFail::Fatal("request disappeared".into()))?;
        if params.temperature == 0.0 {
            accept_greedy(&rows, &proposal.tokens)
        } else {
            accept_stochastic(
                &rows,
                &proposal.tokens,
                &proposal.probs,
                params,
                &mut req.rng,
            )
        }
        .map_err(|err| SpecFail::Fatal(err.to_string()))?
    };
    let keep = keep_kv_len(plan.base_len, decision.accepted);
    let addressed = keep + 1;
    scheduler
        .truncate_blocks(id, addressed)
        .map_err(|err| SpecFail::Fatal(err.to_string()))?;
    if let Some(runtime) = draft.as_mut() {
        let mut table = runtime
            .seqs
            .get_mut(&id)
            .map(|seq| std::mem::take(&mut seq.table))
            .ok_or_else(|| SpecFail::Fatal("draft sequence disappeared".into()))?;
        let truncated = runtime.blocks.truncate(&mut table, addressed);
        if let Some(seq) = runtime.seqs.get_mut(&id) {
            seq.table = table;
            seq.len = keep;
        }
        truncated.map_err(|err| SpecFail::Fatal(err.to_string()))?;
        runtime
            .totals
            .record(k_step as u64, decision.accepted as u64);
    }
    if let Some(stats) = live.get_mut(&id).and_then(|req| req.spec.as_mut()) {
        stats.record(k_step as u64, decision.accepted as u64);
    }
    tracing::debug!(
        request = id,
        drafted = k_step,
        accepted = decision.accepted,
        "speculative step"
    );
    scheduler
        .mark_computed(id, mark_prefix)
        .map_err(|err| SpecFail::Fatal(err.to_string()))?;
    let committed = decision
        .committed(&proposal.tokens)
        .map_err(|err| SpecFail::Fatal(err.to_string()))?;
    commit_tokens(scheduler, draft, live, codec, eos, id, &committed)
}

#[allow(clippy::too_many_arguments)]
fn commit_tokens(
    scheduler: &mut Scheduler,
    draft: &mut DraftSlot,
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    id: u64,
    tokens: &[u32],
) -> Result<(), SpecFail> {
    for (index, &token) in tokens.iter().enumerate() {
        let in_kv = index + 1 < tokens.len();
        match ingest_token(live, codec, eos, id, token) {
            SampleStep::Missing => {
                let _ = scheduler.finish(id);
                release_draft(draft, id);
                return Ok(());
            }
            SampleStep::Cancelled => {
                drop_request(scheduler, draft, live, id, None);
                return Ok(());
            }
            SampleStep::Failed(message) => {
                drop_request(scheduler, draft, live, id, Some(message));
                return Ok(());
            }
            SampleStep::Stopped { tail } => {
                finish_stop(scheduler, draft, live, id, tail);
                return Ok(());
            }
            SampleStep::Token { token, visible } => {
                match push_sampled(scheduler, draft, live, id, token, visible) {
                    PushStatus::Continue => {
                        if in_kv {
                            if let Err(err) = scheduler.mark_computed(id, 1) {
                                drop_request(scheduler, draft, live, id, Some(err.to_string()));
                                return Ok(());
                            }
                        }
                    }
                    PushStatus::Finished | PushStatus::Dropped => return Ok(()),
                }
            }
        }
    }
    Ok(())
}

fn target_table(scheduler: &Scheduler, id: u64) -> Result<Vec<u32>, SpecFail> {
    scheduler
        .block_table(id)
        .map(<[u32]>::to_vec)
        .ok_or_else(|| SpecFail::Fatal("request left the scheduler".into()))
}

fn forward_draft_chunk(
    draft: &mut DraftSlot,
    id: u64,
    tokens: &[u32],
    position: usize,
) -> Result<Vec<f32>, SpecFail> {
    let runtime = draft
        .as_mut()
        .ok_or_else(|| SpecFail::Fatal("draft model disappeared".into()))?;
    let table = runtime
        .seqs
        .get(&id)
        .map(|seq| seq.table.clone())
        .ok_or_else(|| SpecFail::Fatal("draft sequence disappeared".into()))?;
    let rows = runtime
        .model
        .forward_logits(tokens, position, &mut runtime.cache, &table, false)
        .map_err(|err| SpecFail::Fatal(err.to_string()))?;
    if let Some(seq) = runtime.seqs.get_mut(&id) {
        seq.len = position + tokens.len();
    }
    rows.into_iter()
        .next()
        .ok_or_else(|| SpecFail::Fatal("draft prefill returned no logits".into()))
}

fn align_draft(
    draft: &mut DraftSlot,
    id: u64,
    position: usize,
    max_seq: usize,
) -> Result<(), Error> {
    let Some(runtime) = draft.as_mut() else {
        return Ok(());
    };
    let mut table = {
        let seq = runtime.seqs.entry(id).or_insert(DraftSeq {
            table: Vec::new(),
            len: 0,
        });
        if seq.len != position {
            return Err(Error::Inference(format!(
                "draft cache length {} does not match position {position} for request {id}",
                seq.len
            )));
        }
        std::mem::take(&mut seq.table)
    };
    let reserved = runtime.blocks.reserve(&mut table, max_seq);
    if let Some(seq) = runtime.seqs.get_mut(&id) {
        seq.table = table;
    }
    reserved.map_err(|err| Error::Inference(err.to_string()))
}

fn rollback_draft(draft: &mut DraftSlot, id: u64, kv_len: usize, addressed: usize) {
    let Some(runtime) = draft.as_mut() else {
        return;
    };
    let Some(mut table) = runtime.seqs.get_mut(&id).map(|seq| {
        seq.len = kv_len;
        std::mem::take(&mut seq.table)
    }) else {
        return;
    };
    let truncated = runtime.blocks.truncate(&mut table, addressed);
    if let Some(seq) = runtime.seqs.get_mut(&id) {
        seq.table = table;
    }
    if let Err(err) = truncated {
        tracing::error!(%err, request = id, "draft rollback failed");
    }
}

fn sync_draft(draft: &mut DraftSlot, items: &[WorkItem]) -> Result<(), Error> {
    let Some(runtime) = draft.as_mut() else {
        return Ok(());
    };
    if items.is_empty() {
        return Ok(());
    }
    let mut owned = Vec::with_capacity(items.len());
    for item in items {
        let end = item.position + item.tokens.len();
        let mut table = {
            let seq = runtime.seqs.entry(item.request_id).or_insert(DraftSeq {
                table: Vec::new(),
                len: 0,
            });
            if seq.len != item.position {
                return Err(Error::Inference(format!(
                    "draft cache length {} does not match position {} for request {}",
                    seq.len, item.position, item.request_id
                )));
            }
            std::mem::take(&mut seq.table)
        };
        let reserved = runtime.blocks.reserve(&mut table, end);
        let snapshot = table.clone();
        if let Some(seq) = runtime.seqs.get_mut(&item.request_id) {
            seq.table = table;
        }
        reserved.map_err(|err| Error::Inference(err.to_string()))?;
        owned.push((
            item.request_id,
            item.position,
            item.tokens.clone(),
            snapshot,
            end,
        ));
    }
    let chunks: Vec<ForwardChunk<'_>> = owned
        .iter()
        .map(|row| ForwardChunk {
            tokens: &row.2,
            position: row.1,
            block_table: &row.3,
            want_logits: false,
        })
        .collect();
    runtime.model.forward_batch(&chunks, &mut runtime.cache)?;
    for (id, _, _, _, end) in owned {
        if let Some(seq) = runtime.seqs.get_mut(&id) {
            seq.len = end;
        }
    }
    Ok(())
}

fn is_out_of_blocks(err: &Error) -> bool {
    err.to_string().contains("out of KV blocks")
}

fn spec_error(err: Error) -> SpecFail {
    if is_out_of_blocks(&err) {
        SpecFail::Fallback
    } else {
        SpecFail::Fatal(err.to_string())
    }
}

fn release_draft(draft: &mut DraftSlot, id: u64) {
    let Some(runtime) = draft.as_mut() else {
        return;
    };
    let Some(seq) = runtime.seqs.remove(&id) else {
        return;
    };
    if let Err(err) = runtime.blocks.free_ids(&seq.table) {
        tracing::error!(%err, request = id, "draft KV free failed");
    }
}

fn log_spec(id: u64, stats: Option<&SpeculativeStats>, draft: &DraftSlot) {
    let Some(stats) = stats else {
        return;
    };
    let totals = draft
        .as_ref()
        .map(|runtime| runtime.totals)
        .unwrap_or_default();
    tracing::info!(
        request = id,
        drafted = stats.drafted_tokens,
        accepted = stats.accepted_tokens,
        steps = stats.steps,
        acceptance_rate = stats.acceptance_rate(),
        mean_accepted = stats.mean_accepted_per_step(),
        total_drafted = totals.drafted_tokens,
        total_accepted = totals.accepted_tokens,
        "speculative stats"
    );
}

fn accept(
    scheduler: &mut Scheduler,
    live: &mut HashMap<u64, Live>,
    next_id: &mut u64,
    track_spec: bool,
    command: Command,
) {
    let id = *next_id;
    *next_id = next_id.wrapping_add(1);
    let prompt_len = command.tokens.len();
    let max_tokens = command.max_tokens;
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
                    max_tokens,
                    spec: track_spec.then(SpeculativeStats::default),
                },
            );
        }
        Err(err) => {
            let _ = command.events.send(EngineEvent::Error(err.to_string()));
        }
    }
}

fn reap_cancelled(scheduler: &mut Scheduler, draft: &mut DraftSlot, live: &mut HashMap<u64, Live>) {
    let gone: Vec<u64> = live
        .iter()
        .filter(|(_, req)| req.cancel.load(Ordering::Acquire))
        .map(|(id, _)| *id)
        .collect();
    for id in gone {
        drop_request(scheduler, draft, live, id, None);
    }
}

fn cancelled(live: &HashMap<u64, Live>, id: u64) -> bool {
    live.get(&id)
        .is_none_or(|req| req.cancel.load(Ordering::Acquire))
}

fn drop_request(
    scheduler: &mut Scheduler,
    draft: &mut DraftSlot,
    live: &mut HashMap<u64, Live>,
    id: u64,
    message: Option<String>,
) {
    release_draft(draft, id);
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

enum PushStatus {
    Continue,
    Finished,
    Dropped,
}

fn sample_and_emit(
    scheduler: &mut Scheduler,
    draft: &mut DraftSlot,
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
            release_draft(draft, id);
        }
        SampleStep::Cancelled => drop_request(scheduler, draft, live, id, None),
        SampleStep::Failed(message) => drop_request(scheduler, draft, live, id, Some(message)),
        SampleStep::Stopped { tail } => finish_stop(scheduler, draft, live, id, tail),
        SampleStep::Token { token, visible } => {
            let _ = push_sampled(scheduler, draft, live, id, token, visible);
        }
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
    ingest_token(live, codec, eos, id, token)
}

fn ingest_token(
    live: &mut HashMap<u64, Live>,
    codec: &dyn TextCodec,
    eos: &[u32],
    id: u64,
    token: u32,
) -> SampleStep {
    let Some(req) = live.get_mut(&id) else {
        return SampleStep::Missing;
    };
    if req.cancel.load(Ordering::Acquire) {
        return SampleStep::Cancelled;
    }
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

fn finish_stop(
    scheduler: &mut Scheduler,
    draft: &mut DraftSlot,
    live: &mut HashMap<u64, Live>,
    id: u64,
    tail: String,
) {
    let Some((events, prompt_tokens, completion_tokens, spec)) = live.get(&id).map(|req| {
        (
            req.events.clone(),
            req.prompt_len,
            req.completion_tokens,
            req.spec,
        )
    }) else {
        let _ = scheduler.finish(id);
        release_draft(draft, id);
        return;
    };
    log_spec(id, spec.as_ref(), draft);
    if !tail.is_empty() {
        let _ = events.send(EngineEvent::Delta(tail));
    }
    let _ = events.send(EngineEvent::Done {
        reason: FinishReason::Stop,
        prompt_tokens,
        completion_tokens,
        spec,
    });
    drop_request(scheduler, draft, live, id, None);
}

fn push_sampled(
    scheduler: &mut Scheduler,
    draft: &mut DraftSlot,
    live: &mut HashMap<u64, Live>,
    id: u64,
    token: u32,
    visible: String,
) -> PushStatus {
    let sent = match live.get(&id) {
        Some(_) if visible.is_empty() => true,
        Some(req) => req.events.send(EngineEvent::Delta(visible)).is_ok(),
        None => false,
    };
    if !sent {
        drop_request(scheduler, draft, live, id, None);
        return PushStatus::Dropped;
    }
    match scheduler.push_token(id, token) {
        Ok(PushOutcome::Continue) => PushStatus::Continue,
        Ok(PushOutcome::Finished(_)) => {
            let Some((events, tail, prompt_tokens, completion_tokens, spec)) =
                live.get_mut(&id).map(|req| {
                    (
                        req.events.clone(),
                        req.stop.flush(),
                        req.prompt_len,
                        req.completion_tokens,
                        req.spec,
                    )
                })
            else {
                release_draft(draft, id);
                return PushStatus::Dropped;
            };
            log_spec(id, spec.as_ref(), draft);
            if !tail.is_empty() {
                let _ = events.send(EngineEvent::Delta(tail));
            }
            let _ = events.send(EngineEvent::Done {
                reason: FinishReason::Length,
                prompt_tokens,
                completion_tokens,
                spec,
            });
            release_draft(draft, id);
            live.remove(&id);
            PushStatus::Finished
        }
        Err(err) => {
            drop_request(scheduler, draft, live, id, Some(err.to_string()));
            PushStatus::Dropped
        }
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
            None,
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

    fn tiny(layers: usize, hidden: usize, vocab: usize) -> flint_core::ModelConfig {
        flint_core::ModelConfig {
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: vocab,
            tie_word_embeddings: false,
            ..flint_core::ModelConfig::dummy()
        }
    }

    fn sched() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 1,
            max_batched_tokens: 64,
            prefill_chunk_size: 32,
            max_waiting: 1,
        }
    }

    fn drive(
        model: Model,
        draft: Option<Model>,
        prompt: Vec<u32>,
        max_tokens: usize,
    ) -> (String, Option<SpeculativeStats>) {
        let draft = draft.map(|model| DraftSpec {
            model: Box::new(model),
            k: 4,
        });
        let (engine, thread) =
            spawn(Box::new(model), draft, Arc::new(IdCodec), 48, 4, sched()).unwrap();
        let mut submission = engine
            .submit(prompt, SamplingParams::greedy(), max_tokens, Vec::new())
            .unwrap();
        let (text, spec) = futures::executor::block_on(async {
            let mut text = String::new();
            let mut spec = None;
            while let Some(event) = submission.events.recv().await {
                match event {
                    EngineEvent::Delta(delta) => text.push_str(&delta),
                    EngineEvent::Done { spec: stats, .. } => {
                        spec = stats;
                        break;
                    }
                    EngineEvent::Error(message) => panic!("{message}"),
                }
            }
            (text, spec)
        });
        drop(submission);
        drop(engine);
        thread.join().unwrap();
        (text, spec)
    }

    #[test]
    fn speculative_engine_matches_greedy_decoding() {
        let cfg = tiny(2, 32, 32);
        let prompt = vec![1, 4, 2, 7, 3, 5];
        let (baseline, baseline_spec) = drive(
            Model::from_random_with(&cfg, 7, flint_core::QuantMode::None).unwrap(),
            None,
            prompt.clone(),
            8,
        );
        assert!(baseline_spec.is_none());
        let (matched, matched_spec) = drive(
            Model::from_random_with(&cfg, 7, flint_core::QuantMode::None).unwrap(),
            Some(Model::from_random_with(&cfg, 7, flint_core::QuantMode::None).unwrap()),
            prompt.clone(),
            8,
        );
        assert_eq!(matched, baseline);
        let matched_spec = matched_spec.expect("draft run records stats");
        assert!(matched_spec.steps > 0);
        assert_eq!(matched_spec.acceptance_rate(), 1.0);
        let (mixed, mixed_spec) = drive(
            Model::from_random_with(&cfg, 7, flint_core::QuantMode::None).unwrap(),
            Some(
                Model::from_random_with(&tiny(1, 16, 32), 19, flint_core::QuantMode::None).unwrap(),
            ),
            prompt,
            8,
        );
        assert_eq!(mixed, baseline);
        let mixed_spec = mixed_spec.expect("different draft still records stats");
        assert!(mixed_spec.drafted_tokens > 0);

        match spawn(
            Box::new(Model::from_random_with(&cfg, 7, flint_core::QuantMode::None).unwrap()),
            Some(DraftSpec {
                model: Box::new(
                    Model::from_random_with(&tiny(1, 32, 30), 3, flint_core::QuantMode::None)
                        .unwrap(),
                ),
                k: 4,
            }),
            Arc::new(IdCodec),
            16,
            4,
            sched(),
        ) {
            Err(err) => assert!(err.to_string().contains("not compatible"), "{err}"),
            Ok(_) => panic!("incompatible draft was accepted"),
        }
    }
}
