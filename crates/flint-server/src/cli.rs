//! `flint serve`, `flint generate`, and `flint logits`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, Subcommand};
use flint_core::{
    apply_qwen_chat_template, ChatTurn, Error, FlintTokenizer, Model, ModelConfig, QuantMode,
    SamplingParams,
};
use hf_hub::api::sync::Api;

use crate::engine::{
    into_engine_model, spawn, EngineEvent, ExecDType, ExecDevice, FinishReason, TextCodec,
};
use crate::http::{router, AppState};
use crate::serve_runtime;

/// Byte-level codec for `flint --dummy`. Vocabulary is the 256 byte values.
/// Where the forward pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum DeviceKind {
    /// Host reference path. This is the default.
    Cpu,
    /// CUDA weights, cuBLAS GEMMs, and paged attention. Needs `--features cuda`.
    Cuda,
}

impl DeviceKind {
    fn exec(self) -> ExecDevice {
        match self {
            Self::Cpu => ExecDevice::Cpu,
            Self::Cuda => ExecDevice::Cuda,
        }
    }
}

/// Weight storage. `auto` is f32 on CPU and bf16 on CUDA.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum DTypeKind {
    /// f32 on CPU, bf16 on CUDA.
    Auto,
    /// Full precision, including the CUDA parity path.
    F32,
    /// bf16 storage with f32 accumulation. CUDA only.
    Bf16,
}

/// Weight quantization. Activations stay at `--dtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum QuantKind {
    /// Full-precision weights.
    None,
    /// Per-output-channel symmetric INT8 weights.
    Int8,
}

impl QuantKind {
    fn mode(self) -> QuantMode {
        match self {
            Self::None => QuantMode::None,
            Self::Int8 => QuantMode::Int8,
        }
    }
}

fn resolve_dtype(device: DeviceKind, dtype: DTypeKind) -> anyhow::Result<ExecDType> {
    match (device, dtype) {
        (DeviceKind::Cpu, DTypeKind::Bf16) => {
            anyhow::bail!("--dtype bf16 requires --device cuda")
        }
        (DeviceKind::Cpu, DTypeKind::Auto | DTypeKind::F32) => Ok(ExecDType::F32),
        (DeviceKind::Cuda, DTypeKind::F32) => Ok(ExecDType::F32),
        (DeviceKind::Cuda, DTypeKind::Auto | DTypeKind::Bf16) => Ok(ExecDType::Bf16),
    }
}

fn require_device(device: DeviceKind) -> anyhow::Result<()> {
    if device == DeviceKind::Cuda && !cfg!(feature = "cuda") {
        anyhow::bail!("--device cuda requires rebuilding flint with `--features cuda`");
    }
    Ok(())
}

struct ByteCodec;

impl TextCodec for ByteCodec {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(text.bytes().map(|byte| byte as u32).collect())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let bytes: Vec<u8> = ids.iter().map(|id| *id as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn apply_chat_template(&self, messages: &[ChatTurn]) -> Result<String, Error> {
        apply_qwen_chat_template(messages)
    }
}

struct HfCodec {
    inner: FlintTokenizer,
}

impl TextCodec for HfCodec {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.inner.decode(ids)
    }

    fn apply_chat_template(&self, messages: &[ChatTurn]) -> Result<String, Error> {
        apply_qwen_chat_template(messages)
    }
}

#[derive(Parser)]
#[command(name = "flint", version, about = "Flint inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Serve an OpenAI-compatible HTTP API.
    Serve(ServeArgs),
    /// Generate text for one prompt.
    Generate(GenerateArgs),
    /// Print last-token logits for one or more prompts.
    Logits(LogitsArgs),
}

#[derive(Parser)]
struct ModelArgs {
    /// Hugging Face model id used when `--model-dir` is omitted.
    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
    model: String,
    /// Directory containing `config.json`, `tokenizer.json`, and safetensors.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Serve a tiny random model with a byte-level codec. No download.
    #[arg(long)]
    dummy: bool,
    /// Execution device. `cpu` is the default.
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceKind,
    /// Weight storage. `auto` is f32 on CPU and bf16 on CUDA.
    #[arg(long, value_enum, default_value = "auto")]
    dtype: DTypeKind,
    /// Linear-weight quantization. `none` keeps the checkpoint dtype.
    ///
    /// `int8` is per-channel symmetric weight-only INT8. Norms and biases stay
    /// full precision, and activations follow `--dtype`.
    #[arg(long, value_enum, default_value = "none")]
    quant: QuantKind,
}

#[derive(Parser)]
struct ServeArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Port to bind.
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Tokens stored in one KV block.
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    /// Physical KV blocks.
    #[arg(long, default_value_t = 256)]
    num_blocks: usize,
    /// Sequences admitted at once.
    #[arg(long, default_value_t = 8)]
    max_num_seqs: usize,
    /// Tokens scheduled in one iteration.
    #[arg(long, default_value_t = 512)]
    max_batched_tokens: usize,
    /// Largest prefill chunk.
    #[arg(long, default_value_t = 128)]
    prefill_chunk_size: usize,
    /// Maximum prompt length accepted by HTTP.
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,
    /// `max_tokens` when a request omits it.
    #[arg(long, default_value_t = 128)]
    default_max_tokens: usize,
    /// Largest `max_tokens` a request may set.
    #[arg(long, default_value_t = 1024)]
    max_new_tokens: usize,
}

#[derive(Parser)]
struct GenerateArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Raw prompt. Omit when using `--chat`.
    #[arg(long)]
    prompt: Option<String>,
    /// User message. Wrapped in the Qwen chat template.
    #[arg(long)]
    chat: Option<String>,
    /// Tokens to generate.
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    /// `0` is greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// Nucleus probability. `1` disables the filter.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Keep only the k largest logits.
    #[arg(long)]
    top_k: Option<usize>,
    /// Sampling seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Print a JSON object instead of plain text.
    #[arg(long)]
    json: bool,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    #[arg(long, default_value_t = 256)]
    num_blocks: usize,
}

#[derive(Parser)]
struct LogitsArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Prompt to score. Repeat the flag for several prompts.
    #[arg(long, required = true)]
    prompt: Vec<String>,
    /// Write the JSON array here. Defaults to stdout.
    #[arg(long)]
    out: Option<PathBuf>,
    /// KV block size used for the scoring forward pass.
    #[arg(long, default_value_t = 16)]
    block_size: usize,
}

/// Parsed CLI. Called from the binary.
pub async fn cli() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    match cli.command {
        Commands::Serve(args) => serve(args).await,
        Commands::Generate(args) => generate(args).await,
        Commands::Logits(args) => logits(args),
    }
}

async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    require_device(args.model.device)?;
    let dtype = resolve_dtype(args.model.device, args.model.dtype)?;
    let (model, codec, model_name) = load_pair(&args.model)?;
    let num_blocks = args.num_blocks;
    let block_size = args.block_size;
    let model = into_engine_model(
        model,
        args.model.device.exec(),
        dtype,
        num_blocks,
        block_size,
    )?;
    tracing::info!(device = ?args.model.device, "forward device");
    let (engine, thread) = spawn(
        model,
        Arc::clone(&codec),
        num_blocks,
        block_size,
        flint_schedule::SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
            max_batched_tokens: args.max_batched_tokens,
            prefill_chunk_size: args.prefill_chunk_size,
            max_waiting: 64,
        },
    )?;
    let state = AppState {
        engine,
        codec,
        model_name: model_name.clone(),
        max_seq_len: args.max_seq_len,
        default_max_tokens: args.default_max_tokens,
        max_new_tokens: args.max_new_tokens,
        max_cache_tokens: num_blocks * block_size,
    };
    let app = router(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    tracing::info!(%addr, model = %model_name, blocks = num_blocks, block_size, "listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(serve_runtime::shutdown_signal())
        .await
        .context("http server")?;
    drop(thread);
    Ok(())
}

async fn generate(args: GenerateArgs) -> anyhow::Result<()> {
    require_device(args.model.device)?;
    let dtype = resolve_dtype(args.model.device, args.model.dtype)?;
    let (model, codec, _) = load_pair(&args.model)?;
    let model = into_engine_model(
        model,
        args.model.device.exec(),
        dtype,
        args.num_blocks,
        args.block_size,
    )?;
    let prompt = match (&args.prompt, &args.chat) {
        (Some(prompt), None) => prompt.clone(),
        (None, Some(chat)) => apply_qwen_chat_template(&[ChatTurn {
            role: "user".into(),
            content: chat.clone(),
        }])?,
        _ => anyhow::bail!("pass exactly one of --prompt or --chat"),
    };
    let tokens = codec.encode(&prompt)?;
    let sampling = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: Some(args.top_p),
        seed: args.seed,
    };
    sampling.validate()?;
    let (engine, _thread) = spawn(
        model,
        Arc::clone(&codec),
        args.num_blocks,
        args.block_size,
        flint_schedule::SchedulerConfig {
            max_num_seqs: 1,
            max_batched_tokens: 512,
            prefill_chunk_size: 128,
            max_waiting: 1,
        },
    )?;
    let mut submission = engine
        .submit(tokens, sampling, args.max_tokens, Vec::new())
        .map_err(|_| anyhow::anyhow!("engine rejected the prompt"))?;
    let started = std::time::Instant::now();
    let mut text = String::new();
    let mut completion_tokens = 0_usize;
    let mut reason = FinishReason::Length;
    while let Some(event) = submission.events.recv().await {
        match event {
            EngineEvent::Delta(delta) => text.push_str(&delta),
            EngineEvent::Done {
                reason: done_reason,
                completion_tokens: n,
                ..
            } => {
                completion_tokens = n;
                reason = done_reason;
                break;
            }
            EngineEvent::Error(message) => anyhow::bail!(message),
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let tok_s = if elapsed > 0.0 {
        completion_tokens as f64 / elapsed
    } else {
        0.0
    };
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "text": text,
                "completion_tokens": completion_tokens,
                "finish_reason": reason.as_str(),
                "seconds": elapsed,
                "tokens_per_second": tok_s,
            })
        );
    } else {
        println!("{text}");
        eprintln!(
            "finish={reason} tokens={completion_tokens} time={elapsed:.3}s tok/s={tok_s:.2}",
            reason = reason.as_str()
        );
    }
    Ok(())
}

fn logits(args: LogitsArgs) -> anyhow::Result<()> {
    if args.model.dummy {
        anyhow::bail!("--dummy has no checkpoint to score; use generate");
    }
    require_device(args.model.device)?;
    resolve_dtype(args.model.device, args.model.dtype)?;
    let dir = resolve_dir(&args.model)?;
    let model = Model::load_with(&dir, args.model.quant.mode())?;
    let tokenizer = FlintTokenizer::open(&dir)?;
    let encoded: Vec<(String, Vec<u32>)> = args
        .prompt
        .iter()
        .map(|prompt| {
            let token_ids = tokenizer.encode(prompt)?;
            Ok((prompt.clone(), token_ids))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut rows = Vec::new();
    let scored = score_prompts(&args, model, &encoded)?;
    for ScoredPrompt {
        prompt,
        token_ids,
        logits,
        last_hidden,
    } in scored
    {
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        eprintln!(
            "prompt_tokens={} argmax={argmax} logit={:.4}",
            token_ids.len(),
            logits[argmax]
        );
        rows.push(serde_json::json!({
            "prompt": prompt,
            "token_ids": token_ids,
            "argmax": argmax,
            "logits": logits,
            "last_hidden": last_hidden,
        }));
    }
    let text = serde_json::to_string(&rows)?;
    if let Some(path) = &args.out {
        std::fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
    } else {
        println!("{text}");
    }
    Ok(())
}

struct ScoredPrompt {
    prompt: String,
    token_ids: Vec<u32>,
    logits: Vec<f32>,
    last_hidden: Option<Vec<f32>>,
}

fn score_prompts(
    args: &LogitsArgs,
    mut model: Model,
    encoded: &[(String, Vec<u32>)],
) -> anyhow::Result<Vec<ScoredPrompt>> {
    match args.model.device {
        DeviceKind::Cpu => {
            let mut rows = Vec::new();
            for (prompt, token_ids) in encoded {
                let output = model.forward_prompt(token_ids, args.block_size)?;
                let logits = output
                    .logits
                    .context("forward pass did not return logits")?;
                rows.push(ScoredPrompt {
                    prompt: prompt.clone(),
                    token_ids: token_ids.clone(),
                    logits,
                    last_hidden: output.last_hidden,
                });
            }
            Ok(rows)
        }
        DeviceKind::Cuda => score_cuda(args, model, encoded),
    }
}

#[cfg(feature = "cuda")]
fn score_cuda(
    args: &LogitsArgs,
    model: Model,
    encoded: &[(String, Vec<u32>)],
) -> anyhow::Result<Vec<ScoredPrompt>> {
    let max_len = encoded.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1);
    let n_blocks = max_len.div_ceil(args.block_size).max(1);
    let dtype = match resolve_dtype(args.model.device, args.model.dtype)? {
        ExecDType::F32 => flint_core::GpuDType::F32,
        ExecDType::Bf16 => flint_core::GpuDType::Bf16,
    };
    let mut gpu = model.into_gpu_with(n_blocks, args.block_size, dtype)?;
    let mut rows = Vec::new();
    for (prompt, token_ids) in encoded {
        let output = gpu.forward_prompt(token_ids)?;
        let logits = output
            .logits
            .context("forward pass did not return logits")?;
        rows.push(ScoredPrompt {
            prompt: prompt.clone(),
            token_ids: token_ids.clone(),
            logits,
            last_hidden: output.last_hidden,
        });
    }
    Ok(rows)
}

#[cfg(not(feature = "cuda"))]
fn score_cuda(
    _args: &LogitsArgs,
    _model: Model,
    _encoded: &[(String, Vec<u32>)],
) -> anyhow::Result<Vec<ScoredPrompt>> {
    anyhow::bail!("--device cuda requires rebuilding flint with `--features cuda`")
}

fn load_pair(args: &ModelArgs) -> anyhow::Result<(Model, Arc<dyn TextCodec>, String)> {
    if args.dummy {
        let model = Model::from_random_with(&ModelConfig::dummy(), 1, args.quant.mode())?;
        let name = "flint-dummy".to_string();
        return Ok((model, Arc::new(ByteCodec), name));
    }
    let dir = resolve_dir(args)?;
    let model = Model::load_with(&dir, args.quant.mode())?;
    let tokenizer = FlintTokenizer::open(&dir)?;
    let name = args.model.clone();
    Ok((model, Arc::new(HfCodec { inner: tokenizer }), name))
}

fn resolve_dir(args: &ModelArgs) -> anyhow::Result<PathBuf> {
    if let Some(dir) = &args.model_dir {
        return Ok(dir.clone());
    }
    download_model(&args.model)
}

fn download_model(model_id: &str) -> anyhow::Result<PathBuf> {
    tracing::info!(model_id, "downloading weights and tokenizer");
    let api = Api::new().context("huggingface hub api")?;
    let repo = api.model(model_id.to_string());
    let config = repo.get("config.json").context("config.json")?;
    let snapshot = config.parent().context("snapshot directory")?.to_path_buf();
    repo.get("tokenizer.json").context("tokenizer.json")?;
    let _ = repo.get("tokenizer_config.json");
    let _ = repo.get("generation_config.json");
    if repo.get("model.safetensors").is_err() {
        let index = repo
            .get("model.safetensors.index.json")
            .context("model.safetensors or an index")?;
        let text = std::fs::read_to_string(&index)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        let map = value
            .get("weight_map")
            .and_then(|item| item.as_object())
            .context("weight_map")?;
        let mut files: Vec<&str> = map.values().filter_map(|item| item.as_str()).collect();
        files.sort_unstable();
        files.dedup();
        for file in files {
            repo.get(file).with_context(|| format!("shard {file}"))?;
        }
    }
    tracing::info!(path = %snapshot.display(), "snapshot ready");
    Ok(snapshot)
}
