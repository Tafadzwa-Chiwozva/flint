//! `flint-bench` command line.

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use flint_bench::{report_csv, run, BenchConfig, Endpoint};

#[derive(Parser)]
#[command(
    name = "flint-bench",
    version,
    about = "Benchmark an OpenAI-compatible LLM server"
)]
struct Args {
    /// Base URL, for example `http://127.0.0.1:8000`.
    #[arg(long)]
    url: String,
    /// Model name placed in the request body.
    #[arg(long, default_value = "flint")]
    model: String,
    /// Prompt. Repeat to cycle through several. Defaults are used when omitted.
    #[arg(long)]
    prompt: Vec<String>,
    /// `chat` or `completions`.
    #[arg(long, default_value = "chat")]
    endpoint: String,
    /// In-flight requests.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    /// Measured requests.
    #[arg(long, default_value_t = 4)]
    requests: usize,
    /// Unmeasured requests before the timer starts.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Generation cap per request.
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
    /// Sampling temperature. Zero is greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// Bearer token, if the server requires one.
    #[arg(long)]
    api_key: Option<String>,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 600)]
    timeout_secs: u64,
    /// Write the JSON report here.
    #[arg(long)]
    json_out: Option<PathBuf>,
    /// Write the CSV summary here.
    #[arg(long)]
    csv_out: Option<PathBuf>,
    /// Label for the server's `--quant` mode, stored on the JSON report.
    ///
    /// This client does not load weights. Start `flint serve` with the same
    /// value (`none` or `int8`).
    #[arg(long, value_enum, default_value = "none")]
    quant: QuantKind,
}

/// Matches `flint serve --quant`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum QuantKind {
    /// Full-precision weights.
    None,
    /// Per-channel symmetric INT8 weights.
    Int8,
}

impl QuantKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Int8 => "int8",
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(err) = run_cli().await {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

async fn run_cli() -> anyhow::Result<()> {
    let args = Args::parse();
    let endpoint = match args.endpoint.as_str() {
        "chat" => Endpoint::Chat,
        "completions" => Endpoint::Completions,
        other => anyhow::bail!("unknown endpoint '{other}' (expected chat or completions)"),
    };
    let prompts = if args.prompt.is_empty() {
        default_prompts()
    } else {
        args.prompt
    };
    let report = run(&BenchConfig {
        url: args.url,
        model: args.model,
        prompts,
        concurrency: args.concurrency,
        requests: args.requests,
        warmup: args.warmup,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        api_key: args.api_key,
        timeout: Duration::from_secs(args.timeout_secs),
        endpoint,
        quant: args.quant.as_str().to_string(),
    })
    .await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = &args.json_out {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    if let Some(path) = &args.csv_out {
        std::fs::write(path, report_csv(&report))?;
    }
    Ok(())
}

fn default_prompts() -> Vec<String> {
    vec![
        "Hello, my name is".into(),
        "The capital of France is".into(),
        "Explain Rust ownership in two sentences.".into(),
        "Write a haiku about memory.".into(),
    ]
}
