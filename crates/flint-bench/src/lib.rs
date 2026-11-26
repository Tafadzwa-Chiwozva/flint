//! Load generator for OpenAI-compatible chat and completion servers.
//!
//! Point it at Flint, vLLM, or llama.cpp server. Every target is measured
//! through the same streaming protocol, so tokens per second, time to first
//! token, and inter-token latency stay comparable when the GPU phase lands.
//!
//! Inter-token latency is the gap between SSE events that carry text. The
//! output-token count prefers `usage.completion_tokens` when the server sends
//! it (Flint puts that on the final chunk) and otherwise counts text events.

#![deny(missing_docs)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Which route to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    /// `POST /v1/chat/completions`.
    Chat,
    /// `POST /v1/completions`.
    Completions,
}

/// What to measure.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Base URL, without a trailing path. Example: `http://127.0.0.1:8000`.
    pub url: String,
    /// Model name sent in the request body.
    pub model: String,
    /// Prompts cycled across requests.
    pub prompts: Vec<String>,
    /// In-flight requests.
    pub concurrency: usize,
    /// Measured requests, after warmup.
    pub requests: usize,
    /// Requests issued one at a time before the timer starts.
    pub warmup: usize,
    /// `max_tokens` for every request.
    pub max_tokens: usize,
    /// Sampling temperature. `0` is greedy on Flint and on vLLM.
    pub temperature: f32,
    /// Optional bearer token.
    pub api_key: Option<String>,
    /// Per-request HTTP timeout.
    pub timeout: Duration,
    /// Route to call.
    pub endpoint: Endpoint,
    /// Weight quantization the server was started with.
    ///
    /// The client does not quantize. The label is stored on the report so a
    /// bench of `--quant none` and `--quant int8` stays distinguishable.
    pub quant: String,
}

/// One finished request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestSample {
    /// Prompt index into [`BenchConfig::prompts`].
    pub prompt_index: usize,
    /// Output tokens, from usage when present.
    pub output_tokens: usize,
    /// Time to the first text event, milliseconds.
    pub ttft_ms: f64,
    /// Gaps between text events, milliseconds. Empty when the reply is one token.
    pub itl_ms: Vec<f64>,
    /// Request start to the terminal event, milliseconds.
    pub e2e_ms: f64,
}

/// p50 / p99 and the usual range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    /// 50th percentile.
    pub p50: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Smallest sample.
    pub min: f64,
    /// Largest sample.
    pub max: f64,
}

/// Aggregate report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Server base URL.
    pub url: String,
    /// Model field sent to the server.
    pub model: String,
    /// Measured requests.
    pub requests: usize,
    /// Sum of output tokens.
    pub output_tokens: usize,
    /// Wall time of the measured section, seconds.
    pub wall_seconds: f64,
    /// `output_tokens / wall_seconds`.
    pub throughput_tok_s: f64,
    /// Time to first token.
    pub ttft_ms: Percentiles,
    /// Inter-token latency. `None` when no request produced two text events.
    pub itl_ms: Option<Percentiles>,
    /// End to end latency.
    pub e2e_ms: Percentiles,
    /// Quantization label from [`BenchConfig::quant`].
    pub quant: String,
    /// Per-request samples, in completion order.
    pub samples: Vec<RequestSample>,
}

/// Nearest-rank percentile. Index is `round((p / 100) * (n - 1))`.
#[must_use]
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let last = (sorted.len() - 1) as f64;
    let index = ((p / 100.0) * last).round() as usize;
    sorted.get(index.min(sorted.len() - 1)).copied()
}

fn summarize(values: &[f64]) -> Option<Percentiles> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Some(Percentiles {
        p50: percentile(values, 50.0)?,
        p99: percentile(values, 99.0)?,
        mean,
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

/// Incremental SSE decoder. Events are split on a blank line.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: String,
}

impl SseDecoder {
    /// Empty decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Push a chunk of the response body and return `data:` payloads.
    pub fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut payloads = Vec::new();
        while let Some(index) = self.buffer.find("\n\n") {
            let event = self.buffer[..index].to_string();
            self.buffer.drain(..index + 2);
            for line in event.lines() {
                let Some(rest) = line.strip_prefix("data:") else {
                    continue;
                };
                payloads.push(rest.trim_start().to_string());
            }
        }
        payloads
    }
}

/// Run the benchmark.
pub async fn run(config: &BenchConfig) -> anyhow::Result<Report> {
    if config.prompts.is_empty() {
        anyhow::bail!("at least one prompt is required");
    }
    if config.concurrency == 0 || config.requests == 0 {
        anyhow::bail!("concurrency and requests must be at least 1");
    }
    let client = reqwest::Client::builder()
        .timeout(config.timeout)
        .build()
        .context("build http client")?;
    let url = endpoint_url(&config.url, config.endpoint);
    for index in 0..config.warmup {
        let prompt = &config.prompts[index % config.prompts.len()];
        one_request(&client, &url, config, prompt, index)
            .await
            .with_context(|| format!("warmup request {index}"))?;
    }

    let prompts = Arc::new(config.prompts.clone());
    let next = Arc::new(AtomicUsize::new(0));
    let wall_start = Instant::now();
    let mut tasks = Vec::with_capacity(config.concurrency);
    for _ in 0..config.concurrency {
        let client = client.clone();
        let url = url.clone();
        let prompts = Arc::clone(&prompts);
        let next = Arc::clone(&next);
        let config = config.clone();
        tasks.push(tokio::spawn(async move {
            let mut samples = Vec::new();
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= config.requests {
                    break;
                }
                let prompt = prompts[index % prompts.len()].clone();
                let sample = one_request(&client, &url, &config, &prompt, index).await?;
                samples.push(sample);
            }
            Ok::<_, anyhow::Error>(samples)
        }));
    }
    let mut samples = Vec::new();
    for task in tasks {
        samples.extend(task.await.context("bench worker panicked")??);
    }
    let wall_seconds = wall_start.elapsed().as_secs_f64();
    let output_tokens: usize = samples.iter().map(|sample| sample.output_tokens).sum();
    let ttft: Vec<f64> = samples.iter().map(|sample| sample.ttft_ms).collect();
    let e2e: Vec<f64> = samples.iter().map(|sample| sample.e2e_ms).collect();
    let itl: Vec<f64> = samples
        .iter()
        .flat_map(|sample| sample.itl_ms.clone())
        .collect();
    let throughput_tok_s = if wall_seconds > 0.0 {
        output_tokens as f64 / wall_seconds
    } else {
        0.0
    };
    Ok(Report {
        url: config.url.clone(),
        model: config.model.clone(),
        requests: samples.len(),
        output_tokens,
        wall_seconds,
        throughput_tok_s,
        ttft_ms: summarize(&ttft).context("no TTFT samples")?,
        itl_ms: summarize(&itl),
        e2e_ms: summarize(&e2e).context("no end-to-end samples")?,
        quant: config.quant.clone(),
        samples,
    })
}

fn endpoint_url(base: &str, endpoint: Endpoint) -> String {
    let base = base.trim_end_matches('/');
    match endpoint {
        Endpoint::Chat => format!("{base}/v1/chat/completions"),
        Endpoint::Completions => format!("{base}/v1/completions"),
    }
}

fn request_body(config: &BenchConfig, prompt: &str) -> Value {
    match config.endpoint {
        Endpoint::Chat => json!({
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": true,
        }),
        Endpoint::Completions => json!({
            "model": config.model,
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": true,
        }),
    }
}

async fn one_request(
    client: &reqwest::Client,
    url: &str,
    config: &BenchConfig,
    prompt: &str,
    prompt_index: usize,
) -> anyhow::Result<RequestSample> {
    let started = Instant::now();
    let mut builder = client.post(url).json(&request_body(config, prompt));
    if let Some(key) = &config.api_key {
        builder = builder.bearer_auth(key);
    }
    let response = builder.send().await.context("send request")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("{status}: {body}");
    }
    let mut stream = response.bytes_stream();
    let mut raw = Vec::new();
    let mut decoder = SseDecoder::new();
    let mut text_at: Vec<Duration> = Vec::new();
    let mut usage_tokens: Option<usize> = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read response body")?;
        raw.extend_from_slice(&chunk);
        let text = take_utf8(&mut raw);
        for payload in decoder.push(&text) {
            if payload == "[DONE]" {
                break;
            }
            let value: Value = serde_json::from_str(&payload)
                .with_context(|| format!("sse payload is not json: {payload}"))?;
            if let Some(message) = value.get("error").and_then(|err| err.get("message")) {
                anyhow::bail!("server error: {message}");
            }
            if let Some(tokens) = value
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
            {
                usage_tokens = Some(tokens as usize);
            }
            if chunk_has_text(&value) {
                text_at.push(started.elapsed());
            }
        }
    }
    let end = started.elapsed();
    let ttft = text_at.first().copied().unwrap_or(end);
    let mut itl_ms = Vec::new();
    for pair in text_at.windows(2) {
        itl_ms.push((pair[1] - pair[0]).as_secs_f64() * 1000.0);
    }
    let output_tokens = usage_tokens.unwrap_or(text_at.len());
    Ok(RequestSample {
        prompt_index,
        output_tokens,
        ttft_ms: ttft.as_secs_f64() * 1000.0,
        itl_ms,
        e2e_ms: end.as_secs_f64() * 1000.0,
    })
}

fn chunk_has_text(value: &Value) -> bool {
    let choice = &value["choices"][0];
    let delta = choice["delta"]["content"].as_str().unwrap_or("");
    let text = choice["text"].as_str().unwrap_or("");
    !delta.is_empty() || !text.is_empty()
}

fn take_utf8(buf: &mut Vec<u8>) -> String {
    match std::str::from_utf8(buf) {
        Ok(text) => {
            let out = text.to_string();
            buf.clear();
            out
        }
        Err(err) => {
            let valid = err.valid_up_to();
            let out = String::from_utf8_lossy(&buf[..valid]).into_owned();
            buf.drain(..valid);
            out
        }
    }
}

/// One CSV row per metric. Throughput is a single value in the p50 column.
#[must_use]
pub fn report_csv(report: &Report) -> String {
    let mut out = String::from("metric,p50,p99,mean,min,max\n");
    push_percentiles(&mut out, "ttft_ms", &report.ttft_ms);
    if let Some(itl) = &report.itl_ms {
        push_percentiles(&mut out, "itl_ms", itl);
    }
    push_percentiles(&mut out, "e2e_ms", &report.e2e_ms);
    out.push_str(&format!(
        "throughput_tok_s,{:.4},,,,\nrequests,{},,,,\noutput_tokens,{},,,,\nwall_seconds,{:.4},,,,\n",
        report.throughput_tok_s, report.requests, report.output_tokens, report.wall_seconds
    ));
    out
}

fn push_percentiles(out: &mut String, name: &str, stats: &Percentiles) {
    out.push_str(&format!(
        "{name},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
        stats.p50, stats.p99, stats.mean, stats.min, stats.max
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentile() {
        let values = [1.0, 2.0, 3.0, 4.0];
        // round((50/100) * (4 - 1)) = round(1.5) = 2, so the third sample.
        assert_eq!(percentile(&values, 50.0), Some(3.0));
        assert_eq!(percentile(&values, 100.0), Some(4.0));
        assert_eq!(percentile(&[], 50.0), None);
        let one = [7.0];
        assert_eq!(percentile(&one, 99.0), Some(7.0));
    }

    #[test]
    fn sse_decoder_splits_across_chunks() {
        let mut decoder = SseDecoder::new();
        assert!(decoder.push("data: {\"a\":1}\n").is_empty());
        let payloads = decoder.push("\ndata: [DONE]\n\n");
        assert_eq!(
            payloads,
            vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]
        );
    }

    #[test]
    fn csv_contains_throughput() {
        let report = Report {
            url: "http://127.0.0.1".into(),
            model: "m".into(),
            requests: 1,
            output_tokens: 4,
            wall_seconds: 2.0,
            throughput_tok_s: 2.0,
            ttft_ms: Percentiles {
                p50: 1.0,
                p99: 1.0,
                mean: 1.0,
                min: 1.0,
                max: 1.0,
            },
            itl_ms: None,
            e2e_ms: Percentiles {
                p50: 2.0,
                p99: 2.0,
                mean: 2.0,
                min: 2.0,
                max: 2.0,
            },
            quant: "none".into(),
            samples: Vec::new(),
        };
        let csv = report_csv(&report);
        assert!(csv.contains("throughput_tok_s,2.0000"));
        assert!(csv.contains("ttft_ms,"));
    }
}
