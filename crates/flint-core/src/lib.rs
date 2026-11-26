//! Model core for Flint.
//!
//! The math is ours: RMSNorm, rotary embeddings, grouped-query attention, and
//! SwiGLU, on top of a hand-written GEMM. Weights are read from safetensors
//! (bf16, f16, or f32) and the tokenizer is the Hugging Face `tokenizer.json`.
//! Attention reads and writes a [`flint_kv::PagedKvCache`], which is the same
//! cache the HTTP engine serves from.
//!
//! The CPU path is f32 and is the oracle. With `--features cuda`, `GpuModel`
//! uploads those same weights and runs the forward pass on device.

#![deny(missing_docs)]

mod config;
mod error;
mod gemm;
#[cfg(feature = "cuda")]
mod gpu;
mod model;
mod ops;
mod sampling;
mod tokenizer;
mod weights;

pub use config::{EosTokenId, ModelConfig};
pub use error::Error;
#[cfg(feature = "cuda")]
pub use flint_kernels::cuda::GpuDType;
#[cfg(feature = "cuda")]
pub use gpu::GpuModel;
pub use model::{ForwardChunk, ForwardOutput, Model};
pub use sampling::{sample, SamplingParams};
pub use tokenizer::{apply_qwen_chat_template, ChatTurn, FlintTokenizer};
