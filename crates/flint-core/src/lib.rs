//! Model core for Flint.
//!
//! The math is ours: RMSNorm, rotary embeddings, grouped-query attention, and
//! SwiGLU, on top of a hand-written GEMM. Weights are read from safetensors
//! (bf16, f16, or f32) and the tokenizer is the Hugging Face `tokenizer.json`.
//! Attention reads and writes a [`flint_kv::PagedKvCache`], which is the same
//! cache the HTTP engine serves from.
//!
//! Phase 1 keeps this path on CPU and in f32. The `flint-kernels` crate holds
//! the CUDA stubs that will replace [`flint_kernels::rms_norm`] and
//! [`flint_kernels::paged_attention`] without changing this module's structure.

#![deny(missing_docs)]

mod config;
mod error;
mod gemm;
mod model;
mod ops;
mod sampling;
mod tokenizer;
mod weights;

pub use config::{EosTokenId, ModelConfig};
pub use error::Error;
pub use model::{ForwardOutput, Model};
pub use sampling::{sample, SamplingParams};
pub use tokenizer::{apply_qwen_chat_template, ChatTurn, FlintTokenizer};
