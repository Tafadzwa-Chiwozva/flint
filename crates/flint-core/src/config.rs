//! Hugging Face `config.json` for a Qwen2-style causal LM.
//!
//! Llama-architecture models that share this layout (RMSNorm, RoPE, GQA,
//! SwiGLU) can be described by the same struct. Qwen2-specific pieces are the
//! optional Q/K/V bias and the sliding-window flag.

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// `eos_token_id` is a scalar on some configs and a list on others.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum EosTokenId {
    /// A single end-of-sequence id.
    One(u32),
    /// Several ids, any of which ends generation.
    Many(Vec<u32>),
}

impl Default for EosTokenId {
    fn default() -> Self {
        Self::Many(Vec::new())
    }
}

/// Architecture hyperparameters.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Hugging Face architecture names. Informational.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Residual width.
    pub hidden_size: usize,
    /// SwiGLU intermediate width.
    pub intermediate_size: usize,
    /// Transformer layers.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads. Grouped-query attention when this is smaller.
    pub num_key_value_heads: usize,
    /// Explicit head dimension. Defaults to `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    /// RoPE base.
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    /// Maximum position the checkpoint was trained with.
    #[serde(default)]
    pub max_position_embeddings: usize,
    /// When true, the LM head reuses the embedding matrix.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Activation name. Phase 1 implements `silu`.
    #[serde(default)]
    pub hidden_act: String,
    /// Master switch for sliding-window attention.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Window length used by layers at or past [`Self::max_window_layers`].
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// First layer index that uses the sliding window when the flag is on.
    #[serde(default)]
    pub max_window_layers: usize,
    /// Optional beginning-of-sequence id. Qwen2 does not add it by default.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// End-of-sequence ids from `config.json`.
    #[serde(default)]
    pub eos_token_id: EosTokenId,
    /// Present and non-null means a RoPE scaling variant we do not implement yet.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

fn default_eps() -> f64 {
    1e-6
}

fn default_theta() -> f64 {
    10_000.0
}

impl ModelConfig {
    /// Head dimension, from the config or from `hidden_size / num_attention_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    /// Query projection width.
    #[must_use]
    pub fn qo_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    /// Key/value projection width.
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// End-of-sequence ids declared on the model config.
    #[must_use]
    pub fn eos_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            EosTokenId::One(id) => vec![*id],
            EosTokenId::Many(ids) => ids.clone(),
        }
    }

    /// Sliding window for one layer, if this checkpoint uses one.
    ///
    /// Matches Hugging Face Qwen2: the window is disabled unless
    /// `use_sliding_window` is set, and then only layers with
    /// `layer_idx >= max_window_layers` use it.
    #[must_use]
    pub fn sliding_window_for_layer(&self, layer: usize) -> Option<usize> {
        if !self.use_sliding_window {
            return None;
        }
        let window = self.sliding_window.filter(|window| *window > 0)?;
        if layer >= self.max_window_layers {
            Some(window)
        } else {
            None
        }
    }

    /// Reject shapes and features the phase-1 forward pass cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
        {
            return Err(Error::Config(
                "hidden size, intermediate size, layers, heads, and vocab must be non-zero".into(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::Config(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        let head = self.head_dim();
        if head == 0 || !head.is_multiple_of(2) {
            return Err(Error::Config(
                "head_dim must be a positive even number (RoPE pairs dimensions)".into(),
            ));
        }
        if self.hidden_act != "silu" && !self.hidden_act.is_empty() {
            return Err(Error::Config(format!(
                "unsupported hidden_act '{}'; phase 1 implements silu",
                self.hidden_act
            )));
        }
        if self
            .rope_scaling
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Err(Error::Config(
                "rope_scaling is not implemented in phase 1".into(),
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps < 0.0 {
            return Err(Error::Config("rms_norm_eps must be finite and >= 0".into()));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::Config("rope_theta must be finite and > 0".into()));
        }
        Ok(())
    }

    /// Small deterministic architecture for tests and `--dummy` serving.
    #[must_use]
    pub fn dummy() -> Self {
        Self {
            architectures: vec!["FlintDummy".into()],
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: None,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 2048,
            tie_word_embeddings: true,
            hidden_act: "silu".into(),
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 0,
            bos_token_id: None,
            eos_token_id: EosTokenId::Many(vec![]),
            rope_scaling: None,
        }
    }
}
