//! Qwen2 / Llama forward pass.
//!
//! One chunk of tokens is embedded, then each layer does pre-norm attention
//! (RoPE on Q and K, paged GQA, output projection) and a pre-norm SwiGLU MLP.
//! Only the last token of a chunk that is allowed to sample pays for the
//! vocabulary projection. K and V are written into the paged cache before
//! attention, so a later decode step sees them through the block table.

use std::path::Path;

use flint_kernels::{rms_norm, AttentionParams};
use flint_kv::PagedKvCache;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::config::ModelConfig;
use crate::error::Error;
use crate::gemm::linear;
use crate::ops::{add_vec, apply_rope, inv_freq, rope_angles, swiglu};
use crate::weights::TensorMap;

/// Last-token state from a forward chunk.
#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Final RMSNorm output for the last token. Set when logits were requested.
    pub last_hidden: Option<Vec<f32>>,
    /// Vocabulary logits for that token.
    pub logits: Option<Vec<f32>>,
}

struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    out_features: usize,
    in_features: usize,
}

impl Linear {
    fn apply(&self, x: &[f32], y: &mut [f32], rows: usize) {
        linear(
            &self.weight,
            self.bias.as_deref(),
            x,
            y,
            rows,
            self.out_features,
            self.in_features,
        );
    }
}

struct LayerWeights {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

struct Scratch {
    hidden: Vec<f32>,
    residual: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    rope_tmp: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            hidden: Vec::new(),
            residual: Vec::new(),
            normed: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn: Vec::new(),
            proj: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            cos: Vec::new(),
            sin: Vec::new(),
            rope_tmp: Vec::new(),
        }
    }
}

/// A loaded causal language model.
pub struct Model {
    cfg: ModelConfig,
    embed: Vec<f32>,
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>,
    /// `None` when the LM head is tied to [`Self::embed`].
    lm_head: Option<Linear>,
    inv_freq: Vec<f32>,
    eos_ids: Vec<u32>,
    scratch: Scratch,
}

impl Model {
    /// Load `config.json`, optional `generation_config.json`, and safetensors.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path)
            .map_err(|err| Error::Config(format!("read {}: {err}", config_path.display())))?;
        let cfg: ModelConfig = serde_json::from_str(&text)
            .map_err(|err| Error::Config(format!("parse {}: {err}", config_path.display())))?;
        cfg.validate()?;
        let mut eos = cfg.eos_ids();
        for id in read_generation_eos(&dir.join("generation_config.json")) {
            if !eos.contains(&id) {
                eos.push(id);
            }
        }
        let skip: &[&str] = if cfg.tie_word_embeddings {
            &["lm_head.weight"]
        } else {
            &[]
        };
        let mut tensors = TensorMap::load_dir(dir, skip)?;
        let model = assemble(cfg, &mut tensors, eos)?;
        tracing::info!(
            layers = model.cfg.num_hidden_layers,
            params = model.parameter_count(),
            vocab = model.cfg.vocab_size,
            "model ready"
        );
        Ok(model)
    }

    /// Deterministic random weights for tests and `flint serve --dummy`.
    pub fn from_random(cfg: &ModelConfig, seed: u64) -> Result<Self, Error> {
        cfg.validate()?;
        let mut rng = StdRng::seed_from_u64(seed);
        let draw = |rng: &mut StdRng, n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
        };
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let qo = cfg.qo_dim();
        let kv = cfg.kv_dim();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights {
                input_norm: vec![1.0; hidden],
                post_norm: vec![1.0; hidden],
                q: Linear {
                    weight: draw(&mut rng, qo * hidden, 0.05),
                    bias: Some(draw(&mut rng, qo, 0.01)),
                    out_features: qo,
                    in_features: hidden,
                },
                k: Linear {
                    weight: draw(&mut rng, kv * hidden, 0.05),
                    bias: Some(draw(&mut rng, kv, 0.01)),
                    out_features: kv,
                    in_features: hidden,
                },
                v: Linear {
                    weight: draw(&mut rng, kv * hidden, 0.05),
                    bias: Some(draw(&mut rng, kv, 0.01)),
                    out_features: kv,
                    in_features: hidden,
                },
                o: Linear {
                    weight: draw(&mut rng, hidden * qo, 0.05),
                    bias: None,
                    out_features: hidden,
                    in_features: qo,
                },
                gate: Linear {
                    weight: draw(&mut rng, inter * hidden, 0.05),
                    bias: None,
                    out_features: inter,
                    in_features: hidden,
                },
                up: Linear {
                    weight: draw(&mut rng, inter * hidden, 0.05),
                    bias: None,
                    out_features: inter,
                    in_features: hidden,
                },
                down: Linear {
                    weight: draw(&mut rng, hidden * inter, 0.05),
                    bias: None,
                    out_features: hidden,
                    in_features: inter,
                },
            });
        }
        let embed = draw(&mut rng, cfg.vocab_size * hidden, 0.05);
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(Linear {
                weight: draw(&mut rng, cfg.vocab_size * hidden, 0.05),
                bias: None,
                out_features: cfg.vocab_size,
                in_features: hidden,
            })
        };
        Ok(Self {
            inv_freq: inv_freq(cfg.head_dim(), cfg.rope_theta as f32),
            eos_ids: cfg.eos_ids(),
            cfg: cfg.clone(),
            embed,
            layers,
            final_norm: vec![1.0; hidden],
            lm_head,
            scratch: Scratch::new(),
        })
    }

    /// Borrowed architecture.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Eos ids from the config and, when present, `generation_config.json`.
    #[must_use]
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Trainable parameters. A tied LM head is counted once.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let mut total = self.embed.len() + self.final_norm.len();
        if let Some(head) = &self.lm_head {
            total += head.weight.len();
        }
        for layer in &self.layers {
            total += layer.input_norm.len() + layer.post_norm.len();
            for linear in [
                &layer.q,
                &layer.k,
                &layer.v,
                &layer.o,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                total += linear.weight.len();
                if let Some(bias) = &linear.bias {
                    total += bias.len();
                }
            }
        }
        total
    }

    /// Run `tokens` as a fresh prompt and return last-token logits.
    ///
    /// `block_size` selects the page size of the temporary cache. Callers that
    /// already own a cache should use [`Self::forward`].
    pub fn forward_prompt(
        &mut self,
        tokens: &[u32],
        block_size: usize,
    ) -> Result<ForwardOutput, Error> {
        if tokens.is_empty() {
            return Err(Error::Inference("empty prompt".into()));
        }
        if block_size == 0 {
            return Err(Error::Inference("block_size must be non-zero".into()));
        }
        let n_blocks = tokens.len().div_ceil(block_size);
        let mut cache = PagedKvCache::with_dims(
            n_blocks,
            block_size,
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
        )
        .map_err(|err| Error::Inference(err.to_string()))?;
        let table: Vec<u32> = (0..n_blocks as u32).collect();
        self.forward(tokens, 0, &mut cache, &table, true)
    }

    /// Run one prefill chunk or decode step.
    ///
    /// `position` is the absolute index of `tokens[0]`. KV for those tokens is
    /// written into `cache` at the slots described by `block_table`. When
    /// `want_logits` is false the vocabulary projection is skipped; intermediate
    /// prefill chunks use that to avoid the LM head.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        position: usize,
        cache: &mut PagedKvCache,
        block_table: &[u32],
        want_logits: bool,
    ) -> Result<ForwardOutput, Error> {
        if tokens.is_empty() {
            return Err(Error::Inference("forward chunk is empty".into()));
        }
        if cache.num_layers() != self.cfg.num_hidden_layers
            || cache.num_kv_heads() != self.cfg.num_key_value_heads
            || cache.head_dim() != self.cfg.head_dim()
        {
            return Err(Error::Inference(
                "KV cache geometry does not match the model".into(),
            ));
        }
        let chunk = tokens.len();
        self.ensure_scratch(chunk);
        embed_tokens(
            &self.embed,
            tokens,
            self.cfg.hidden_size,
            &mut self.scratch.hidden,
        )?;

        let hidden = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let scale = 1.0 / (self.cfg.head_dim() as f32).sqrt();
        let n_q = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        let head_dim = self.cfg.head_dim();

        for layer_idx in 0..self.layers.len() {
            let args = LayerArgs {
                inv_freq: &self.inv_freq,
                position,
                chunk,
                n_q,
                n_kv,
                head_dim,
                hidden,
                eps,
                scale,
                window: self.cfg.sliding_window_for_layer(layer_idx),
                layer_idx,
            };
            // Reborrow per layer so the cache and scratch stay disjoint from `self.layers`.
            layer_forward(
                &self.layers[layer_idx],
                &mut self.scratch,
                cache,
                block_table,
                &args,
            )?;
        }

        if !want_logits {
            return Ok(ForwardOutput {
                last_hidden: None,
                logits: None,
            });
        }
        let start = (chunk - 1) * hidden;
        rms_norm(
            &mut self.scratch.normed[..hidden],
            &self.scratch.hidden[start..start + hidden],
            &self.final_norm,
            eps,
        );
        let last_hidden = self.scratch.normed[..hidden].to_vec();
        let mut logits = vec![0.0; self.cfg.vocab_size];
        if let Some(head) = &self.lm_head {
            head.apply(&last_hidden, &mut logits, 1);
        } else {
            linear(
                &self.embed,
                None,
                &last_hidden,
                &mut logits,
                1,
                self.cfg.vocab_size,
                hidden,
            );
        }
        Ok(ForwardOutput {
            last_hidden: Some(last_hidden),
            logits: Some(logits),
        })
    }

    fn ensure_scratch(&mut self, chunk: usize) {
        let hidden = self.cfg.hidden_size;
        let qo = self.cfg.qo_dim();
        let kv = self.cfg.kv_dim();
        let inter = self.cfg.intermediate_size;
        let head = self.cfg.head_dim();
        resize(&mut self.scratch.hidden, chunk * hidden);
        resize(&mut self.scratch.residual, chunk * hidden);
        resize(&mut self.scratch.normed, chunk * hidden);
        resize(&mut self.scratch.q, chunk * qo);
        resize(&mut self.scratch.k, chunk * kv);
        resize(&mut self.scratch.v, chunk * kv);
        resize(&mut self.scratch.attn, chunk * qo);
        resize(&mut self.scratch.proj, chunk * hidden);
        resize(&mut self.scratch.gate, chunk * inter);
        resize(&mut self.scratch.up, chunk * inter);
        resize(&mut self.scratch.cos, head);
        resize(&mut self.scratch.sin, head);
        resize(&mut self.scratch.rope_tmp, head);
    }
}

fn resize(buf: &mut Vec<f32>, len: usize) {
    if buf.len() != len {
        buf.resize(len, 0.0);
    }
}

fn embed_tokens(
    table: &[f32],
    tokens: &[u32],
    hidden: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    let vocab = table.len() / hidden;
    for (row, &id) in tokens.iter().enumerate() {
        if id as usize >= vocab {
            return Err(Error::Inference(format!(
                "token id {id} is outside vocab {vocab}"
            )));
        }
        let src = id as usize * hidden;
        let dst = row * hidden;
        out[dst..dst + hidden].copy_from_slice(&table[src..src + hidden]);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct LayerArgs<'a> {
    inv_freq: &'a [f32],
    position: usize,
    chunk: usize,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    hidden: usize,
    eps: f32,
    scale: f32,
    window: Option<usize>,
    layer_idx: usize,
}

fn layer_forward(
    layer: &LayerWeights,
    scratch: &mut Scratch,
    cache: &mut PagedKvCache,
    block_table: &[u32],
    args: &LayerArgs<'_>,
) -> Result<(), Error> {
    let LayerArgs {
        inv_freq,
        position,
        chunk,
        n_q,
        n_kv,
        head_dim,
        hidden,
        eps,
        scale,
        window,
        layer_idx,
    } = *args;
    scratch.residual.copy_from_slice(&scratch.hidden);
    rms_norm(
        &mut scratch.normed[..chunk * hidden],
        &scratch.hidden[..chunk * hidden],
        &layer.input_norm,
        eps,
    );

    layer.q.apply(
        &scratch.normed[..chunk * hidden],
        &mut scratch.q[..chunk * n_q * head_dim],
        chunk,
    );
    layer.k.apply(
        &scratch.normed[..chunk * hidden],
        &mut scratch.k[..chunk * n_kv * head_dim],
        chunk,
    );
    layer.v.apply(
        &scratch.normed[..chunk * hidden],
        &mut scratch.v[..chunk * n_kv * head_dim],
        chunk,
    );

    let block_size = cache.block_size();
    let qo = n_q * head_dim;
    let kv = n_kv * head_dim;
    for t in 0..chunk {
        let pos = position + t;
        let block_index = pos / block_size;
        let slot = pos % block_size;
        let block = *block_table.get(block_index).ok_or_else(|| {
            Error::Inference(format!(
                "block table length {} does not cover position {pos}",
                block_table.len()
            ))
        })?;
        rope_angles(pos, inv_freq, &mut scratch.cos, &mut scratch.sin);
        for head in 0..n_q {
            let start = t * qo + head * head_dim;
            apply_rope(
                &mut scratch.q[start..start + head_dim],
                &scratch.cos,
                &scratch.sin,
                &mut scratch.rope_tmp,
            );
        }
        for head in 0..n_kv {
            let start = t * kv + head * head_dim;
            apply_rope(
                &mut scratch.k[start..start + head_dim],
                &scratch.cos,
                &scratch.sin,
                &mut scratch.rope_tmp,
            );
        }
        cache
            .write_token(
                layer_idx,
                block,
                slot,
                &scratch.k[t * kv..(t + 1) * kv],
                &scratch.v[t * kv..(t + 1) * kv],
            )
            .map_err(|err| Error::Inference(err.to_string()))?;
        cache
            .attention(
                &scratch.q[t * qo..(t + 1) * qo],
                block_table,
                AttentionParams {
                    layer: layer_idx,
                    seq_len: pos + 1,
                    num_qo_heads: n_q,
                    scale,
                    sliding_window: window,
                },
                &mut scratch.attn[t * qo..(t + 1) * qo],
            )
            .map_err(|err| Error::Inference(err.to_string()))?;
    }

    layer.o.apply(
        &scratch.attn[..chunk * qo],
        &mut scratch.proj[..chunk * hidden],
        chunk,
    );
    add_vec(
        &mut scratch.hidden[..chunk * hidden],
        &scratch.residual[..chunk * hidden],
        &scratch.proj[..chunk * hidden],
    );

    scratch.residual.copy_from_slice(&scratch.hidden);
    rms_norm(
        &mut scratch.normed[..chunk * hidden],
        &scratch.hidden[..chunk * hidden],
        &layer.post_norm,
        eps,
    );
    let inter = layer.gate.out_features;
    layer.gate.apply(
        &scratch.normed[..chunk * hidden],
        &mut scratch.gate[..chunk * inter],
        chunk,
    );
    layer.up.apply(
        &scratch.normed[..chunk * hidden],
        &mut scratch.up[..chunk * inter],
        chunk,
    );
    swiglu(
        &mut scratch.gate[..chunk * inter],
        &scratch.up[..chunk * inter],
    );
    layer.down.apply(
        &scratch.gate[..chunk * inter],
        &mut scratch.proj[..chunk * hidden],
        chunk,
    );
    add_vec(
        &mut scratch.hidden[..chunk * hidden],
        &scratch.residual[..chunk * hidden],
        &scratch.proj[..chunk * hidden],
    );
    Ok(())
}

fn assemble(cfg: ModelConfig, tensors: &mut TensorMap, eos_ids: Vec<u32>) -> Result<Model, Error> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let qo = cfg.qo_dim();
    let kv = cfg.kv_dim();
    let embed = tensors.take("model.embed_tokens.weight", cfg.vocab_size * hidden)?;
    let final_norm = tensors.take("model.norm.weight", hidden)?;
    let lm_head = if cfg.tie_word_embeddings {
        None
    } else {
        Some(Linear {
            weight: tensors.take("lm_head.weight", cfg.vocab_size * hidden)?,
            bias: None,
            out_features: cfg.vocab_size,
            in_features: hidden,
        })
    };
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        layers.push(LayerWeights {
            input_norm: tensors.take(&format!("{p}.input_layernorm.weight"), hidden)?,
            post_norm: tensors.take(&format!("{p}.post_attention_layernorm.weight"), hidden)?,
            q: take_linear(tensors, &format!("{p}.self_attn.q_proj"), qo, hidden)?,
            k: take_linear(tensors, &format!("{p}.self_attn.k_proj"), kv, hidden)?,
            v: take_linear(tensors, &format!("{p}.self_attn.v_proj"), kv, hidden)?,
            o: take_linear(tensors, &format!("{p}.self_attn.o_proj"), hidden, qo)?,
            gate: take_linear(tensors, &format!("{p}.mlp.gate_proj"), inter, hidden)?,
            up: take_linear(tensors, &format!("{p}.mlp.up_proj"), inter, hidden)?,
            down: take_linear(tensors, &format!("{p}.mlp.down_proj"), hidden, inter)?,
        });
    }
    Ok(Model {
        inv_freq: inv_freq(cfg.head_dim(), cfg.rope_theta as f32),
        eos_ids,
        cfg,
        embed,
        layers,
        final_norm,
        lm_head,
        scratch: Scratch::new(),
    })
}

fn take_linear(
    tensors: &mut TensorMap,
    prefix: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Linear, Error> {
    Ok(Linear {
        weight: tensors.take(&format!("{prefix}.weight"), out_features * in_features)?,
        bias: tensors.take_optional(&format!("{prefix}.bias"), out_features)?,
        out_features,
        in_features,
    })
}

fn read_generation_eos(path: &Path) -> Vec<u32> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    match value.get("eos_token_id") {
        Some(serde_json::Value::Number(number)) => number
            .as_u64()
            .map(|id| vec![id as u32])
            .unwrap_or_default(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_u64().map(|id| id as u32))
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    fn tiny() -> ModelConfig {
        ModelConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: 32,
            ..ModelConfig::dummy()
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        let mut max = 0.0_f32;
        for (x, y) in a.iter().zip(b) {
            max = max.max((x - y).abs());
        }
        assert!(max <= tol, "max abs diff {max} > {tol}");
    }

    #[test]
    fn forward_is_finite_and_deterministic() {
        let cfg = tiny();
        let mut a = Model::from_random(&cfg, 7).unwrap();
        let mut b = Model::from_random(&cfg, 7).unwrap();
        let tokens = [1_u32, 4, 9, 2, 8, 3];
        let left = a.forward_prompt(&tokens, 4).unwrap().logits.unwrap();
        let right = b.forward_prompt(&tokens, 4).unwrap().logits.unwrap();
        assert_eq!(left.len(), cfg.vocab_size);
        assert!(left.iter().all(|v| v.is_finite()));
        assert_eq!(left, right);
    }

    #[test]
    fn block_size_and_table_do_not_change_logits() {
        let cfg = tiny();
        let tokens = [1_u32, 2, 3, 4, 5, 6];
        let mut model = Model::from_random(&cfg, 3).unwrap();
        let identity = model.forward_prompt(&tokens, 2).unwrap().logits.unwrap();

        let mut cache = PagedKvCache::with_dims(
            3,
            2,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        )
        .unwrap();
        let scrambled = [2_u32, 0, 1];
        let other = model
            .forward(&tokens, 0, &mut cache, &scrambled, true)
            .unwrap()
            .logits
            .unwrap();
        assert_close(&identity, &other, 1e-5);
    }

    #[test]
    fn chunked_prefill_matches_full_prefill() {
        let cfg = tiny();
        let tokens = [2_u32, 5, 1, 7, 4, 6, 3];
        let mut full = Model::from_random(&cfg, 11).unwrap();
        let expected = full.forward_prompt(&tokens, 4).unwrap().logits.unwrap();

        let mut model = Model::from_random(&cfg, 11).unwrap();
        let n_blocks = tokens.len().div_ceil(4);
        let mut cache = PagedKvCache::with_dims(
            n_blocks,
            4,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        )
        .unwrap();
        let table: Vec<u32> = (0..n_blocks as u32).collect();
        model
            .forward(&tokens[..3], 0, &mut cache, &table, false)
            .unwrap();
        let got = model
            .forward(&tokens[3..], 3, &mut cache, &table, true)
            .unwrap()
            .logits
            .unwrap();
        assert_close(&expected, &got, 1e-5);
    }

    #[test]
    fn decode_matches_prefill_of_the_same_tokens() {
        let cfg = tiny();
        let tokens = [4_u32, 1, 6, 2, 9];
        let mut full = Model::from_random(&cfg, 5).unwrap();
        let expected = full.forward_prompt(&tokens, 2).unwrap().logits.unwrap();

        let mut model = Model::from_random(&cfg, 5).unwrap();
        let n_blocks = tokens.len().div_ceil(2);
        let mut cache = PagedKvCache::with_dims(
            n_blocks,
            2,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        )
        .unwrap();
        let table: Vec<u32> = (0..n_blocks as u32).collect();
        model
            .forward(&tokens[..tokens.len() - 1], 0, &mut cache, &table, false)
            .unwrap();
        let got = model
            .forward(
                &tokens[tokens.len() - 1..],
                tokens.len() - 1,
                &mut cache,
                &table,
                true,
            )
            .unwrap()
            .logits
            .unwrap();
        assert_close(&expected, &got, 1e-5);
    }

    #[test]
    fn safetensors_roundtrip_preserves_a_known_weight() {
        let cfg = tiny();
        let dir = std::env::temp_dir().join(format!("flint-tiny-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();

        let hidden = cfg.hidden_size;
        let mut embed = vec![0.0_f32; cfg.vocab_size * hidden];
        embed[0] = 1.5;
        let mut tensors: Vec<(&str, Vec<u8>, Vec<usize>)> = Vec::new();
        let push = |tensors: &mut Vec<(&str, Vec<u8>, Vec<usize>)>,
                    name: &str,
                    values: &[f32],
                    shape: &[usize]| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            // Leak the name so the serialize call can borrow it. The test is short.
            let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
            tensors.push((leaked, bytes, shape.to_vec()));
        };
        push(
            &mut tensors,
            "model.embed_tokens.weight",
            &embed,
            &[cfg.vocab_size, hidden],
        );
        push(
            &mut tensors,
            "model.norm.weight",
            &vec![1.0; hidden],
            &[hidden],
        );
        let qo = cfg.qo_dim();
        let kv = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            push(
                &mut tensors,
                &format!("{prefix}.input_layernorm.weight"),
                &vec![1.0; hidden],
                &[hidden],
            );
            push(
                &mut tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &vec![1.0; hidden],
                &[hidden],
            );
            push(
                &mut tensors,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &vec![0.01; qo * hidden],
                &[qo, hidden],
            );
            push(
                &mut tensors,
                &format!("{prefix}.self_attn.q_proj.bias"),
                &vec![0.0; qo],
                &[qo],
            );
            push(
                &mut tensors,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &vec![0.01; kv * hidden],
                &[kv, hidden],
            );
            push(
                &mut tensors,
                &format!("{prefix}.self_attn.k_proj.bias"),
                &vec![0.0; kv],
                &[kv],
            );
            push(
                &mut tensors,
                &format!("{prefix}.self_attn.v_proj.weight"),
                &vec![0.01; kv * hidden],
                &[kv, hidden],
            );
            push(
                &mut tensors,
                &format!("{prefix}.self_attn.v_proj.bias"),
                &vec![0.0; kv],
                &[kv],
            );
            push(
                &mut tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &vec![0.01; hidden * qo],
                &[hidden, qo],
            );
            push(
                &mut tensors,
                &format!("{prefix}.mlp.gate_proj.weight"),
                &vec![0.01; inter * hidden],
                &[inter, hidden],
            );
            push(
                &mut tensors,
                &format!("{prefix}.mlp.up_proj.weight"),
                &vec![0.01; inter * hidden],
                &[inter, hidden],
            );
            push(
                &mut tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                &vec![0.01; hidden * inter],
                &[hidden, inter],
            );
        }
        let views: Vec<(&str, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, bytes, shape)| {
                (
                    *name,
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        let blob = serialize(views, &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), blob).unwrap();

        let loaded = Model::load(&dir).unwrap();
        assert!((loaded.embed[0] - 1.5).abs() < 1e-6);
        assert!(loaded.eos_token_ids().is_empty());
        let mut model = loaded;
        let logits = model.forward_prompt(&[0, 1, 2], 2).unwrap().logits.unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "requires FLINT_MODEL_DIR and FLINT_HF_LOGITS"]
    fn compare_against_hf_logits_file() {
        let dir = std::env::var("FLINT_MODEL_DIR").expect("FLINT_MODEL_DIR");
        let file = std::env::var("FLINT_HF_LOGITS").expect("FLINT_HF_LOGITS");
        let text = std::fs::read_to_string(file).unwrap();
        let cases: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        let mut model = Model::load(Path::new(&dir)).unwrap();
        for case in cases {
            let ids: Vec<u32> = case["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let hf: Vec<f32> = case["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let ours = model.forward_prompt(&ids, 16).unwrap().logits.unwrap();
            let mut max = 0.0_f32;
            for (a, b) in ours.iter().zip(&hf) {
                max = max.max((a - b).abs());
            }
            let our_top = ours
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            let hf_top = hf
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert_eq!(our_top, hf_top, "top token mismatch, max abs {max}");
            assert!(
                max < 1e-3,
                "max abs logit diff {max} >= 1e-3 (bf16 upcast plus f32 GEMM)"
            );
        }
    }
}
