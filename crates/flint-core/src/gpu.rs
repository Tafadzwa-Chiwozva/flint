//! CUDA forward pass.
//!
//! Weights are uploaded once. Every linear layer is a cuBLAS GEMM in the
//! Hugging Face layout (`Y = X @ W.T`). RMSNorm, RoPE, SwiGLU, the residual
//! add, and attention are the kernels in `flint-kernels`. KV lives on the
//! device in the same paged layout as [`flint_kv::PagedKvCache`]. A scheduler
//! step is one launch sequence over every sequence in the step: GEMMs see the
//! concatenated token rows, and attention is a batched decode kernel for
//! single-token rows plus a causal prefill kernel for longer chunks.

use flint_kernels::cuda::{AttnShape, Device, F32Buf, I32Buf, MAX_HEAD_DIM};
use flint_kernels::CacheGeometry;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::model::{ForwardChunk, ForwardOutput};

/// Host weights moved out of [`crate::Model`] before the upload.
pub(crate) struct OwnedLinear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub out_features: usize,
    pub in_features: usize,
}

/// One transformer block, still on the host.
pub(crate) struct OwnedLayer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q: OwnedLinear,
    pub k: OwnedLinear,
    pub v: OwnedLinear,
    pub o: OwnedLinear,
    pub gate: OwnedLinear,
    pub up: OwnedLinear,
    pub down: OwnedLinear,
}

/// Everything [`GpuModel::from_owned`] uploads.
pub(crate) struct OwnedWeights {
    pub cfg: ModelConfig,
    pub embed: Vec<f32>,
    pub layers: Vec<OwnedLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<OwnedLinear>,
    pub inv_freq: Vec<f32>,
    pub eos_ids: Vec<u32>,
}

struct DevLinear {
    weight: F32Buf,
    bias: Option<F32Buf>,
    out_features: usize,
    in_features: usize,
}

struct DevLayer {
    input_norm: F32Buf,
    post_norm: F32Buf,
    q: DevLinear,
    k: DevLinear,
    v: DevLinear,
    o: DevLinear,
    gate: DevLinear,
    up: DevLinear,
    down: DevLinear,
}

struct Scratch {
    hidden: F32Buf,
    residual: F32Buf,
    normed: F32Buf,
    q: F32Buf,
    k: F32Buf,
    v: F32Buf,
    attn: F32Buf,
    proj: F32Buf,
    gate: F32Buf,
    up: F32Buf,
    gathered: F32Buf,
    logits: F32Buf,
}

impl Scratch {
    fn empty() -> Self {
        Self {
            hidden: F32Buf::empty(),
            residual: F32Buf::empty(),
            normed: F32Buf::empty(),
            q: F32Buf::empty(),
            k: F32Buf::empty(),
            v: F32Buf::empty(),
            attn: F32Buf::empty(),
            proj: F32Buf::empty(),
            gate: F32Buf::empty(),
            up: F32Buf::empty(),
            gathered: F32Buf::empty(),
            logits: F32Buf::empty(),
        }
    }
}

struct SeqPlan {
    row0: usize,
    len: usize,
    position: usize,
    table_off: i32,
    want_logits: bool,
}

struct BatchPlan {
    rows: usize,
    seqs: Vec<SeqPlan>,
    tokens: I32Buf,
    positions: I32Buf,
    row_table_off: I32Buf,
    tables: I32Buf,
    decode_rows: I32Buf,
    decode_offs: I32Buf,
    decode_lens: I32Buf,
    decode_count: usize,
}

/// Qwen2 / Llama forward pass on one GPU.
///
/// Construct with [`crate::Model::into_gpu`]. The host KV cache the engine
/// still allocates is not read: this model keeps its own device cache and
/// addresses it with the same block tables.
pub struct GpuModel {
    cfg: ModelConfig,
    dev: Device,
    embed: F32Buf,
    layers: Vec<DevLayer>,
    final_norm: F32Buf,
    lm_head: Option<DevLinear>,
    inv_freq: F32Buf,
    key_cache: F32Buf,
    value_cache: F32Buf,
    scratch: Scratch,
    eos_ids: Vec<u32>,
    block_size: usize,
    num_blocks: usize,
    hidden: usize,
    inter: usize,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    eps: f32,
    scale: f32,
}

impl GpuModel {
    pub(crate) fn from_owned(
        owned: OwnedWeights,
        num_blocks: usize,
        block_size: usize,
    ) -> Result<Self, Error> {
        if num_blocks == 0 || block_size == 0 {
            return Err(Error::Inference(
                "GPU KV cache needs a non-zero block pool".into(),
            ));
        }
        let head_dim = owned.cfg.head_dim();
        if head_dim > MAX_HEAD_DIM {
            return Err(Error::Inference(format!(
                "head_dim {head_dim} exceeds the CUDA attention kernel limit of {MAX_HEAD_DIM}"
            )));
        }
        let dev = Device::new().map_err(gpu_err)?;
        let geometry = CacheGeometry {
            num_blocks,
            num_layers: owned.cfg.num_hidden_layers,
            num_kv_heads: owned.cfg.num_key_value_heads,
            block_size,
            head_dim,
        };
        let elems = geometry
            .elements()
            .ok_or_else(|| Error::Inference("GPU KV cache size overflowed".into()))?;
        let key_cache = dev.alloc_f32(elems).map_err(gpu_err)?;
        let value_cache = dev.alloc_f32(elems).map_err(gpu_err)?;
        let embed = dev.upload_f32(&owned.embed).map_err(gpu_err)?;
        let final_norm = dev.upload_f32(&owned.final_norm).map_err(gpu_err)?;
        let inv_freq = dev.upload_f32(&owned.inv_freq).map_err(gpu_err)?;
        let lm_head = owned
            .lm_head
            .map(|lin| upload_linear(&dev, lin))
            .transpose()?;
        let mut layers = Vec::with_capacity(owned.layers.len());
        for layer in owned.layers {
            layers.push(DevLayer {
                input_norm: dev.upload_f32(&layer.input_norm).map_err(gpu_err)?,
                post_norm: dev.upload_f32(&layer.post_norm).map_err(gpu_err)?,
                q: upload_linear(&dev, layer.q)?,
                k: upload_linear(&dev, layer.k)?,
                v: upload_linear(&dev, layer.v)?,
                o: upload_linear(&dev, layer.o)?,
                gate: upload_linear(&dev, layer.gate)?,
                up: upload_linear(&dev, layer.up)?,
                down: upload_linear(&dev, layer.down)?,
            });
        }
        tracing::info!(
            layers = owned.cfg.num_hidden_layers,
            num_blocks,
            block_size,
            "GPU weights uploaded"
        );
        Ok(Self {
            hidden: owned.cfg.hidden_size,
            inter: owned.cfg.intermediate_size,
            n_q: owned.cfg.num_attention_heads,
            n_kv: owned.cfg.num_key_value_heads,
            head_dim,
            eps: owned.cfg.rms_norm_eps as f32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            block_size,
            num_blocks,
            eos_ids: owned.eos_ids,
            cfg: owned.cfg,
            dev,
            embed,
            layers,
            final_norm,
            lm_head,
            inv_freq,
            key_cache,
            value_cache,
            scratch: Scratch::empty(),
        })
    }

    /// Borrowed architecture.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Eos ids carried over from the CPU model.
    #[must_use]
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Vocabulary width.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    /// Transformer layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }

    /// KV heads stored per token.
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.n_kv
    }

    /// Channels per head.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Run `tokens` as a fresh prompt on the device cache and return logits.
    ///
    /// Blocks are the identity table starting at physical block 0. A later
    /// call overwrites those slots from position 0.
    pub fn forward_prompt(&mut self, tokens: &[u32]) -> Result<ForwardOutput, Error> {
        if tokens.is_empty() {
            return Err(Error::Inference("empty prompt".into()));
        }
        let n_blocks = tokens.len().div_ceil(self.block_size);
        if n_blocks > self.num_blocks {
            return Err(Error::Inference(format!(
                "prompt needs {n_blocks} KV blocks but the GPU cache has {}",
                self.num_blocks
            )));
        }
        let table: Vec<u32> = (0..n_blocks as u32).collect();
        let req = [ForwardChunk {
            tokens,
            position: 0,
            block_table: &table,
            want_logits: true,
        }];
        let mut rows = self.forward_batch(&req)?;
        let logits = rows
            .pop()
            .flatten()
            .ok_or_else(|| Error::Inference("GPU forward did not return logits".into()))?;
        Ok(ForwardOutput {
            last_hidden: None,
            logits: Some(logits),
        })
    }

    /// One batched forward for every chunk in `reqs`.
    ///
    /// GEMMs run over the concatenated rows. Single-token chunks share one
    /// paged-decode launch. Longer chunks each get a causal prefill launch.
    /// The returned vector lines up with `reqs`: `None` when `want_logits` is
    /// false.
    pub fn forward_batch(
        &mut self,
        reqs: &[ForwardChunk<'_>],
    ) -> Result<Vec<Option<Vec<f32>>>, Error> {
        if reqs.is_empty() {
            return Ok(Vec::new());
        }
        let plan = self.build_plan(reqs)?;
        self.ensure_scratch(plan.rows)?;
        self.embed_plan(&plan)?;
        for layer_idx in 0..self.layers.len() {
            self.run_layer(layer_idx, &plan)?;
        }
        self.project_logits(&plan)
    }

    fn build_plan(&self, reqs: &[ForwardChunk<'_>]) -> Result<BatchPlan, Error> {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut row_table_off = Vec::new();
        let mut tables = Vec::new();
        let mut seqs = Vec::with_capacity(reqs.len());
        let mut decode_rows = Vec::new();
        let mut decode_offs = Vec::new();
        let mut decode_lens = Vec::new();
        let mut row0 = 0usize;
        for req in reqs {
            if req.tokens.is_empty() {
                return Err(Error::Inference("forward chunk is empty".into()));
            }
            let len = req.tokens.len();
            let end = req
                .position
                .checked_add(len)
                .ok_or_else(|| Error::Inference("token position overflowed".into()))?;
            let need = end.div_ceil(self.block_size);
            if req.block_table.len() < need {
                return Err(Error::Inference(format!(
                    "block table length {} does not cover position {}",
                    req.block_table.len(),
                    end - 1
                )));
            }
            for &id in req.block_table {
                if id as usize >= self.num_blocks {
                    return Err(Error::Inference(format!(
                        "block id {id} is outside the GPU cache of {} blocks",
                        self.num_blocks
                    )));
                }
            }
            for &id in req.tokens {
                if id as usize >= self.cfg.vocab_size {
                    return Err(Error::Inference(format!(
                        "token id {id} is outside vocab {}",
                        self.cfg.vocab_size
                    )));
                }
            }
            let table_off = as_i32(tables.len(), "block table offset")?;
            for &id in req.block_table {
                tables.push(as_i32(id as usize, "block id")?);
            }
            for (offset, &id) in req.tokens.iter().enumerate() {
                tokens.push(as_i32(id as usize, "token id")?);
                positions.push(as_i32(req.position + offset, "position")?);
                row_table_off.push(table_off);
            }
            if len == 1 {
                decode_rows.push(as_i32(row0, "decode row")?);
                decode_offs.push(table_off);
                decode_lens.push(as_i32(req.position + 1, "decode length")?);
            }
            seqs.push(SeqPlan {
                row0,
                len,
                position: req.position,
                table_off,
                want_logits: req.want_logits,
            });
            row0 += len;
        }
        let dev = &self.dev;
        Ok(BatchPlan {
            rows: row0,
            decode_count: decode_rows.len(),
            seqs,
            tokens: dev.upload_i32(&tokens).map_err(gpu_err)?,
            positions: dev.upload_i32(&positions).map_err(gpu_err)?,
            row_table_off: dev.upload_i32(&row_table_off).map_err(gpu_err)?,
            tables: dev.upload_i32(&tables).map_err(gpu_err)?,
            decode_rows: dev.upload_i32(&decode_rows).map_err(gpu_err)?,
            decode_offs: dev.upload_i32(&decode_offs).map_err(gpu_err)?,
            decode_lens: dev.upload_i32(&decode_lens).map_err(gpu_err)?,
        })
    }

    fn ensure_scratch(&mut self, rows: usize) -> Result<(), Error> {
        let qo = self.n_q * self.head_dim;
        let kv = self.n_kv * self.head_dim;
        let dev = &self.dev;
        dev.ensure_f32(&mut self.scratch.hidden, rows * self.hidden)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.residual, rows * self.hidden)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.normed, rows * self.hidden)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.q, rows * qo)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.k, rows * kv)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.v, rows * kv)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.attn, rows * qo)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.proj, rows * self.hidden)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.gate, rows * self.inter)
            .map_err(gpu_err)?;
        dev.ensure_f32(&mut self.scratch.up, rows * self.inter)
            .map_err(gpu_err)?;
        Ok(())
    }

    fn embed_plan(&mut self, plan: &BatchPlan) -> Result<(), Error> {
        self.dev
            .embed(
                &mut self.scratch.hidden,
                &self.embed,
                &plan.tokens,
                plan.rows,
                self.hidden,
            )
            .map_err(gpu_err)
    }

    fn run_layer(&mut self, layer_idx: usize, plan: &BatchPlan) -> Result<(), Error> {
        let rows = plan.rows;
        let hidden = self.hidden;
        let n_q = self.n_q;
        let n_kv = self.n_kv;
        let head_dim = self.head_dim;
        let eps = self.eps;
        let scale = self.scale;
        let block_size = self.block_size;
        let num_layers = self.cfg.num_hidden_layers;
        let inter = self.inter;
        let window = self
            .cfg
            .sliding_window_for_layer(layer_idx)
            .map(|w| w as i32)
            .unwrap_or(-1);
        let shape = AttnShape {
            layer: layer_idx as i32,
            num_qo_heads: n_q as i32,
            num_kv_heads: n_kv as i32,
            head_dim: head_dim as i32,
            block_size: block_size as i32,
            num_layers: num_layers as i32,
            scale,
            sliding_window: window,
        };

        self.dev
            .copy_f32(
                &self.scratch.hidden,
                &mut self.scratch.residual,
                rows * hidden,
            )
            .map_err(gpu_err)?;
        self.dev
            .rms_norm(
                &mut self.scratch.normed,
                &self.scratch.hidden,
                &self.layers[layer_idx].input_norm,
                rows,
                hidden,
                eps,
            )
            .map_err(gpu_err)?;
        apply_linear(
            &self.dev,
            &self.layers[layer_idx].q,
            &self.scratch.normed,
            &mut self.scratch.q,
            rows,
        )?;
        apply_linear(
            &self.dev,
            &self.layers[layer_idx].k,
            &self.scratch.normed,
            &mut self.scratch.k,
            rows,
        )?;
        apply_linear(
            &self.dev,
            &self.layers[layer_idx].v,
            &self.scratch.normed,
            &mut self.scratch.v,
            rows,
        )?;
        self.dev
            .rope(
                &mut self.scratch.q,
                &plan.positions,
                &self.inv_freq,
                rows,
                n_q,
                head_dim,
            )
            .map_err(gpu_err)?;
        self.dev
            .rope(
                &mut self.scratch.k,
                &plan.positions,
                &self.inv_freq,
                rows,
                n_kv,
                head_dim,
            )
            .map_err(gpu_err)?;
        self.dev
            .write_kv(
                &mut self.key_cache,
                &mut self.value_cache,
                &self.scratch.k,
                &self.scratch.v,
                &plan.positions,
                &plan.row_table_off,
                &plan.tables,
                rows,
                layer_idx as i32,
                n_kv,
                head_dim,
                block_size,
                num_layers,
            )
            .map_err(gpu_err)?;
        if plan.decode_count > 0 {
            self.dev
                .paged_decode(
                    &mut self.scratch.attn,
                    &self.scratch.q,
                    &self.key_cache,
                    &self.value_cache,
                    &plan.tables,
                    &plan.decode_offs,
                    &plan.decode_lens,
                    &plan.decode_rows,
                    plan.decode_count,
                    shape,
                )
                .map_err(gpu_err)?;
        }
        for seq in &plan.seqs {
            if seq.len > 1 {
                self.dev
                    .paged_prefill(
                        &mut self.scratch.attn,
                        &self.scratch.q,
                        &self.key_cache,
                        &self.value_cache,
                        &plan.tables,
                        seq.table_off,
                        seq.row0,
                        seq.len,
                        seq.position,
                        shape,
                    )
                    .map_err(gpu_err)?;
            }
        }
        apply_linear(
            &self.dev,
            &self.layers[layer_idx].o,
            &self.scratch.attn,
            &mut self.scratch.proj,
            rows,
        )?;
        self.dev
            .add(
                &mut self.scratch.hidden,
                &self.scratch.residual,
                &self.scratch.proj,
                rows * hidden,
            )
            .map_err(gpu_err)?;

        self.dev
            .copy_f32(
                &self.scratch.hidden,
                &mut self.scratch.residual,
                rows * hidden,
            )
            .map_err(gpu_err)?;
        self.dev
            .rms_norm(
                &mut self.scratch.normed,
                &self.scratch.hidden,
                &self.layers[layer_idx].post_norm,
                rows,
                hidden,
                eps,
            )
            .map_err(gpu_err)?;
        apply_linear(
            &self.dev,
            &self.layers[layer_idx].gate,
            &self.scratch.normed,
            &mut self.scratch.gate,
            rows,
        )?;
        apply_linear(
            &self.dev,
            &self.layers[layer_idx].up,
            &self.scratch.normed,
            &mut self.scratch.up,
            rows,
        )?;
        self.dev
            .swiglu(&mut self.scratch.gate, &self.scratch.up, rows * inter)
            .map_err(gpu_err)?;
        apply_linear(
            &self.dev,
            &self.layers[layer_idx].down,
            &self.scratch.gate,
            &mut self.scratch.proj,
            rows,
        )?;
        self.dev
            .add(
                &mut self.scratch.hidden,
                &self.scratch.residual,
                &self.scratch.proj,
                rows * hidden,
            )
            .map_err(gpu_err)?;
        Ok(())
    }

    fn project_logits(&mut self, plan: &BatchPlan) -> Result<Vec<Option<Vec<f32>>>, Error> {
        let mut out = vec![None; plan.seqs.len()];
        let mut index = Vec::new();
        let mut which = Vec::new();
        for (seq_i, seq) in plan.seqs.iter().enumerate() {
            if seq.want_logits {
                index.push(as_i32(seq.row0 + seq.len - 1, "logit row")?);
                which.push(seq_i);
            }
        }
        let n = index.len();
        if n == 0 {
            return Ok(out);
        }
        let hidden = self.hidden;
        let vocab = self.cfg.vocab_size;
        self.dev
            .ensure_f32(&mut self.scratch.gathered, n * hidden)
            .map_err(gpu_err)?;
        self.dev
            .ensure_f32(&mut self.scratch.logits, n * vocab)
            .map_err(gpu_err)?;
        let rows = self.dev.upload_i32(&index).map_err(gpu_err)?;
        self.dev
            .gather_rows(
                &mut self.scratch.gathered,
                &self.scratch.hidden,
                &rows,
                n,
                hidden,
            )
            .map_err(gpu_err)?;
        self.dev
            .rms_norm(
                &mut self.scratch.normed,
                &self.scratch.gathered,
                &self.final_norm,
                n,
                hidden,
                self.eps,
            )
            .map_err(gpu_err)?;
        if let Some(head) = &self.lm_head {
            apply_linear(
                &self.dev,
                head,
                &self.scratch.normed,
                &mut self.scratch.logits,
                n,
            )?;
        } else {
            self.dev
                .gemm_nt(
                    &self.scratch.normed,
                    &self.embed,
                    &mut self.scratch.logits,
                    n,
                    vocab,
                    hidden,
                )
                .map_err(gpu_err)?;
        }
        let host = self
            .dev
            .download_f32(&self.scratch.logits, n * vocab)
            .map_err(gpu_err)?;
        for (slot, seq_i) in which.into_iter().enumerate() {
            let start = slot * vocab;
            out[seq_i] = Some(host[start..start + vocab].to_vec());
        }
        Ok(out)
    }
}

fn apply_linear(
    dev: &Device,
    lin: &DevLinear,
    x: &F32Buf,
    y: &mut F32Buf,
    rows: usize,
) -> Result<(), Error> {
    dev.gemm_nt(x, &lin.weight, y, rows, lin.out_features, lin.in_features)
        .map_err(gpu_err)?;
    if let Some(bias) = &lin.bias {
        dev.bias(y, bias, rows, lin.out_features).map_err(gpu_err)?;
    }
    Ok(())
}

fn upload_linear(dev: &Device, lin: OwnedLinear) -> Result<DevLinear, Error> {
    let bias = match lin.bias {
        Some(values) => Some(dev.upload_f32(&values).map_err(gpu_err)?),
        None => None,
    };
    Ok(DevLinear {
        weight: dev.upload_f32(&lin.weight).map_err(gpu_err)?,
        bias,
        out_features: lin.out_features,
        in_features: lin.in_features,
    })
}

fn as_i32(n: usize, what: &str) -> Result<i32, Error> {
    i32::try_from(n).map_err(|_| Error::Inference(format!("{what} ({n}) does not fit in i32")))
}

fn gpu_err(err: impl std::fmt::Display) -> Error {
    Error::Inference(format!("cuda: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;

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

    fn assert_close(got: &[f32], expect: &[f32], tol: f32) {
        assert_eq!(got.len(), expect.len());
        let max = got
            .iter()
            .zip(expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max <= tol, "max abs diff {max} > {tol}");
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_logits_match_cpu() {
        let cfg = tiny();
        let tokens = [1_u32, 4, 9, 2, 8, 3];
        let mut cpu = Model::from_random(&cfg, 7).unwrap();
        let expect = cpu.forward_prompt(&tokens, 4).unwrap().logits.unwrap();
        let mut gpu = Model::from_random(&cfg, 7).unwrap().into_gpu(8, 4).unwrap();
        let got = gpu.forward_prompt(&tokens).unwrap().logits.unwrap();
        assert_eq!(argmax(&got), argmax(&expect));
        assert_close(&got, &expect, 2e-3);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_block_table_does_not_change_logits() {
        let cfg = tiny();
        let tokens = [1_u32, 2, 3, 4, 5, 6];
        let mut identity = Model::from_random(&cfg, 3).unwrap().into_gpu(8, 2).unwrap();
        let expect = identity.forward_prompt(&tokens).unwrap().logits.unwrap();
        let mut scrambled = Model::from_random(&cfg, 3).unwrap().into_gpu(8, 2).unwrap();
        let table = [2_u32, 0, 1];
        let got = scrambled
            .forward_batch(&[ForwardChunk {
                tokens: &tokens,
                position: 0,
                block_table: &table,
                want_logits: true,
            }])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_close(&got, &expect, 2e-3);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_chunked_prefill_and_decode_match_cpu() {
        let cfg = tiny();
        let tokens = [2_u32, 5, 1, 7, 4, 6, 3];
        let mut cpu = Model::from_random(&cfg, 11).unwrap();
        let expect = cpu.forward_prompt(&tokens, 4).unwrap().logits.unwrap();

        let mut chunked = Model::from_random(&cfg, 11)
            .unwrap()
            .into_gpu(4, 4)
            .unwrap();
        let table = [0_u32, 1];
        chunked
            .forward_batch(&[ForwardChunk {
                tokens: &tokens[..3],
                position: 0,
                block_table: &table,
                want_logits: false,
            }])
            .unwrap();
        let got = chunked
            .forward_batch(&[ForwardChunk {
                tokens: &tokens[3..],
                position: 3,
                block_table: &table,
                want_logits: true,
            }])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_eq!(argmax(&got), argmax(&expect));
        assert_close(&got, &expect, 2e-3);

        let mut decoded = Model::from_random(&cfg, 11)
            .unwrap()
            .into_gpu(4, 4)
            .unwrap();
        decoded
            .forward_batch(&[ForwardChunk {
                tokens: &tokens[..tokens.len() - 1],
                position: 0,
                block_table: &table,
                want_logits: false,
            }])
            .unwrap();
        let last = &tokens[tokens.len() - 1..];
        let decoded_logits = decoded
            .forward_batch(&[ForwardChunk {
                tokens: last,
                position: tokens.len() - 1,
                block_table: &table,
                want_logits: true,
            }])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_close(&decoded_logits, &expect, 2e-3);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_batch_matches_two_solo_forwards() {
        let cfg = tiny();
        let a_tokens = [1_u32, 2, 3, 4];
        let b_tokens = [5_u32, 6, 1, 2, 3];
        let mut solo_a = Model::from_random(&cfg, 9).unwrap().into_gpu(8, 4).unwrap();
        let mut solo_b = Model::from_random(&cfg, 9).unwrap().into_gpu(8, 4).unwrap();
        let mut both = Model::from_random(&cfg, 9).unwrap().into_gpu(8, 4).unwrap();
        let expect_a = solo_a.forward_prompt(&a_tokens).unwrap().logits.unwrap();
        let expect_b = solo_b.forward_prompt(&b_tokens).unwrap().logits.unwrap();
        // Block size 4. The decode step grows A to 5 tokens, so A needs two blocks.
        let table_a = [0_u32, 3];
        let table_b = [1_u32, 2];
        let got = both
            .forward_batch(&[
                ForwardChunk {
                    tokens: &a_tokens,
                    position: 0,
                    block_table: &table_a,
                    want_logits: true,
                },
                ForwardChunk {
                    tokens: &b_tokens,
                    position: 0,
                    block_table: &table_b,
                    want_logits: true,
                },
            ])
            .unwrap();
        assert_close(got[0].as_ref().unwrap(), &expect_a, 2e-3);
        assert_close(got[1].as_ref().unwrap(), &expect_b, 2e-3);

        // Decode step for both sequences in one launch, against a full CPU prefill.
        let mut cpu_a = Model::from_random(&cfg, 9).unwrap();
        let mut cpu_b = Model::from_random(&cfg, 9).unwrap();
        let full_a = cpu_a
            .forward_prompt(&[1, 2, 3, 4, 8], 4)
            .unwrap()
            .logits
            .unwrap();
        let full_b = cpu_b
            .forward_prompt(&[5, 6, 1, 2, 3, 7], 4)
            .unwrap()
            .logits
            .unwrap();
        let mut step = Model::from_random(&cfg, 9).unwrap().into_gpu(8, 4).unwrap();
        step.forward_batch(&[
            ForwardChunk {
                tokens: &a_tokens,
                position: 0,
                block_table: &table_a,
                want_logits: false,
            },
            ForwardChunk {
                tokens: &b_tokens,
                position: 0,
                block_table: &table_b,
                want_logits: false,
            },
        ])
        .unwrap();
        let next_a = [8_u32];
        let next_b = [7_u32];
        let decoded = step
            .forward_batch(&[
                ForwardChunk {
                    tokens: &next_a,
                    position: a_tokens.len(),
                    block_table: &table_a,
                    want_logits: true,
                },
                ForwardChunk {
                    tokens: &next_b,
                    position: b_tokens.len(),
                    block_table: &table_b,
                    want_logits: true,
                },
            ])
            .unwrap();
        assert_eq!(argmax(decoded[0].as_ref().unwrap()), argmax(&full_a));
        assert_eq!(argmax(decoded[1].as_ref().unwrap()), argmax(&full_b));
        assert_close(decoded[0].as_ref().unwrap(), &full_a, 2e-3);
        assert_close(decoded[1].as_ref().unwrap(), &full_b, 2e-3);
    }
}
