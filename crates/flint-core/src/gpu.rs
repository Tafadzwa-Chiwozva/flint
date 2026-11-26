//! CUDA forward pass.
//!
//! Weights are uploaded once. Every linear layer is a cuBLAS GEMM in the
//! Hugging Face layout (`Y = X @ W.T`). RMSNorm, RoPE, SwiGLU, the residual
//! add, and attention are the kernels in `flint-kernels`. KV lives on the
//! device in the same paged layout as [`flint_kv::PagedKvCache`]. A scheduler
//! step is one launch sequence over every sequence in the step: GEMMs see the
//! concatenated token rows, and attention is a batched decode kernel for
//! single-token rows plus a causal prefill kernel for longer chunks.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use flint_kernels::cuda::{
    AttnShape, Bf16Buf, CapturedGraph, Device, F32Buf, GpuDType, I32Buf, I8Buf, MAX_HEAD_DIM,
};
use flint_kernels::CacheGeometry;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::model::{ForwardChunk, ForwardOutput};
use crate::quant::PackedWeight;

/// Rows at or below this use the fused dequant GEMV, including every CUDA-graph
/// decode width. Larger prefill rows dequantize into scratch and call cuBLAS.
const INT8_GEMV_MAX_ROWS: usize = 64;

/// Host weights moved out of [`crate::Model`] before the upload.
pub(crate) struct OwnedLinear {
    pub weight: PackedWeight,
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
    pub embed: PackedWeight,
    pub layers: Vec<OwnedLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<OwnedLinear>,
    pub inv_freq: Vec<f32>,
    pub eos_ids: Vec<u32>,
}

/// Device tensor. f32 is the parity path. bf16 is the fast path.
enum Act {
    F32(F32Buf),
    Bf16(Bf16Buf),
}

impl Act {
    fn empty(dtype: GpuDType) -> Self {
        match dtype {
            GpuDType::F32 => Self::F32(F32Buf::empty()),
            GpuDType::Bf16 => Self::Bf16(Bf16Buf::empty()),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::F32(inner) => inner.capacity(),
            Self::Bf16(inner) => inner.capacity(),
        }
    }
}

enum DevWeight {
    Dense(Act),
    Int8 { q: I8Buf, scales: F32Buf },
}

struct DevLinear {
    weight: DevWeight,
    bias: Option<Act>,
    out_features: usize,
    in_features: usize,
}

struct DevLayer {
    input_norm: Act,
    post_norm: Act,
    q: DevLinear,
    k: DevLinear,
    v: DevLinear,
    o: DevLinear,
    gate: DevLinear,
    up: DevLinear,
    down: DevLinear,
}

struct Scratch {
    hidden: Act,
    residual: Act,
    normed: Act,
    q: Act,
    k: Act,
    v: Act,
    attn: Act,
    proj: Act,
    gate: Act,
    up: Act,
    gathered: Act,
    /// Logits stay f32 in both modes so the sampler reads full precision.
    logits: F32Buf,
    /// Dequantized copy of one INT8 matrix for prefill GEMMs. Decode uses the
    /// fused GEMV and does not read this buffer.
    dequant: Act,
    /// Split-K workspace. Unused when `splits == 1`.
    partial_acc: F32Buf,
    partial_max: F32Buf,
    partial_sum: F32Buf,
}

impl Scratch {
    fn empty(dtype: GpuDType) -> Self {
        Self {
            hidden: Act::empty(dtype),
            residual: Act::empty(dtype),
            normed: Act::empty(dtype),
            q: Act::empty(dtype),
            k: Act::empty(dtype),
            v: Act::empty(dtype),
            attn: Act::empty(dtype),
            proj: Act::empty(dtype),
            gate: Act::empty(dtype),
            up: Act::empty(dtype),
            gathered: Act::empty(dtype),
            dequant: Act::empty(dtype),
            logits: F32Buf::empty(),
            partial_acc: F32Buf::empty(),
            partial_max: F32Buf::empty(),
            partial_sum: F32Buf::empty(),
        }
    }
}

/// Integer plan buffers reused across steps. Growing one drops CUDA graphs.
struct Meta {
    tokens: I32Buf,
    positions: I32Buf,
    row_table_off: I32Buf,
    tables: I32Buf,
    decode_rows: I32Buf,
    decode_offs: I32Buf,
    decode_lens: I32Buf,
    logit_rows: I32Buf,
    grew: bool,
}

impl Default for Meta {
    fn default() -> Self {
        Self {
            tokens: I32Buf::empty(),
            positions: I32Buf::empty(),
            row_table_off: I32Buf::empty(),
            tables: I32Buf::empty(),
            decode_rows: I32Buf::empty(),
            decode_offs: I32Buf::empty(),
            decode_lens: I32Buf::empty(),
            logit_rows: I32Buf::empty(),
            grew: false,
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

struct BatchPlan<'a> {
    rows: usize,
    seqs: Vec<SeqPlan>,
    tokens: &'a I32Buf,
    positions: &'a I32Buf,
    row_table_off: &'a I32Buf,
    tables: &'a I32Buf,
    decode_rows: &'a I32Buf,
    decode_offs: &'a I32Buf,
    decode_lens: &'a I32Buf,
    logit_rows: &'a I32Buf,
    decode_count: usize,
    logit_count: usize,
    max_decode_len: i32,
    grew: bool,
}

/// Qwen2 / Llama forward pass on one GPU.
///
/// Construct with [`crate::Model::into_gpu`]. The host KV cache the engine
/// still allocates is not read: this model keeps its own device cache and
/// addresses it with the same block tables.
pub struct GpuModel {
    cfg: ModelConfig,
    dev: Device,
    dtype: GpuDType,
    embed: DevWeight,
    layers: Vec<DevLayer>,
    final_norm: Act,
    lm_head: Option<DevLinear>,
    /// RoPE frequencies stay f32. The bf16 rope kernel reads them as float.
    inv_freq: F32Buf,
    key_cache: Act,
    value_cache: Act,
    scratch: Scratch,
    meta: Meta,
    /// Pure-decode graphs keyed by `(batch, split-K)`.
    graphs: HashMap<(usize, i32), CapturedGraph>,
    graphs_ok: bool,
    /// Largest token-row scratch allocated so far. Growth drops graphs.
    scratch_rows: usize,
    logit_slots: usize,
    partial_cap: usize,
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
        dtype: GpuDType,
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
        if dtype == GpuDType::Bf16 {
            dev.use_tensor_cores().map_err(gpu_err)?;
        }
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
        let key_cache = alloc_act(&dev, dtype, elems)?;
        let value_cache = alloc_act(&dev, dtype, elems)?;
        let (matmul_elems, matmul_bytes, max_matrix, quant_i8) = weight_stats(&owned);
        let embed = upload_weight(&dev, dtype, owned.embed)?;
        let final_norm = upload_act(&dev, dtype, &owned.final_norm)?;
        let inv_freq = dev.upload_f32(&owned.inv_freq).map_err(gpu_err)?;
        let lm_head = owned
            .lm_head
            .map(|lin| upload_linear(&dev, dtype, lin))
            .transpose()?;
        let mut layers = Vec::with_capacity(owned.layers.len());
        for layer in owned.layers {
            layers.push(DevLayer {
                input_norm: upload_act(&dev, dtype, &layer.input_norm)?,
                post_norm: upload_act(&dev, dtype, &layer.post_norm)?,
                q: upload_linear(&dev, dtype, layer.q)?,
                k: upload_linear(&dev, dtype, layer.k)?,
                v: upload_linear(&dev, dtype, layer.v)?,
                o: upload_linear(&dev, dtype, layer.o)?,
                gate: upload_linear(&dev, dtype, layer.gate)?,
                up: upload_linear(&dev, dtype, layer.up)?,
                down: upload_linear(&dev, dtype, layer.down)?,
            });
        }
        let mut scratch = Scratch::empty(dtype);
        if quant_i8 {
            // One copy of the largest INT8 matrix. Prefill then dequantizes that
            // matrix once and issues one cuBLAS GEMM. A 2048-wide tile turned
            // the 0.5B LM head into 75 sequential GEMMs. Decode never reads
            // this buffer, so its size does not change a captured graph.
            scratch.dequant = alloc_act(&dev, dtype, max_matrix.max(1))?;
        }
        tracing::info!(
            layers = owned.cfg.num_hidden_layers,
            num_blocks,
            block_size,
            ?dtype,
            quant = if quant_i8 { "int8" } else { "none" },
            matmul_bytes,
            bf16_bytes = matmul_elems * 2,
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
            dtype,
            embed,
            layers,
            final_norm,
            lm_head,
            inv_freq,
            key_cache,
            value_cache,
            scratch,
            meta: Meta::default(),
            graphs: HashMap::new(),
            graphs_ok: true,
            scratch_rows: 0,
            logit_slots: 0,
            partial_cap: 0,
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
        let mut meta = std::mem::take(&mut self.meta);
        let built = self.build_plan(reqs, &mut meta, false);
        let result = match built {
            Ok(plan) => self.execute(&plan),
            Err(err) => Err(err),
        };
        self.meta = meta;
        result
    }

    /// Logits for every token in one chunk, or only the last token.
    ///
    /// A single token uses the decode path, including a captured CUDA graph
    /// when the batch shape is one of the recorded widths. A longer chunk is
    /// the speculative verification forward: one prefill launch scores every
    /// position and does not replay a decode graph.
    pub fn forward_token_logits(
        &mut self,
        tokens: &[u32],
        position: usize,
        block_table: &[u32],
        all_positions: bool,
    ) -> Result<Vec<Vec<f32>>, Error> {
        if tokens.is_empty() {
            return Err(Error::Inference("forward chunk is empty".into()));
        }
        if !all_positions || tokens.len() == 1 {
            let req = [ForwardChunk {
                tokens,
                position,
                block_table,
                want_logits: true,
            }];
            let mut rows = self.forward_batch(&req)?;
            let logits = rows
                .pop()
                .flatten()
                .ok_or_else(|| Error::Inference("GPU forward did not return logits".into()))?;
            return Ok(vec![logits]);
        }
        let mut meta = std::mem::take(&mut self.meta);
        let req = [ForwardChunk {
            tokens,
            position,
            block_table,
            want_logits: true,
        }];
        let built = self.build_plan(&req, &mut meta, true);
        let result = match built {
            Ok(plan) => {
                let n = plan.logit_count;
                if n != tokens.len() {
                    Err(Error::Inference(
                        "GPU verification forward did not score every token".into(),
                    ))
                } else {
                    self.launch(&plan).and_then(|_| self.read_row_logits(n))
                }
            }
            Err(err) => Err(err),
        };
        self.meta = meta;
        result
    }

    fn build_plan<'a>(
        &self,
        reqs: &[ForwardChunk<'_>],
        meta: &'a mut Meta,
        per_token_logits: bool,
    ) -> Result<BatchPlan<'a>, Error> {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut row_table_off = Vec::new();
        let mut tables = Vec::new();
        let mut seqs = Vec::with_capacity(reqs.len());
        let mut decode_rows = Vec::new();
        let mut decode_offs = Vec::new();
        let mut decode_lens = Vec::new();
        let mut logit_rows = Vec::new();
        let mut max_decode_len = 0i32;
        let mut row0 = 0usize;
        meta.grew = false;
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
                let decode_len = as_i32(req.position + 1, "decode length")?;
                decode_lens.push(decode_len);
                max_decode_len = max_decode_len.max(decode_len);
            }
            if per_token_logits {
                for offset in 0..len {
                    logit_rows.push(as_i32(row0 + offset, "logit row")?);
                }
            } else if req.want_logits {
                logit_rows.push(as_i32(row0 + len - 1, "logit row")?);
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
        meta.grew |= dev.fill_i32(&mut meta.tokens, &tokens).map_err(gpu_err)?;
        meta.grew |= dev
            .fill_i32(&mut meta.positions, &positions)
            .map_err(gpu_err)?;
        meta.grew |= dev
            .fill_i32(&mut meta.row_table_off, &row_table_off)
            .map_err(gpu_err)?;
        meta.grew |= dev.fill_i32(&mut meta.tables, &tables).map_err(gpu_err)?;
        meta.grew |= dev
            .fill_i32(&mut meta.decode_rows, &decode_rows)
            .map_err(gpu_err)?;
        meta.grew |= dev
            .fill_i32(&mut meta.decode_offs, &decode_offs)
            .map_err(gpu_err)?;
        meta.grew |= dev
            .fill_i32(&mut meta.decode_lens, &decode_lens)
            .map_err(gpu_err)?;
        meta.grew |= dev
            .fill_i32(&mut meta.logit_rows, &logit_rows)
            .map_err(gpu_err)?;
        Ok(BatchPlan {
            rows: row0,
            decode_count: decode_rows.len(),
            logit_count: logit_rows.len(),
            max_decode_len,
            grew: meta.grew,
            seqs,
            tokens: &meta.tokens,
            positions: &meta.positions,
            row_table_off: &meta.row_table_off,
            tables: &meta.tables,
            decode_rows: &meta.decode_rows,
            decode_offs: &meta.decode_offs,
            decode_lens: &meta.decode_lens,
            logit_rows: &meta.logit_rows,
        })
    }

    fn ensure_scratch(&mut self, rows: usize) -> Result<(), Error> {
        let qo = self.n_q * self.head_dim;
        let kv = self.n_kv * self.head_dim;
        let dev = &self.dev;
        grow(dev, &mut self.scratch.hidden, rows * self.hidden)?;
        grow(dev, &mut self.scratch.residual, rows * self.hidden)?;
        grow(dev, &mut self.scratch.normed, rows * self.hidden)?;
        grow(dev, &mut self.scratch.q, rows * qo)?;
        grow(dev, &mut self.scratch.k, rows * kv)?;
        grow(dev, &mut self.scratch.v, rows * kv)?;
        grow(dev, &mut self.scratch.attn, rows * qo)?;
        grow(dev, &mut self.scratch.proj, rows * self.hidden)?;
        grow(dev, &mut self.scratch.gate, rows * self.inter)?;
        grow(dev, &mut self.scratch.up, rows * self.inter)?;
        Ok(())
    }

    fn embed_plan(&mut self, plan: &BatchPlan<'_>) -> Result<(), Error> {
        match &self.embed {
            DevWeight::Dense(table) => embed_act(
                &self.dev,
                self.dtype,
                &mut self.scratch.hidden,
                table,
                plan.tokens,
                plan.rows,
                self.hidden,
            ),
            DevWeight::Int8 { q, scales } => embed_i8(
                &self.dev,
                self.dtype,
                &mut self.scratch.hidden,
                q,
                scales,
                plan.tokens,
                plan.rows,
                self.hidden,
            ),
        }
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

        let dtype = self.dtype;
        if plan.decode_count > 0 {
            let splits = decode_splits(dtype, plan.max_decode_len);
            self.ensure_partials(plan.decode_count, splits)?;
        }
        let dev = &self.dev;
        fused_residual_rms(
            dev,
            dtype,
            &mut self.scratch.residual,
            &mut self.scratch.normed,
            &self.scratch.hidden,
            &self.layers[layer_idx].input_norm,
            rows,
            hidden,
            eps,
        )?;
        apply_linear(
            dev,
            dtype,
            &self.layers[layer_idx].q,
            &self.scratch.normed,
            &mut self.scratch.q,
            &mut self.scratch.dequant,
            rows,
        )?;
        apply_linear(
            dev,
            dtype,
            &self.layers[layer_idx].k,
            &self.scratch.normed,
            &mut self.scratch.k,
            &mut self.scratch.dequant,
            rows,
        )?;
        apply_linear(
            dev,
            dtype,
            &self.layers[layer_idx].v,
            &self.scratch.normed,
            &mut self.scratch.v,
            &mut self.scratch.dequant,
            rows,
        )?;
        rope_act(
            dev,
            dtype,
            &mut self.scratch.q,
            plan.positions,
            &self.inv_freq,
            rows,
            n_q,
            head_dim,
        )?;
        rope_act(
            dev,
            dtype,
            &mut self.scratch.k,
            plan.positions,
            &self.inv_freq,
            rows,
            n_kv,
            head_dim,
        )?;
        write_kv_act(
            dev,
            dtype,
            &mut self.key_cache,
            &mut self.value_cache,
            &self.scratch.k,
            &self.scratch.v,
            plan.positions,
            plan.row_table_off,
            plan.tables,
            rows,
            layer_idx as i32,
            n_kv,
            head_dim,
            block_size,
            num_layers,
        )?;
        if plan.decode_count > 0 {
            let splits = decode_splits(dtype, plan.max_decode_len);
            decode_act(
                &self.dev,
                dtype,
                &mut self.scratch.attn,
                &mut self.scratch.partial_acc,
                &mut self.scratch.partial_max,
                &mut self.scratch.partial_sum,
                &self.scratch.q,
                &self.key_cache,
                &self.value_cache,
                plan,
                splits,
                shape,
            )?;
        }
        for seq in &plan.seqs {
            if seq.len > 1 {
                prefill_act(
                    &self.dev,
                    dtype,
                    &mut self.scratch.attn,
                    &self.scratch.q,
                    &self.key_cache,
                    &self.value_cache,
                    plan.tables,
                    seq,
                    shape,
                )?;
            }
        }
        apply_linear(
            dev,
            dtype,
            &self.layers[layer_idx].o,
            &self.scratch.attn,
            &mut self.scratch.proj,
            &mut self.scratch.dequant,
            rows,
        )?;
        // Post-attention residual and the MLP norm are one launch.
        // `hidden` becomes the sum. The MLP skip is that sum, added in place below.
        fused_add_rms(
            dev,
            dtype,
            &mut self.scratch.hidden,
            &mut self.scratch.normed,
            &self.scratch.residual,
            &self.scratch.proj,
            &self.layers[layer_idx].post_norm,
            rows,
            hidden,
            eps,
        )?;
        apply_linear(
            dev,
            dtype,
            &self.layers[layer_idx].gate,
            &self.scratch.normed,
            &mut self.scratch.gate,
            &mut self.scratch.dequant,
            rows,
        )?;
        apply_linear(
            dev,
            dtype,
            &self.layers[layer_idx].up,
            &self.scratch.normed,
            &mut self.scratch.up,
            &mut self.scratch.dequant,
            rows,
        )?;
        swiglu_act(
            dev,
            dtype,
            &mut self.scratch.gate,
            &self.scratch.up,
            rows * inter,
        )?;
        apply_linear(
            dev,
            dtype,
            &self.layers[layer_idx].down,
            &self.scratch.gate,
            &mut self.scratch.proj,
            &mut self.scratch.dequant,
            rows,
        )?;
        // `residual` still holds the pre-attention value. Refresh it, then add.
        copy_act(
            dev,
            dtype,
            &self.scratch.hidden,
            &mut self.scratch.residual,
            rows * hidden,
        )?;
        add_act(
            dev,
            dtype,
            &mut self.scratch.hidden,
            &self.scratch.residual,
            &self.scratch.proj,
            rows * hidden,
        )?;
        Ok(())
    }

    fn execute(&mut self, plan: &BatchPlan<'_>) -> Result<Vec<Option<Vec<f32>>>, Error> {
        self.launch(plan)?;
        self.read_logits(plan)
    }

    fn launch(&mut self, plan: &BatchPlan<'_>) -> Result<(), Error> {
        let logit_n = plan.logit_count;
        let scratch_grew = plan.rows > self.scratch_rows;
        let logit_grew = logit_n > self.logit_slots;
        if scratch_grew || logit_grew || plan.grew {
            self.graphs.clear();
        }
        self.ensure_scratch(plan.rows)?;
        self.scratch_rows = self.scratch_rows.max(plan.rows);
        if logit_n > 0 {
            grow(&self.dev, &mut self.scratch.gathered, logit_n * self.hidden)?;
            self.dev
                .ensure_f32(&mut self.scratch.logits, logit_n * self.cfg.vocab_size)
                .map_err(gpu_err)?;
        }
        self.logit_slots = self.logit_slots.max(logit_n);
        let key = self.graph_key(plan);
        if let Some(key) = key {
            if self.graphs.contains_key(&key) {
                self.dev
                    .launch_graph(self.graphs.get(&key).expect("graph key present"))
                    .map_err(gpu_err)?;
                return Ok(());
            }
        }
        self.run_forward(plan)?;
        if let Some(key) = key {
            if self.graphs_ok {
                match self.capture(plan) {
                    Ok(graph) => {
                        self.graphs.insert(key, graph);
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "disabling CUDA graphs after capture failed");
                        self.graphs_ok = false;
                        self.graphs.clear();
                    }
                }
            }
        }
        Ok(())
    }

    fn read_row_logits(&self, n: usize) -> Result<Vec<Vec<f32>>, Error> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let vocab = self.cfg.vocab_size;
        let host = self
            .dev
            .download_f32(&self.scratch.logits, n * vocab)
            .map_err(gpu_err)?;
        Ok(host.chunks(vocab).map(|row| row.to_vec()).collect())
    }

    fn graph_key(&self, plan: &BatchPlan<'_>) -> Option<(usize, i32)> {
        if !self.graphs_ok || plan.decode_count != plan.seqs.len() || plan.rows == 0 {
            return None;
        }
        if !plan.seqs.iter().all(|seq| seq.len == 1 && seq.want_logits) {
            return None;
        }
        if !is_captured_decode_batch(plan.rows) {
            return None;
        }
        Some((plan.rows, decode_splits(self.dtype, plan.max_decode_len)))
    }

    fn run_forward(&mut self, plan: &BatchPlan<'_>) -> Result<(), Error> {
        self.embed_plan(plan)?;
        for layer_idx in 0..self.layers.len() {
            self.run_layer(layer_idx, plan)?;
        }
        self.project_device(plan)
    }

    fn capture(&mut self, plan: &BatchPlan<'_>) -> Result<CapturedGraph, Error> {
        self.dev.begin_capture().map_err(gpu_err)?;
        let ran = self.run_forward(plan);
        match ran {
            Ok(()) => self.dev.end_capture().map_err(gpu_err),
            Err(err) => {
                let _ = self.dev.end_capture();
                Err(err)
            }
        }
    }

    fn project_device(&mut self, plan: &BatchPlan<'_>) -> Result<(), Error> {
        let n = plan.logit_count;
        if n == 0 {
            return Ok(());
        }
        let hidden = self.hidden;
        let vocab = self.cfg.vocab_size;
        let dtype = self.dtype;
        let eps = self.eps;
        gather_act(
            &self.dev,
            dtype,
            &mut self.scratch.gathered,
            &self.scratch.hidden,
            plan.logit_rows,
            n,
            hidden,
        )?;
        rms_act(
            &self.dev,
            dtype,
            &mut self.scratch.normed,
            &self.scratch.gathered,
            &self.final_norm,
            n,
            hidden,
            eps,
        )?;
        if let Some(head) = &self.lm_head {
            gemm_logits(
                &self.dev,
                dtype,
                &self.scratch.normed,
                &head.weight,
                &mut self.scratch.logits,
                &mut self.scratch.dequant,
                n,
                head.out_features,
                head.in_features,
            )?;
            if let Some(bias) = &head.bias {
                bias_logits(
                    &self.dev,
                    dtype,
                    &mut self.scratch.logits,
                    bias,
                    n,
                    head.out_features,
                )?;
            }
        } else {
            gemm_logits(
                &self.dev,
                dtype,
                &self.scratch.normed,
                &self.embed,
                &mut self.scratch.logits,
                &mut self.scratch.dequant,
                n,
                vocab,
                hidden,
            )?;
        }
        Ok(())
    }

    fn read_logits(&self, plan: &BatchPlan<'_>) -> Result<Vec<Option<Vec<f32>>>, Error> {
        let mut out = vec![None; plan.seqs.len()];
        let n = plan.logit_count;
        if n == 0 {
            return Ok(out);
        }
        let vocab = self.cfg.vocab_size;
        let host = self
            .dev
            .download_f32(&self.scratch.logits, n * vocab)
            .map_err(gpu_err)?;
        let mut slot = 0usize;
        for (seq_i, seq) in plan.seqs.iter().enumerate() {
            if seq.want_logits {
                let start = slot * vocab;
                out[seq_i] = Some(host[start..start + vocab].to_vec());
                slot += 1;
            }
        }
        Ok(out)
    }

    fn ensure_partials(&mut self, batch: usize, splits: i32) -> Result<(), Error> {
        let n = if splits <= 1 {
            1
        } else {
            splits as usize * batch * self.n_q
        };
        let acc = if splits <= 1 { 1 } else { n * self.head_dim };
        if acc > self.partial_cap {
            self.graphs.clear();
            self.partial_cap = acc;
        }
        self.dev
            .ensure_f32(&mut self.scratch.partial_acc, acc)
            .map_err(gpu_err)?;
        self.dev
            .ensure_f32(&mut self.scratch.partial_max, n)
            .map_err(gpu_err)?;
        self.dev
            .ensure_f32(&mut self.scratch.partial_sum, n)
            .map_err(gpu_err)?;
        Ok(())
    }
}

fn grow(dev: &Device, buf: &mut Act, n: usize) -> Result<(), Error> {
    match buf {
        Act::F32(inner) => dev.ensure_f32(inner, n).map_err(gpu_err),
        Act::Bf16(inner) => dev.ensure_bf16(inner, n).map_err(gpu_err),
    }
}

/// Batch sizes recorded as CUDA graphs. Every one of these is at or below
/// [`INT8_GEMV_MAX_ROWS`], so a captured INT8 decode never calls cuBLAS.
fn is_captured_decode_batch(rows: usize) -> bool {
    matches!(rows, 1 | 2 | 4 | 8 | 16 | 32 | 64)
}

fn decode_splits(dtype: GpuDType, max_len: i32) -> i32 {
    if dtype != GpuDType::Bf16 {
        return 1;
    }
    if max_len >= 2048 {
        8
    } else if max_len >= 512 {
        4
    } else if max_len >= 128 {
        2
    } else {
        1
    }
}

fn alloc_act(dev: &Device, dtype: GpuDType, n: usize) -> Result<Act, Error> {
    match dtype {
        GpuDType::F32 => Ok(Act::F32(dev.alloc_f32(n).map_err(gpu_err)?)),
        GpuDType::Bf16 => Ok(Act::Bf16(dev.alloc_bf16(n).map_err(gpu_err)?)),
    }
}

fn upload_act(dev: &Device, dtype: GpuDType, data: &[f32]) -> Result<Act, Error> {
    match dtype {
        GpuDType::F32 => Ok(Act::F32(dev.upload_f32(data).map_err(gpu_err)?)),
        GpuDType::Bf16 => Ok(Act::Bf16(dev.upload_bf16_from_f32(data).map_err(gpu_err)?)),
    }
}

fn upload_weight(dev: &Device, dtype: GpuDType, weight: PackedWeight) -> Result<DevWeight, Error> {
    match weight {
        PackedWeight::F32(values) => Ok(DevWeight::Dense(upload_act(dev, dtype, &values)?)),
        PackedWeight::Int8 { q, scales } => Ok(DevWeight::Int8 {
            q: dev.upload_i8(&q).map_err(gpu_err)?,
            scales: dev.upload_f32(&scales).map_err(gpu_err)?,
        }),
    }
}

fn upload_linear(dev: &Device, dtype: GpuDType, lin: OwnedLinear) -> Result<DevLinear, Error> {
    let bias = match lin.bias {
        Some(values) => Some(upload_act(dev, dtype, &values)?),
        None => None,
    };
    Ok(DevLinear {
        weight: upload_weight(dev, dtype, lin.weight)?,
        bias,
        out_features: lin.out_features,
        in_features: lin.in_features,
    })
}

fn weight_stats(owned: &OwnedWeights) -> (usize, usize, usize, bool) {
    let mut elems = 0usize;
    let mut bytes = 0usize;
    let mut max_matrix = 0usize;
    let mut int8 = false;
    let mut note = |weight: &PackedWeight| {
        elems += weight.elements();
        bytes += weight.storage_bytes();
        if matches!(weight, PackedWeight::Int8 { .. }) {
            int8 = true;
            max_matrix = max_matrix.max(weight.elements());
        }
    };
    note(&owned.embed);
    if let Some(head) = &owned.lm_head {
        note(&head.weight);
    }
    for layer in &owned.layers {
        for lin in [
            &layer.q,
            &layer.k,
            &layer.v,
            &layer.o,
            &layer.gate,
            &layer.up,
            &layer.down,
        ] {
            note(&lin.weight);
        }
    }
    (elems, bytes, max_matrix, int8)
}

fn apply_linear(
    dev: &Device,
    dtype: GpuDType,
    lin: &DevLinear,
    x: &Act,
    y: &mut Act,
    dequant: &mut Act,
    rows: usize,
) -> Result<(), Error> {
    match &lin.weight {
        DevWeight::Dense(weight) => gemm_act(
            dev,
            dtype,
            x,
            weight,
            y,
            rows,
            lin.out_features,
            lin.in_features,
        )?,
        DevWeight::Int8 { q, scales } => gemm_int8(
            dev,
            dtype,
            x,
            q,
            scales,
            Int8Out::Act(y),
            dequant,
            rows,
            lin.out_features,
            lin.in_features,
        )?,
    }
    if let Some(bias) = &lin.bias {
        bias_act(dev, dtype, y, bias, rows, lin.out_features)?;
    }
    Ok(())
}

fn pair<'a>(a: &'a Act, b: &'a Act) -> Result<(&'a F32Buf, &'a F32Buf), Error> {
    match (a, b) {
        (Act::F32(a), Act::F32(b)) => Ok((a, b)),
        _ => Err(Error::Inference(
            "internal: expected f32 activations".into(),
        )),
    }
}

fn pair_bf16<'a>(a: &'a Act, b: &'a Act) -> Result<(&'a Bf16Buf, &'a Bf16Buf), Error> {
    match (a, b) {
        (Act::Bf16(a), Act::Bf16(b)) => Ok((a, b)),
        _ => Err(Error::Inference(
            "internal: expected bf16 activations".into(),
        )),
    }
}

fn as_f32_mut(buf: &mut Act) -> Result<&mut F32Buf, Error> {
    match buf {
        Act::F32(inner) => Ok(inner),
        Act::Bf16(_) => Err(Error::Inference("internal: expected f32 buffer".into())),
    }
}

fn as_bf16_mut(buf: &mut Act) -> Result<&mut Bf16Buf, Error> {
    match buf {
        Act::Bf16(inner) => Ok(inner),
        Act::F32(_) => Err(Error::Inference("internal: expected bf16 buffer".into())),
    }
}

fn gemm_act(
    dev: &Device,
    dtype: GpuDType,
    x: &Act,
    w: &Act,
    y: &mut Act,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => {
            let (x, w) = pair(x, w)?;
            dev.gemm_nt(x, w, as_f32_mut(y)?, rows, out_features, in_features)
                .map_err(gpu_err)
        }
        GpuDType::Bf16 => {
            let (x, w) = pair_bf16(x, w)?;
            dev.gemm_bf16(x, w, as_bf16_mut(y)?, rows, out_features, in_features)
                .map_err(gpu_err)
        }
    }
}

fn gemm_logits(
    dev: &Device,
    dtype: GpuDType,
    x: &Act,
    w: &DevWeight,
    y: &mut F32Buf,
    dequant: &mut Act,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Error> {
    match w {
        DevWeight::Dense(weight) => match dtype {
            GpuDType::F32 => {
                let (x, w) = pair(x, weight)?;
                dev.gemm_nt(x, w, y, rows, out_features, in_features)
                    .map_err(gpu_err)
            }
            GpuDType::Bf16 => {
                let (x, w) = pair_bf16(x, weight)?;
                dev.gemm_bf16_f32(x, w, y, rows, out_features, in_features)
                    .map_err(gpu_err)
            }
        },
        DevWeight::Int8 { q, scales } => gemm_int8(
            dev,
            dtype,
            x,
            q,
            scales,
            Int8Out::Logits(y),
            dequant,
            rows,
            out_features,
            in_features,
        ),
    }
}

enum Int8Out<'a> {
    Act(&'a mut Act),
    Logits(&'a mut F32Buf),
}

#[allow(clippy::too_many_arguments)]
fn gemm_int8(
    dev: &Device,
    dtype: GpuDType,
    x: &Act,
    q: &I8Buf,
    scales: &F32Buf,
    mut y: Int8Out<'_>,
    dequant: &mut Act,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Error> {
    if rows <= INT8_GEMV_MAX_ROWS {
        return gemv_int8(dev, dtype, x, q, scales, y, rows, out_features, in_features);
    }
    // One span when scratch holds the matrix. A short scratch still tiles, so
    // a smaller allocation cannot write past the buffer.
    let tiles = crate::quant::dequant_tiles(out_features, in_features, dequant.capacity());
    if tiles.iter().sum::<usize>() != out_features {
        return Err(Error::Inference(
            "INT8 dequant scratch cannot hold one output row".into(),
        ));
    }
    let mut col = 0;
    for tile in tiles {
        dequant_tile(dev, dtype, dequant, q, scales, col, tile, in_features)?;
        gemm_tile(
            dev,
            dtype,
            x,
            dequant,
            &mut y,
            rows,
            tile,
            in_features,
            out_features,
            col,
        )?;
        col += tile;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemv_int8(
    dev: &Device,
    dtype: GpuDType,
    x: &Act,
    q: &I8Buf,
    scales: &F32Buf,
    y: Int8Out<'_>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Error> {
    match (dtype, y) {
        (GpuDType::F32, Int8Out::Act(y)) => dev
            .w8a_gemv_f32(
                as_f32_mut(y)?,
                expect_f32(x)?,
                q,
                scales,
                rows,
                out_features,
                in_features,
            )
            .map_err(gpu_err),
        (GpuDType::F32, Int8Out::Logits(y)) => dev
            .w8a_gemv_f32(
                y,
                expect_f32(x)?,
                q,
                scales,
                rows,
                out_features,
                in_features,
            )
            .map_err(gpu_err),
        (GpuDType::Bf16, Int8Out::Act(y)) => dev
            .w8a_gemv_bf16(
                as_bf16_mut(y)?,
                expect_bf16(x)?,
                q,
                scales,
                rows,
                out_features,
                in_features,
            )
            .map_err(gpu_err),
        (GpuDType::Bf16, Int8Out::Logits(y)) => dev
            .w8a_gemv_bf16_f32(
                y,
                expect_bf16(x)?,
                q,
                scales,
                rows,
                out_features,
                in_features,
            )
            .map_err(gpu_err),
    }
}

#[allow(clippy::too_many_arguments)]
fn dequant_tile(
    dev: &Device,
    dtype: GpuDType,
    dst: &mut Act,
    q: &I8Buf,
    scales: &F32Buf,
    src_row: usize,
    rows: usize,
    cols: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .dequant_i8_f32(as_f32_mut(dst)?, q, scales, src_row, rows, cols)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .dequant_i8_bf16(as_bf16_mut(dst)?, q, scales, src_row, rows, cols)
            .map_err(gpu_err),
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_tile(
    dev: &Device,
    dtype: GpuDType,
    x: &Act,
    w: &Act,
    y: &mut Int8Out<'_>,
    rows: usize,
    tile_out: usize,
    in_features: usize,
    ldc: usize,
    col: usize,
) -> Result<(), Error> {
    match (dtype, y) {
        (GpuDType::F32, Int8Out::Act(y)) => dev
            .gemm_nt_strided(
                expect_f32(x)?,
                expect_f32(w)?,
                as_f32_mut(y)?,
                rows,
                tile_out,
                in_features,
                ldc,
                col,
            )
            .map_err(gpu_err),
        (GpuDType::F32, Int8Out::Logits(y)) => dev
            .gemm_nt_strided(
                expect_f32(x)?,
                expect_f32(w)?,
                y,
                rows,
                tile_out,
                in_features,
                ldc,
                col,
            )
            .map_err(gpu_err),
        (GpuDType::Bf16, Int8Out::Act(y)) => dev
            .gemm_bf16_strided(
                expect_bf16(x)?,
                expect_bf16(w)?,
                as_bf16_mut(y)?,
                rows,
                tile_out,
                in_features,
                ldc,
                col,
            )
            .map_err(gpu_err),
        (GpuDType::Bf16, Int8Out::Logits(y)) => dev
            .gemm_bf16_f32_strided(
                expect_bf16(x)?,
                expect_bf16(w)?,
                y,
                rows,
                tile_out,
                in_features,
                ldc,
                col,
            )
            .map_err(gpu_err),
    }
}

fn embed_i8(
    dev: &Device,
    dtype: GpuDType,
    out: &mut Act,
    table: &I8Buf,
    scales: &F32Buf,
    tokens: &I32Buf,
    rows: usize,
    hidden: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .embed_i8_f32(as_f32_mut(out)?, table, scales, tokens, rows, hidden)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .embed_i8_bf16(as_bf16_mut(out)?, table, scales, tokens, rows, hidden)
            .map_err(gpu_err),
    }
}

fn bias_act(
    dev: &Device,
    dtype: GpuDType,
    y: &mut Act,
    bias: &Act,
    rows: usize,
    cols: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .bias(as_f32_mut(y)?, expect_f32(bias)?, rows, cols)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .bias_bf16(as_bf16_mut(y)?, expect_bf16(bias)?, rows, cols)
            .map_err(gpu_err),
    }
}

fn bias_logits(
    dev: &Device,
    dtype: GpuDType,
    y: &mut F32Buf,
    bias: &Act,
    rows: usize,
    cols: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev.bias(y, expect_f32(bias)?, rows, cols).map_err(gpu_err),
        // Logits are f32. A bf16 bias is added after the f32 gemm by casting
        // through the bf16 add only when the destination matches. The LM head
        // bias, when present, is applied by converting on the fly in f32 via
        // the f32 kernel after an upload. Qwen2.5 ties the head and has no
        // bias here. Reject a bf16 bias on f32 logits rather than silently
        // skipping it.
        GpuDType::Bf16 => Err(Error::Inference(
            "bf16 LM-head bias is not used by the tied Qwen head".into(),
        )),
    }
}

fn expect_f32(buf: &Act) -> Result<&F32Buf, Error> {
    match buf {
        Act::F32(inner) => Ok(inner),
        Act::Bf16(_) => Err(Error::Inference("internal: expected f32 buffer".into())),
    }
}

fn expect_bf16(buf: &Act) -> Result<&Bf16Buf, Error> {
    match buf {
        Act::Bf16(inner) => Ok(inner),
        Act::F32(_) => Err(Error::Inference("internal: expected bf16 buffer".into())),
    }
}

fn fused_residual_rms(
    dev: &Device,
    dtype: GpuDType,
    residual: &mut Act,
    out: &mut Act,
    x: &Act,
    weight: &Act,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .fused_residual_rms_f32(
                as_f32_mut(residual)?,
                as_f32_mut(out)?,
                expect_f32(x)?,
                expect_f32(weight)?,
                rows,
                cols,
                eps,
            )
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .fused_residual_rms_bf16(
                as_bf16_mut(residual)?,
                as_bf16_mut(out)?,
                expect_bf16(x)?,
                expect_bf16(weight)?,
                rows,
                cols,
                eps,
            )
            .map_err(gpu_err),
    }
}

fn fused_add_rms(
    dev: &Device,
    dtype: GpuDType,
    sum: &mut Act,
    normed: &mut Act,
    a: &Act,
    b: &Act,
    weight: &Act,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .fused_add_rms_f32(
                as_f32_mut(sum)?,
                as_f32_mut(normed)?,
                expect_f32(a)?,
                expect_f32(b)?,
                expect_f32(weight)?,
                rows,
                cols,
                eps,
            )
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .fused_add_rms_bf16(
                as_bf16_mut(sum)?,
                as_bf16_mut(normed)?,
                expect_bf16(a)?,
                expect_bf16(b)?,
                expect_bf16(weight)?,
                rows,
                cols,
                eps,
            )
            .map_err(gpu_err),
    }
}

fn rope_act(
    dev: &Device,
    dtype: GpuDType,
    x: &mut Act,
    positions: &I32Buf,
    inv_freq: &F32Buf,
    rows: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .rope(as_f32_mut(x)?, positions, inv_freq, rows, n_heads, head_dim)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .rope_bf16(
                as_bf16_mut(x)?,
                positions,
                inv_freq,
                rows,
                n_heads,
                head_dim,
            )
            .map_err(gpu_err),
    }
}

fn swiglu_act(
    dev: &Device,
    dtype: GpuDType,
    gate: &mut Act,
    up: &Act,
    n: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .swiglu(as_f32_mut(gate)?, expect_f32(up)?, n)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .swiglu_bf16(as_bf16_mut(gate)?, expect_bf16(up)?, n)
            .map_err(gpu_err),
    }
}

fn add_act(
    dev: &Device,
    dtype: GpuDType,
    dst: &mut Act,
    a: &Act,
    b: &Act,
    n: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .add(as_f32_mut(dst)?, expect_f32(a)?, expect_f32(b)?, n)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .add_bf16(as_bf16_mut(dst)?, expect_bf16(a)?, expect_bf16(b)?, n)
            .map_err(gpu_err),
    }
}

fn copy_act(
    dev: &Device,
    dtype: GpuDType,
    src: &Act,
    dst: &mut Act,
    n: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .copy_f32(expect_f32(src)?, as_f32_mut(dst)?, n)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .copy_bf16(expect_bf16(src)?, as_bf16_mut(dst)?, n)
            .map_err(gpu_err),
    }
}

fn embed_act(
    dev: &Device,
    dtype: GpuDType,
    out: &mut Act,
    table: &Act,
    tokens: &I32Buf,
    rows: usize,
    hidden: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .embed(as_f32_mut(out)?, expect_f32(table)?, tokens, rows, hidden)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .embed_bf16(as_bf16_mut(out)?, expect_bf16(table)?, tokens, rows, hidden)
            .map_err(gpu_err),
    }
}

fn gather_act(
    dev: &Device,
    dtype: GpuDType,
    dst: &mut Act,
    src: &Act,
    rows: &I32Buf,
    n: usize,
    cols: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .gather_rows(as_f32_mut(dst)?, expect_f32(src)?, rows, n, cols)
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .gather_rows_bf16(as_bf16_mut(dst)?, expect_bf16(src)?, rows, n, cols)
            .map_err(gpu_err),
    }
}

fn rms_act(
    dev: &Device,
    dtype: GpuDType,
    out: &mut Act,
    x: &Act,
    weight: &Act,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .rms_norm(
                as_f32_mut(out)?,
                expect_f32(x)?,
                expect_f32(weight)?,
                rows,
                cols,
                eps,
            )
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .rms_norm_bf16(
                as_bf16_mut(out)?,
                expect_bf16(x)?,
                expect_bf16(weight)?,
                rows,
                cols,
                eps,
            )
            .map_err(gpu_err),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_kv_act(
    dev: &Device,
    dtype: GpuDType,
    key_cache: &mut Act,
    value_cache: &mut Act,
    k: &Act,
    v: &Act,
    positions: &I32Buf,
    table_offsets: &I32Buf,
    block_tables: &I32Buf,
    rows: usize,
    layer: i32,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
    num_layers: usize,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .write_kv(
                as_f32_mut(key_cache)?,
                as_f32_mut(value_cache)?,
                expect_f32(k)?,
                expect_f32(v)?,
                positions,
                table_offsets,
                block_tables,
                rows,
                layer,
                n_kv,
                head_dim,
                block_size,
                num_layers,
            )
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .write_kv_bf16(
                as_bf16_mut(key_cache)?,
                as_bf16_mut(value_cache)?,
                expect_bf16(k)?,
                expect_bf16(v)?,
                positions,
                table_offsets,
                block_tables,
                rows,
                layer,
                n_kv,
                head_dim,
                block_size,
                num_layers,
            )
            .map_err(gpu_err),
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_act(
    dev: &Device,
    dtype: GpuDType,
    out: &mut Act,
    partial_acc: &mut F32Buf,
    partial_max: &mut F32Buf,
    partial_sum: &mut F32Buf,
    query: &Act,
    key_cache: &Act,
    value_cache: &Act,
    plan: &BatchPlan<'_>,
    splits: i32,
    shape: AttnShape,
) -> Result<(), Error> {
    match dtype {
        GpuDType::F32 => dev
            .paged_decode(
                as_f32_mut(out)?,
                expect_f32(query)?,
                expect_f32(key_cache)?,
                expect_f32(value_cache)?,
                plan.tables,
                plan.decode_offs,
                plan.decode_lens,
                plan.decode_rows,
                plan.decode_count,
                shape,
            )
            .map_err(gpu_err),
        GpuDType::Bf16 => {
            dev.paged_decode_bf16(
                as_bf16_mut(out)?,
                partial_acc,
                partial_max,
                partial_sum,
                expect_bf16(query)?,
                expect_bf16(key_cache)?,
                expect_bf16(value_cache)?,
                plan.tables,
                plan.decode_offs,
                plan.decode_lens,
                plan.decode_rows,
                plan.decode_count,
                splits,
                shape,
            )
            .map_err(gpu_err)?;
            dev.decode_combine_bf16(
                as_bf16_mut(out)?,
                partial_acc,
                partial_max,
                partial_sum,
                plan.decode_rows,
                plan.decode_count,
                splits,
                shape.num_qo_heads,
                shape.head_dim,
            )
            .map_err(gpu_err)
        }
    }
}

fn prefill_act(
    dev: &Device,
    dtype: GpuDType,
    out: &mut Act,
    query: &Act,
    key_cache: &Act,
    value_cache: &Act,
    tables: &I32Buf,
    seq: &SeqPlan,
    shape: AttnShape,
) -> Result<(), Error> {
    let row0 = as_i32(seq.row0, "prefill row")?;
    let chunk = as_i32(seq.len, "prefill chunk")?;
    let position = as_i32(seq.position, "prefill position")?;
    match dtype {
        GpuDType::F32 => dev
            .paged_prefill(
                as_f32_mut(out)?,
                expect_f32(query)?,
                expect_f32(key_cache)?,
                expect_f32(value_cache)?,
                tables,
                seq.table_off,
                seq.row0,
                seq.len,
                seq.position,
                shape,
            )
            .map_err(gpu_err),
        GpuDType::Bf16 => dev
            .paged_prefill_bf16(
                as_bf16_mut(out)?,
                expect_bf16(query)?,
                expect_bf16(key_cache)?,
                expect_bf16(value_cache)?,
                tables,
                seq.table_off,
                row0,
                chunk,
                position,
                shape,
            )
            .map_err(gpu_err),
    }
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

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_second_decode_replays_graph() {
        let cfg = tiny();
        let prompt = [1_u32, 4, 9, 2];
        let mut cpu_full = Model::from_random(&cfg, 7).unwrap();
        let full = cpu_full
            .forward_prompt(&[1, 4, 9, 2, 8, 3], 4)
            .unwrap()
            .logits
            .unwrap();
        let mut gpu = Model::from_random(&cfg, 7).unwrap().into_gpu(8, 4).unwrap();
        let table = [0_u32, 1];
        gpu.forward_batch(&[ForwardChunk {
            tokens: &prompt,
            position: 0,
            block_table: &table,
            want_logits: false,
        }])
        .unwrap();
        let first = [8_u32];
        let mid = gpu
            .forward_batch(&[ForwardChunk {
                tokens: &first,
                position: prompt.len(),
                block_table: &table,
                want_logits: true,
            }])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_graph_captured(&gpu);
        let mut cpu_mid = Model::from_random(&cfg, 7).unwrap();
        let expect_mid = cpu_mid
            .forward_prompt(&[1, 4, 9, 2, 8], 4)
            .unwrap()
            .logits
            .unwrap();
        assert_close(&mid, &expect_mid, 2e-3);
        let second = [3_u32];
        let got = gpu
            .forward_batch(&[ForwardChunk {
                tokens: &second,
                position: prompt.len() + 1,
                block_table: &table,
                want_logits: true,
            }])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_eq!(argmax(&got), argmax(&full));
        assert_close(&got, &full, 2e-3);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_bf16_logits_agree_with_cpu() {
        let cfg = tiny();
        let tokens = [1_u32, 4, 9, 2, 8, 3];
        let mut cpu = Model::from_random(&cfg, 7).unwrap();
        let expect = cpu.forward_prompt(&tokens, 4).unwrap().logits.unwrap();
        let mut gpu = Model::from_random(&cfg, 7)
            .unwrap()
            .into_gpu_with(8, 4, GpuDType::Bf16)
            .unwrap();
        let got = gpu.forward_prompt(&tokens).unwrap().logits.unwrap();
        assert_eq!(argmax(&got), argmax(&expect));
        assert_close(&got, &expect, 5e-2);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_bf16_second_decode_replays_graph() {
        let cfg = tiny();
        let prompt = [1_u32, 4, 9, 2];
        let mut cpu = Model::from_random(&cfg, 7).unwrap();
        let full = cpu
            .forward_prompt(&[1, 4, 9, 2, 8, 3], 4)
            .unwrap()
            .logits
            .unwrap();
        let mut gpu = Model::from_random(&cfg, 7)
            .unwrap()
            .into_gpu_with(8, 4, GpuDType::Bf16)
            .unwrap();
        let table = [0_u32, 1];
        gpu.forward_batch(&[ForwardChunk {
            tokens: &prompt,
            position: 0,
            block_table: &table,
            want_logits: false,
        }])
        .unwrap();
        let first = [8_u32];
        gpu.forward_batch(&[ForwardChunk {
            tokens: &first,
            position: prompt.len(),
            block_table: &table,
            want_logits: true,
        }])
        .unwrap();
        assert_graph_captured(&gpu);
        let second = [3_u32];
        let got = gpu
            .forward_batch(&[ForwardChunk {
                tokens: &second,
                position: prompt.len() + 1,
                block_table: &table,
                want_logits: true,
            }])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_eq!(argmax(&got), argmax(&full));
        assert_close(&got, &full, 5e-2);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_int8_matches_cpu_int8() {
        let cfg = tiny();
        let short = [1_u32, 4, 9, 2, 8, 3];
        let mut cpu = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8).unwrap();
        let expect_short = cpu.forward_prompt(&short, 4).unwrap().logits.unwrap();
        let mut gpu = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8)
            .unwrap()
            .into_gpu_with(8, 4, GpuDType::F32)
            .unwrap();
        let got_short = gpu.forward_prompt(&short).unwrap().logits.unwrap();
        assert_eq!(argmax(&got_short), argmax(&expect_short));
        assert_close(&got_short, &expect_short, 2e-2);

        // Longer than the fused GEMV cutoff, so prefill takes dequant + cuBLAS.
        let long: Vec<u32> = (0..80).map(|i| (i % 31) + 1).collect();
        let mut cpu_long = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8).unwrap();
        let expect_long = cpu_long.forward_prompt(&long, 4).unwrap().logits.unwrap();
        let mut gpu_long = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8)
            .unwrap()
            .into_gpu_with(24, 4, GpuDType::F32)
            .unwrap();
        let got_long = gpu_long.forward_prompt(&long).unwrap().logits.unwrap();
        assert_eq!(argmax(&got_long), argmax(&expect_long));
        assert_close(&got_long, &expect_long, 2e-2);

        // Second decode replays the CUDA graph captured on the first single-token step.
        let prompt = [1_u32, 4, 9, 2];
        let mut cpu_dec = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8).unwrap();
        let expect_dec = cpu_dec
            .forward_prompt(&[1, 4, 9, 2, 8, 3], 4)
            .unwrap()
            .logits
            .unwrap();
        let mut gpu_dec = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8)
            .unwrap()
            .into_gpu_with(8, 4, GpuDType::F32)
            .unwrap();
        let table = [0_u32, 1];
        gpu_dec
            .forward_batch(&[ForwardChunk {
                tokens: &prompt,
                position: 0,
                block_table: &table,
                want_logits: false,
            }])
            .unwrap();
        let first = [8_u32];
        gpu_dec
            .forward_batch(&[ForwardChunk {
                tokens: &first,
                position: prompt.len(),
                block_table: &table,
                want_logits: true,
            }])
            .unwrap();
        assert_graph_captured(&gpu_dec);
        let second = [3_u32];
        let got_dec = gpu_dec
            .forward_batch(&[ForwardChunk {
                tokens: &second,
                position: prompt.len() + 1,
                block_table: &table,
                want_logits: true,
            }])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_eq!(argmax(&got_dec), argmax(&expect_dec));
        assert_close(&got_dec, &expect_dec, 2e-2);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_int8_logits_close_to_bf16() {
        let cfg = tiny();
        let tokens = [1_u32, 4, 9, 2, 8, 3];
        let mut bf16 = Model::from_random(&cfg, 7)
            .unwrap()
            .into_gpu_with(8, 4, GpuDType::Bf16)
            .unwrap();
        let expect = bf16.forward_prompt(&tokens).unwrap().logits.unwrap();
        let mut int8 = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8)
            .unwrap()
            .into_gpu_with(8, 4, GpuDType::Bf16)
            .unwrap();
        let got = int8.forward_prompt(&tokens).unwrap().logits.unwrap();
        assert_eq!(argmax(&got), argmax(&expect));
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for (x, y) in got.iter().zip(&expect) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let sim = dot / (na.sqrt() * nb.sqrt());
        assert!(sim > 0.98, "cosine {sim}");
        assert_close(&got, &expect, 0.25);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_bf16_int8_second_decode_replays_graph() {
        let cfg = tiny();
        let prompt = [1_u32, 4, 9, 2];
        let mut cpu = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8).unwrap();
        let expect = cpu
            .forward_prompt(&[1, 4, 9, 2, 8, 3], 4)
            .unwrap()
            .logits
            .unwrap();
        let mut gpu = Model::from_random_with(&cfg, 7, crate::QuantMode::Int8)
            .unwrap()
            .into_gpu_with(8, 4, GpuDType::Bf16)
            .unwrap();
        let table = [0_u32, 1];
        gpu.forward_batch(&[ForwardChunk {
            tokens: &prompt,
            position: 0,
            block_table: &table,
            want_logits: false,
        }])
        .unwrap();
        let first = [8_u32];
        gpu.forward_batch(&[ForwardChunk {
            tokens: &first,
            position: prompt.len(),
            block_table: &table,
            want_logits: true,
        }])
        .unwrap();
        assert_graph_captured(&gpu);
        let second = [3_u32];
        let got = gpu
            .forward_batch(&[ForwardChunk {
                tokens: &second,
                position: prompt.len() + 1,
                block_table: &table,
                want_logits: true,
            }])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_eq!(argmax(&got), argmax(&expect));
        assert_close(&got, &expect, 0.25);
    }

    #[test]
    fn captured_decode_widths_stay_on_the_int8_gemv() {
        let mut saw_graph_width = false;
        for rows in 0..=128 {
            if is_captured_decode_batch(rows) {
                saw_graph_width = true;
                assert!(
                    rows <= INT8_GEMV_MAX_ROWS,
                    "graph width {rows} would enter the cuBLAS dequant path"
                );
            }
        }
        assert!(saw_graph_width);
    }

    fn assert_graph_captured(model: &GpuModel) {
        assert!(
            model.graphs_ok && !model.graphs.is_empty(),
            "decode step did not capture a CUDA graph (graphs_ok={}, captured={})",
            model.graphs_ok,
            model.graphs.len()
        );
    }
}
