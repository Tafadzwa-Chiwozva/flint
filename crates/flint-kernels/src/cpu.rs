//! Reference CPU kernels. They are numerically straightforward on purpose:
//! phase 2 replaces the bodies with CUDA launches without changing callers.

/// Physical layout of one KV cache tensor (keys or values).
///
/// ```text
/// index = (((block * num_layers + layer) * num_kv_heads + head) * block_size + slot) * head_dim + d
/// ```
///
/// Keys and values are stored in separate allocations with this same geometry.
/// Logical token `t` lives at `block = block_table[t / block_size]`, `slot = t % block_size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheGeometry {
    /// Number of physical blocks allocated in the cache.
    pub num_blocks: usize,
    /// Transformer layers packed into each block.
    pub num_layers: usize,
    /// KV heads (not query heads). GQA repeats these across query groups.
    pub num_kv_heads: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// Per-head channel count.
    pub head_dim: usize,
}

impl CacheGeometry {
    /// Element offset of `head_dim` values at one token slot.
    #[must_use]
    pub fn offset(&self, block: usize, layer: usize, head: usize, slot: usize) -> usize {
        let slot_stride = self.head_dim;
        let head_stride = self.block_size * slot_stride;
        let layer_stride = self.num_kv_heads * head_stride;
        let block_stride = self.num_layers * layer_stride;
        block * block_stride + layer * layer_stride + head * head_stride + slot * slot_stride
    }

    /// Elements in a single physical block (one of keys or values).
    #[must_use]
    pub fn block_stride(&self) -> usize {
        self.num_layers
            .saturating_mul(self.num_kv_heads)
            .saturating_mul(self.block_size)
            .saturating_mul(self.head_dim)
    }

    /// Total elements in one cache tensor, or `None` on overflow.
    #[must_use]
    pub fn elements(&self) -> Option<usize> {
        self.num_blocks.checked_mul(self.block_stride())
    }
}

/// Parameters for one paged-attention query (one token, all query heads).
#[derive(Clone, Copy, Debug)]
pub struct AttentionParams {
    /// Layer whose K/V pages are read.
    pub layer: usize,
    /// Number of valid tokens visible to this query, including itself.
    pub seq_len: usize,
    /// Query heads. Must be a multiple of `geometry.num_kv_heads`.
    pub num_qo_heads: usize,
    /// Score scale, typically `1 / sqrt(head_dim)`.
    pub scale: f32,
    /// If set, attend only to the most recent `w` tokens (including the current one).
    pub sliding_window: Option<usize>,
}

/// RMSNorm over the last dimension.
///
/// `weight.len()` is the row width. `x` and `out` hold one or more rows.
/// `out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i % width]`.
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let cols = weight.len();
    assert!(cols > 0, "rms_norm width must be non-zero");
    assert!(
        x.len().is_multiple_of(cols) && out.len() == x.len(),
        "rms_norm shape mismatch"
    );
    for (xrow, orow) in x.chunks_exact(cols).zip(out.chunks_exact_mut(cols)) {
        let mut square_sum = 0.0_f32;
        for v in xrow {
            square_sum += v * v;
        }
        let variance = square_sum / cols as f32;
        let scale = (variance + eps).sqrt().recip();
        for i in 0..cols {
            orow[i] = xrow[i] * scale * weight[i];
        }
    }
}

/// Grouped-query causal attention over a paged KV cache.
///
/// `query` and `out` are `[num_qo_heads, head_dim]` row-major. Query head `h`
/// reads KV head `h / (num_qo_heads / num_kv_heads)`, which is the same mapping
/// as Hugging Face `repeat_kv` (`repeat_interleave` over heads).
///
/// Tokens outside `sliding_window` (when set) are skipped. Softmax is computed
/// in f32 with a max subtraction.
pub fn paged_attention(
    out: &mut [f32],
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    block_table: &[u32],
    geometry: CacheGeometry,
    params: AttentionParams,
) {
    let head_dim = geometry.head_dim;
    assert!(head_dim > 0, "head_dim must be non-zero");
    assert_eq!(query.len(), params.num_qo_heads * head_dim);
    assert_eq!(out.len(), query.len());
    assert!(
        params.num_qo_heads.is_multiple_of(geometry.num_kv_heads),
        "query heads must be divisible by kv heads"
    );
    if params.seq_len == 0 {
        out.fill(0.0);
        return;
    }
    let last_block = (params.seq_len - 1) / geometry.block_size;
    assert!(
        block_table.len() > last_block,
        "block table does not cover seq_len"
    );

    let group = params.num_qo_heads / geometry.num_kv_heads;
    let start = match params.sliding_window {
        Some(window) if window < params.seq_len => params.seq_len - window,
        _ => 0,
    };

    let mut scores = vec![0.0_f32; params.seq_len];
    for kv_h in 0..geometry.num_kv_heads {
        for g in 0..group {
            let qh = kv_h * group + g;
            let q = &query[qh * head_dim..(qh + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for (t, score_slot) in scores.iter_mut().enumerate().skip(start) {
                let (block, slot) = locate(&geometry, block_table, t);
                let off = geometry.offset(block, params.layer, kv_h, slot);
                let key = &key_cache[off..off + head_dim];
                let mut dot = 0.0_f32;
                for (q_d, key_d) in q.iter().zip(key.iter()) {
                    dot += q_d * key_d;
                }
                let score = dot * params.scale;
                *score_slot = score;
                max_score = max_score.max(score);
            }
            let mut sum = 0.0_f32;
            for score in scores.iter_mut().skip(start) {
                let e = (*score - max_score).exp();
                *score = e;
                sum += e;
            }
            let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            let out_h = &mut out[qh * head_dim..(qh + 1) * head_dim];
            out_h.fill(0.0);
            for (t, score) in scores.iter().enumerate().skip(start) {
                let prob = score * inv;
                let (block, slot) = locate(&geometry, block_table, t);
                let off = geometry.offset(block, params.layer, kv_h, slot);
                let value = &value_cache[off..off + head_dim];
                for (out_d, value_d) in out_h.iter_mut().zip(value.iter()) {
                    *out_d += prob * value_d;
                }
            }
        }
    }
}

fn locate(geometry: &CacheGeometry, block_table: &[u32], token: usize) -> (usize, usize) {
    let index = token / geometry.block_size;
    let slot = token % geometry.block_size;
    (block_table[index] as usize, slot)
}
