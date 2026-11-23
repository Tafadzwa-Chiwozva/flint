//! CPU kernels used by the reference forward pass, plus the CUDA device runtime.
//!
//! The CPU path calls two kernels:
//!
//! * [`rms_norm`] — row-wise RMSNorm used before attention, before the MLP, and
//!   before the LM head.
//! * [`paged_attention`] — grouped-query attention that reads K and V through a
//!   block table instead of a contiguous cache.
//!
//! With `--features cuda`, `kernels/ops.cu` is compiled to PTX and the `cuda`
//! module loads it. That module also owns the cuBLAS GEMM. The CPU functions
//! stay the oracle; the GPU path is selected at runtime with `--device cuda`.

#![deny(missing_docs)]

mod cpu;

pub use cpu::{paged_attention, rms_norm, AttentionParams, CacheGeometry};

/// Returns whether this build compiled the CUDA translation units.
#[must_use]
pub fn cuda_compiled() -> bool {
    cfg!(feature = "cuda")
}

/// CUDA entry points. Present only when the crate is built with `--features cuda`.
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_feature_matches_this_build() {
        assert_eq!(cuda_compiled(), cfg!(feature = "cuda"));
    }

    #[test]
    fn rms_norm_matches_definition() {
        // x = [1, 2, 3], weight = [1, 1, 1], eps = 1e-6
        // variance = (1+4+9)/3 = 14/3
        let x = [1.0_f32, 2.0, 3.0];
        let w = [1.0_f32, 1.0, 1.0];
        let mut out = [0.0; 3];
        rms_norm(&mut out, &x, &w, 1e-6);
        let var = 14.0 / 3.0;
        let scale = (var + 1e-6_f32).sqrt().recip();
        let expect = [scale, 2.0 * scale, 3.0 * scale];
        for (got, exp) in out.iter().zip(expect) {
            assert!((got - exp).abs() < 1e-6, "{got} vs {exp}");
        }
    }

    #[test]
    fn rms_norm_is_rowwise() {
        let x = [1.0_f32, 0.0, 0.0, 1.0];
        let w = [2.0_f32, 3.0];
        let mut out = [0.0; 4];
        rms_norm(&mut out, &x, &w, 0.0);
        // Two rows of width 2. Variance of each row is 0.5.
        // Weight is per channel, so column 0 is scaled by 2 and column 1 by 3.
        let scale = (0.5_f32).sqrt().recip();
        assert!((out[0] - scale * 2.0).abs() < 1e-5);
        assert!(out[1].abs() < 1e-6);
        assert!(out[2].abs() < 1e-6);
        assert!((out[3] - scale * 3.0).abs() < 1e-5);
    }

    #[test]
    fn paged_attention_matches_dense_gqa() {
        let geometry = CacheGeometry {
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 2,
            block_size: 2,
            head_dim: 4,
        };
        let num_qo = 4; // group size 2
        let seq_len = 5;
        let scale = 1.0 / (geometry.head_dim as f32).sqrt();

        // Deterministic values, not all zeros.
        let mut query = vec![0.0; num_qo * geometry.head_dim];
        for (i, q) in query.iter_mut().enumerate() {
            *q = ((i * 17 + 3) % 11) as f32 * 0.1 - 0.4;
        }
        let mut keys_tok = vec![0.0; seq_len * geometry.num_kv_heads * geometry.head_dim];
        let mut vals_tok = vec![0.0; keys_tok.len()];
        for (i, k) in keys_tok.iter_mut().enumerate() {
            *k = ((i * 5 + 1) % 7) as f32 * 0.15 - 0.3;
        }
        for (i, v) in vals_tok.iter_mut().enumerate() {
            *v = ((i * 3 + 2) % 5) as f32 * 0.2;
        }

        let mut reference = vec![0.0; num_qo * geometry.head_dim];
        dense_gqa(
            &query,
            &keys_tok,
            &vals_tok,
            DenseSpec {
                seq_len,
                num_qo,
                num_kv: geometry.num_kv_heads,
                head_dim: geometry.head_dim,
                scale,
                window: None,
            },
            &mut reference,
        );

        // Scrambled block table: logical blocks 0,1,2 live at physical 3,1,0.
        let block_table = [3_u32, 1, 0];
        let elems = geometry.elements().unwrap();
        let mut key_cache = vec![0.0; elems];
        let mut value_cache = vec![0.0; elems];
        for t in 0..seq_len {
            let block = block_table[t / geometry.block_size] as usize;
            let slot = t % geometry.block_size;
            for h in 0..geometry.num_kv_heads {
                let src = (t * geometry.num_kv_heads + h) * geometry.head_dim;
                let dst = geometry.offset(block, 0, h, slot);
                key_cache[dst..dst + geometry.head_dim]
                    .copy_from_slice(&keys_tok[src..src + geometry.head_dim]);
                value_cache[dst..dst + geometry.head_dim]
                    .copy_from_slice(&vals_tok[src..src + geometry.head_dim]);
            }
        }

        let mut out = vec![0.0; reference.len()];
        paged_attention(
            &mut out,
            &query,
            &key_cache,
            &value_cache,
            &block_table,
            geometry,
            AttentionParams {
                layer: 0,
                seq_len,
                num_qo_heads: num_qo,
                scale,
                sliding_window: None,
            },
        );
        assert_close(&out, &reference, 1e-5);

        // Sliding window of 2 attends only to the last two tokens.
        let mut window_ref = vec![0.0; reference.len()];
        dense_gqa(
            &query,
            &keys_tok,
            &vals_tok,
            DenseSpec {
                seq_len,
                num_qo,
                num_kv: geometry.num_kv_heads,
                head_dim: geometry.head_dim,
                scale,
                window: Some(2),
            },
            &mut window_ref,
        );
        let mut window_out = vec![0.0; reference.len()];
        paged_attention(
            &mut window_out,
            &query,
            &key_cache,
            &value_cache,
            &block_table,
            geometry,
            AttentionParams {
                layer: 0,
                seq_len,
                num_qo_heads: num_qo,
                scale,
                sliding_window: Some(2),
            },
        );
        assert_close(&window_out, &window_ref, 1e-5);
        assert!(
            max_abs(&window_out, &reference) > 1e-4,
            "windowed attention should differ from full attention"
        );
    }

    struct DenseSpec {
        seq_len: usize,
        num_qo: usize,
        num_kv: usize,
        head_dim: usize,
        scale: f32,
        window: Option<usize>,
    }

    fn dense_gqa(query: &[f32], keys: &[f32], values: &[f32], spec: DenseSpec, out: &mut [f32]) {
        let DenseSpec {
            seq_len,
            num_qo,
            num_kv,
            head_dim,
            scale,
            window,
        } = spec;
        let group = num_qo / num_kv;
        let start = match window {
            Some(w) if w < seq_len => seq_len - w,
            _ => 0,
        };
        for qh in 0..num_qo {
            let kvh = qh / group;
            let q = &query[qh * head_dim..(qh + 1) * head_dim];
            let mut scores = vec![0.0; seq_len];
            let mut max = f32::NEG_INFINITY;
            for t in start..seq_len {
                let k = &keys[(t * num_kv + kvh) * head_dim..(t * num_kv + kvh + 1) * head_dim];
                let mut dot = 0.0;
                for (q_d, k_d) in q.iter().zip(k.iter()) {
                    dot += q_d * k_d;
                }
                scores[t] = dot * scale;
                max = max.max(scores[t]);
            }
            let mut sum = 0.0;
            for score in scores.iter_mut().skip(start) {
                *score = (*score - max).exp();
                sum += *score;
            }
            let inv = 1.0 / sum;
            let o = &mut out[qh * head_dim..(qh + 1) * head_dim];
            o.fill(0.0);
            for t in start..seq_len {
                let p = scores[t] * inv;
                let v = &values[(t * num_kv + kvh) * head_dim..(t * num_kv + kvh + 1) * head_dim];
                for (slot, v_d) in o.iter_mut().zip(v.iter()) {
                    *slot += p * v_d;
                }
            }
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        let max = max_abs(a, b);
        assert!(max <= tol, "max abs diff {max} > {tol}\na={a:?}\nb={b:?}");
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
}
