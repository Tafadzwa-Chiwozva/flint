//! Per-channel symmetric INT8 weight quantization.
//!
//! Each output row of a linear weight `[out, in]` gets one f32 scale,
//! `max(|row|) / 127`. Values are rounded to the nearest integer in
//! `[-127, 127]`. Zero stays zero, and the positive and negative ranges match,
//! which is why the grid stops at 127 rather than using the extra negative
//! code `-128`. Norms and biases are not quantized.
//!
//! The matmul is W8A32: activations stay f32 (or the CUDA activation dtype)
//! and the product `scale * sum(x * q)` is accumulated in f32.

/// How linear weights are stored after load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantMode {
    /// Full-precision f32 weights.
    #[default]
    None,
    /// Per-output-channel symmetric INT8 weights.
    Int8,
}

/// One weight matrix, either f32 or per-channel INT8.
#[derive(Clone, Debug)]
pub(crate) enum PackedWeight {
    /// Row-major f32, `[rows, cols]`.
    F32(Vec<f32>),
    /// Row-major `i8` with one positive scale per row.
    Int8 { q: Vec<i8>, scales: Vec<f32> },
}

impl PackedWeight {
    pub(crate) fn f32(values: Vec<f32>) -> Self {
        Self::F32(values)
    }

    /// Parameter count, not the stored byte size.
    pub(crate) fn elements(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Int8 { q, .. } => q.len(),
        }
    }

    /// Bytes of the weight payload, including per-channel scales.
    pub(crate) fn storage_bytes(&self) -> usize {
        match self {
            Self::F32(values) => values.len() * std::mem::size_of::<f32>(),
            Self::Int8 { q, scales } => q.len() + scales.len() * std::mem::size_of::<f32>(),
        }
    }

    /// Quantize an f32 matrix in place. An already quantized matrix is unchanged.
    pub(crate) fn quantize(self, rows: usize, cols: usize) -> Self {
        match self {
            Self::F32(values) => {
                let (q, scales) = quantize_per_channel(&values, rows, cols);
                Self::Int8 { q, scales }
            }
            other => other,
        }
    }
}

/// Symmetric per-row INT8 quantization of a row-major `[rows, cols]` matrix.
pub(crate) fn quantize_per_channel(
    weight: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(weight.len(), rows.saturating_mul(cols), "quantize shape");
    let mut q = vec![0_i8; weight.len()];
    let mut scales = vec![0.0_f32; rows];
    if cols == 0 {
        return (q, scales);
    }
    for (row, (slice, scale_slot)) in weight.chunks(cols).zip(scales.iter_mut()).enumerate() {
        let mut max_abs = 0.0_f32;
        for value in slice {
            max_abs = max_abs.max(value.abs());
        }
        if max_abs == 0.0 || !max_abs.is_finite() {
            continue;
        }
        let scale = max_abs / 127.0;
        *scale_slot = scale;
        let inv = scale.recip();
        for (code, value) in q[row * cols..(row + 1) * cols].iter_mut().zip(slice) {
            let rounded = (value * inv).round().clamp(-127.0, 127.0);
            *code = rounded as i8;
        }
    }
    (q, scales)
}

/// Expand per-channel INT8 back to f32. `q` is row-major `[scales.len(), cols]`.
///
/// The forward path dequantizes one row or one product at a time. This copy is
/// the reference used by the accuracy tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn dequantize(q: &[i8], scales: &[f32], cols: usize) -> Vec<f32> {
    assert_eq!(q.len(), scales.len().saturating_mul(cols), "dequant shape");
    let mut out = vec![0.0_f32; q.len()];
    if cols == 0 {
        return out;
    }
    for (row, &scale) in scales.iter().enumerate() {
        let src = &q[row * cols..(row + 1) * cols];
        for (slot, code) in out[row * cols..(row + 1) * cols].iter_mut().zip(src) {
            *slot = f32::from(*code) * scale;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{linear, linear_i8};

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for (x, y) in a.iter().zip(b) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn endpoints_and_zero_round_trip() {
        // scale = 1/127. 0.5 * 127 = 63.5, and f32::round takes that to 64.
        let weight = [1.0_f32, -1.0, 0.5, 0.0];
        let (q, scales) = quantize_per_channel(&weight, 1, 4);
        assert!((scales[0] - 1.0 / 127.0).abs() < 1e-7);
        assert_eq!(q, [127, -127, 64, 0]);
        let back = dequantize(&q, &scales, 4);
        let limit = scales[0] * 0.5 + 1e-6;
        for (got, src) in back.iter().zip(&weight) {
            assert!((got - src).abs() <= limit, "{got} vs {src} limit {limit}");
        }
    }

    #[test]
    fn zero_row_stays_zero_and_error_is_half_a_bin() {
        let rows = 3;
        let cols = 17;
        let mut weight = vec![0.0_f32; rows * cols];
        for (i, value) in weight.iter_mut().enumerate().take(cols) {
            *value = ((i % 5) as f32) * 0.2 - 0.4;
        }
        // Row 1 is left at zero. Row 2 is a larger range, including a negative peak.
        for (i, value) in weight.iter_mut().skip(2 * cols).enumerate() {
            *value = if i == 3 { -2.5 } else { (i as f32) * 0.01 };
        }
        let (q, scales) = quantize_per_channel(&weight, rows, cols);
        assert_eq!(scales[1], 0.0);
        assert!(q[cols..2 * cols].iter().all(|v| *v == 0));
        let back = dequantize(&q, &scales, cols);
        for (row, &scale) in scales.iter().enumerate() {
            let limit = if scale == 0.0 {
                0.0
            } else {
                scale * 0.5 + 1e-5
            };
            for col in 0..cols {
                let idx = row * cols + col;
                let err = (back[idx] - weight[idx]).abs();
                assert!(err <= limit, "row {row} col {col}: err {err} > {limit}");
            }
        }
        assert!(cosine(&weight, &back) > 0.999);
    }

    #[test]
    fn int8_matmul_matches_dequant_and_stays_close_to_f32() {
        let rows = 3;
        let out = 40;
        let k = 17;
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let w: Vec<f32> = (0..out * k)
            .map(|i| ((i % 11) as f32) * 0.13 - 0.6)
            .collect();
        let (q, scales) = quantize_per_channel(&w, out, k);
        let mut bias = vec![0.25_f32; out];
        bias[1] = -0.5;
        let mut y_i8 = vec![0.0; rows * out];
        linear_i8(&q, &scales, Some(&bias), &x, &mut y_i8, rows, out, k);

        let dequant = dequantize(&q, &scales, k);
        let mut y_dequant = vec![0.0; rows * out];
        linear(&dequant, Some(&bias), &x, &mut y_dequant, rows, out, k);
        for (got, expect) in y_i8.iter().zip(&y_dequant) {
            assert!(
                (got - expect).abs() < 1e-3,
                "on-the-fly {got} vs dequant gemm {expect}"
            );
        }

        let mut y_f32 = vec![0.0; rows * out];
        linear(&w, Some(&bias), &x, &mut y_f32, rows, out, k);
        assert!(
            cosine(&y_i8, &y_f32) > 0.99,
            "cosine {}",
            cosine(&y_i8, &y_f32)
        );

        // Bias is applied once per row, on top of the scaled product.
        let mut y_plain = vec![0.0; rows * out];
        linear_i8(&q, &scales, None, &x, &mut y_plain, rows, out, k);
        assert!((y_i8[0] - (y_plain[0] + 0.25)).abs() < 1e-5);
        assert!((y_i8[1] - (y_plain[1] - 0.5)).abs() < 1e-5);
    }
}
