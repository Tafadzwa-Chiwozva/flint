//! Row-wise GEMM for `Y = X @ W.T` (+ optional bias).
//!
//! Weights stay in the Hugging Face layout `[out_features, in_features]`, so
//! each output is a dot product of two contiguous rows. The dot product uses
//! AVX2 FMA when the CPU has it and a scalar loop otherwise. Parallelism is
//! across outputs, never inside a single dot, so the reduction order (and
//! therefore the result) does not depend on the thread count.

use std::sync::atomic::{AtomicU8, Ordering};

use rayon::prelude::*;

/// `c[m, n] = a[m, k] @ b[n, k].T` where `b` is stored row-major `[n, k]`.
pub fn gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "gemm A shape");
    assert_eq!(b.len(), n * k, "gemm B shape");
    assert_eq!(c.len(), m * n, "gemm C shape");
    if m == 0 || n == 0 {
        return;
    }
    if m == 1 && n >= 256 {
        gemm_row_parallel(&a[..k], b, c, n, k);
        return;
    }
    if m >= 4 && n * k >= 4096 {
        c.par_chunks_mut(n).enumerate().for_each(|(row, crow)| {
            let arow = &a[row * k..(row + 1) * k];
            gemm_row_serial(arow, b, crow, n, k);
        });
        return;
    }
    for row in 0..m {
        let arow = &a[row * k..(row + 1) * k];
        gemm_row_serial(arow, b, &mut c[row * n..(row + 1) * n], n, k);
    }
}

/// `Y = X @ W.T + bias`, with `W` shaped `[n, k]` and optional bias of length `n`.
///
/// `qweight` is the same layout as an f32 weight, one `i8` per element, and
/// `scales` has one f32 per output channel. The dot is `scale * sum(x * q)`
/// in f32 (W8A32). Activations are not quantized.
#[allow(clippy::too_many_arguments)]
pub fn linear_i8(
    qweight: &[i8],
    scales: &[f32],
    bias: Option<&[f32]>,
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    out_features: usize,
    in_features: usize,
) {
    assert_eq!(
        qweight.len(),
        out_features * in_features,
        "int8 weight shape"
    );
    assert_eq!(scales.len(), out_features, "int8 scale shape");
    assert_eq!(x.len(), rows * in_features, "int8 activation shape");
    assert_eq!(y.len(), rows * out_features, "int8 output shape");
    if rows == 0 || out_features == 0 {
        return;
    }
    if rows == 1 && out_features >= 256 {
        gemm_i8_row_parallel(
            &x[..in_features],
            qweight,
            scales,
            y,
            out_features,
            in_features,
        );
    } else if rows >= 4 && out_features * in_features >= 4096 {
        y.par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(row, crow)| {
                let arow = &x[row * in_features..(row + 1) * in_features];
                gemm_i8_row_serial(arow, qweight, scales, crow, out_features, in_features);
            });
    } else {
        for row in 0..rows {
            let arow = &x[row * in_features..(row + 1) * in_features];
            gemm_i8_row_serial(
                arow,
                qweight,
                scales,
                &mut y[row * out_features..(row + 1) * out_features],
                out_features,
                in_features,
            );
        }
    }
    if let Some(bias) = bias {
        assert_eq!(bias.len(), out_features);
        for row in y.chunks_mut(out_features) {
            for (slot, &b) in row.iter_mut().zip(bias.iter()) {
                *slot += b;
            }
        }
    }
}

/// `Y = X @ W.T + bias`, with `W` shaped `[n, k]` and optional bias of length `n`.
pub fn linear(
    weight: &[f32],
    bias: Option<&[f32]>,
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    out_features: usize,
    in_features: usize,
) {
    gemm_nt(x, weight, y, rows, out_features, in_features);
    if let Some(bias) = bias {
        assert_eq!(bias.len(), out_features);
        for row in y.chunks_mut(out_features) {
            for (slot, &b) in row.iter_mut().zip(bias.iter()) {
                *slot += b;
            }
        }
    }
}

fn gemm_row_serial(arow: &[f32], b: &[f32], crow: &mut [f32], n: usize, k: usize) {
    for col in 0..n {
        crow[col] = dot(arow, &b[col * k..(col + 1) * k]);
    }
}

fn gemm_i8_row_serial(
    arow: &[f32],
    q: &[i8],
    scales: &[f32],
    crow: &mut [f32],
    n: usize,
    k: usize,
) {
    for col in 0..n {
        crow[col] = dot_f32_i8(arow, &q[col * k..(col + 1) * k]) * scales[col];
    }
}

fn gemm_i8_row_parallel(
    arow: &[f32],
    q: &[i8],
    scales: &[f32],
    crow: &mut [f32],
    n: usize,
    k: usize,
) {
    const TILE: usize = 128;
    crow.par_chunks_mut(TILE)
        .enumerate()
        .for_each(|(tile, out)| {
            let col0 = tile * TILE;
            for (offset, slot) in out.iter_mut().enumerate() {
                let col = col0 + offset;
                if col < n {
                    *slot = dot_f32_i8(arow, &q[col * k..(col + 1) * k]) * scales[col];
                }
            }
        });
}

fn dot_f32_i8(a: &[f32], b: &[i8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_fma() {
            // Safety: both slices have the same length, and the AVX2 kernel
            // only loads elements inside that range.
            return unsafe { dot_f32_i8_avx2(a, b) };
        }
    }
    dot_f32_i8_scalar(a, b)
}

fn dot_f32_i8_scalar(a: &[f32], b: &[i8]) -> f32 {
    let mut acc = 0.0_f32;
    for (x, q) in a.iter().zip(b) {
        acc += x * f32::from(*q);
    }
    acc
}

fn gemm_row_parallel(arow: &[f32], b: &[f32], crow: &mut [f32], n: usize, k: usize) {
    const TILE: usize = 128;
    crow.par_chunks_mut(TILE)
        .enumerate()
        .for_each(|(tile, out)| {
            let col0 = tile * TILE;
            for (offset, slot) in out.iter_mut().enumerate() {
                let col = col0 + offset;
                if col < n {
                    *slot = dot(arow, &b[col * k..(col + 1) * k]);
                }
            }
        });
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_fma() {
            // Safety: both slices have the same length, and the AVX2 kernel
            // only loads elements inside that range.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mut i = 0;
    let n = a.len();
    while i + 4 <= n {
        acc += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

fn avx2_fma() -> bool {
    static FLAG: AtomicU8 = AtomicU8::new(0);
    let flag = FLAG.load(Ordering::Relaxed);
    if flag == 0 {
        let enabled = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        FLAG.store(if enabled { 1 } else { 2 }, Ordering::Relaxed);
        enabled
    } else {
        flag == 1
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
    };

    let n = a.len();
    let mut i = 0;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    let mut out = _mm_cvtss_f32(sum);
    while i < n {
        out += a[i] * b[i];
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_avx2(a: &[f32], b: &[i8]) -> f32 {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_extractf128_ps,
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32,
        _mm_hadd_ps, _mm_loadl_epi64,
    };

    let n = a.len();
    let mut i = 0;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb8 = _mm_loadl_epi64(b.as_ptr().add(i).cast());
        let vb32 = _mm256_cvtepi8_epi32(vb8);
        let vb = _mm256_cvtepi32_ps(vb32);
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    let mut out = _mm_cvtss_f32(sum);
    while i < n {
        out += a[i] * f32::from(b[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0;
                for t in 0..k {
                    acc += a[row * k + t] * b[col * k + t];
                }
                c[row * n + col] = acc;
            }
        }
        c
    }

    #[test]
    fn gemm_matches_small_integers() {
        // A is [2, 2], B is [3, 2] (the weight matrix).
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut c = [0.0; 6];
        gemm_nt(&a, &b, &mut c, 2, 3, 2);
        assert_eq!(c, [17.0, 23.0, 29.0, 39.0, 53.0, 67.0]);
    }

    #[test]
    fn linear_adds_bias_per_row() {
        let weight = [1.0, 0.0, 0.0, 1.0]; // identity [2, 2]
        let x = [3.0, 4.0, 5.0, 6.0];
        let mut y = [0.0; 4];
        linear(&weight, Some(&[10.0, -1.0]), &x, &mut y, 2, 2, 2);
        assert_eq!(y, [13.0, 3.0, 15.0, 5.0]);
    }

    #[test]
    fn gemm_matches_naive_on_a_wider_problem() {
        let m = 3;
        let n = 40;
        let k = 17;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
        let mut c = vec![0.0; m * n];
        gemm_nt(&a, &b, &mut c, m, n, k);
        let expect = naive(&a, &b, m, n, k);
        for (got, exp) in c.iter().zip(&expect) {
            assert!((got - exp).abs() < 1e-4, "{got} vs {exp}");
        }
        let mut again = vec![0.0; m * n];
        gemm_nt(&a, &b, &mut again, m, n, k);
        assert_eq!(c, again);
    }
}
