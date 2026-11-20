//! Elementwise pieces of the transformer block.

/// `x * sigmoid(x)`, stable for large positive and negative inputs.
#[must_use]
pub fn silu(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        x / (1.0 + z)
    } else {
        let z = x.exp();
        x * z / (1.0 + z)
    }
}

/// In-place SwiGLU: `gate = silu(gate) * up`.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = silu(*g) * u;
    }
}

/// `dst[i] = a[i] + b[i]`.
pub fn add_vec(dst: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(dst.len(), b.len());
    for i in 0..dst.len() {
        dst[i] = a[i] + b[i];
    }
}

/// Duplicate-frequency RoPE angles used by Llama and Qwen2.
///
/// `inv_freq` has length `head_dim / 2`. `cos` and `sin` have length `head_dim`
/// and store the angle twice, so the rotate-half product lines up.
pub fn rope_angles(pos: usize, inv_freq: &[f32], cos: &mut [f32], sin: &mut [f32]) {
    let half = inv_freq.len();
    assert_eq!(cos.len(), half * 2);
    assert_eq!(sin.len(), half * 2);
    let pos = pos as f32;
    for i in 0..half {
        let (s, c) = (pos * inv_freq[i]).sin_cos();
        cos[i] = c;
        sin[i] = s;
        cos[i + half] = c;
        sin[i + half] = s;
    }
}

/// Apply Llama-style rotate-half RoPE to one head, in place.
pub fn apply_rope(x: &mut [f32], cos: &[f32], sin: &[f32], tmp: &mut [f32]) {
    assert_eq!(x.len(), cos.len());
    assert_eq!(x.len(), sin.len());
    assert!(tmp.len() >= x.len());
    let half = x.len() / 2;
    for i in 0..half {
        let x1 = x[i];
        let x2 = x[i + half];
        tmp[i] = x1 * cos[i] - x2 * sin[i];
        tmp[i + half] = x2 * cos[i + half] + x1 * sin[i + half];
    }
    x.copy_from_slice(&tmp[..x.len()]);
}

/// Build the RoPE inverse frequencies in f32, matching the PyTorch expression
/// `1 / theta ** (arange(0, dim, 2) / dim)`.
#[must_use]
pub fn inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    let dim = head_dim as f32;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / theta.powf((i as f32) / dim))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_of_one() {
        // sigmoid(1) ≈ 0.73105857863
        let got = silu(1.0);
        assert!((got - 0.731_058_6).abs() < 1e-5, "{got}");
    }

    #[test]
    fn swiglu_multiplies_after_silu() {
        let mut gate = [1.0_f32];
        swiglu(&mut gate, &[2.0]);
        assert!((gate[0] - silu(1.0) * 2.0).abs() < 1e-6);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let freq = inv_freq(4, 10_000.0);
        let mut cos = [0.0; 4];
        let mut sin = [0.0; 4];
        rope_angles(0, &freq, &mut cos, &mut sin);
        let mut x = [0.2, -0.4, 0.6, 1.5];
        let original = x;
        let mut tmp = [0.0; 4];
        apply_rope(&mut x, &cos, &sin, &mut tmp);
        for (a, b) in x.iter().zip(original) {
            assert!((a - b).abs() < 1e-6, "{x:?}");
        }
    }

    #[test]
    fn rope_matches_rotate_half_formula() {
        // head_dim = 4, theta = 10000, pos = 1
        // inv_freq = [1, 0.01]
        let freq = inv_freq(4, 10_000.0);
        assert!((freq[0] - 1.0).abs() < 1e-6);
        assert!((freq[1] - 0.01).abs() < 1e-6);
        let mut cos = [0.0; 4];
        let mut sin = [0.0; 4];
        rope_angles(1, &freq, &mut cos, &mut sin);
        let mut x = [1.0, 2.0, 3.0, 4.0];
        let mut tmp = [0.0; 4];
        apply_rope(&mut x, &cos, &sin, &mut tmp);
        // Hand-evaluated: out = x * cos + rotate_half(x) * sin
        // rotate_half([1, 2, 3, 4]) = [-3, -4, 1, 2]
        let expect = [
            1.0 * cos[0] + (-3.0) * sin[0],
            2.0 * cos[1] + (-4.0) * sin[1],
            3.0 * cos[2] + 1.0 * sin[2],
            4.0 * cos[3] + 2.0 * sin[3],
        ];
        for (got, exp) in x.iter().zip(expect) {
            assert!((got - exp).abs() < 1e-5, "{x:?} vs {expect:?}");
        }
    }
}
