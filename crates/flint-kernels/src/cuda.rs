//! Host launchers for the CUDA kernels compiled by `build.rs`.
//!
//! These are not on the phase-1 forward path. They exist so the device code,
//! the ABI, and the link line are reviewed and compiled before a GPU box is
//! available. Phase 2 will call them from the model after a device KV cache
//! and a cuBLAS GEMM wrapper land.

use crate::{AttentionParams, CacheGeometry};

unsafe extern "C" {
    fn flint_rms_norm_launch(
        out: *mut f32,
        x: *const f32,
        weight: *const f32,
        rows: i32,
        cols: i32,
        eps: f32,
    );

    #[allow(dead_code)]
    fn flint_paged_attention_launch(
        out: *mut f32,
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        block_table: *const i32,
        layer: i32,
        seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        num_layers: i32,
        scale: f32,
        sliding_window: i32,
    );
}

/// Launch fused RMSNorm on device pointers. The caller owns synchronization
/// only in the sense that this function syncs before returning.
///
/// # Safety
///
/// Pointers must be device allocations covering `rows * cols` floats for `x`
/// and `out`, and `cols` floats for `weight`.
#[allow(dead_code)]
pub unsafe fn launch_rms_norm(
    out: *mut f32,
    x: *const f32,
    weight: *const f32,
    rows: i32,
    cols: i32,
    eps: f32,
) {
    unsafe {
        flint_rms_norm_launch(out, x, weight, rows, cols, eps);
    }
}

/// Launch paged attention. `sliding_window < 0` disables the window.
///
/// # Safety
///
/// All pointers are device pointers. `block_table` holds `ceil(seq_len / block_size)`
/// physical block ids. Cache layout matches [`CacheGeometry::offset`](crate::CacheGeometry::offset).
#[allow(dead_code)]
pub unsafe fn launch_paged_attention(
    out: *mut f32,
    query: *const f32,
    key_cache: *const f32,
    value_cache: *const f32,
    block_table: *const i32,
    geometry: CacheGeometry,
    params: AttentionParams,
) {
    let window = params.sliding_window.map(|w| w as i32).unwrap_or(-1);
    unsafe {
        flint_paged_attention_launch(
            out,
            query,
            key_cache,
            value_cache,
            block_table,
            params.layer as i32,
            params.seq_len as i32,
            params.num_qo_heads as i32,
            geometry.num_kv_heads as i32,
            geometry.head_dim as i32,
            geometry.block_size as i32,
            geometry.num_layers as i32,
            params.scale,
            window,
        );
    }
}
