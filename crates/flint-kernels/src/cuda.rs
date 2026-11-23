//! CUDA runtime for the forward pass.
//!
//! Kernels are the PTX image `build.rs` produces from `kernels/ops.cu`. This
//! module loads that image with the CUDA driver API (via `cudarc`) and runs
//! GEMMs through cuBLAS in full f32 (`CUBLAS_PEDANTIC_MATH`, so tensor-core
//! TF32 does not change the result). Nothing here runs unless a caller
//! constructs a [`Device`], which needs a live NVIDIA driver.

use std::ffi::c_int;
use std::sync::Arc;

use cudarc::cublas::sys::{cublasMath_t, cublasOperation_t, cublasSetMathMode, cublasStatus_t};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

const FLINT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/flint.ptx"));

/// Largest head dimension the attention kernel keeps in registers.
pub const MAX_HEAD_DIM: usize = 256;

/// Failure from the driver, cuBLAS, or a launch the host refused.
#[derive(Debug)]
pub struct CudaError {
    message: String,
}

impl CudaError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CudaError {}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        Self::new(err.to_string())
    }
}

impl From<cudarc::cublas::result::CublasError> for CudaError {
    fn from(err: cudarc::cublas::result::CublasError) -> Self {
        Self::new(err.to_string())
    }
}

/// Device buffer of `f32`. The allocation may be longer than the prefix a
/// kernel uses; launchers take an explicit length.
pub struct F32Buf {
    data: Option<CudaSlice<f32>>,
    cap: usize,
}

/// Device buffer of `i32` (token ids, positions, block tables).
pub struct I32Buf {
    data: Option<CudaSlice<i32>>,
    cap: usize,
}

impl F32Buf {
    /// Empty allocation. [`Device::ensure_f32`] grows it on first use.
    #[must_use]
    pub fn empty() -> Self {
        Self { data: None, cap: 0 }
    }

    fn view(&self, n: usize) -> Result<CudaView<'_, f32>, CudaError> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| CudaError::new(format!("empty f32 buffer, need {n} elements")))?;
        if n > self.cap {
            return Err(CudaError::new(format!(
                "f32 buffer cap {} is shorter than {n}",
                self.cap
            )));
        }
        data.try_slice(0..n)
            .ok_or_else(|| CudaError::new(format!("f32 slice 0..{n} rejected")))
    }

    fn view_mut(&mut self, n: usize) -> Result<CudaViewMut<'_, f32>, CudaError> {
        let cap = self.cap;
        let data = self
            .data
            .as_mut()
            .ok_or_else(|| CudaError::new(format!("empty f32 buffer, need {n} elements")))?;
        if n > cap {
            return Err(CudaError::new(format!(
                "f32 buffer cap {cap} is shorter than {n}"
            )));
        }
        data.try_slice_mut(0..n)
            .ok_or_else(|| CudaError::new(format!("f32 slice 0..{n} rejected")))
    }
}

impl I32Buf {
    fn view(&self, n: usize) -> Result<CudaView<'_, i32>, CudaError> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| CudaError::new(format!("empty i32 buffer, need {n} elements")))?;
        if n > self.cap {
            return Err(CudaError::new(format!(
                "i32 buffer cap {} is shorter than {n}",
                self.cap
            )));
        }
        data.try_slice(0..n)
            .ok_or_else(|| CudaError::new(format!("i32 slice 0..{n} rejected")))
    }
}

/// Static shape shared by the decode and prefill attention launches.
#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    /// Layer whose pages are read.
    pub layer: i32,
    /// Query heads.
    pub num_qo_heads: i32,
    /// KV heads. Must divide [`Self::num_qo_heads`].
    pub num_kv_heads: i32,
    /// Channels per head. Must be in `1..=`[`MAX_HEAD_DIM`].
    pub head_dim: i32,
    /// Tokens per KV block.
    pub block_size: i32,
    /// Layers packed into each physical block.
    pub num_layers: i32,
    /// Score scale, typically `1/sqrt(head_dim)`.
    pub scale: f32,
    /// Attend only to the last `w` tokens when `>= 0`. `-1` is the full prefix.
    pub sliding_window: i32,
}

/// One CUDA device, its stream, cuBLAS handle, and the Flint kernels.
pub struct Device {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    rms_norm: CudaFunction,
    rope: CudaFunction,
    swiglu: CudaFunction,
    add: CudaFunction,
    bias: CudaFunction,
    embed: CudaFunction,
    gather_rows: CudaFunction,
    write_kv: CudaFunction,
    paged_decode: CudaFunction,
    paged_prefill: CudaFunction,
}

impl Device {
    /// Retain the primary context on GPU 0 and load the Flint PTX.
    pub fn new() -> Result<Self, CudaError> {
        let ctx = CudaContext::new(0).map_err(|err| {
            CudaError::new(format!(
                "create CUDA context on device 0 ({err}). A loaded NVIDIA driver is required to run, not to compile."
            ))
        })?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        let status =
            unsafe { cublasSetMathMode(*blas.handle(), cublasMath_t::CUBLAS_PEDANTIC_MATH) };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::new(format!(
                "cublasSetMathMode(PEDANTIC) returned {status:?}"
            )));
        }
        let module = ctx.load_module(Ptx::from_src(FLINT_PTX))?;
        Ok(Self {
            stream,
            blas,
            rms_norm: function(&module, "flint_rms_norm")?,
            rope: function(&module, "flint_rope")?,
            swiglu: function(&module, "flint_swiglu")?,
            add: function(&module, "flint_add")?,
            bias: function(&module, "flint_bias")?,
            embed: function(&module, "flint_embed")?,
            gather_rows: function(&module, "flint_gather_rows")?,
            write_kv: function(&module, "flint_write_kv")?,
            paged_decode: function(&module, "flint_paged_decode")?,
            paged_prefill: function(&module, "flint_paged_prefill")?,
        })
    }

    /// Allocate `n` zeroed floats. `n == 0` yields an empty buffer.
    pub fn alloc_f32(&self, n: usize) -> Result<F32Buf, CudaError> {
        if n == 0 {
            return Ok(F32Buf { data: None, cap: 0 });
        }
        let data = self.stream.alloc_zeros::<f32>(n)?;
        Ok(F32Buf {
            data: Some(data),
            cap: n,
        })
    }

    /// Grow `buf` so it can hold `n` floats. Existing contents are dropped.
    pub fn ensure_f32(&self, buf: &mut F32Buf, n: usize) -> Result<(), CudaError> {
        if buf.cap >= n {
            return Ok(());
        }
        *buf = self.alloc_f32(n)?;
        Ok(())
    }

    /// Copy `data` to a fresh device buffer.
    pub fn upload_f32(&self, data: &[f32]) -> Result<F32Buf, CudaError> {
        if data.is_empty() {
            return Ok(F32Buf { data: None, cap: 0 });
        }
        let slice = self.stream.clone_htod(data)?;
        Ok(F32Buf {
            data: Some(slice),
            cap: data.len(),
        })
    }

    /// Copy `data` to a fresh device buffer of `i32`.
    pub fn upload_i32(&self, data: &[i32]) -> Result<I32Buf, CudaError> {
        if data.is_empty() {
            return Ok(I32Buf { data: None, cap: 0 });
        }
        let slice = self.stream.clone_htod(data)?;
        Ok(I32Buf {
            data: Some(slice),
            cap: data.len(),
        })
    }

    /// Copy `n` floats back to the host. Waits for the stream.
    pub fn download_f32(&self, buf: &F32Buf, n: usize) -> Result<Vec<f32>, CudaError> {
        self.stream.synchronize()?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let view = buf.view(n)?;
        Ok(self.stream.clone_dtoh(&view)?)
    }

    /// `dst[0..n] = src[0..n]`.
    pub fn copy_f32(&self, src: &F32Buf, dst: &mut F32Buf, n: usize) -> Result<(), CudaError> {
        if n == 0 {
            return Ok(());
        }
        let from = src.view(n)?;
        let mut to = dst.view_mut(n)?;
        self.stream.memcpy_dtod(&from, &mut to)?;
        Ok(())
    }

    /// Row-wise RMSNorm. `weight` has `cols` elements. `x` and `out` hold `rows` rows.
    pub fn rms_norm(
        &self,
        out: &mut F32Buf,
        x: &F32Buf,
        weight: &F32Buf,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "rms rows")?;
        let cols_i = as_i32(cols, "rms cols")?;
        let mut out_v = out.view_mut(rows * cols)?;
        let x_v = x.view(rows * cols)?;
        let w_v = weight.view(cols)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm)
                .arg(&mut out_v)
                .arg(&x_v)
                .arg(&w_v)
                .arg(&rows_i)
                .arg(&cols_i)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// In-place rotate-half RoPE. `x` is `[rows, n_heads, head_dim]`.
    /// `positions` has one absolute position per row. `inv_freq` has `head_dim/2` entries.
    pub fn rope(
        &self,
        x: &mut F32Buf,
        positions: &I32Buf,
        inv_freq: &F32Buf,
        rows: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let total = rows * n_heads;
        let rows_i = as_i32(rows, "rope rows")?;
        let heads_i = as_i32(n_heads, "rope heads")?;
        let dim_i = as_i32(head_dim, "rope head_dim")?;
        let mut x_v = x.view_mut(rows * n_heads * head_dim)?;
        let pos_v = positions.view(rows)?;
        let freq_v = inv_freq.view(head_dim / 2)?;
        let cfg = LaunchConfig {
            grid_dim: (total as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.rope)
                .arg(&mut x_v)
                .arg(&pos_v)
                .arg(&freq_v)
                .arg(&rows_i)
                .arg(&heads_i)
                .arg(&dim_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// In-place SwiGLU over `n` elements: `gate = silu(gate) * up`.
    pub fn swiglu(&self, gate: &mut F32Buf, up: &F32Buf, n: usize) -> Result<(), CudaError> {
        if n == 0 {
            return Ok(());
        }
        let n_i = as_i32(n, "swiglu n")?;
        let mut gate_v = gate.view_mut(n)?;
        let up_v = up.view(n)?;
        let cfg = linear_cfg(n)?;
        unsafe {
            self.stream
                .launch_builder(&self.swiglu)
                .arg(&mut gate_v)
                .arg(&up_v)
                .arg(&n_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `dst[i] = a[i] + b[i]` for `n` elements.
    pub fn add(&self, dst: &mut F32Buf, a: &F32Buf, b: &F32Buf, n: usize) -> Result<(), CudaError> {
        if n == 0 {
            return Ok(());
        }
        let n_i = as_i32(n, "add n")?;
        let mut dst_v = dst.view_mut(n)?;
        let a_v = a.view(n)?;
        let b_v = b.view(n)?;
        let cfg = linear_cfg(n)?;
        unsafe {
            self.stream
                .launch_builder(&self.add)
                .arg(&mut dst_v)
                .arg(&a_v)
                .arg(&b_v)
                .arg(&n_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Add `bias` (length `cols`) to every row of `y`.
    pub fn bias(
        &self,
        y: &mut F32Buf,
        bias: &F32Buf,
        rows: usize,
        cols: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "bias rows")?;
        let cols_i = as_i32(cols, "bias cols")?;
        let mut y_v = y.view_mut(rows * cols)?;
        let b_v = bias.view(cols)?;
        let cfg = linear_cfg(rows * cols)?;
        unsafe {
            self.stream
                .launch_builder(&self.bias)
                .arg(&mut y_v)
                .arg(&b_v)
                .arg(&rows_i)
                .arg(&cols_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Gather embedding rows. `tokens` are ids, already checked against `vocab`.
    pub fn embed(
        &self,
        out: &mut F32Buf,
        table: &F32Buf,
        tokens: &I32Buf,
        rows: usize,
        hidden: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "embed rows")?;
        let hidden_i = as_i32(hidden, "embed hidden")?;
        let table_n = table.cap;
        let mut out_v = out.view_mut(rows * hidden)?;
        let table_v = table.view(table_n)?;
        let tok_v = tokens.view(rows)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.embed)
                .arg(&mut out_v)
                .arg(&table_v)
                .arg(&tok_v)
                .arg(&rows_i)
                .arg(&hidden_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `dst[i] = src[rows[i]]` for `n` rows of `cols` floats.
    pub fn gather_rows(
        &self,
        dst: &mut F32Buf,
        src: &F32Buf,
        rows: &I32Buf,
        n: usize,
        cols: usize,
    ) -> Result<(), CudaError> {
        if n == 0 {
            return Ok(());
        }
        let n_i = as_i32(n, "gather n")?;
        let cols_i = as_i32(cols, "gather cols")?;
        let src_n = src.cap;
        let mut dst_v = dst.view_mut(n * cols)?;
        let src_v = src.view(src_n)?;
        let rows_v = rows.view(n)?;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.gather_rows)
                .arg(&mut dst_v)
                .arg(&src_v)
                .arg(&rows_v)
                .arg(&n_i)
                .arg(&cols_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Write each row of `k` and `v` into the paged cache.
    ///
    /// `positions[row]` is the absolute token index. `table_offsets[row]` indexes
    /// `block_tables`, which is the concatenation of every sequence's table.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv(
        &self,
        key_cache: &mut F32Buf,
        value_cache: &mut F32Buf,
        k: &F32Buf,
        v: &F32Buf,
        positions: &I32Buf,
        table_offsets: &I32Buf,
        block_tables: &I32Buf,
        rows: usize,
        layer: i32,
        n_kv: usize,
        head_dim: usize,
        block_size: usize,
        num_layers: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let total = rows * n_kv;
        let rows_i = as_i32(rows, "write_kv rows")?;
        let n_kv_i = as_i32(n_kv, "write_kv heads")?;
        let dim_i = as_i32(head_dim, "write_kv head_dim")?;
        let block_i = as_i32(block_size, "write_kv block_size")?;
        let layers_i = as_i32(num_layers, "write_kv layers")?;
        let key_n = key_cache.cap;
        let val_n = value_cache.cap;
        let table_n = block_tables.cap;
        let mut key_v = key_cache.view_mut(key_n)?;
        let mut val_v = value_cache.view_mut(val_n)?;
        let k_v = k.view(rows * n_kv * head_dim)?;
        let v_v = v.view(rows * n_kv * head_dim)?;
        let pos_v = positions.view(rows)?;
        let off_v = table_offsets.view(rows)?;
        let table_v = block_tables.view(table_n)?;
        let cfg = LaunchConfig {
            grid_dim: (total as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.write_kv)
                .arg(&mut key_v)
                .arg(&mut val_v)
                .arg(&k_v)
                .arg(&v_v)
                .arg(&pos_v)
                .arg(&off_v)
                .arg(&table_v)
                .arg(&rows_i)
                .arg(&layer)
                .arg(&n_kv_i)
                .arg(&dim_i)
                .arg(&block_i)
                .arg(&layers_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `Y = X @ W.T` with `W` stored row-major `[out_features, in_features]`.
    ///
    /// cuBLAS is column-major, so this is `Y_cm = W_cm.T @ X_cm` with both
    /// leading dimensions equal to `in_features`.
    pub fn gemm_nt(
        &self,
        x: &F32Buf,
        w: &F32Buf,
        y: &mut F32Buf,
        rows: usize,
        out_features: usize,
        in_features: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || out_features == 0 {
            return Ok(());
        }
        let x_v = x.view(rows * in_features)?;
        let w_v = w.view(out_features * in_features)?;
        let mut y_v = y.view_mut(rows * out_features)?;
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: as_i32(out_features, "gemm n")?,
            n: as_i32(rows, "gemm m")?,
            k: as_i32(in_features, "gemm k")?,
            alpha: 1.0_f32,
            lda: as_i32(in_features, "gemm lda")? as c_int,
            ldb: as_i32(in_features, "gemm ldb")? as c_int,
            beta: 0.0_f32,
            ldc: as_i32(out_features, "gemm ldc")? as c_int,
        };
        unsafe { self.blas.gemm(cfg, &w_v, &x_v, &mut y_v)? };
        Ok(())
    }

    /// Batched decode attention. One query token per sequence.
    ///
    /// `row_index[b]` selects the row of `query` and `out` (`[rows, qo, dim]`).
    /// `table_offsets[b]` indexes `block_tables`. `seq_lens[b]` is the number
    /// of visible tokens, including the one just written.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode(
        &self,
        out: &mut F32Buf,
        query: &F32Buf,
        key_cache: &F32Buf,
        value_cache: &F32Buf,
        block_tables: &I32Buf,
        table_offsets: &I32Buf,
        seq_lens: &I32Buf,
        row_index: &I32Buf,
        batch: usize,
        shape: AttnShape,
    ) -> Result<(), CudaError> {
        if batch == 0 {
            return Ok(());
        }
        check_shape(shape)?;
        let batch_i = as_i32(batch, "decode batch")?;
        let total = shape.num_qo_heads as usize * batch;
        let out_n = out.cap;
        let q_n = query.cap;
        let k_n = key_cache.cap;
        let v_n = value_cache.cap;
        let table_n = block_tables.cap;
        let mut out_v = out.view_mut(out_n)?;
        let q_v = query.view(q_n)?;
        let k_v = key_cache.view(k_n)?;
        let v_v = value_cache.view(v_n)?;
        let tables = block_tables.view(table_n)?;
        let offs = table_offsets.view(batch)?;
        let lens = seq_lens.view(batch)?;
        let rows = row_index.view(batch)?;
        let cfg = LaunchConfig {
            grid_dim: (total as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.paged_decode)
                .arg(&mut out_v)
                .arg(&q_v)
                .arg(&k_v)
                .arg(&v_v)
                .arg(&tables)
                .arg(&offs)
                .arg(&lens)
                .arg(&rows)
                .arg(&batch_i)
                .arg(&shape.layer)
                .arg(&shape.num_qo_heads)
                .arg(&shape.num_kv_heads)
                .arg(&shape.head_dim)
                .arg(&shape.block_size)
                .arg(&shape.num_layers)
                .arg(&shape.scale)
                .arg(&shape.sliding_window)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Causal prefill attention for one sequence chunk.
    ///
    /// Query row `row0 + i` attends to tokens `[0, position + i]`. `table_offset`
    /// indexes `block_tables` for this sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_prefill(
        &self,
        out: &mut F32Buf,
        query: &F32Buf,
        key_cache: &F32Buf,
        value_cache: &F32Buf,
        block_tables: &I32Buf,
        table_offset: i32,
        row0: usize,
        chunk: usize,
        position: usize,
        shape: AttnShape,
    ) -> Result<(), CudaError> {
        if chunk == 0 {
            return Ok(());
        }
        check_shape(shape)?;
        let total = shape.num_qo_heads as usize * chunk;
        let row0_i = as_i32(row0, "prefill row0")?;
        let chunk_i = as_i32(chunk, "prefill chunk")?;
        let pos_i = as_i32(position, "prefill position")?;
        let out_n = out.cap;
        let q_n = query.cap;
        let k_n = key_cache.cap;
        let v_n = value_cache.cap;
        let table_n = block_tables.cap;
        let mut out_v = out.view_mut(out_n)?;
        let q_v = query.view(q_n)?;
        let k_v = key_cache.view(k_n)?;
        let v_v = value_cache.view(v_n)?;
        let tables = block_tables.view(table_n)?;
        let cfg = LaunchConfig {
            grid_dim: (total as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.paged_prefill)
                .arg(&mut out_v)
                .arg(&q_v)
                .arg(&k_v)
                .arg(&v_v)
                .arg(&tables)
                .arg(&table_offset)
                .arg(&row0_i)
                .arg(&chunk_i)
                .arg(&pos_i)
                .arg(&shape.layer)
                .arg(&shape.num_qo_heads)
                .arg(&shape.num_kv_heads)
                .arg(&shape.head_dim)
                .arg(&shape.block_size)
                .arg(&shape.num_layers)
                .arg(&shape.scale)
                .arg(&shape.sliding_window)
                .launch(cfg)?;
        }
        Ok(())
    }
}

fn function(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction, CudaError> {
    module
        .load_function(name)
        .map_err(|err| CudaError::new(format!("load kernel {name}: {err}")))
}

fn as_i32(n: usize, what: &str) -> Result<i32, CudaError> {
    i32::try_from(n).map_err(|_| CudaError::new(format!("{what} ({n}) does not fit in i32")))
}

fn linear_cfg(n: usize) -> Result<LaunchConfig, CudaError> {
    let n =
        u32::try_from(n).map_err(|_| CudaError::new(format!("launch width {n} exceeds u32")))?;
    let block = 256u32;
    let grid = n.div_ceil(block);
    Ok(LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn check_shape(shape: AttnShape) -> Result<(), CudaError> {
    if shape.num_kv_heads <= 0 || shape.num_qo_heads % shape.num_kv_heads != 0 {
        return Err(CudaError::new(
            "query heads must be a positive multiple of kv heads",
        ));
    }
    if shape.head_dim <= 0 || shape.head_dim as usize > MAX_HEAD_DIM {
        return Err(CudaError::new(format!(
            "head_dim {} is outside 1..={MAX_HEAD_DIM}",
            shape.head_dim
        )));
    }
    if shape.block_size <= 0 || shape.num_layers <= 0 {
        return Err(CudaError::new("block_size and num_layers must be positive"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{paged_attention, rms_norm, AttentionParams, CacheGeometry};

    #[test]
    fn ptx_exports_kernels() {
        for name in [
            "flint_rms_norm",
            "flint_rope",
            "flint_swiglu",
            "flint_add",
            "flint_bias",
            "flint_embed",
            "flint_gather_rows",
            "flint_write_kv",
            "flint_paged_decode",
            "flint_paged_prefill",
        ] {
            assert!(FLINT_PTX.contains(name), "PTX is missing kernel {name}");
        }
    }

    fn gpu() -> Device {
        Device::new().expect("NVIDIA GPU and driver")
    }

    fn assert_close(got: &[f32], expect: &[f32], tol: f32) {
        assert_eq!(got.len(), expect.len());
        let max = got
            .iter()
            .zip(expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max <= tol,
            "max abs diff {max} > {tol}\ngot={got:?}\nexpect={expect:?}"
        );
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn rms_norm_matches_cpu() {
        let dev = gpu();
        let cols = 8;
        let x = vec![
            1.0_f32, 2.0, 3.0, 0.5, -1.0, 0.25, 4.0, -2.0, 0.0, 1.0, -1.0, 2.0, 0.5, 0.5, 0.5, 0.5,
        ];
        let w = vec![1.0_f32, 0.5, 2.0, 1.0, 1.0, -1.0, 0.25, 3.0];
        let mut expect = vec![0.0; x.len()];
        rms_norm(&mut expect, &x, &w, 1e-5);
        let x_d = dev.upload_f32(&x).unwrap();
        let w_d = dev.upload_f32(&w).unwrap();
        let mut out = dev.alloc_f32(x.len()).unwrap();
        dev.rms_norm(&mut out, &x_d, &w_d, 2, cols, 1e-5).unwrap();
        let got = dev.download_f32(&out, x.len()).unwrap();
        assert_close(&got, &expect, 1e-5);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn rope_matches_rotate_half() {
        let dev = gpu();
        let head_dim = 4;
        let n_heads = 2;
        let rows = 2;
        let inv_freq = [1.0_f32, 0.01];
        let x = vec![
            1.0_f32, 2.0, 3.0, 4.0, 0.5, -1.0, 1.5, 2.5, 1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 1.5, 2.5,
        ];
        let positions = [0_i32, 1];
        let mut expect = x.clone();
        apply_rope_ref(&mut expect[8..12], 1, &inv_freq);
        apply_rope_ref(&mut expect[12..16], 1, &inv_freq);
        let mut x_d = dev.upload_f32(&x).unwrap();
        let pos_d = dev.upload_i32(&positions).unwrap();
        let freq_d = dev.upload_f32(&inv_freq).unwrap();
        dev.rope(&mut x_d, &pos_d, &freq_d, rows, n_heads, head_dim)
            .unwrap();
        let got = dev.download_f32(&x_d, x.len()).unwrap();
        assert_close(&got, &expect, 1e-5);
    }

    fn apply_rope_ref(x: &mut [f32], pos: usize, inv_freq: &[f32]) {
        let half = x.len() / 2;
        let mut tmp = vec![0.0; x.len()];
        for i in 0..half {
            let (s, c) = ((pos as f32) * inv_freq[i]).sin_cos();
            let x1 = x[i];
            let x2 = x[i + half];
            tmp[i] = x1 * c - x2 * s;
            tmp[i + half] = x2 * c + x1 * s;
        }
        x.copy_from_slice(&tmp);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn swiglu_and_residual_match_cpu() {
        let dev = gpu();
        let gate = vec![1.0_f32, -2.0, 0.0, 0.5];
        let up = vec![2.0_f32, 3.0, -1.0, 4.0];
        let mut expect_g = gate.clone();
        for (g, u) in expect_g.iter_mut().zip(&up) {
            *g = silu_ref(*g) * u;
        }
        let mut gate_d = dev.upload_f32(&gate).unwrap();
        let up_d = dev.upload_f32(&up).unwrap();
        dev.swiglu(&mut gate_d, &up_d, gate.len()).unwrap();
        let got = dev.download_f32(&gate_d, gate.len()).unwrap();
        assert_close(&got, &expect_g, 1e-5);

        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![0.5_f32, -1.0, 4.0];
        let a_d = dev.upload_f32(&a).unwrap();
        let b_d = dev.upload_f32(&b).unwrap();
        let mut dst = dev.alloc_f32(a.len()).unwrap();
        dev.add(&mut dst, &a_d, &b_d, a.len()).unwrap();
        let sum = dev.download_f32(&dst, a.len()).unwrap();
        assert_close(&sum, &[1.5, 1.0, 7.0], 1e-6);
    }

    fn silu_ref(x: f32) -> f32 {
        if x >= 0.0 {
            x / (1.0 + (-x).exp())
        } else {
            let z = x.exp();
            x * z / (1.0 + z)
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gemm_matches_reference_layout() {
        let dev = gpu();
        // Y = X @ W.T, X is [2, 3], W is [2, 3].
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let bias = vec![0.5_f32, -1.0];
        let x_d = dev.upload_f32(&x).unwrap();
        let w_d = dev.upload_f32(&w).unwrap();
        let bias_d = dev.upload_f32(&bias).unwrap();
        let mut y = dev.alloc_f32(4).unwrap();
        dev.gemm_nt(&x_d, &w_d, &mut y, 2, 2, 3).unwrap();
        dev.bias(&mut y, &bias_d, 2, 2).unwrap();
        let got = dev.download_f32(&y, 4).unwrap();
        // Rows [1, 2] and [4, 5], plus bias.
        assert_close(&got, &[1.5, 1.0, 4.5, 4.0], 1e-5);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn paged_decode_matches_cpu_gqa() {
        let dev = gpu();
        let geometry = CacheGeometry {
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 2,
            block_size: 2,
            head_dim: 4,
        };
        let num_qo = 4;
        let scale = 1.0 / (geometry.head_dim as f32).sqrt();
        let elems = geometry.elements().unwrap();
        let mut key_cache = vec![0.0_f32; elems];
        let mut value_cache = vec![0.0_f32; elems];

        // Two sequences, scrambled tables, different lengths.
        // block_size 2: 4 tokens need two blocks, 3 tokens need two blocks.
        let tables = [3_i32, 1, 0, 2];
        let seqs = [(4_usize, 0_i32, 2_usize), (3_usize, 2_i32, 2_usize)];
        let mut query = vec![0.0_f32; 2 * num_qo * geometry.head_dim];
        let mut expect = vec![0.0_f32; query.len()];
        for (s, &(seq_len, table_off, table_len)) in seqs.iter().enumerate() {
            let table_u: Vec<u32> = tables[table_off as usize..table_off as usize + table_len]
                .iter()
                .map(|v| *v as u32)
                .collect();
            fill_seq_cache(
                &mut key_cache,
                &mut value_cache,
                &geometry,
                &table_u,
                seq_len,
                s + 1,
            );
            let q =
                &mut query[s * num_qo * geometry.head_dim..(s + 1) * num_qo * geometry.head_dim];
            for (i, slot) in q.iter_mut().enumerate() {
                *slot = ((i * 13 + s * 5) % 9) as f32 * 0.1 - 0.3;
            }
            paged_attention(
                &mut expect[s * num_qo * geometry.head_dim..(s + 1) * num_qo * geometry.head_dim],
                q,
                &key_cache,
                &value_cache,
                &table_u,
                geometry,
                AttentionParams {
                    layer: 0,
                    seq_len,
                    num_qo_heads: num_qo,
                    scale,
                    sliding_window: None,
                },
            );
        }

        let q_d = dev.upload_f32(&query).unwrap();
        let k_d = dev.upload_f32(&key_cache).unwrap();
        let v_d = dev.upload_f32(&value_cache).unwrap();
        let tables_d = dev.upload_i32(&tables).unwrap();
        let offs_d = dev.upload_i32(&[0, 2]).unwrap();
        let lens_d = dev.upload_i32(&[4, 3]).unwrap();
        let rows_d = dev.upload_i32(&[0, 1]).unwrap();
        let mut out = dev.alloc_f32(query.len()).unwrap();
        dev.paged_decode(
            &mut out,
            &q_d,
            &k_d,
            &v_d,
            &tables_d,
            &offs_d,
            &lens_d,
            &rows_d,
            2,
            AttnShape {
                layer: 0,
                num_qo_heads: num_qo as i32,
                num_kv_heads: geometry.num_kv_heads as i32,
                head_dim: geometry.head_dim as i32,
                block_size: geometry.block_size as i32,
                num_layers: geometry.num_layers as i32,
                scale,
                sliding_window: -1,
            },
        )
        .unwrap();
        let got = dev.download_f32(&out, query.len()).unwrap();
        assert_close(&got, &expect, 1e-4);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn paged_prefill_matches_cpu_and_window() {
        let dev = gpu();
        let geometry = CacheGeometry {
            num_blocks: 4,
            num_layers: 2,
            num_kv_heads: 2,
            block_size: 2,
            head_dim: 4,
        };
        let num_qo = 4;
        let scale = 1.0 / (geometry.head_dim as f32).sqrt();
        let chunk = 3;
        let position = 1; // chunk starts at token 1; token 0 is already cached
        let seq_total = position + chunk;
        let table = [2_u32, 0];
        let elems = geometry.elements().unwrap();
        let mut key_cache = vec![0.0_f32; elems];
        let mut value_cache = vec![0.0_f32; elems];
        fill_seq_cache(
            &mut key_cache,
            &mut value_cache,
            &geometry,
            &table,
            seq_total,
            3,
        );
        let mut query = vec![0.0_f32; chunk * num_qo * geometry.head_dim];
        for (i, slot) in query.iter_mut().enumerate() {
            *slot = ((i * 7) % 11) as f32 * 0.15 - 0.4;
        }
        let mut expect = vec![0.0_f32; query.len()];
        for t in 0..chunk {
            let q = &query[t * num_qo * geometry.head_dim..(t + 1) * num_qo * geometry.head_dim];
            paged_attention(
                &mut expect[t * num_qo * geometry.head_dim..(t + 1) * num_qo * geometry.head_dim],
                q,
                &key_cache,
                &value_cache,
                &table,
                geometry,
                AttentionParams {
                    layer: 1,
                    seq_len: position + t + 1,
                    num_qo_heads: num_qo,
                    scale,
                    sliding_window: Some(2),
                },
            );
        }
        let q_d = dev.upload_f32(&query).unwrap();
        let k_d = dev.upload_f32(&key_cache).unwrap();
        let v_d = dev.upload_f32(&value_cache).unwrap();
        let table_i: Vec<i32> = table.iter().map(|v| *v as i32).collect();
        let tables_d = dev.upload_i32(&table_i).unwrap();
        let mut out = dev.alloc_f32(query.len()).unwrap();
        dev.paged_prefill(
            &mut out,
            &q_d,
            &k_d,
            &v_d,
            &tables_d,
            0,
            0,
            chunk,
            position,
            AttnShape {
                layer: 1,
                num_qo_heads: num_qo as i32,
                num_kv_heads: geometry.num_kv_heads as i32,
                head_dim: geometry.head_dim as i32,
                block_size: geometry.block_size as i32,
                num_layers: geometry.num_layers as i32,
                scale,
                sliding_window: 2,
            },
        )
        .unwrap();
        let got = dev.download_f32(&out, query.len()).unwrap();
        assert_close(&got, &expect, 1e-4);
    }

    fn fill_seq_cache(
        keys: &mut [f32],
        values: &mut [f32],
        geometry: &CacheGeometry,
        table: &[u32],
        seq_len: usize,
        seed: usize,
    ) {
        for t in 0..seq_len {
            let block = table[t / geometry.block_size] as usize;
            let slot = t % geometry.block_size;
            for head in 0..geometry.num_kv_heads {
                let off = geometry.offset(block, 0, head, slot);
                // Layer 1 is a shifted copy so a wrong layer index fails the prefill test.
                let off1 = geometry.offset(block, 1.min(geometry.num_layers - 1), head, slot);
                for d in 0..geometry.head_dim {
                    let k = ((t * 5 + head * 3 + d + seed) % 7) as f32 * 0.2 - 0.4;
                    let v = ((t * 3 + head + d * 2 + seed) % 5) as f32 * 0.25;
                    keys[off + d] = k;
                    values[off + d] = v;
                    if geometry.num_layers > 1 {
                        keys[off1 + d] = k + 0.1;
                        values[off1 + d] = v + 0.2;
                    }
                }
            }
        }
    }
}
