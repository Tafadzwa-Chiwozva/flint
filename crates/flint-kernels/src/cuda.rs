//! CUDA runtime for the forward pass.
//!
//! Kernels are the PTX image `build.rs` produces from `kernels/ops.cu`. This
//! module loads that image with the CUDA driver API (via `cudarc`) and runs
//! GEMMs through cuBLAS in full f32 (`CUBLAS_PEDANTIC_MATH`, so tensor-core
//! TF32 does not change the result). All of that work is submitted to one
//! non-blocking stream. The legacy default stream cannot be captured, and it
//! implicitly synchronizes every cuBLAS launch. Nothing here runs unless a
//! caller constructs a [`Device`], which needs a live NVIDIA driver.

use std::ffi::{c_int, c_void};
use std::sync::Arc;

use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasMath_t, cublasOperation_t, cublasSetMathMode,
    cublasSetWorkspace_v2, cublasStatus_t, cudaDataType_t,
};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::sys::CUgraphInstantiate_flags;
use cudarc::driver::CudaGraph;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr,
    DevicePtrMut, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

const FLINT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/flint.ptx"));

/// Bytes reserved for cuBLAS so a captured GEMM does not call `cudaMalloc`.
/// 32 MiB is the size cuBLAS documents for Ada and Hopper.
const CUBLAS_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

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

/// Device buffer of bf16, stored as 16-bit words. Kernels interpret the bits.
pub struct Bf16Buf {
    data: Option<CudaSlice<u16>>,
    cap: usize,
}

/// Device buffer of `i32` (token ids, positions, block tables).
pub struct I32Buf {
    data: Option<CudaSlice<i32>>,
    cap: usize,
}

/// Device buffer of `i8` weight values.
pub struct I8Buf {
    data: Option<CudaSlice<i8>>,
    cap: usize,
}

impl F32Buf {
    /// Empty allocation. [`Device::ensure_f32`] grows it on first use.
    #[must_use]
    pub fn empty() -> Self {
        Self { data: None, cap: 0 }
    }

    /// Elements allocated, which may exceed the prefix a kernel uses.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
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

impl Bf16Buf {
    /// Empty allocation. Grown by [`Device::ensure_bf16`].
    #[must_use]
    pub fn empty() -> Self {
        Self { data: None, cap: 0 }
    }

    /// Elements allocated, which may exceed the prefix a kernel uses.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    fn view(&self, n: usize) -> Result<CudaView<'_, u16>, CudaError> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| CudaError::new(format!("empty bf16 buffer, need {n} elements")))?;
        if n > self.cap {
            return Err(CudaError::new(format!(
                "bf16 buffer cap {} is shorter than {n}",
                self.cap
            )));
        }
        data.try_slice(0..n)
            .ok_or_else(|| CudaError::new(format!("bf16 slice 0..{n} rejected")))
    }

    fn view_mut(&mut self, n: usize) -> Result<CudaViewMut<'_, u16>, CudaError> {
        let cap = self.cap;
        let data = self
            .data
            .as_mut()
            .ok_or_else(|| CudaError::new(format!("empty bf16 buffer, need {n} elements")))?;
        if n > cap {
            return Err(CudaError::new(format!(
                "bf16 buffer cap {cap} is shorter than {n}"
            )));
        }
        data.try_slice_mut(0..n)
            .ok_or_else(|| CudaError::new(format!("bf16 slice 0..{n} rejected")))
    }
}

impl I32Buf {
    /// Empty allocation. [`Device::fill_i32`] grows it on first use.
    #[must_use]
    pub fn empty() -> Self {
        Self { data: None, cap: 0 }
    }

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

    fn view_mut(&mut self, n: usize) -> Result<CudaViewMut<'_, i32>, CudaError> {
        let cap = self.cap;
        let data = self
            .data
            .as_mut()
            .ok_or_else(|| CudaError::new(format!("empty i32 buffer, need {n} elements")))?;
        if n > cap {
            return Err(CudaError::new(format!(
                "i32 buffer cap {cap} is shorter than {n}"
            )));
        }
        data.try_slice_mut(0..n)
            .ok_or_else(|| CudaError::new(format!("i32 slice 0..{n} rejected")))
    }
}

impl I8Buf {
    /// Empty allocation.
    #[must_use]
    pub fn empty() -> Self {
        Self { data: None, cap: 0 }
    }

    fn view(&self, n: usize) -> Result<CudaView<'_, i8>, CudaError> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| CudaError::new(format!("empty i8 buffer, need {n} elements")))?;
        if n > self.cap {
            return Err(CudaError::new(format!(
                "i8 buffer cap {} is shorter than {n}",
                self.cap
            )));
        }
        data.try_slice(0..n)
            .ok_or_else(|| CudaError::new(format!("i8 slice 0..{n} rejected")))
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
    /// User workspace for the cuBLAS handle. Captured GEMMs read it, so it
    /// has to stay mapped for the life of the device.
    workspace: CudaSlice<u8>,
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
    residual_rms_f32: CudaFunction,
    add_rms_f32: CudaFunction,
    rms_bf16: CudaFunction,
    rope_bf16: CudaFunction,
    swiglu_bf16: CudaFunction,
    add_bf16: CudaFunction,
    bias_bf16: CudaFunction,
    embed_bf16: CudaFunction,
    gather_bf16: CudaFunction,
    write_kv_bf16: CudaFunction,
    residual_rms_bf16: CudaFunction,
    add_rms_bf16: CudaFunction,
    cast_to_bf16: CudaFunction,
    paged_decode_bf16: CudaFunction,
    decode_combine_bf16: CudaFunction,
    paged_prefill_bf16: CudaFunction,
    w8a_gemv_f32: CudaFunction,
    w8a_gemv_bf16: CudaFunction,
    w8a_gemv_bf16_f32: CudaFunction,
    dequant_i8_f32: CudaFunction,
    dequant_i8_bf16: CudaFunction,
    embed_i8_f32: CudaFunction,
    embed_i8_bf16: CudaFunction,
}

/// A captured decode step. Replayed with [`Device::launch_graph`].
///
/// The engine thread owns the model, so the graph is sent once and then
/// launched from that thread only.
pub struct CapturedGraph {
    inner: CudaGraph,
}

unsafe impl Send for CapturedGraph {}

/// Storage format for GPU weights and activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuDType {
    /// Full f32, pedantic cuBLAS. This is the parity path.
    F32,
    /// bf16 storage with f32 accumulation and tensor cores.
    Bf16,
}

/// Bind `workspace` to the handle. `cublasSetStream` clears this, so call it
/// after [`CudaBlas::new`] and do not change the stream later.
fn set_cublas_workspace(
    blas: &CudaBlas,
    workspace: &CudaSlice<u8>,
    stream: &CudaStream,
) -> Result<(), CudaError> {
    let (ptr, _guard) = workspace.device_ptr(stream);
    let status =
        unsafe { cublasSetWorkspace_v2(*blas.handle(), ptr as *mut c_void, workspace.len()) };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(CudaError::new(format!(
            "cublasSetWorkspace returned {status:?}"
        )));
    }
    Ok(())
}

impl Drop for Device {
    fn drop(&mut self) {
        // The cuBLAS handle is destroyed with the device. Wait so an in-flight
        // GEMM cannot outlive the workspace it reads.
        let _ = self.stream.synchronize();
    }
}

impl Device {
    /// Retain the primary context on GPU 0 and load the Flint PTX.
    ///
    /// Work goes to a non-blocking stream. The legacy default stream cannot be
    /// captured, and each cuBLAS call on it waits for every other blocking
    /// stream before returning.
    pub fn new() -> Result<Self, CudaError> {
        let ctx = CudaContext::new(0).map_err(|err| {
            CudaError::new(format!(
                "create CUDA context on device 0 ({err}). A loaded NVIDIA driver is required to run, not to compile."
            ))
        })?;
        // Safety: this device owns one stream and never overlaps launches.
        // Automatic event tracking waits on events recorded outside a capture,
        // which the driver rejects while the stream is capturing.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let blas = CudaBlas::new(stream.clone())?;
        let status =
            unsafe { cublasSetMathMode(*blas.handle(), cublasMath_t::CUBLAS_PEDANTIC_MATH) };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::new(format!(
                "cublasSetMathMode(PEDANTIC) returned {status:?}"
            )));
        }
        // `CudaBlas::new` calls `cublasSetStream`, which clears any workspace.
        // Set it after that, or the first captured GEMM allocates.
        let workspace = stream.alloc_zeros::<u8>(CUBLAS_WORKSPACE_BYTES)?;
        set_cublas_workspace(&blas, &workspace, &stream)?;
        let module = ctx.load_module(Ptx::from_src(FLINT_PTX))?;
        Ok(Self {
            stream,
            blas,
            workspace,
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
            residual_rms_f32: function(&module, "flint_residual_rms_f32")?,
            add_rms_f32: function(&module, "flint_add_rms_f32")?,
            rms_bf16: function(&module, "flint_rms_norm_bf16")?,
            rope_bf16: function(&module, "flint_rope_bf16")?,
            swiglu_bf16: function(&module, "flint_swiglu_bf16")?,
            add_bf16: function(&module, "flint_add_bf16")?,
            bias_bf16: function(&module, "flint_bias_bf16")?,
            embed_bf16: function(&module, "flint_embed_bf16")?,
            gather_bf16: function(&module, "flint_gather_rows_bf16")?,
            write_kv_bf16: function(&module, "flint_write_kv_bf16")?,
            residual_rms_bf16: function(&module, "flint_residual_rms_bf16")?,
            add_rms_bf16: function(&module, "flint_add_rms_bf16")?,
            cast_to_bf16: function(&module, "flint_f32_to_bf16")?,
            paged_decode_bf16: function(&module, "flint_paged_decode_bf16")?,
            decode_combine_bf16: function(&module, "flint_decode_combine_bf16")?,
            paged_prefill_bf16: function(&module, "flint_paged_prefill_bf16")?,
            w8a_gemv_f32: function(&module, "flint_w8a_gemv_f32")?,
            w8a_gemv_bf16: function(&module, "flint_w8a_gemv_bf16")?,
            w8a_gemv_bf16_f32: function(&module, "flint_w8a_gemv_bf16_f32")?,
            dequant_i8_f32: function(&module, "flint_dequant_i8_f32")?,
            dequant_i8_bf16: function(&module, "flint_dequant_i8_bf16")?,
            embed_i8_f32: function(&module, "flint_embed_i8_f32")?,
            embed_i8_bf16: function(&module, "flint_embed_i8_bf16")?,
        })
    }

    /// Tensor-core math for the bf16 path. f32 parity stays on pedantic SGEMM.
    pub fn use_tensor_cores(&self) -> Result<(), CudaError> {
        let status =
            unsafe { cublasSetMathMode(*self.blas.handle(), cublasMath_t::CUBLAS_TENSOR_OP_MATH) };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::new(format!(
                "cublasSetMathMode(TENSOR_OP) returned {status:?}"
            )));
        }
        Ok(())
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

    /// Copy `data` to a fresh device buffer of `i8`.
    pub fn upload_i8(&self, data: &[i8]) -> Result<I8Buf, CudaError> {
        if data.is_empty() {
            return Ok(I8Buf::empty());
        }
        let slice = self.stream.clone_htod(data)?;
        Ok(I8Buf {
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
        if n == 0 {
            return Ok(Vec::new());
        }
        let view = buf.view(n)?;
        // `clone_dtoh` queues the copy. On the legacy NULL stream a pageable
        // D2H finished before the call returned; this stream does not.
        let host = self.stream.clone_dtoh(&view)?;
        self.stream.synchronize()?;
        Ok(host)
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

    /// Copy `n` bf16 values.
    pub fn copy_bf16(&self, src: &Bf16Buf, dst: &mut Bf16Buf, n: usize) -> Result<(), CudaError> {
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

impl Device {
    /// Copy `data` into `buf`, allocating only when the buffer is too small.
    /// The bool is true when the device address changed.
    pub fn fill_i32(&self, buf: &mut I32Buf, data: &[i32]) -> Result<bool, CudaError> {
        if data.is_empty() {
            return Ok(false);
        }
        if buf.data.is_none() || buf.cap < data.len() {
            *buf = self.upload_i32(data)?;
            return Ok(true);
        }
        let mut view = buf.view_mut(data.len())?;
        self.stream.memcpy_htod(data, &mut view)?;
        Ok(false)
    }

    /// Allocate `n` zeroed bf16 values.
    pub fn alloc_bf16(&self, n: usize) -> Result<Bf16Buf, CudaError> {
        if n == 0 {
            return Ok(Bf16Buf::empty());
        }
        let data = self.stream.alloc_zeros::<u16>(n)?;
        Ok(Bf16Buf {
            data: Some(data),
            cap: n,
        })
    }

    /// Grow `buf` so it can hold `n` bf16 values. Existing contents are dropped.
    pub fn ensure_bf16(&self, buf: &mut Bf16Buf, n: usize) -> Result<(), CudaError> {
        if buf.cap >= n && (n == 0 || buf.data.is_some()) {
            return Ok(());
        }
        *buf = self.alloc_bf16(n)?;
        Ok(())
    }

    /// Upload f32 host weights and store them as bf16.
    pub fn upload_bf16_from_f32(&self, data: &[f32]) -> Result<Bf16Buf, CudaError> {
        if data.is_empty() {
            return Ok(Bf16Buf::empty());
        }
        let src = self.upload_f32(data)?;
        let mut dst = self.alloc_bf16(data.len())?;
        self.cast_f32_to_bf16(&src, &mut dst, data.len())?;
        Ok(dst)
    }

    fn cast_f32_to_bf16(&self, src: &F32Buf, dst: &mut Bf16Buf, n: usize) -> Result<(), CudaError> {
        let n_i = as_i32(n, "cast n")?;
        let mut dst_v = dst.view_mut(n)?;
        let src_v = src.view(n)?;
        let cfg = linear_cfg(n)?;
        unsafe {
            self.stream
                .launch_builder(&self.cast_to_bf16)
                .arg(&mut dst_v)
                .arg(&src_v)
                .arg(&n_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Copy `x` into `residual` and write RMSNorm(`x`) into `out`. One launch.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_residual_rms_f32(
        &self,
        residual: &mut F32Buf,
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
        let rows_i = as_i32(rows, "residual rows")?;
        let cols_i = as_i32(cols, "residual cols")?;
        let mut residual_v = residual.view_mut(rows * cols)?;
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
                .launch_builder(&self.residual_rms_f32)
                .arg(&mut residual_v)
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

    /// bf16 copy plus RMSNorm. The sum of squares is accumulated in f32.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_residual_rms_bf16(
        &self,
        residual: &mut Bf16Buf,
        out: &mut Bf16Buf,
        x: &Bf16Buf,
        weight: &Bf16Buf,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "residual rows")?;
        let cols_i = as_i32(cols, "residual cols")?;
        let mut residual_v = residual.view_mut(rows * cols)?;
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
                .launch_builder(&self.residual_rms_bf16)
                .arg(&mut residual_v)
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

    /// `sum = a + b`, then RMSNorm(`sum`) into `normed`. One launch.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_add_rms_f32(
        &self,
        sum: &mut F32Buf,
        normed: &mut F32Buf,
        a: &F32Buf,
        b: &F32Buf,
        weight: &F32Buf,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "add-rms rows")?;
        let cols_i = as_i32(cols, "add-rms cols")?;
        let mut sum_v = sum.view_mut(rows * cols)?;
        let mut normed_v = normed.view_mut(rows * cols)?;
        let a_v = a.view(rows * cols)?;
        let b_v = b.view(rows * cols)?;
        let w_v = weight.view(cols)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.add_rms_f32)
                .arg(&mut sum_v)
                .arg(&mut normed_v)
                .arg(&a_v)
                .arg(&b_v)
                .arg(&w_v)
                .arg(&rows_i)
                .arg(&cols_i)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// bf16 residual add plus RMSNorm. Accumulation is f32.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_add_rms_bf16(
        &self,
        sum: &mut Bf16Buf,
        normed: &mut Bf16Buf,
        a: &Bf16Buf,
        b: &Bf16Buf,
        weight: &Bf16Buf,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "add-rms rows")?;
        let cols_i = as_i32(cols, "add-rms cols")?;
        let mut sum_v = sum.view_mut(rows * cols)?;
        let mut normed_v = normed.view_mut(rows * cols)?;
        let a_v = a.view(rows * cols)?;
        let b_v = b.view(rows * cols)?;
        let w_v = weight.view(cols)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.add_rms_bf16)
                .arg(&mut sum_v)
                .arg(&mut normed_v)
                .arg(&a_v)
                .arg(&b_v)
                .arg(&w_v)
                .arg(&rows_i)
                .arg(&cols_i)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `Y = X @ W.T` with bf16 inputs, f32 accumulation, and a bf16 result.
    pub fn gemm_bf16(
        &self,
        x: &Bf16Buf,
        w: &Bf16Buf,
        y: &mut Bf16Buf,
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
        let (w_ptr, w_sync) = w_v.device_ptr(&self.stream);
        let (x_ptr, x_sync) = x_v.device_ptr(&self.stream);
        let (y_ptr, y_sync) = y_v.device_ptr_mut(&self.stream);
        self.gemm_ex_raw(
            w_ptr,
            x_ptr,
            y_ptr,
            rows,
            out_features,
            in_features,
            out_features,
            cudaDataType_t::CUDA_R_16BF,
            cudaDataType_t::CUDA_R_16BF,
        )?;
        drop((w_sync, x_sync, y_sync));
        Ok(())
    }

    /// Same product as [`Self::gemm_bf16`], writing an f32 matrix (the logits).
    pub fn gemm_bf16_f32(
        &self,
        x: &Bf16Buf,
        w: &Bf16Buf,
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
        let (w_ptr, w_sync) = w_v.device_ptr(&self.stream);
        let (x_ptr, x_sync) = x_v.device_ptr(&self.stream);
        let (y_ptr, y_sync) = y_v.device_ptr_mut(&self.stream);
        self.gemm_ex_raw(
            w_ptr,
            x_ptr,
            y_ptr,
            rows,
            out_features,
            in_features,
            out_features,
            cudaDataType_t::CUDA_R_16BF,
            cudaDataType_t::CUDA_R_32F,
        )?;
        drop((w_sync, x_sync, y_sync));
        Ok(())
    }

    /// RMSNorm over bf16 rows. The sum of squares is f32.
    pub fn rms_norm_bf16(
        &self,
        out: &mut Bf16Buf,
        x: &Bf16Buf,
        weight: &Bf16Buf,
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
                .launch_builder(&self.rms_bf16)
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

    /// In-place rotate-half RoPE on bf16. `inv_freq` stays f32.
    pub fn rope_bf16(
        &self,
        x: &mut Bf16Buf,
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
                .launch_builder(&self.rope_bf16)
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

    /// In-place SwiGLU over `n` bf16 elements.
    pub fn swiglu_bf16(&self, gate: &mut Bf16Buf, up: &Bf16Buf, n: usize) -> Result<(), CudaError> {
        if n == 0 {
            return Ok(());
        }
        let n_i = as_i32(n, "swiglu n")?;
        let mut gate_v = gate.view_mut(n)?;
        let up_v = up.view(n)?;
        let cfg = linear_cfg(n)?;
        unsafe {
            self.stream
                .launch_builder(&self.swiglu_bf16)
                .arg(&mut gate_v)
                .arg(&up_v)
                .arg(&n_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `dst[i] = a[i] + b[i]` for `n` bf16 elements. `dst` may alias `a`.
    pub fn add_bf16(
        &self,
        dst: &mut Bf16Buf,
        a: &Bf16Buf,
        b: &Bf16Buf,
        n: usize,
    ) -> Result<(), CudaError> {
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
                .launch_builder(&self.add_bf16)
                .arg(&mut dst_v)
                .arg(&a_v)
                .arg(&b_v)
                .arg(&n_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Add a bf16 bias of length `cols` to every row.
    pub fn bias_bf16(
        &self,
        y: &mut Bf16Buf,
        bias: &Bf16Buf,
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
                .launch_builder(&self.bias_bf16)
                .arg(&mut y_v)
                .arg(&b_v)
                .arg(&rows_i)
                .arg(&cols_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Gather bf16 embedding rows.
    pub fn embed_bf16(
        &self,
        out: &mut Bf16Buf,
        table: &Bf16Buf,
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
                .launch_builder(&self.embed_bf16)
                .arg(&mut out_v)
                .arg(&table_v)
                .arg(&tok_v)
                .arg(&rows_i)
                .arg(&hidden_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `dst[i] = src[rows[i]]` for `n` bf16 rows of `cols`.
    pub fn gather_rows_bf16(
        &self,
        dst: &mut Bf16Buf,
        src: &Bf16Buf,
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
                .launch_builder(&self.gather_bf16)
                .arg(&mut dst_v)
                .arg(&src_v)
                .arg(&rows_v)
                .arg(&n_i)
                .arg(&cols_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `Y[:, col:col+tile] = X @ W_tile.T` with `Y` row-major `[rows, ldc]`.
    ///
    /// `w` is a contiguous dequantized tile `[tile_out, in_features]`. The
    /// write uses cuBLAS leading dimension `ldc` so the tile lands in the
    /// right columns without a second scatter.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_nt_strided(
        &self,
        x: &F32Buf,
        w: &F32Buf,
        y: &mut F32Buf,
        rows: usize,
        tile_out: usize,
        in_features: usize,
        ldc: usize,
        col: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || tile_out == 0 {
            return Ok(());
        }
        let x_v = x.view(rows * in_features)?;
        let w_v = w.view(tile_out * in_features)?;
        let mut y_v = y.view_mut(rows * ldc)?;
        let (w_ptr, w_sync) = w_v.device_ptr(&self.stream);
        let (x_ptr, x_sync) = x_v.device_ptr(&self.stream);
        let (y_ptr, y_sync) = y_v.device_ptr_mut(&self.stream);
        // f32 stays on cublasSgemm so the pedantic math mode (no TF32) matches
        // the dense f32 GEMM. The pointer is the column-major start of this tile.
        let y_off = y_ptr + (col * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
        let m = as_i32(tile_out, "gemm n")?;
        let n = as_i32(rows, "gemm m")?;
        let k = as_i32(in_features, "gemm k")?;
        let ld = as_i32(in_features, "gemm lda")?;
        let ldc_i = as_i32(ldc, "gemm ldc")?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        unsafe {
            cudarc::cublas::result::sgemm(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                w_ptr as *const f32,
                ld,
                x_ptr as *const f32,
                ld,
                &beta,
                y_off as *mut f32,
                ldc_i,
            )?;
        }
        drop((w_sync, x_sync, y_sync));
        Ok(())
    }

    /// bf16 form of [`Self::gemm_nt_strided`]. `y` is bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_strided(
        &self,
        x: &Bf16Buf,
        w: &Bf16Buf,
        y: &mut Bf16Buf,
        rows: usize,
        tile_out: usize,
        in_features: usize,
        ldc: usize,
        col: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || tile_out == 0 {
            return Ok(());
        }
        let x_v = x.view(rows * in_features)?;
        let w_v = w.view(tile_out * in_features)?;
        let mut y_v = y.view_mut(rows * ldc)?;
        let (w_ptr, w_sync) = w_v.device_ptr(&self.stream);
        let (x_ptr, x_sync) = x_v.device_ptr(&self.stream);
        let (y_ptr, y_sync) = y_v.device_ptr_mut(&self.stream);
        let y_off = y_ptr + (col * std::mem::size_of::<u16>()) as cudarc::driver::sys::CUdeviceptr;
        self.gemm_ex_raw(
            w_ptr,
            x_ptr,
            y_off,
            rows,
            tile_out,
            in_features,
            ldc,
            cudaDataType_t::CUDA_R_16BF,
            cudaDataType_t::CUDA_R_16BF,
        )?;
        drop((w_sync, x_sync, y_sync));
        Ok(())
    }

    /// bf16 inputs, f32 output, strided into a logit matrix.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_f32_strided(
        &self,
        x: &Bf16Buf,
        w: &Bf16Buf,
        y: &mut F32Buf,
        rows: usize,
        tile_out: usize,
        in_features: usize,
        ldc: usize,
        col: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || tile_out == 0 {
            return Ok(());
        }
        let x_v = x.view(rows * in_features)?;
        let w_v = w.view(tile_out * in_features)?;
        let mut y_v = y.view_mut(rows * ldc)?;
        let (w_ptr, w_sync) = w_v.device_ptr(&self.stream);
        let (x_ptr, x_sync) = x_v.device_ptr(&self.stream);
        let (y_ptr, y_sync) = y_v.device_ptr_mut(&self.stream);
        let y_off = y_ptr + (col * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
        self.gemm_ex_raw(
            w_ptr,
            x_ptr,
            y_off,
            rows,
            tile_out,
            in_features,
            ldc,
            cudaDataType_t::CUDA_R_16BF,
            cudaDataType_t::CUDA_R_32F,
        )?;
        drop((w_sync, x_sync, y_sync));
        Ok(())
    }

    /// Fused dequant GEMV. `Y = X @ dequant(Q).T` with f32 activations.
    #[allow(clippy::too_many_arguments)]
    pub fn w8a_gemv_f32(
        &self,
        y: &mut F32Buf,
        x: &F32Buf,
        q: &I8Buf,
        scales: &F32Buf,
        rows: usize,
        out_features: usize,
        in_features: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || out_features == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "gemv rows")?;
        let n_i = as_i32(out_features, "gemv n")?;
        let k_i = as_i32(in_features, "gemv k")?;
        let mut y_v = y.view_mut(rows * out_features)?;
        let x_v = x.view(rows * in_features)?;
        let q_v = q.view(out_features * in_features)?;
        let s_v = scales.view(out_features)?;
        let cfg = row_cfg(out_features, 128)?;
        unsafe {
            self.stream
                .launch_builder(&self.w8a_gemv_f32)
                .arg(&mut y_v)
                .arg(&x_v)
                .arg(&q_v)
                .arg(&s_v)
                .arg(&rows_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Fused dequant GEMV with bf16 activations and a bf16 result.
    #[allow(clippy::too_many_arguments)]
    pub fn w8a_gemv_bf16(
        &self,
        y: &mut Bf16Buf,
        x: &Bf16Buf,
        q: &I8Buf,
        scales: &F32Buf,
        rows: usize,
        out_features: usize,
        in_features: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || out_features == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "gemv rows")?;
        let n_i = as_i32(out_features, "gemv n")?;
        let k_i = as_i32(in_features, "gemv k")?;
        let mut y_v = y.view_mut(rows * out_features)?;
        let x_v = x.view(rows * in_features)?;
        let q_v = q.view(out_features * in_features)?;
        let s_v = scales.view(out_features)?;
        let cfg = row_cfg(out_features, 128)?;
        unsafe {
            self.stream
                .launch_builder(&self.w8a_gemv_bf16)
                .arg(&mut y_v)
                .arg(&x_v)
                .arg(&q_v)
                .arg(&s_v)
                .arg(&rows_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Fused dequant GEMV with bf16 activations and an f32 result (logits).
    #[allow(clippy::too_many_arguments)]
    pub fn w8a_gemv_bf16_f32(
        &self,
        y: &mut F32Buf,
        x: &Bf16Buf,
        q: &I8Buf,
        scales: &F32Buf,
        rows: usize,
        out_features: usize,
        in_features: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || out_features == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "gemv rows")?;
        let n_i = as_i32(out_features, "gemv n")?;
        let k_i = as_i32(in_features, "gemv k")?;
        let mut y_v = y.view_mut(rows * out_features)?;
        let x_v = x.view(rows * in_features)?;
        let q_v = q.view(out_features * in_features)?;
        let s_v = scales.view(out_features)?;
        let cfg = row_cfg(out_features, 128)?;
        unsafe {
            self.stream
                .launch_builder(&self.w8a_gemv_bf16_f32)
                .arg(&mut y_v)
                .arg(&x_v)
                .arg(&q_v)
                .arg(&s_v)
                .arg(&rows_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Dequantize output rows `[src_row, src_row+rows)` into a contiguous f32 tile.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_i8_f32(
        &self,
        dst: &mut F32Buf,
        q: &I8Buf,
        scales: &F32Buf,
        src_row: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "dequant rows")?;
        let cols_i = as_i32(cols, "dequant cols")?;
        let src_i = as_i32(src_row, "dequant src")?;
        let mut dst_v = dst.view_mut(rows * cols)?;
        let q_v = q.view((src_row + rows) * cols)?;
        let s_v = scales.view(src_row + rows)?;
        let cfg = row_cfg(rows, 128)?;
        unsafe {
            self.stream
                .launch_builder(&self.dequant_i8_f32)
                .arg(&mut dst_v)
                .arg(&q_v)
                .arg(&s_v)
                .arg(&rows_i)
                .arg(&cols_i)
                .arg(&src_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Dequantize output rows `[src_row, src_row+rows)` into a contiguous bf16 tile.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_i8_bf16(
        &self,
        dst: &mut Bf16Buf,
        q: &I8Buf,
        scales: &F32Buf,
        src_row: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "dequant rows")?;
        let cols_i = as_i32(cols, "dequant cols")?;
        let src_i = as_i32(src_row, "dequant src")?;
        let mut dst_v = dst.view_mut(rows * cols)?;
        let q_v = q.view((src_row + rows) * cols)?;
        let s_v = scales.view(src_row + rows)?;
        let cfg = row_cfg(rows, 128)?;
        unsafe {
            self.stream
                .launch_builder(&self.dequant_i8_bf16)
                .arg(&mut dst_v)
                .arg(&q_v)
                .arg(&s_v)
                .arg(&rows_i)
                .arg(&cols_i)
                .arg(&src_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Gather INT8 embedding rows into f32, scaling each row by its channel scale.
    pub fn embed_i8_f32(
        &self,
        out: &mut F32Buf,
        table: &I8Buf,
        scales: &F32Buf,
        tokens: &I32Buf,
        rows: usize,
        hidden: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "embed rows")?;
        let hidden_i = as_i32(hidden, "embed hidden")?;
        let mut out_v = out.view_mut(rows * hidden)?;
        let table_v = table.view(table.cap)?;
        let scale_v = scales.view(scales.cap)?;
        let tok_v = tokens.view(rows)?;
        let cfg = row_cfg(rows, 128)?;
        unsafe {
            self.stream
                .launch_builder(&self.embed_i8_f32)
                .arg(&mut out_v)
                .arg(&table_v)
                .arg(&scale_v)
                .arg(&tok_v)
                .arg(&rows_i)
                .arg(&hidden_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Gather INT8 embedding rows into bf16.
    pub fn embed_i8_bf16(
        &self,
        out: &mut Bf16Buf,
        table: &I8Buf,
        scales: &F32Buf,
        tokens: &I32Buf,
        rows: usize,
        hidden: usize,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        let rows_i = as_i32(rows, "embed rows")?;
        let hidden_i = as_i32(hidden, "embed hidden")?;
        let mut out_v = out.view_mut(rows * hidden)?;
        let table_v = table.view(table.cap)?;
        let scale_v = scales.view(scales.cap)?;
        let tok_v = tokens.view(rows)?;
        let cfg = row_cfg(rows, 128)?;
        unsafe {
            self.stream
                .launch_builder(&self.embed_i8_bf16)
                .arg(&mut out_v)
                .arg(&table_v)
                .arg(&scale_v)
                .arg(&tok_v)
                .arg(&rows_i)
                .arg(&hidden_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_ex_raw(
        &self,
        w_ptr: cudarc::driver::sys::CUdeviceptr,
        x_ptr: cudarc::driver::sys::CUdeviceptr,
        y_ptr: cudarc::driver::sys::CUdeviceptr,
        rows: usize,
        out_features: usize,
        in_features: usize,
        ldc: usize,
        ab_type: cudaDataType_t,
        c_type: cudaDataType_t,
    ) -> Result<(), CudaError> {
        let m = as_i32(out_features, "gemm n")?;
        let n = as_i32(rows, "gemm m")?;
        let k = as_i32(in_features, "gemm k")?;
        let ldc_i = as_i32(ldc, "gemm ldc")?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                (&alpha) as *const f32 as *const c_void,
                w_ptr as *const c_void,
                ab_type,
                k,
                x_ptr as *const c_void,
                ab_type,
                k,
                (&beta) as *const f32 as *const c_void,
                y_ptr as *mut c_void,
                c_type,
                ldc_i,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )?;
        }
        Ok(())
    }

    /// Write each row of bf16 `k` and `v` into the paged cache.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_bf16(
        &self,
        key_cache: &mut Bf16Buf,
        value_cache: &mut Bf16Buf,
        k: &Bf16Buf,
        v: &Bf16Buf,
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
                .launch_builder(&self.write_kv_bf16)
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

    /// Batched bf16 decode attention. `splits > 1` writes f32 partials that
    /// [`Self::decode_combine_bf16`] merges. `splits == 1` writes `out` directly.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_bf16(
        &self,
        out: &mut Bf16Buf,
        partial_acc: &mut F32Buf,
        partial_max: &mut F32Buf,
        partial_sum: &mut F32Buf,
        query: &Bf16Buf,
        key_cache: &Bf16Buf,
        value_cache: &Bf16Buf,
        block_tables: &I32Buf,
        table_offsets: &I32Buf,
        seq_lens: &I32Buf,
        row_index: &I32Buf,
        batch: usize,
        splits: i32,
        shape: AttnShape,
    ) -> Result<(), CudaError> {
        if batch == 0 {
            return Ok(());
        }
        let batch_i = as_i32(batch, "decode batch")?;
        let out_n = out.cap;
        let key_n = key_cache.cap;
        let val_n = value_cache.cap;
        let table_n = block_tables.cap;
        let q_n = query.cap;
        let mut out_v = out.view_mut(out_n)?;
        let mut acc_v = partial_acc.view_mut(partial_acc.cap.max(1))?;
        let mut max_v = partial_max.view_mut(partial_max.cap.max(1))?;
        let mut sum_v = partial_sum.view_mut(partial_sum.cap.max(1))?;
        let q_v = query.view(q_n)?;
        let k_v = key_cache.view(key_n)?;
        let v_v = value_cache.view(val_n)?;
        let tables = block_tables.view(table_n)?;
        let offs = table_offsets.view(batch)?;
        let lens = seq_lens.view(batch)?;
        let rows = row_index.view(batch)?;
        let grid = shape.num_qo_heads as usize * batch * splits.max(1) as usize;
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.paged_decode_bf16)
                .arg(&mut out_v)
                .arg(&mut acc_v)
                .arg(&mut max_v)
                .arg(&mut sum_v)
                .arg(&q_v)
                .arg(&k_v)
                .arg(&v_v)
                .arg(&tables)
                .arg(&offs)
                .arg(&lens)
                .arg(&rows)
                .arg(&batch_i)
                .arg(&splits)
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

    /// Merge split-K bf16 decode partials into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_combine_bf16(
        &self,
        out: &mut Bf16Buf,
        partial_acc: &F32Buf,
        partial_max: &F32Buf,
        partial_sum: &F32Buf,
        row_index: &I32Buf,
        batch: usize,
        splits: i32,
        num_qo_heads: i32,
        head_dim: i32,
    ) -> Result<(), CudaError> {
        if batch == 0 || splits <= 1 {
            return Ok(());
        }
        let batch_i = as_i32(batch, "combine batch")?;
        let out_n = out.cap;
        let mut out_v = out.view_mut(out_n)?;
        let acc_v = partial_acc.view(partial_acc.cap)?;
        let max_v = partial_max.view(partial_max.cap)?;
        let sum_v = partial_sum.view(partial_sum.cap)?;
        let rows = row_index.view(batch)?;
        let grid = num_qo_heads as usize * batch;
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.decode_combine_bf16)
                .arg(&mut out_v)
                .arg(&acc_v)
                .arg(&max_v)
                .arg(&sum_v)
                .arg(&rows)
                .arg(&batch_i)
                .arg(&splits)
                .arg(&num_qo_heads)
                .arg(&head_dim)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// One query token of a bf16 prefill chunk. Same layout as the f32 kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_prefill_bf16(
        &self,
        out: &mut Bf16Buf,
        query: &Bf16Buf,
        key_cache: &Bf16Buf,
        value_cache: &Bf16Buf,
        block_tables: &I32Buf,
        table_offset: i32,
        row0: i32,
        chunk: i32,
        position: i32,
        shape: AttnShape,
    ) -> Result<(), CudaError> {
        if chunk <= 0 {
            return Ok(());
        }
        let out_n = out.cap;
        let key_n = key_cache.cap;
        let val_n = value_cache.cap;
        let table_n = block_tables.cap;
        let q_n = query.cap;
        let mut out_v = out.view_mut(out_n)?;
        let q_v = query.view(q_n)?;
        let k_v = key_cache.view(key_n)?;
        let v_v = value_cache.view(val_n)?;
        let tables = block_tables.view(table_n)?;
        let total = shape.num_qo_heads as usize * chunk as usize;
        let cfg = LaunchConfig {
            grid_dim: (total as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.paged_prefill_bf16)
                .arg(&mut out_v)
                .arg(&q_v)
                .arg(&k_v)
                .arg(&v_v)
                .arg(&tables)
                .arg(&table_offset)
                .arg(&row0)
                .arg(&chunk)
                .arg(&position)
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

    /// Record launches until [`Self::end_capture`]. The recorded work does not run.
    pub fn begin_capture(&self) -> Result<(), CudaError> {
        if self.stream.cu_stream().is_null() {
            return Err(CudaError::new(
                "CUDA graph capture is not supported on the legacy default stream",
            ));
        }
        // Captured GEMMs read this allocation. Touching it here keeps that
        // dependency visible; dropping it would let cuBLAS allocate instead.
        let _workspace_bytes = self.workspace.len();
        self.stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )?;
        Ok(())
    }

    /// Finish a capture started by [`Self::begin_capture`].
    pub fn end_capture(&self) -> Result<CapturedGraph, CudaError> {
        let graph = self
            .stream
            .end_capture(CUgraphInstantiate_flags(0))?
            .ok_or_else(|| CudaError::new("cuda graph capture produced an empty graph"))?;
        Ok(CapturedGraph { inner: graph })
    }

    /// Replay a captured decode step.
    pub fn launch_graph(&self, graph: &CapturedGraph) -> Result<(), CudaError> {
        graph.inner.launch()?;
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

fn row_cfg(rows: usize, block: u32) -> Result<LaunchConfig, CudaError> {
    let rows = u32::try_from(rows)
        .map_err(|_| CudaError::new(format!("launch rows {rows} exceed u32")))?;
    Ok(LaunchConfig {
        grid_dim: (rows.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    })
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
            "flint_residual_rms_f32",
            "flint_add_rms_f32",
            "flint_rms_norm_bf16",
            "flint_rope_bf16",
            "flint_swiglu_bf16",
            "flint_add_bf16",
            "flint_bias_bf16",
            "flint_embed_bf16",
            "flint_gather_rows_bf16",
            "flint_write_kv_bf16",
            "flint_residual_rms_bf16",
            "flint_add_rms_bf16",
            "flint_f32_to_bf16",
            "flint_paged_decode_bf16",
            "flint_decode_combine_bf16",
            "flint_paged_prefill_bf16",
            "flint_w8a_gemv_f32",
            "flint_w8a_gemv_bf16",
            "flint_w8a_gemv_bf16_f32",
            "flint_dequant_i8_f32",
            "flint_dequant_i8_bf16",
            "flint_embed_i8_f32",
            "flint_embed_i8_bf16",
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
