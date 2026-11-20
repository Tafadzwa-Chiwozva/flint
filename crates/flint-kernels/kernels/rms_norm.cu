// Fused RMSNorm.
//
// One CUDA block per row. Threads cooperatively reduce the sum of squares,
// then scale the row by rsqrt(mean + eps) * weight. Phase 2 fuses this with
// the residual add (hidden = residual + norm(x) path is separate today) and
// with the QKV projection that consumes the normalized row.
//
// Launch: flint_rms_norm_launch. rows blocks, 256 threads.

#include <cuda_runtime.h>

namespace {

__device__ float block_sum(float value) {
    constexpr int WARP = 32;
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP;
    int warp = threadIdx.x / WARP;

    unsigned mask = 0xffffffffu;
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    float warp_sum = 0.f;
    if (warp == 0) {
        warp_sum = (lane < (blockDim.x + WARP - 1) / WARP) ? shared[lane] : 0.f;
        for (int offset = WARP / 2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }
    }
    if (threadIdx.x == 0) {
        shared[0] = warp_sum;
    }
    __syncthreads();
    return shared[0];
}

__global__ void flint_rms_norm_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int rows,
    int cols,
    float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* row_x = x + static_cast<long long>(row) * cols;
    float* row_o = out + static_cast<long long>(row) * cols;

    float partial = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row_x[i];
        partial += v * v;
    }
    float sum = block_sum(partial);
    float scale = rsqrtf(sum / static_cast<float>(cols) + eps);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_o[i] = row_x[i] * scale * weight[i];
    }
}

}  // namespace

extern "C" void flint_rms_norm_launch(
    float* out,
    const float* x,
    const float* weight,
    int rows,
    int cols,
    float eps) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    flint_rms_norm_kernel<<<rows, 256>>>(out, x, weight, rows, cols, eps);
    cudaDeviceSynchronize();
}
