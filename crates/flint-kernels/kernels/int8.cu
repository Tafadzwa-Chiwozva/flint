// INT8 weight-only matmul and the prefill dequant used before cuBLAS.
//
// Weights are row-major `[out, in]` int8 with one f32 scale per output row.
// The product is scale * sum(activation * q), accumulated in f32. Included by
// ops.cu so load_float, store_one, block_sum, and warp_sum stay shared.
//
// The GEMV reloads each int8 row for every group of four batch rows. That is
// fine for a decode of 8 or fewer. Wider batches use the weight-stationary
// GEMM below: one warp per output column. When the int8 rows fit, the block
// stages them once and reuses an 8-row activation tile. `w8a_plan` in cuda.rs
// sizes that shared memory.

template <typename In>
__device__ void accum_i8x4(
    const In* __restrict__ x,
    int row,
    int k,
    int base,
    float w0,
    float w1,
    float w2,
    float w3,
    float& acc) {
    const In* ptr = x + static_cast<long long>(row) * k + base;
    acc += load_float(ptr[0]) * w0;
    acc += load_float(ptr[1]) * w1;
    acc += load_float(ptr[2]) * w2;
    acc += load_float(ptr[3]) * w3;
}

// One block per output channel. Batch rows are consumed four at a time so the
// int8 row is reused across the decode batch without a 64-wide register file.
template <typename In, typename Out>
__device__ void flint_w8a_gemv(
    Out* __restrict__ y,
    const In* __restrict__ x,
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    int m,
    int n,
    int k) {
    const int col = static_cast<int>(blockIdx.x);
    if (col >= n || m <= 0 || k <= 0) {
        return;
    }
    const int8_t* wrow = q + static_cast<long long>(col) * k;
    const float scale = scales[col];
    for (int row0 = 0; row0 < m; row0 += 4) {
        const int rows_here = (m - row0) < 4 ? (m - row0) : 4;
        float acc0 = 0.f;
        float acc1 = 0.f;
        float acc2 = 0.f;
        float acc3 = 0.f;
        if ((k & 3) == 0) {
            const int k4 = k >> 2;
            for (int t = static_cast<int>(threadIdx.x); t < k4; t += static_cast<int>(blockDim.x)) {
                const int base = t << 2;
                const int packed = *reinterpret_cast<const int*>(wrow + base);
                const float w0 = static_cast<float>(static_cast<int8_t>(packed & 0xff));
                const float w1 = static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xff));
                const float w2 = static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xff));
                const float w3 = static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xff));
                accum_i8x4(x, row0, k, base, w0, w1, w2, w3, acc0);
                if (rows_here > 1) {
                    accum_i8x4(x, row0 + 1, k, base, w0, w1, w2, w3, acc1);
                }
                if (rows_here > 2) {
                    accum_i8x4(x, row0 + 2, k, base, w0, w1, w2, w3, acc2);
                }
                if (rows_here > 3) {
                    accum_i8x4(x, row0 + 3, k, base, w0, w1, w2, w3, acc3);
                }
            }
        } else {
            for (int t = static_cast<int>(threadIdx.x); t < k; t += static_cast<int>(blockDim.x)) {
                const float w = static_cast<float>(wrow[t]);
                acc0 += load_float(x[static_cast<long long>(row0) * k + t]) * w;
                if (rows_here > 1) {
                    acc1 += load_float(x[static_cast<long long>(row0 + 1) * k + t]) * w;
                }
                if (rows_here > 2) {
                    acc2 += load_float(x[static_cast<long long>(row0 + 2) * k + t]) * w;
                }
                if (rows_here > 3) {
                    acc3 += load_float(x[static_cast<long long>(row0 + 3) * k + t]) * w;
                }
            }
        }
        const float s0 = block_sum(acc0);
        const float s1 = rows_here > 1 ? block_sum(acc1) : 0.f;
        const float s2 = rows_here > 2 ? block_sum(acc2) : 0.f;
        const float s3 = rows_here > 3 ? block_sum(acc3) : 0.f;
        if (threadIdx.x == 0) {
            store_one(y + static_cast<long long>(row0) * n + col, s0 * scale);
            if (rows_here > 1) {
                store_one(y + static_cast<long long>(row0 + 1) * n + col, s1 * scale);
            }
            if (rows_here > 2) {
                store_one(y + static_cast<long long>(row0 + 2) * n + col, s2 * scale);
            }
            if (rows_here > 3) {
                store_one(y + static_cast<long long>(row0 + 3) * n + col, s3 * scale);
            }
        }
        __syncthreads();
    }
}

template <typename Out>
__device__ void flint_dequant_i8(
    Out* __restrict__ dst,
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    int rows,
    int cols,
    int src_row) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows || cols <= 0) {
        return;
    }
    const float scale = scales[src_row + row];
    const int8_t* src = q + (static_cast<long long>(src_row) + row) * cols;
    Out* out = dst + static_cast<long long>(row) * cols;
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += static_cast<int>(blockDim.x)) {
        store_one(out + col, static_cast<float>(src[col]) * scale);
    }
}

template <typename Out>
__device__ void flint_embed_i8(
    Out* __restrict__ out,
    const int8_t* __restrict__ table,
    const float* __restrict__ scales,
    const int* __restrict__ tokens,
    int rows,
    int hidden) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows || hidden <= 0) {
        return;
    }
    const int id = tokens[row];
    const float scale = scales[id];
    const int8_t* src = table + static_cast<long long>(id) * hidden;
    Out* dst = out + static_cast<long long>(row) * hidden;
    for (int col = static_cast<int>(threadIdx.x); col < hidden; col += static_cast<int>(blockDim.x)) {
        store_one(dst + col, static_cast<float>(src[col]) * scale);
    }
}

extern "C" __global__ void flint_w8a_gemv_f32(
    float* y, const float* x, const int8_t* q, const float* scales, int m, int n, int k) {
    flint_w8a_gemv<float, float>(y, x, q, scales, m, n, k);
}

extern "C" __global__ void flint_w8a_gemv_bf16(
    __nv_bfloat16* y, const __nv_bfloat16* x, const int8_t* q, const float* scales, int m, int n, int k) {
    flint_w8a_gemv<__nv_bfloat16, __nv_bfloat16>(y, x, q, scales, m, n, k);
}

extern "C" __global__ void flint_w8a_gemv_bf16_f32(
    float* y, const __nv_bfloat16* x, const int8_t* q, const float* scales, int m, int n, int k) {
    flint_w8a_gemv<__nv_bfloat16, float>(y, x, q, scales, m, n, k);
}

extern "C" __global__ void flint_dequant_i8_f32(
    float* dst, const int8_t* q, const float* scales, int rows, int cols, int src_row) {
    flint_dequant_i8<float>(dst, q, scales, rows, cols, src_row);
}

extern "C" __global__ void flint_dequant_i8_bf16(
    __nv_bfloat16* dst, const int8_t* q, const float* scales, int rows, int cols, int src_row) {
    flint_dequant_i8<__nv_bfloat16>(dst, q, scales, rows, cols, src_row);
}

extern "C" __global__ void flint_embed_i8_f32(
    float* out, const int8_t* table, const float* scales, const int* tokens, int rows, int hidden) {
    flint_embed_i8<float>(out, table, scales, tokens, rows, hidden);
}

extern "C" __global__ void flint_embed_i8_bf16(
    __nv_bfloat16* out, const int8_t* table, const float* scales, const int* tokens, int rows, int hidden) {
    flint_embed_i8<__nv_bfloat16>(out, table, scales, tokens, rows, hidden);
}

// Dot one weight-tile against the staged activation rows. `rows_here == 8` is
// the steady path (decode 32 and the c=32 arrival prefill are multiples of 8).
template <int Rows>
__device__ void accum_wtile(
    float acc[8],
    const float* __restrict__ xs,
    int bk,
    int rows_here,
    const int8_t* __restrict__ src,
    int bk_here,
    int lane) {
    const bool vec = (bk_here & 3) == 0 && (reinterpret_cast<uintptr_t>(src) & 3u) == 0;
    if (vec) {
        const int groups = bk_here >> 2;
        for (int t = lane; t < groups; t += 32) {
            const int base = t << 2;
            const int packed = *reinterpret_cast<const int*>(src + base);
            const float w0 = static_cast<float>(static_cast<int8_t>(packed & 0xff));
            const float w1 = static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xff));
            const float w2 = static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xff));
            const float w3 = static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xff));
            if (rows_here == 8) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float* row = xs + r * bk + base;
                    acc[r] += row[0] * w0 + row[1] * w1 + row[2] * w2 + row[3] * w3;
                }
            } else {
                for (int r = 0; r < rows_here && r < Rows; ++r) {
                    const float* row = xs + r * bk + base;
                    acc[r] += row[0] * w0 + row[1] * w1 + row[2] * w2 + row[3] * w3;
                }
            }
        }
        return;
    }
    for (int t = lane; t < bk_here; t += 32) {
        const float w = static_cast<float>(src[t]);
        if (rows_here == 8) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[r] += xs[r * bk + t] * w;
            }
        } else {
            for (int r = 0; r < rows_here && r < Rows; ++r) {
                acc[r] += xs[r * bk + t] * w;
            }
        }
    }
}

// One warp per output column. When `wbytes` is non-zero the block stages every
// warp's int8 row once, then an 8-row activation tile (`bk` columns) in the
// rest of dynamic shared memory. A zero `wbytes` means the row did not fit, so
// each row tile re-reads the int8 row. Launch with block (32, warps) and grid
// ceil(n / warps). Captured decode widths above the GEMV cutoff come through
// here and never call cuBLAS.
template <typename In, typename Out>
__device__ void flint_w8a_gemm(
    Out* __restrict__ y,
    const In* __restrict__ x,
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    int m,
    int n,
    int k,
    int bk,
    int wbytes,
    char* __restrict__ smem) {
    if (m <= 0 || n <= 0 || k <= 0 || bk <= 0) {
        return;
    }
    int8_t* wsmem = reinterpret_cast<int8_t*>(smem);
    float* xs = reinterpret_cast<float*>(smem + wbytes);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int warps = static_cast<int>(blockDim.y);
    const int col = static_cast<int>(blockIdx.x) * warps + warp;
    const bool active = col < n;
    const float scale = active ? scales[col] : 0.f;
    const int8_t* wrow = active ? q + static_cast<long long>(col) * k : nullptr;
    const int tid = warp * static_cast<int>(blockDim.x) + lane;
    const int threads = static_cast<int>(blockDim.x) * warps;

    if (wbytes > 0) {
        if (active) {
            int8_t* dst = wsmem + static_cast<long long>(warp) * k;
            for (int t = lane; t < k; t += 32) {
                dst[t] = wrow[t];
            }
        }
        __syncthreads();
    }

    for (int row0 = 0; row0 < m; row0 += 8) {
        const int rows_here = (m - row0) < 8 ? (m - row0) : 8;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int k0 = 0; k0 < k; k0 += bk) {
            const int bk_here = (k - k0) < bk ? (k - k0) : bk;
            for (int r = 0; r < rows_here; ++r) {
                const In* row = x + (static_cast<long long>(row0 + r) * k) + k0;
                for (int c = tid; c < bk_here; c += threads) {
                    xs[r * bk + c] = load_float(row[c]);
                }
            }
            __syncthreads();
            if (active) {
                const int8_t* src = wbytes > 0 ? wsmem + static_cast<long long>(warp) * k + k0
                                                : wrow + k0;
                accum_wtile<8>(acc, xs, bk, rows_here, src, bk_here, lane);
            }
            __syncthreads();
        }
        if (active) {
            for (int r = 0; r < rows_here; ++r) {
                const float sum = warp_sum(acc[r]) * scale;
                if (lane == 0) {
                    store_one(y + static_cast<long long>(row0 + r) * n + col, sum);
                }
            }
        }
    }
}

extern "C" __global__ void flint_w8a_gemm_f32(
    float* y,
    const float* x,
    const int8_t* q,
    const float* scales,
    int m,
    int n,
    int k,
    int bk,
    int wbytes) {
    extern __shared__ __align__(16) char flint_w8a_smem[];
    flint_w8a_gemm<float, float>(y, x, q, scales, m, n, k, bk, wbytes, flint_w8a_smem);
}

extern "C" __global__ void flint_w8a_gemm_bf16(
    __nv_bfloat16* y,
    const __nv_bfloat16* x,
    const int8_t* q,
    const float* scales,
    int m,
    int n,
    int k,
    int bk,
    int wbytes) {
    extern __shared__ __align__(16) char flint_w8a_smem[];
    flint_w8a_gemm<__nv_bfloat16, __nv_bfloat16>(y, x, q, scales, m, n, k, bk, wbytes, flint_w8a_smem);
}

extern "C" __global__ void flint_w8a_gemm_bf16_f32(
    float* y,
    const __nv_bfloat16* x,
    const int8_t* q,
    const float* scales,
    int m,
    int n,
    int k,
    int bk,
    int wbytes) {
    extern __shared__ __align__(16) char flint_w8a_smem[];
    flint_w8a_gemm<__nv_bfloat16, float>(y, x, q, scales, m, n, k, bk, wbytes, flint_w8a_smem);
}
