// Device kernels for the CUDA forward path.
//
// Compiled to PTX by build.rs (`nvcc -ptx`) and loaded with the driver API.
// Layout of the KV cache matches CacheGeometry::offset:
//   index = (((block * num_layers + layer) * num_kv_heads + head) * block_size + slot) * head_dim
//
// Attention is one warp per query head with an online softmax. That keeps the
// score vector out of local memory. It is not a flash-attention tile kernel:
// each query walks its own key range. Prefill launches one warp per query
// token; decode launches one warp per (sequence, query head).

#include <cuda_runtime.h>

namespace {

constexpr int kMaxPerLane = 8; // head_dim <= 256

__device__ float block_sum(float value) {
    constexpr int kWarp = 32;
    __shared__ float shared[32];
    int lane = threadIdx.x % kWarp;
    int warp = threadIdx.x / kWarp;
    unsigned mask = 0xffffffffu;

    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    float warp_sum = 0.f;
    if (warp == 0) {
        int nwarps = (blockDim.x + kWarp - 1) / kWarp;
        warp_sum = (lane < nwarps) ? shared[lane] : 0.f;
        for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }
    }
    if (threadIdx.x == 0) {
        shared[0] = warp_sum;
    }
    __syncthreads();
    return shared[0];
}

__device__ float warp_sum(float value) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return __shfl_sync(mask, value, 0);
}

__device__ long long cache_index(
    int block,
    int layer,
    int kv_h,
    int slot,
    int block_size,
    int num_layers,
    int num_kv_heads,
    int head_dim) {
    long long head_stride = static_cast<long long>(block_size) * head_dim;
    long long layer_stride = static_cast<long long>(num_kv_heads) * head_stride;
    long long block_stride = static_cast<long long>(num_layers) * layer_stride;
    return static_cast<long long>(block) * block_stride +
           static_cast<long long>(layer) * layer_stride +
           static_cast<long long>(kv_h) * head_stride +
           static_cast<long long>(slot) * head_dim;
}

// One query head. `lane` is threadIdx.x in a 32-thread block. `table` points
// at this sequence's block table.
__device__ void attend_one(
    float* __restrict__ out_head,
    const float* __restrict__ query_head,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int* __restrict__ table,
    int seq_len,
    int kv_h,
    int layer,
    int head_dim,
    int block_size,
    int num_layers,
    int num_kv_heads,
    float scale,
    int sliding_window,
    int lane) {
    int per = (head_dim + 31) >> 5;
    if (per > kMaxPerLane || seq_len <= 0) {
        if (per <= kMaxPerLane) {
            for (int i = 0; i < per; ++i) {
                int d = lane + i * 32;
                if (d < head_dim) {
                    out_head[d] = 0.f;
                }
            }
        }
        return;
    }

    float qreg[kMaxPerLane];
    float acc[kMaxPerLane];
    for (int i = 0; i < per; ++i) {
        int d = lane + i * 32;
        qreg[i] = d < head_dim ? query_head[d] : 0.f;
        acc[i] = 0.f;
    }

    int start = 0;
    if (sliding_window >= 0 && sliding_window < seq_len) {
        start = seq_len - sliding_window;
    }

    // `seen` starts false so the first score sets the running max. expf (not
    // the fast approx) stays close to the f32 CPU softmax.
    int seen = 0;
    float max_score = 0.f;
    float sum = 0.f;
    for (int t = start; t < seq_len; ++t) {
        int block = table[t / block_size];
        int slot = t % block_size;
        const float* key = key_cache + cache_index(
                                            block, layer, kv_h, slot, block_size, num_layers,
                                            num_kv_heads, head_dim);
        const float* value = value_cache + cache_index(
                                                block, layer, kv_h, slot, block_size, num_layers,
                                                num_kv_heads, head_dim);
        float partial = 0.f;
        for (int i = 0; i < per; ++i) {
            int d = lane + i * 32;
            if (d < head_dim) {
                partial += qreg[i] * key[d];
            }
        }
        float score = warp_sum(partial) * scale;
        float next_max = seen ? fmaxf(max_score, score) : score;
        float alpha = seen ? expf(max_score - next_max) : 0.f;
        float prob = expf(score - next_max);
        sum = sum * alpha + prob;
        max_score = next_max;
        seen = 1;
        for (int i = 0; i < per; ++i) {
            int d = lane + i * 32;
            float v = d < head_dim ? value[d] : 0.f;
            acc[i] = acc[i] * alpha + prob * v;
        }
    }
    float inv = sum == 0.f ? 0.f : 1.f / sum;
    for (int i = 0; i < per; ++i) {
        int d = lane + i * 32;
        if (d < head_dim) {
            out_head[d] = acc[i] * inv;
        }
    }
}

}  // namespace

extern "C" __global__ void flint_rms_norm(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int rows,
    int cols,
    float eps) {
    int row = blockIdx.x;
    if (row >= rows || cols <= 0) {
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

// In-place Llama/Qwen rotate-half RoPE. inv_freq has length head_dim/2.
// x is [rows, n_heads, head_dim].
extern "C" __global__ void flint_rope(
    float* __restrict__ x,
    const int* __restrict__ positions,
    const float* __restrict__ inv_freq,
    int rows,
    int n_heads,
    int head_dim) {
    int index = blockIdx.x;
    int total = rows * n_heads;
    if (index >= total || head_dim <= 0) {
        return;
    }
    int row = index / n_heads;
    int head = index % n_heads;
    int half = head_dim >> 1;
    float* h = x + (static_cast<long long>(row) * n_heads + head) * head_dim;
    float pos = static_cast<float>(positions[row]);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = pos * inv_freq[i];
        float s, c;
        sincosf(angle, &s, &c);
        float x1 = h[i];
        float x2 = h[i + half];
        h[i] = x1 * c - x2 * s;
        h[i + half] = x2 * c + x1 * s;
    }
}

// In-place SwiGLU: gate = silu(gate) * up.
extern "C" __global__ void flint_swiglu(float* __restrict__ gate, const float* __restrict__ up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float x = gate[i];
    float sig = (x >= 0.f) ? (1.f / (1.f + expf(-x))) : (expf(x) / (1.f + expf(x)));
    gate[i] = x * sig * up[i];
}

extern "C" __global__ void flint_add(
    float* __restrict__ dst,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = a[i] + b[i];
    }
}

extern "C" __global__ void flint_bias(float* __restrict__ y, const float* __restrict__ bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i < n) {
        y[i] += bias[i % cols];
    }
}

extern "C" __global__ void flint_embed(
    float* __restrict__ out,
    const float* __restrict__ table,
    const int* __restrict__ tokens,
    int rows,
    int hidden) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int id = tokens[row];
    const float* src = table + static_cast<long long>(id) * hidden;
    float* dst = out + static_cast<long long>(row) * hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        dst[d] = src[d];
    }
}

extern "C" __global__ void flint_gather_rows(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ rows,
    int n,
    int cols) {
    int i = blockIdx.x;
    if (i >= n) {
        return;
    }
    int row = rows[i];
    const float* s = src + static_cast<long long>(row) * cols;
    float* d = dst + static_cast<long long>(i) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        d[c] = s[c];
    }
}

extern "C" __global__ void flint_write_kv(
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ positions,
    const int* __restrict__ table_offsets,
    const int* __restrict__ block_tables,
    int rows,
    int layer,
    int n_kv,
    int head_dim,
    int block_size,
    int num_layers) {
    int index = blockIdx.x;
    int total = rows * n_kv;
    if (index >= total) {
        return;
    }
    int row = index / n_kv;
    int head = index % n_kv;
    int pos = positions[row];
    int block = block_tables[table_offsets[row] + pos / block_size];
    int slot = pos % block_size;
    const float* src_k = k + (static_cast<long long>(row) * n_kv + head) * head_dim;
    const float* src_v = v + (static_cast<long long>(row) * n_kv + head) * head_dim;
    float* dst_k = key_cache + cache_index(block, layer, head, slot, block_size, num_layers, n_kv, head_dim);
    float* dst_v = value_cache + cache_index(block, layer, head, slot, block_size, num_layers, n_kv, head_dim);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst_k[d] = src_k[d];
        dst_v[d] = src_v[d];
    }
}

// One token per sequence. Grid is a single dimension of num_qo_heads * batch,
// block is one warp.
extern "C" __global__ void flint_paged_decode(
    float* __restrict__ out,
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ table_offsets,
    const int* __restrict__ seq_lens,
    const int* __restrict__ row_index,
    int batch,
    int layer,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_layers,
    float scale,
    int sliding_window) {
    int index = blockIdx.x;
    int total = num_qo_heads * batch;
    if (index >= total || num_kv_heads <= 0) {
        return;
    }
    int qh = index % num_qo_heads;
    int b = index / num_qo_heads;
    int group = num_qo_heads / num_kv_heads;
    int kv_h = qh / group;
    int row = row_index[b];
    const float* q = query + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    float* o = out + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    const int* table = block_tables + table_offsets[b];
    attend_one(
        o,
        q,
        key_cache,
        value_cache,
        table,
        seq_lens[b],
        kv_h,
        layer,
        head_dim,
        block_size,
        num_layers,
        num_kv_heads,
        scale,
        sliding_window,
        threadIdx.x);
}

// Causal prefill of one sequence. Query token q_i (absolute position
// `position + q_i`) attends to tokens [0, position + q_i] inclusive. KV for
// the chunk has already been written.
extern "C" __global__ void flint_paged_prefill(
    float* __restrict__ out,
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    int table_offset,
    int row0,
    int chunk,
    int position,
    int layer,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_layers,
    float scale,
    int sliding_window) {
    int index = blockIdx.x;
    int total = num_qo_heads * chunk;
    if (index >= total || num_kv_heads <= 0) {
        return;
    }
    int qh = index % num_qo_heads;
    int q_i = index / num_qo_heads;
    int group = num_qo_heads / num_kv_heads;
    int kv_h = qh / group;
    int row = row0 + q_i;
    const float* q = query + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    float* o = out + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    const int* table = block_tables + table_offset;
    int seq_len = position + q_i + 1;
    attend_one(
        o,
        q,
        key_cache,
        value_cache,
        table,
        seq_len,
        kv_h,
        layer,
        head_dim,
        block_size,
        num_layers,
        num_kv_heads,
        scale,
        sliding_window,
        threadIdx.x);
}
