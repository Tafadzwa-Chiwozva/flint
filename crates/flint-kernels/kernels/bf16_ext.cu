// bf16 entry points and fused norm kernels. Included by ops.cu so the
// helpers in that file (load_float, attend_aligned, block_sum) stay shared.
// Templates are __device__ functions; the extern "C" wrappers are the launches.

template <typename T>
__device__ void rms_norm_body(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ weight,
    int rows,
    int cols,
    float eps) {
    int row = blockIdx.x;
    if (row >= rows || cols <= 0) {
        return;
    }
    const T* row_x = x + static_cast<long long>(row) * cols;
    T* row_o = out + static_cast<long long>(row) * cols;
    float partial = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = load_float(row_x[i]);
        partial += v * v;
    }
    float sum = block_sum(partial);
    float scale = rsqrtf(sum / static_cast<float>(cols) + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        store_one(&row_o[i], load_float(row_x[i]) * scale * load_float(weight[i]));
    }
}

template <typename T>
__device__ void rope_body(
    T* __restrict__ x,
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
    T* h = x + (static_cast<long long>(row) * n_heads + head) * head_dim;
    float pos = static_cast<float>(positions[row]);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = pos * inv_freq[i];
        float s, c;
        sincosf(angle, &s, &c);
        float x1 = load_float(h[i]);
        float x2 = load_float(h[i + half]);
        store_one(&h[i], x1 * c - x2 * s);
        store_one(&h[i + half], x2 * c + x1 * s);
    }
}

template <typename T>
__device__ void swiglu_body(T* __restrict__ gate, const T* __restrict__ up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float x = load_float(gate[i]);
    float sig = (x >= 0.f) ? (1.f / (1.f + expf(-x))) : (expf(x) / (1.f + expf(x)));
    store_one(&gate[i], x * sig * load_float(up[i]));
}

template <typename T>
__device__ void add_body(T* __restrict__ dst, const T* __restrict__ a, const T* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        store_one(&dst[i], load_float(a[i]) + load_float(b[i]));
    }
}

template <typename T>
__device__ void bias_body(T* __restrict__ y, const T* __restrict__ bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i < n) {
        store_one(&y[i], load_float(y[i]) + load_float(bias[i % cols]));
    }
}

template <typename T>
__device__ void embed_body(
    T* __restrict__ out, const T* __restrict__ table, const int* __restrict__ tokens, int rows, int hidden) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int id = tokens[row];
    const T* src = table + static_cast<long long>(id) * hidden;
    T* dst = out + static_cast<long long>(row) * hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        dst[d] = src[d];
    }
}

template <typename T>
__device__ void gather_body(
    T* __restrict__ dst, const T* __restrict__ src, const int* __restrict__ rows, int n, int cols) {
    int i = blockIdx.x;
    if (i >= n) {
        return;
    }
    int row = rows[i];
    const T* s = src + static_cast<long long>(row) * cols;
    T* d = dst + static_cast<long long>(i) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        d[c] = s[c];
    }
}

template <typename T>
__device__ void write_kv_body(
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const T* __restrict__ k,
    const T* __restrict__ v,
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
    const T* src_k = k + (static_cast<long long>(row) * n_kv + head) * head_dim;
    const T* src_v = v + (static_cast<long long>(row) * n_kv + head) * head_dim;
    T* dst_k =
        key_cache + cache_index(block, layer, head, slot, block_size, num_layers, n_kv, head_dim);
    T* dst_v = value_cache +
               cache_index(block, layer, head, slot, block_size, num_layers, n_kv, head_dim);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst_k[d] = src_k[d];
        dst_v[d] = src_v[d];
    }
}

template <typename T>
__device__ void residual_rms_body(
    T* __restrict__ residual,
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ weight,
    int rows,
    int cols,
    float eps) {
    int row = blockIdx.x;
    if (row >= rows || cols <= 0) {
        return;
    }
    const T* row_x = x + static_cast<long long>(row) * cols;
    T* row_r = residual + static_cast<long long>(row) * cols;
    T* row_o = out + static_cast<long long>(row) * cols;
    float partial = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = load_float(row_x[i]);
        row_r[i] = row_x[i];
        partial += v * v;
    }
    float sum = block_sum(partial);
    float scale = rsqrtf(sum / static_cast<float>(cols) + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        store_one(&row_o[i], load_float(row_x[i]) * scale * load_float(weight[i]));
    }
}

template <typename T>
__device__ void add_rms_body(
    T* __restrict__ sum,
    T* __restrict__ normed,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ weight,
    int rows,
    int cols,
    float eps) {
    int row = blockIdx.x;
    if (row >= rows || cols <= 0) {
        return;
    }
    T* row_s = sum + static_cast<long long>(row) * cols;
    T* row_o = normed + static_cast<long long>(row) * cols;
    const T* row_a = a + static_cast<long long>(row) * cols;
    const T* row_b = b + static_cast<long long>(row) * cols;
    float partial = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float s = load_float(row_a[i]) + load_float(row_b[i]);
        store_one(&row_s[i], s);
        partial += s * s;
    }
    float reduced = block_sum(partial);
    float scale = rsqrtf(reduced / static_cast<float>(cols) + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        store_one(&row_o[i], load_float(row_s[i]) * scale * load_float(weight[i]));
    }
}

__device__ void token_span(int seq_len, int sliding_window, int splits, int split, int* t0, int* t1) {
    int start = 0;
    if (sliding_window >= 0 && sliding_window < seq_len) {
        start = seq_len - sliding_window;
    }
    if (seq_len <= 0) {
        *t0 = 0;
        *t1 = 0;
        return;
    }
    if (splits <= 1) {
        *t0 = split == 0 ? start : seq_len;
        *t1 = seq_len;
        return;
    }
    int span = seq_len - start;
    int chunk = (span + splits - 1) / splits;
    int begin = start + split * chunk;
    if (begin > seq_len) {
        begin = seq_len;
    }
    int end = begin + chunk;
    if (end > seq_len) {
        end = seq_len;
    }
    *t0 = begin;
    *t1 = end;
}

template <typename T>
__device__ void attend_scalar_span(
    T* __restrict__ out_head,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    const T* __restrict__ query_head,
    const T* __restrict__ key_cache,
    const T* __restrict__ value_cache,
    const int* __restrict__ table,
    int t0,
    int t1,
    int kv_h,
    int layer,
    int head_dim,
    int block_size,
    int num_layers,
    int num_kv_heads,
    float scale,
    int lane) {
    int per = (head_dim + 31) >> 5;
    float qreg[8];
    float acc[8];
    for (int i = 0; i < per; ++i) {
        int d = lane + i * 32;
        qreg[i] = d < head_dim ? load_float(query_head[d]) : 0.f;
        acc[i] = 0.f;
    }
    if (t1 <= t0) {
        if (partial_max != nullptr && lane == 0) {
            *partial_max = -1e30f;
            *partial_sum = 0.f;
        }
        for (int i = 0; i < per; ++i) {
            int d = lane + i * 32;
            if (d < head_dim) {
                if (partial_acc != nullptr) {
                    partial_acc[d] = 0.f;
                } else if (out_head != nullptr) {
                    store_one(&out_head[d], 0.f);
                }
            }
        }
        return;
    }
    int seen = 0;
    float max_score = 0.f;
    float sum = 0.f;
    for (int t = t0; t < t1; ++t) {
        int block = table[t / block_size];
        int slot = t % block_size;
        const T* key = key_cache + cache_index(
                                       block, layer, kv_h, slot, block_size, num_layers, num_kv_heads,
                                       head_dim);
        const T* value = value_cache + cache_index(
                                           block, layer, kv_h, slot, block_size, num_layers,
                                           num_kv_heads, head_dim);
        float partial = 0.f;
        for (int i = 0; i < per; ++i) {
            int d = lane + i * 32;
            if (d < head_dim) {
                partial += qreg[i] * load_float(key[d]);
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
            float v = d < head_dim ? load_float(value[d]) : 0.f;
            acc[i] = acc[i] * alpha + prob * v;
        }
    }
    if (partial_max != nullptr) {
        if (lane == 0) {
            *partial_max = max_score;
            *partial_sum = sum;
        }
        for (int i = 0; i < per; ++i) {
            int d = lane + i * 32;
            if (d < head_dim) {
                partial_acc[d] = acc[i];
            }
        }
        return;
    }
    float inv = sum == 0.f ? 0.f : 1.f / sum;
    for (int i = 0; i < per; ++i) {
        int d = lane + i * 32;
        if (d < head_dim && out_head != nullptr) {
            store_one(&out_head[d], acc[i] * inv);
        }
    }
}

template <typename T>
__device__ void attend_dispatch(
    T* out_head,
    float* partial_acc,
    float* partial_max,
    float* partial_sum,
    const T* query_head,
    const T* key_cache,
    const T* value_cache,
    const int* table,
    int t0,
    int t1,
    int kv_h,
    int layer,
    int head_dim,
    int block_size,
    int num_layers,
    int num_kv_heads,
    float scale,
    int lane) {
    if ((head_dim & 31) == 0 && head_dim > 0 && head_dim <= 256) {
        attend_aligned<T>(
            out_head, partial_acc, partial_max, partial_sum, query_head, key_cache, value_cache, table,
            t0, t1, kv_h, layer, head_dim, block_size, num_layers, num_kv_heads, scale, lane);
    } else {
        attend_scalar_span<T>(
            out_head, partial_acc, partial_max, partial_sum, query_head, key_cache, value_cache, table,
            t0, t1, kv_h, layer, head_dim, block_size, num_layers, num_kv_heads, scale, lane);
    }
}

extern "C" __global__ void flint_rms_norm_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight, int rows, int cols, float eps) {
    rms_norm_body<__nv_bfloat16>(out, x, weight, rows, cols, eps);
}

extern "C" __global__ void flint_rope_bf16(
    __nv_bfloat16* x, const int* positions, const float* inv_freq, int rows, int n_heads, int head_dim) {
    rope_body<__nv_bfloat16>(x, positions, inv_freq, rows, n_heads, head_dim);
}

extern "C" __global__ void flint_swiglu_bf16(__nv_bfloat16* gate, const __nv_bfloat16* up, int n) {
    swiglu_body<__nv_bfloat16>(gate, up, n);
}

extern "C" __global__ void flint_add_bf16(
    __nv_bfloat16* dst, const __nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    add_body<__nv_bfloat16>(dst, a, b, n);
}

extern "C" __global__ void flint_bias_bf16(__nv_bfloat16* y, const __nv_bfloat16* bias, int rows, int cols) {
    bias_body<__nv_bfloat16>(y, bias, rows, cols);
}

extern "C" __global__ void flint_embed_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* table, const int* tokens, int rows, int hidden) {
    embed_body<__nv_bfloat16>(out, table, tokens, rows, hidden);
}

extern "C" __global__ void flint_gather_rows_bf16(
    __nv_bfloat16* dst, const __nv_bfloat16* src, const int* rows, int n, int cols) {
    gather_body<__nv_bfloat16>(dst, src, rows, n, cols);
}

extern "C" __global__ void flint_write_kv_bf16(
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const int* positions,
    const int* table_offsets,
    const int* block_tables,
    int rows,
    int layer,
    int n_kv,
    int head_dim,
    int block_size,
    int num_layers) {
    write_kv_body<__nv_bfloat16>(
        key_cache, value_cache, k, v, positions, table_offsets, block_tables, rows, layer, n_kv,
        head_dim, block_size, num_layers);
}

extern "C" __global__ void flint_residual_rms_f32(
    float* residual, float* out, const float* x, const float* weight, int rows, int cols, float eps) {
    residual_rms_body<float>(residual, out, x, weight, rows, cols, eps);
}

extern "C" __global__ void flint_residual_rms_bf16(
    __nv_bfloat16* residual,
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    int rows,
    int cols,
    float eps) {
    residual_rms_body<__nv_bfloat16>(residual, out, x, weight, rows, cols, eps);
}

extern "C" __global__ void flint_add_rms_f32(
    float* sum,
    float* normed,
    const float* a,
    const float* b,
    const float* weight,
    int rows,
    int cols,
    float eps) {
    add_rms_body<float>(sum, normed, a, b, weight, rows, cols, eps);
}

extern "C" __global__ void flint_add_rms_bf16(
    __nv_bfloat16* sum,
    __nv_bfloat16* normed,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* weight,
    int rows,
    int cols,
    float eps) {
    add_rms_body<__nv_bfloat16>(sum, normed, a, b, weight, rows, cols, eps);
}

extern "C" __global__ void flint_f32_to_bf16(__nv_bfloat16* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

extern "C" __global__ void flint_paged_decode_bf16(
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ table_offsets,
    const int* __restrict__ seq_lens,
    const int* __restrict__ row_index,
    int batch,
    int splits,
    int layer,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_layers,
    float scale,
    int sliding_window) {
    int index = blockIdx.x;
    int group = num_qo_heads * batch;
    int total = group * splits;
    if (index >= total || num_kv_heads <= 0 || splits <= 0) {
        return;
    }
    int split = index / group;
    int local = index - split * group;
    int qh = local % num_qo_heads;
    int b = local / num_qo_heads;
    int groups = num_qo_heads / num_kv_heads;
    int kv_h = qh / groups;
    int row = row_index[b];
    const __nv_bfloat16* q = query + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    __nv_bfloat16* o = out + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    const int* table = block_tables + table_offsets[b];
    int t0, t1;
    token_span(seq_lens[b], sliding_window, splits, split, &t0, &t1);
    float* acc = nullptr;
    float* mx = nullptr;
    float* sm = nullptr;
    __nv_bfloat16* dest = o;
    if (splits > 1) {
        long long base = (static_cast<long long>(split) * batch + b) * num_qo_heads + qh;
        acc = partial_acc + base * head_dim;
        mx = partial_max + base;
        sm = partial_sum + base;
        dest = nullptr;
    }
    attend_dispatch<__nv_bfloat16>(
        dest, acc, mx, sm, q, key_cache, value_cache, table, t0, t1, kv_h, layer, head_dim, block_size,
        num_layers, num_kv_heads, scale, threadIdx.x);
}

extern "C" __global__ void flint_decode_combine_bf16(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const int* __restrict__ row_index,
    int batch,
    int splits,
    int num_qo_heads,
    int head_dim) {
    int index = blockIdx.x;
    int total = num_qo_heads * batch;
    if (index >= total || splits <= 0) {
        return;
    }
    int qh = index % num_qo_heads;
    int b = index / num_qo_heads;
    int row = row_index[b];
    int lane = threadIdx.x;
    float mx = -1e30f;
    for (int s = 0; s < splits; ++s) {
        long long base = (static_cast<long long>(s) * batch + b) * num_qo_heads + qh;
        mx = fmaxf(mx, partial_max[base]);
    }
    float sum = 0.f;
    for (int s = 0; s < splits; ++s) {
        long long base = (static_cast<long long>(s) * batch + b) * num_qo_heads + qh;
        sum += partial_sum[base] * expf(partial_max[base] - mx);
    }
    float inv = sum == 0.f ? 0.f : 1.f / sum;
    __nv_bfloat16* o = out + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    for (int d = lane; d < head_dim; d += 32) {
        float acc = 0.f;
        for (int s = 0; s < splits; ++s) {
            long long base = (static_cast<long long>(s) * batch + b) * num_qo_heads + qh;
            float rescale = expf(partial_max[base] - mx);
            acc += partial_acc[base * head_dim + d] * rescale;
        }
        o[d] = __float2bfloat16(acc * inv);
    }
}

extern "C" __global__ void flint_paged_prefill_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
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
    int groups = num_qo_heads / num_kv_heads;
    int kv_h = qh / groups;
    int row = row0 + q_i;
    const __nv_bfloat16* q = query + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    __nv_bfloat16* o = out + (static_cast<long long>(row) * num_qo_heads + qh) * head_dim;
    const int* table = block_tables + table_offset;
    int seq_len = position + q_i + 1;
    int t0, t1;
    token_span(seq_len, sliding_window, 1, 0, &t0, &t1);
    attend_dispatch<__nv_bfloat16>(
        o, nullptr, nullptr, nullptr, q, key_cache, value_cache, table, t0, t1, kv_h, layer, head_dim,
        block_size, num_layers, num_kv_heads, scale, threadIdx.x);
}
