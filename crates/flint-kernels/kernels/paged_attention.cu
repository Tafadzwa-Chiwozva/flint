// Paged-attention scaffold.
//
// Layout matches CacheGeometry::offset in src/cpu.rs:
//   index = (((block * num_layers + layer) * num_kv_heads + head) * block_size + slot) * head_dim
// Keys and values are separate device buffers.
//
// This kernel is a correct, readable baseline: one block per query head, one
// thread walks the block table, computes softmax, and writes the head output.
// It is intentionally not the phase-2 kernel. The production kernel should:
//
//   1. Online softmax (FlashAttention-style) so the score vector is never
//      materialized and long contexts stay in registers/shared memory.
//   2. One warp per query head, vectorized 128-bit loads of K and V.
//   3. A grid of (query_heads, sequences_in_batch) so a whole continuous-batch
//      iteration is one launch. Block tables are gathered from a ragged
//      tensor of shape [batch, max_blocks].
//   4. Optional sliding-window early exit, already plumbed through
//      `sliding_window` (-1 disables it).
//   5. Later: split-K for very long sequences and FP8 KV pages.
//
// GQA mapping matches the CPU kernel: query head h reads KV head
// h / (num_qo_heads / num_kv_heads).

#include <cuda_runtime.h>

namespace {

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
    return block * block_stride + static_cast<long long>(layer) * layer_stride +
           static_cast<long long>(kv_h) * head_stride + static_cast<long long>(slot) * head_dim;
}

__global__ void flint_paged_attention_kernel(
    float* __restrict__ out,
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int* __restrict__ block_table,
    int layer,
    int seq_len,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_layers,
    float scale,
    int sliding_window) {
    int qh = blockIdx.x;
    if (qh >= num_qo_heads || seq_len <= 0) {
        return;
    }
    int group = num_qo_heads / num_kv_heads;
    int kv_h = qh / group;
    const float* q = query + static_cast<long long>(qh) * head_dim;
    float* o = out + static_cast<long long>(qh) * head_dim;

    int start = 0;
    if (sliding_window >= 0 && sliding_window < seq_len) {
        start = seq_len - sliding_window;
    }

    // Single-thread baseline. The score buffer lives in local memory; the
    // flash rewrite will not allocate it.
    // seq_len for phase-1 GPU bring-up is expected to be modest. A hard cap
    // keeps the stub from blowing the local-memory budget if mis-launched.
    constexpr int kMaxSeq = 4096;
    float scores[kMaxSeq];
    int n = seq_len < kMaxSeq ? seq_len : kMaxSeq;
    if (start >= n) {
        start = 0;
    }

    float max_score = -INFINITY;
    for (int t = start; t < n; ++t) {
        int block = block_table[t / block_size];
        int slot = t % block_size;
        const float* key = key_cache + cache_index(
                                           block, layer, kv_h, slot, block_size, num_layers,
                                           num_kv_heads, head_dim);
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * key[d];
        }
        float score = dot * scale;
        scores[t] = score;
        max_score = fmaxf(max_score, score);
    }

    float sum = 0.f;
    for (int t = start; t < n; ++t) {
        float e = expf(scores[t] - max_score);
        scores[t] = e;
        sum += e;
    }
    float inv = sum == 0.f ? 0.f : 1.f / sum;

    for (int d = 0; d < head_dim; ++d) {
        o[d] = 0.f;
    }
    for (int t = start; t < n; ++t) {
        int block = block_table[t / block_size];
        int slot = t % block_size;
        const float* value = value_cache + cache_index(
                                             block, layer, kv_h, slot, block_size, num_layers,
                                             num_kv_heads, head_dim);
        float prob = scores[t] * inv;
        for (int d = 0; d < head_dim; ++d) {
            o[d] += prob * value[d];
        }
    }
}

}  // namespace

extern "C" void flint_paged_attention_launch(
    float* out,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_table,
    int layer,
    int seq_len,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_layers,
    float scale,
    int sliding_window) {
    if (num_qo_heads <= 0) {
        return;
    }
    flint_paged_attention_kernel<<<num_qo_heads, 1>>>(
        out,
        query,
        key_cache,
        value_cache,
        block_table,
        layer,
        seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        block_size,
        num_layers,
        scale,
        sliding_window);
    cudaDeviceSynchronize();
}
