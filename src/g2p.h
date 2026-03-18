#pragma once
// g2p_model_cuda.h — CUDA inference for G2P V3 Conformer CTC model.
//
// Same binary format as g2p_model.h, but runs on GPU using Cutlass GEMM for
// linear projections and custom kernels for RMSNorm, RoPE, softmax, SiLU.
//
// Architecture: char_emb → N × (RMSNorm→MHA→RMSNorm→SwiGLU) → upsample → head → CTC decode
//
// Optimizations over naive implementation:
//   - Cutlass GEMM with TF32 TensorOp + SIMT fallback + operator caching
//   - Cutlass batched GEMM for multi-head attention (1 call vs 4 per-head)
//   - Attention scale folded into GEMM alpha (eliminates scale kernel)
//   - Residual adds fused into GEMM via beta=1 (eliminates add kernels)
//   - Softmax batched across all heads (1 launch vs 4)
//
// Usage:
//   G2PModelCuda model;
//   model.load("data/g2p_model.bin", stream);
//   std::string phonemes = model.infer("hello world", stream);

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

// Cutlass GEMM (defined in cutlass_gemm.cu)
extern "C" int cutlass_gemm_tn(int M, int N, int K,
    const float* A, int lda, const float* B, int ldb,
    float* C, int ldc, float alpha, float beta,
    float* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" int cutlass_gemm_nn(int M, int N, int K,
    const float* A, int lda, const float* B, int ldb,
    float* C, int ldc, float alpha, float beta,
    float* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" int cutlass_gemm_batched_tn(int M, int N, int K,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float* C, int ldc, long long strideC,
    int batch_count, float alpha, float beta,
    float* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" int cutlass_gemm_batched_nn(int M, int N, int K,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float* C, int ldc, long long strideC,
    int batch_count, float alpha, float beta,
    float* workspace, size_t workspace_bytes, cudaStream_t stream);

// ── CUDA kernels ────────────────────────────────────────────────────────────

// RMSNorm: out[t,i] = weight[i] * x[t,i] / sqrt(mean(x[t,:]^2) + eps)
__global__ void g2p_rms_norm_kernel(const float* __restrict__ x,
                                      const float* __restrict__ weight,
                                      float* __restrict__ out,
                                      int T, int d, float eps) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float* xt = x + t * d;
    float* ot = out + t * d;

    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        local_sum += xt[i] * xt[i];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv = rsqrtf(sdata[0] / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        ot[i] = weight[i] * xt[i] * inv;
}

// RoPE: apply rotary position embeddings in-place.
// data: column-major [stride, T], where head h occupies rows [h*head_dim..(h+1)*head_dim-1]
__global__ void g2p_rope_kernel(float* __restrict__ data,
                                  const float* __restrict__ cos_table,
                                  const float* __restrict__ sin_table,
                                  int T, int stride, int heads, int head_dim) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= T || h >= heads) return;

    int d2 = head_dim / 2;
    float* base = data + t * stride + h * head_dim;
    const float* rc = cos_table + t * d2;
    const float* rs = sin_table + t * d2;

    for (int i = threadIdx.x; i < d2; i += blockDim.x) {
        float x0 = base[i], x1 = base[d2 + i];
        base[i]      = x0 * rc[i] - x1 * rs[i];
        base[d2 + i] = x1 * rc[i] + x0 * rs[i];
    }
}

// Batched softmax: scores has `num_rows` rows each of length `row_len`.
// In-place: row[i] = softmax(row[i]) for each row.
__global__ void g2p_softmax_kernel(float* __restrict__ scores, int row_len, int num_rows) {
    int t = blockIdx.x;
    if (t >= num_rows) return;
    float* row = scores + t * row_len;
    extern __shared__ float sdata[];

    // Find max
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x)
        local_max = fmaxf(local_max, row[i]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        float v = expf(row[i] - max_val);
        row[i] = v;
        local_sum += v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int i = threadIdx.x; i < row_len; i += blockDim.x)
        row[i] *= inv_sum;
}

// SwiGLU with fused bias on interleaved [2*ff, T] column-major layout.
// Applies bias to both gate and up halves, then SwiGLU, storing result in gate half.
__global__ void g2p_swiglu_bias_kernel(float* __restrict__ data,
                                        const float* __restrict__ bias,
                                        int ff, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * ff) return;
    int t = idx / ff;
    int i = idx % ff;
    int base = t * 2 * ff;
    float g = data[base + i] + bias[i];
    float u = data[base + ff + i] + bias[ff + i];
    data[base + i] = (g / (1.0f + expf(-g))) * u;
}

// Add bias to column-major matrix: data[M, N] += bias[M] (broadcast over columns)
__global__ void g2p_bias_kernel(float* __restrict__ data,
                                  const float* __restrict__ bias,
                                  int M, int N) {
    int col = blockIdx.x;
    if (col >= N) return;
    float* col_data = data + col * M;
    for (int i = threadIdx.x; i < M; i += blockDim.x)
        col_data[i] += bias[i];
}

// Fused QKV bias + RoPE for Q and K.
// QKV is column-major [3*d, T]. Adds bias[3*d] to each column, then applies
// rotary position embeddings to the Q (rows 0..d-1) and K (rows d..2d-1) portions.
__global__ void g2p_qkv_bias_rope_kernel(float* __restrict__ QKV,
                                           const float* __restrict__ bias,
                                           const float* __restrict__ cos_table,
                                           const float* __restrict__ sin_table,
                                           int T, int d3, int d, int heads, int head_dim) {
    int t = blockIdx.x;
    if (t >= T) return;
    float* col = QKV + t * d3;

    // Add bias to all 3*d elements
    for (int i = threadIdx.x; i < d3; i += blockDim.x)
        col[i] += bias[i];
    __syncthreads();

    // Apply RoPE to Q and K
    int d2 = head_dim / 2;
    const float* rc = cos_table + t * d2;
    const float* rs = sin_table + t * d2;
    for (int idx = threadIdx.x; idx < heads * d2; idx += blockDim.x) {
        int h = idx / d2;
        int i = idx % d2;
        // Q
        float* q = col + h * head_dim;
        float qr = q[i], qi = q[d2 + i];
        q[i]      = qr * rc[i] - qi * rs[i];
        q[d2 + i] = qi * rc[i] + qr * rs[i];
        // K
        float* k = col + d + h * head_dim;
        float kr = k[i], ki = k[d2 + i];
        k[i]      = kr * rc[i] - ki * rs[i];
        k[d2 + i] = ki * rc[i] + kr * rs[i];
    }
}

// Fused bias-add + RMSNorm: adds bias to x in-place, then computes RMSNorm.
// Used to fuse output projection bias with the FFN layer norm.
__global__ void g2p_bias_rms_norm_kernel(float* __restrict__ x,
                                           const float* __restrict__ bias,
                                           const float* __restrict__ weight,
                                           float* __restrict__ out,
                                           int T, int d, float eps) {
    int t = blockIdx.x;
    if (t >= T) return;
    float* xt = x + t * d;
    float* ot = out + t * d;

    extern __shared__ float sdata[];

    // Add bias in-place, accumulate sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = xt[i] + bias[i];
        xt[i] = v;
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv = rsqrtf(sdata[0] / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        ot[i] = weight[i] * xt[i] * inv;
}

// Fused FFN block: out_bias + RMSNorm + gate_up_GEMV + bias_SwiGLU + down_GEMV + bias + residual.
// One thread block per column of T. Replaces 5 separate kernel launches per layer.
// Shared memory: (d + 2*ff) * sizeof(float) bytes.
__global__ void g2p_fused_ffn_kernel(
    float* __restrict__ X,
    const float* __restrict__ out_b,
    const float* __restrict__ n2_w,
    const float* __restrict__ gate_up_w,
    const float* __restrict__ gate_up_b,
    const float* __restrict__ down_w,
    const float* __restrict__ down_b,
    int T, int d, int ff, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;

    extern __shared__ float smem[];
    float* s_x   = smem;          // [d] — biased X, then normed
    float* s_ffn = smem + d;      // [2*ff] — GEMV output, then SwiGLU result

    float* x_col = X + t * d;

    // ── Bias add: X[:, t] += out_b[:] ──
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = x_col[i] + out_b[i];
        s_x[i] = v;
        x_col[i] = v;  // write back for residual in final phase
    }
    __syncthreads();

    // ── RMSNorm: s_x → normed (in-place) ──
    float local_sum = 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        local_sum += s_x[i] * s_x[i];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) s_ffn[warp_id] = local_sum;  // reuse s_ffn as scratch
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < blockDim.x / 32) ? s_ffn[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0) s_ffn[0] = v;
    }
    __syncthreads();

    float inv = rsqrtf(s_ffn[0] / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        s_x[i] = n2_w[i] * s_x[i] * inv;
    __syncthreads();

    // ── Gate+Up GEMV: s_ffn[j] = dot(gate_up_w[:, j], s_x[:]) ──
    // gate_up_w is transposed to [d, 2*ff] row-major: row k at offset k*2*ff
    // Adjacent threads (adjacent j) read adjacent memory addresses → coalesced
    for (int j = threadIdx.x; j < 2 * ff; j += blockDim.x) {
        float sum = 0;
        for (int k = 0; k < d; k++)
            sum += gate_up_w[k * 2 * ff + j] * s_x[k];
        s_ffn[j] = sum;
    }
    __syncthreads();

    // ── Bias + SwiGLU: in-place on s_ffn ──
    for (int i = threadIdx.x; i < ff; i += blockDim.x) {
        float g = s_ffn[i] + gate_up_b[i];
        float u = s_ffn[ff + i] + gate_up_b[ff + i];
        s_ffn[i] = (g / (1.0f + expf(-g))) * u;
    }
    __syncthreads();

    // ── Down GEMV + bias + residual: X[:, t] += down_w^T × s_ffn[0:ff] + down_b ──
    // down_w is transposed to [ff, d] row-major: row k at offset k*d
    // Adjacent threads (adjacent j) read adjacent memory addresses → coalesced
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float sum = 0;
        for (int k = 0; k < ff; k++)
            sum += down_w[k * d + j] * s_ffn[k];
        x_col[j] += sum + down_b[j];
    }
}

// Embedding lookup: out[t, :] = emb[ids[t], :]
__global__ void g2p_embed_kernel(const int* __restrict__ ids, const float* __restrict__ emb,
                                   float* __restrict__ out, int T, int d) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float* src = emb + ids[t] * d;
    float* dst = out + t * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        dst[i] = src[i];
}

// Fused bias + reshape upsample: adds bias to proj[d*up, T], then reshapes to X_up[d, T*up]
__global__ void g2p_upsample_bias_reshape_kernel(const float* __restrict__ proj,
                                                    const float* __restrict__ bias,
                                                    float* __restrict__ X_up,
                                                    int T, int d, int up) {
    int idx = blockIdx.x;
    if (idx >= T * up) return;
    int t = idx / up;
    int u = idx % up;
    const float* src = proj + t * (d * up) + u * d;
    float* dst = X_up + idx * d;
    const float* b = bias + u * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        dst[i] = src[i] + b[i];
}

// Fused bias + CTC argmax per timestep: out[t] = argmax(logits[t, :] + bias[:])
__global__ void g2p_bias_ctc_argmax_kernel(const float* __restrict__ logits,
                                             const float* __restrict__ bias,
                                             int* __restrict__ out,
                                             int T, int n_phones) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* row = logits + t * n_phones;
    int best = 0;
    float best_v = row[0] + bias[0];
    for (int c = 1; c < n_phones; c++) {
        float v = row[c] + bias[c];
        if (v > best_v) { best_v = v; best = c; }
    }
    out[t] = best;
}


// ── Model struct ────────────────────────────────────────────────────────────

struct G2PModelCuda {
    bool load(const char* path, cudaStream_t stream);
    bool load(const void* data, size_t size, cudaStream_t stream);
    std::string infer(const std::string& text, cudaStream_t stream) const;
    void free();

    bool loaded() const { return d_ > 0; }
    size_t param_bytes() const { return total_bytes_; }

private:
    // Config
    int d_ = 0, heads_ = 0, n_layers_ = 0, ff_ = 0, up_ = 0;
    int n_chars_ = 0, n_phones_ = 0;
    int head_dim_ = 0;
    bool use_rope_ = false;

    // Vocab mappings (CPU)
    std::unordered_map<uint32_t, int> char2id_;
    std::unordered_map<int, uint32_t> id2phone_;

    // GPU weights (all contiguous in one allocation)
    float* weights_gpu_ = nullptr;
    size_t total_bytes_ = 0;

    // Pointers into weights_gpu_
    float* char_emb_ = nullptr;    // [n_chars, d]
    float* rope_cos_ = nullptr;    // [max_pos, head_dim/2]
    float* rope_sin_ = nullptr;    // [max_pos, head_dim/2]

    struct Layer {
        float* n1_w;               // [d]
        float* qkv_w, *qkv_b;     // [3d, d], [3d]
        float* out_w, *out_b;      // [d, d], [d]
        float* n2_w;               // [d]
        float* gate_up_w;          // [2*ff, d] (gate and up concatenated)
        float* gate_up_b;          // [2*ff]
        float* down_w, *down_b;    // [d, ff], [d]
    };
    std::vector<Layer> layers_;

    float* up_w_ = nullptr, *up_b_ = nullptr;     // [d*up, d], [d*up]
    float* head_w_ = nullptr, *head_b_ = nullptr;  // [n_phones, d], [n_phones]

    int max_pos_ = 2048;

    // Cached workspace (avoids cudaMalloc per call)
    mutable float* workspace_ = nullptr;
    mutable size_t workspace_bytes_ = 0;

    // CUDA graph cache: keyed by input length T for zero-overhead replay.
    // Eliminates ~150 kernel dispatch overheads per inference call.
    mutable std::unordered_map<int, cudaGraphExec_t> graph_cache_;

    // Attention scale (value, not pointer — Cutlass uses value-based alpha/beta)
    float attn_scale_ = 0.0f;

    bool load_from_file_(FILE* f, const char* label, cudaStream_t stream);
};

// ── Implementation ──────────────────────────────────────────────────────────

inline bool G2PModelCuda::load(const void* data, size_t size, cudaStream_t stream) {
    FILE* f = fmemopen(const_cast<void*>(data), size, "rb");
    if (!f) return false;
    bool ok = load_from_file_(f, "<bundle:g2p>", stream);
    fclose(f);
    return ok;
}

inline bool G2PModelCuda::load(const char* path, cudaStream_t stream) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    bool ok = load_from_file_(f, path, stream);
    fclose(f);
    return ok;
}

inline bool G2PModelCuda::load_from_file_(FILE* f, const char* label, cudaStream_t stream) {

    // Magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "G2P3", 4) != 0) {
        fprintf(stderr, "g2p_cuda: expected G2P3 format\n");
        return false;
    }

    // Header
    uint32_t hdr[10];
    if (fread(hdr, 4, 10, f) != 10) { return false; }
    d_ = hdr[0]; heads_ = hdr[1]; n_layers_ = hdr[2]; ff_ = hdr[3]; up_ = hdr[4];
    n_chars_ = hdr[5]; n_phones_ = hdr[6];
    uint32_t flags = hdr[9];
    use_rope_ = (flags & 1) != 0;
    bool use_qk_norm = (flags & 2) != 0;
    bool use_conv = (flags & 4) != 0;
    head_dim_ = d_ / heads_;

    bool use_rmsnorm = (flags & 8) != 0;
    if (use_qk_norm || use_conv) {
        fprintf(stderr, "g2p_cuda: QK-Norm and ConvModule not supported\n");
        return false;
    }
    if (!use_rmsnorm) {
        fprintf(stderr, "g2p_cuda: only RMSNorm supported (got LayerNorm model)\n");
        return false;
    }

    // Char vocab
    uint32_t n_cv;
    if (fread(&n_cv, 4, 1, f) != 1) { return false; }
    for (uint32_t i = 0; i < n_cv; i++) {
        uint32_t pair[2];
        if (fread(pair, 4, 2, f) != 2) { return false; }
        char2id_[pair[0]] = pair[1];
    }

    // Phone vocab
    uint32_t n_pv;
    if (fread(&n_pv, 4, 1, f) != 1) { return false; }
    for (uint32_t i = 0; i < n_pv; i++) {
        uint32_t pair[2];
        if (fread(pair, 4, 2, f) != 2) { return false; }
        id2phone_[pair[1]] = pair[0];
    }

    // Calculate total weight size
    int d2 = head_dim_ / 2;
    size_t total_floats = 0;
    total_floats += n_chars_ * d_;                      // char_emb
    total_floats += max_pos_ * d2 * 2;                  // rope cos + sin
    for (int i = 0; i < n_layers_; i++) {
        total_floats += d_;                              // n1_w
        total_floats += 3 * d_ * d_ + 3 * d_;           // qkv_w, qkv_b
        total_floats += d_ * d_ + d_;                    // out_w, out_b
        total_floats += d_;                              // n2_w
        total_floats += ff_ * d_ + ff_;                  // gate_w, gate_b
        total_floats += ff_ * d_ + ff_;                  // up_w, up_b
        total_floats += d_ * ff_ + d_;                   // down_w, down_b
    }
    total_floats += d_ * up_ * d_ + d_ * up_;           // up_w, up_b
    total_floats += n_phones_ * d_ + n_phones_;          // head_w, head_b

    total_bytes_ = total_floats * sizeof(float);

    // Allocate GPU memory (single contiguous block)
    auto err = cudaMalloc(&weights_gpu_, total_bytes_);
    if (err != cudaSuccess) {
        fprintf(stderr, "g2p_cuda: cudaMalloc failed for weights (%.1f MB): %s\n",
                total_bytes_ / (1024.0f * 1024.0f), cudaGetErrorString(err));
        return false;
    }

    // Read weights to CPU staging buffer, then upload
    std::vector<float> staging(total_floats);
    float* ptr = staging.data();
    auto read_w = [&](int n) -> float* {
        float* p = ptr;
        if (fread(p, sizeof(float), n, f) != (size_t)n) return nullptr;
        ptr += n;
        return p;
    };

    // Char embedding
    float* char_emb_cpu = read_w(n_chars_ * d_);
    if (!char_emb_cpu) { return false; }

    // Layers
    struct LayerCPU { float *n1_w, *qkv_w, *qkv_b, *out_w, *out_b, *n2_w,
                            *gate_w, *gate_b, *up_w, *up_b, *down_w, *down_b; };
    std::vector<LayerCPU> layers_cpu(n_layers_);
    for (int i = 0; i < n_layers_; i++) {
        auto& L = layers_cpu[i];
        L.n1_w   = read_w(d_);
        L.qkv_w  = read_w(3 * d_ * d_);
        L.qkv_b  = read_w(3 * d_);
        L.out_w  = read_w(d_ * d_);
        L.out_b  = read_w(d_);
        L.n2_w   = read_w(d_);
        L.gate_w = read_w(ff_ * d_);
        L.gate_b = read_w(ff_);
        L.up_w   = read_w(ff_ * d_);
        L.up_b   = read_w(ff_);
        L.down_w = read_w(d_ * ff_);
        L.down_b = read_w(d_);
        if (!L.down_b) { return false; }
    }

    float* up_w_cpu = read_w(d_ * up_ * d_);
    float* up_b_cpu = read_w(d_ * up_);
    float* head_w_cpu = read_w(n_phones_ * d_);
    float* head_b_cpu = read_w(n_phones_);
    if (!head_b_cpu) { return false; }

    // Compute RoPE tables (CPU)
    size_t rope_floats = max_pos_ * d2 * 2;
    std::vector<float> rope_staging(rope_floats);
    float* rope_cos_cpu = rope_staging.data();
    float* rope_sin_cpu = rope_staging.data() + max_pos_ * d2;
    if (use_rope_) {
        for (int t = 0; t < max_pos_; t++) {
            for (int i = 0; i < d2; i++) {
                float freq = (float)t / std::pow(10000.0f, (float)(2 * i) / head_dim_);
                rope_cos_cpu[t * d2 + i] = std::cos(freq);
                rope_sin_cpu[t * d2 + i] = std::sin(freq);
            }
        }
    }

    // Upload everything to GPU
    float* gpu = weights_gpu_;
    auto upload = [&](const float* src, int n) -> float* {
        float* dst = gpu;
        cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyHostToDevice, stream);
        gpu += n;
        return dst;
    };

    char_emb_ = upload(char_emb_cpu, n_chars_ * d_);
    rope_cos_ = upload(rope_cos_cpu, max_pos_ * d2);
    rope_sin_ = upload(rope_sin_cpu, max_pos_ * d2);

    // Transpose gate_w[ff,d] + up_w[ff,d] → gate_up_wt[d, 2*ff] row-major
    // and down_w[d,ff] col-major → down_wt[ff, d] row-major
    // so that adjacent threads read adjacent memory in the fused FFN kernel.
    std::vector<float> gate_up_transposed(2 * ff_ * d_);
    std::vector<float> down_transposed(d_ * ff_);

    layers_.resize(n_layers_);
    for (int i = 0; i < n_layers_; i++) {
        auto& L = layers_[i];
        auto& C = layers_cpu[i];
        L.n1_w      = upload(C.n1_w,   d_);
        L.qkv_w     = upload(C.qkv_w,  3 * d_ * d_);
        L.qkv_b     = upload(C.qkv_b,  3 * d_);
        L.out_w     = upload(C.out_w,   d_ * d_);
        L.out_b     = upload(C.out_b,   d_);
        L.n2_w      = upload(C.n2_w,    d_);
        // Transpose gate_w[ff,d] and up_w[ff,d] into gate_up_wt[d, 2*ff] row-major.
        // Original col-major: gate_w[j*d + k] = weight at row k, col j.
        // Transposed row-major: gate_up_wt[k*2*ff + j] = same weight.
        for (int k = 0; k < d_; k++) {
            for (int j = 0; j < ff_; j++) {
                gate_up_transposed[k * 2 * ff_ + j]      = C.gate_w[j * d_ + k];
                gate_up_transposed[k * 2 * ff_ + ff_ + j] = C.up_w[j * d_ + k];
            }
        }
        L.gate_up_w = upload(gate_up_transposed.data(), 2 * ff_ * d_);
        // Upload gate_b then up_b contiguously → gate_up_b[2*ff]
        L.gate_up_b = upload(C.gate_b,  ff_);
                      upload(C.up_b,    ff_);
        // Transpose down_w[d,ff] col-major → down_wt[ff, d] row-major.
        // Original col-major: down_w[j*ff + k] = weight at row k, col j.
        // Transposed row-major: down_wt[k*d + j] = same weight.
        for (int k = 0; k < ff_; k++) {
            for (int j = 0; j < d_; j++) {
                down_transposed[k * d_ + j] = C.down_w[j * ff_ + k];
            }
        }
        L.down_w    = upload(down_transposed.data(),  d_ * ff_);
        L.down_b = upload(C.down_b,  d_);
    }
    up_w_   = upload(up_w_cpu,   d_ * up_ * d_);
    up_b_   = upload(up_b_cpu,   d_ * up_);
    head_w_ = upload(head_w_cpu, n_phones_ * d_);
    head_b_ = upload(head_b_cpu, n_phones_);

    attn_scale_ = 1.0f / std::sqrt((float)head_dim_);

    // Pre-allocate workspace at max_pos_ size so that CUDA graphs are never
    // invalidated by a larger T arriving later.
    {
        int Tm = max_pos_;
        int Tm_up = Tm * up_;
        auto a = [](size_t off, size_t bytes) -> size_t {
            return ((off + 255) & ~(size_t)255) + bytes;
        };
        workspace_bytes_ = 0;
        workspace_bytes_ = a(workspace_bytes_, Tm * sizeof(int));
        workspace_bytes_ = a(workspace_bytes_, Tm * d_ * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, Tm * 3 * d_ * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, heads_ * Tm * Tm * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, Tm * d_ * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, Tm * d_ * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, Tm * 2 * ff_ * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, Tm * d_ * up_ * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, Tm_up * d_ * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, Tm_up * n_phones_ * sizeof(float));
        workspace_bytes_ = a(workspace_bytes_, Tm_up * sizeof(int));
        auto ws_err = cudaMalloc(&workspace_, workspace_bytes_);
        if (ws_err != cudaSuccess) {
            fprintf(stderr, "g2p_cuda: workspace alloc failed (%zu bytes): %s\n",
                    workspace_bytes_, cudaGetErrorString(ws_err));
            workspace_ = nullptr; workspace_bytes_ = 0;
            return false;
        }
    }

    cudaStreamSynchronize(stream);

    fprintf(stderr, "g2p_cuda: loaded %s (d=%d, %d layers, %d heads, %d ff, %dx up, %.1f MB, ws=%.1f MB)\n",
            label, d_, n_layers_, heads_, ff_, up_,
            total_bytes_ / (1024.0f * 1024.0f), workspace_bytes_ / (1024.0f * 1024.0f));
    return true;
}

inline void G2PModelCuda::free() {
    for (auto& [t, exec] : graph_cache_) if (exec) cudaGraphExecDestroy(exec);
    graph_cache_.clear();
    if (weights_gpu_) { cudaFree(weights_gpu_); weights_gpu_ = nullptr; }
    if (workspace_) { cudaFree(workspace_); workspace_ = nullptr; workspace_bytes_ = 0; }
    d_ = 0;
}

inline std::string G2PModelCuda::infer(const std::string& text,
                                         cudaStream_t stream) const {
    if (d_ == 0) return "";

    // Encode input to char IDs
    std::vector<int> ids;
    const uint8_t* p = (const uint8_t*)text.data();
    const uint8_t* end = p + text.size();
    while (p < end) {
        uint32_t cp;
        if (*p < 0x80) { cp = *p++; }
        else if ((*p & 0xE0) == 0xC0) { cp = (*p++ & 0x1F) << 6; if (p < end) cp |= (*p++ & 0x3F); }
        else if ((*p & 0xF0) == 0xE0) { cp = (*p++ & 0x0F) << 12; if (p < end) cp |= (*p++ & 0x3F) << 6; if (p < end) cp |= (*p++ & 0x3F); }
        else { cp = (*p++ & 0x07) << 18; if (p < end) cp |= (*p++ & 0x3F) << 12; if (p < end) cp |= (*p++ & 0x3F) << 6; if (p < end) cp |= (*p++ & 0x3F); }
        auto it = char2id_.find(cp);
        ids.push_back(it != char2id_.end() ? it->second : 0);
    }

    int T = (int)ids.size();
    if (T == 0 || T > max_pos_) return "";
    int d = d_, h = heads_, dk = head_dim_, ff = ff_;
    int T_up = T * up_;

    // Workspace layout (each buffer 256-byte aligned for Cutlass compatibility):
    auto a = [](size_t off, size_t bytes) -> size_t {
        return ((off + 255) & ~(size_t)255) + bytes;
    };
    size_t ws_bytes = 0;
    ws_bytes = a(ws_bytes, T * sizeof(int));                    // ids_gpu
    ws_bytes = a(ws_bytes, T * d * sizeof(float));              // X
    ws_bytes = a(ws_bytes, T * 3 * d * sizeof(float));          // QKV
    ws_bytes = a(ws_bytes, h * T * T * sizeof(float));          // attn_scores
    ws_bytes = a(ws_bytes, T * d * sizeof(float));              // attn_out
    ws_bytes = a(ws_bytes, T * d * sizeof(float));              // normed
    ws_bytes = a(ws_bytes, T * 2 * ff * sizeof(float));         // ffn_out
    ws_bytes = a(ws_bytes, T * d * up_ * sizeof(float));        // X_up_proj
    ws_bytes = a(ws_bytes, T_up * d * sizeof(float));           // X_up
    ws_bytes = a(ws_bytes, T_up * n_phones_ * sizeof(float));   // logits
    ws_bytes = a(ws_bytes, T_up * sizeof(int));                 // argmax

    if (ws_bytes > workspace_bytes_) {
        // Should never happen — workspace is pre-allocated at max_pos_ size
        fprintf(stderr, "g2p_cuda: BUG: ws_bytes %zu > workspace_bytes_ %zu (T=%d)\n",
                ws_bytes, workspace_bytes_, T);
        return "";
    }

    // Assign pointers (256-byte aligned for Cutlass TensorOp compatibility)
    char* wp = (char*)workspace_;
    auto align256 = [&]() { wp = (char*)(((uintptr_t)wp + 255) & ~(uintptr_t)255); };
    auto wallocf = [&](size_t n) -> float* { align256(); float* p = (float*)wp; wp += n * sizeof(float); return p; };
    auto walloci = [&](size_t n) -> int*   { align256(); int* p = (int*)wp; wp += n * sizeof(int); return p; };

    int* ids_gpu         = walloci(T);
    float* X             = wallocf(T * d);
    float* QKV           = wallocf(T * 3 * d);
    float* attn_scores   = wallocf(h * T * T);
    float* attn_out      = wallocf(T * d);
    float* normed        = wallocf(T * d);
    float* ffn_out       = wallocf(T * 2 * ff);
    float* X_up_proj     = wallocf(T * d * up_);
    float* X_up          = wallocf(T_up * d);
    float* logits        = wallocf(T_up * n_phones_);
    int* argmax          = walloci(T_up);

    // Upload input IDs (before graph — stream ordering guarantees completion)
    cudaMemcpyAsync(ids_gpu, ids.data(), T * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Lambda: run all inference kernels on the stream
    auto run_kernels = [&]() {

        g2p_embed_kernel<<<T, 128, 0, stream>>>(ids_gpu, char_emb_, X, T, d);

        int block = 128;
        for (int li = 0; li < n_layers_; li++) {
            const auto& L = layers_[li];

            // 1. RMSNorm (attn)
            g2p_rms_norm_kernel<<<T, block, block * sizeof(float), stream>>>(
                X, L.n1_w, normed, T, d, 1e-6f);

            // 2. QKV projection
            cutlass_gemm_tn(3 * d, T, d, L.qkv_w, d, normed, d, QKV, 3 * d,
                             1.0f, 0.0f, nullptr, 0, stream);


            // 3. Fused QKV bias + RoPE Q&K
            if (use_rope_) {
                g2p_qkv_bias_rope_kernel<<<T, 256, 0, stream>>>(
                    QKV, L.qkv_b, rope_cos_, rope_sin_, T, 3 * d, d, h, dk);
            } else {
                g2p_bias_kernel<<<T, 256, 0, stream>>>(QKV, L.qkv_b, 3 * d, T);
            }

            // 4. Batched attention scores
            cutlass_gemm_batched_tn(T, T, dk,
                QKV + d, 3 * d, (long long)dk,
                QKV,     3 * d, (long long)dk,
                attn_scores, T, (long long)T * T,
                h, attn_scale_, 0.0f, nullptr, 0, stream);


            // 5. Softmax
            g2p_softmax_kernel<<<h * T, block, block * sizeof(float), stream>>>(
                attn_scores, T, h * T);

            // 6. Batched value weighted sum
            cutlass_gemm_batched_nn(dk, T, T,
                QKV + 2 * d, 3 * d, (long long)dk,
                attn_scores, T,     (long long)T * T,
                attn_out, d, (long long)dk,
                h, 1.0f, 0.0f, nullptr, 0, stream);


            // 7. Output projection with fused residual (beta=1)
            cutlass_gemm_tn(d, T, d, L.out_w, d, attn_out, d, X, d,
                             1.0f, 1.0f, nullptr, 0, stream);


            // 8a. Fused out_bias + RMSNorm
            g2p_bias_rms_norm_kernel<<<T, block, block * sizeof(float), stream>>>(
                X, L.out_b, L.n2_w, normed, T, d, 1e-6f);

            // 8b. Gate+Up GEMM: ffn_out[2*ff, T] = gate_up_w[2*ff, d] × normed[d, T]
            cutlass_gemm_nn(2 * ff, T, d, L.gate_up_w, 2 * ff, normed, d, ffn_out, 2 * ff,
                             1.0f, 0.0f, nullptr, 0, stream);

            // 8c. Fused bias + SwiGLU
            g2p_swiglu_bias_kernel<<<(T * ff + 255) / 256, 256, 0, stream>>>(
                ffn_out, L.gate_up_b, ff, T);

            // 8d. Down GEMM with fused residual: X += down_w[d, ff] × ffn_out[ff, T]
            cutlass_gemm_nn(d, T, ff, L.down_w, d, ffn_out, 2 * ff, X, d,
                             1.0f, 1.0f, nullptr, 0, stream);

            // 8e. Down bias
            g2p_bias_kernel<<<T, 256, 0, stream>>>(X, L.down_b, d, T);
        }

        // Upsample projection (no bias — fused into reshape)
        cutlass_gemm_tn(d * up_, T, d, up_w_, d, X, d, X_up_proj, d * up_,
                         1.0f, 0.0f, nullptr, 0, stream);

        // Fused upsample bias + reshape
        g2p_upsample_bias_reshape_kernel<<<T_up, 128, 0, stream>>>(
            X_up_proj, up_b_, X_up, T, d, up_);

        // Output head (no bias — fused into CTC argmax)
        cutlass_gemm_tn(n_phones_, T_up, d, head_w_, d, X_up, d, logits, n_phones_,
                         1.0f, 0.0f, nullptr, 0, stream);

        // Fused head bias + CTC argmax
        g2p_bias_ctc_argmax_kernel<<<(T_up + 255) / 256, 256, 0, stream>>>(
            logits, head_b_, argmax, T_up, n_phones_);

    };

    // CUDA graph: first call runs directly (populates Cutlass operator caches),
    // second call captures the graph, third+ replays.
    auto git = graph_cache_.find(T);
    if (git == graph_cache_.end()) {
        // First call: run directly. Cutlass initialize() populates operator
        // caches. Output is valid — returned to caller.
        run_kernels();
        // Mark as "seen but not yet captured" with nullptr
        graph_cache_[T] = nullptr;
    } else if (git->second == nullptr) {
        // Second call: capture graph (all Cutlass operators hit cache)
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        run_kernels();
        cudaGraph_t graph;
        cudaStreamEndCapture(stream, &graph);
        cudaGraphExec_t exec;
        cudaGraphInstantiateWithFlags(&exec, graph, 0);
        cudaGraphDestroy(graph);
        git->second = exec;
        cudaGraphLaunch(exec, stream);
    } else {
        // Third+: replay cached graph
        cudaGraphLaunch(git->second, stream);
    }

    // Copy argmax back to CPU
    std::vector<int> argmax_cpu(T_up);
    cudaMemcpyAsync(argmax_cpu.data(), argmax, T_up * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // CTC decode on CPU
    std::string result;
    int prev = -1;
    for (int t = 0; t < T_up; t++) {
        int best = argmax_cpu[t];
        if (best != 0 && best != prev) {
            auto it = id2phone_.find(best);
            if (it != id2phone_.end()) {
                uint32_t cp = it->second;
                if (cp < 0x80) { result += (char)cp; }
                else if (cp < 0x800) { result += (char)(0xC0 | (cp >> 6)); result += (char)(0x80 | (cp & 0x3F)); }
                else if (cp < 0x10000) { result += (char)(0xE0 | (cp >> 12)); result += (char)(0x80 | ((cp >> 6) & 0x3F)); result += (char)(0x80 | (cp & 0x3F)); }
                else { result += (char)(0xF0 | (cp >> 18)); result += (char)(0x80 | ((cp >> 12) & 0x3F)); result += (char)(0x80 | ((cp >> 6) & 0x3F)); result += (char)(0x80 | (cp & 0x3F)); }
            }
        }
        prev = best;
    }

    return result;
}
