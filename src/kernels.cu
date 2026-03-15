// kernels.cu — Custom CUDA kernels for Rokoko TTS (FP32)
//
// All kernels use FP32.

#include "kernels.h"
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Embedding gather: y[i, :] = table[ids[i], :]
// ---------------------------------------------------------------------------

__global__ void embedding_gather_kernel(const float* table, const int* ids,
                                         float* y, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    int idx = ids[row];
    const float* src = table + idx * D;
    float* dst = y + row * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        dst[d] = src[d];
    }
}

void embedding_gather(const float* table, const int* ids, float* y,
                      int N, int D, cudaStream_t stream) {
    int threads = (D + 31) / 32 * 32;
    if (threads > 256) threads = 256;
    embedding_gather_kernel<<<N, threads, 0, stream>>>(table, ids, y, N, D);
}

// ---------------------------------------------------------------------------
// Element-wise add: y = a + b
// ---------------------------------------------------------------------------

__global__ void add_kernel(const float* a, const float* b, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = a[i] + b[i];
    }
}

void add_f32(const float* a, const float* b, float* y, int N,
             cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(a, b, y, N);
}

// ---------------------------------------------------------------------------
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
//   One warp (32 threads) per row.
// ---------------------------------------------------------------------------

__global__ void layer_norm_kernel(const float* x, const float* gamma,
                                   const float* beta, float* y,
                                   int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32) {
        sum += x_row[d];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    float mean = __shfl_sync(0xFFFFFFFF, sum, 0) / D;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32) {
        float diff = x_row[d] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    float inv_std = rsqrtf(__shfl_sync(0xFFFFFFFF, var_sum, 0) / D + eps);

    // Pass 3: normalize
    for (int d = threadIdx.x; d < D; d += 32) {
        float val = (x_row[d] - mean) * inv_std;
        y_row[d] = val * gamma[d] + beta[d];
    }
}

void layer_norm_f32(const float* x, const float* gamma, const float* beta,
                    float* y, int N, int D, float eps, cudaStream_t stream) {
    layer_norm_kernel<<<N, 32, 0, stream>>>(x, gamma, beta, y, N, D, eps);
}

// ---------------------------------------------------------------------------
// Fused Residual + LayerNorm: y = LayerNorm(a + b; gamma, beta, eps)
//   One warp (32 threads) per row. Saves a separate add kernel launch.
// ---------------------------------------------------------------------------

__global__ void residual_layer_norm_kernel(const float* a, const float* b,
                                            const float* gamma, const float* beta,
                                            float* y, int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* a_row = a + row * D;
    const float* b_row = b + row * D;
    float* y_row = y + row * D;

    // Pass 1: compute mean of (a + b)
    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32) {
        sum += a_row[d] + b_row[d];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    float mean = __shfl_sync(0xFFFFFFFF, sum, 0) / D;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32) {
        float val = a_row[d] + b_row[d];
        float diff = val - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    float inv_std = rsqrtf(__shfl_sync(0xFFFFFFFF, var_sum, 0) / D + eps);

    // Pass 3: normalize and write
    for (int d = threadIdx.x; d < D; d += 32) {
        float val = (a_row[d] + b_row[d] - mean) * inv_std;
        y_row[d] = val * gamma[d] + beta[d];
    }
}

void residual_layer_norm_f32(const float* a, const float* b,
                              const float* gamma, const float* beta,
                              float* y, int N, int D, float eps,
                              cudaStream_t stream) {
    residual_layer_norm_kernel<<<N, 32, 0, stream>>>(a, b, gamma, beta, y, N, D, eps);
}

// ---------------------------------------------------------------------------
// GELU (tanh approximation): y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   This matches PyTorch's "gelu_new" / approximate='tanh'.
// ---------------------------------------------------------------------------

__global__ void gelu_kernel(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = x[i];
        float inner = 0.7978845608028654f * (val + 0.044715f * val * val * val);
        y[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

void gelu_f32(const float* x, float* y, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(x, y, N);
}

// ---------------------------------------------------------------------------
// LeakyReLU
// ---------------------------------------------------------------------------

__global__ void leaky_relu_kernel(const float* x, float* y, int N, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = x[i];
        y[i] = val > 0.0f ? val : alpha * val;
    }
}

void leaky_relu_f32(const float* x, float* y, int N, float alpha,
                    cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(x, y, N, alpha);
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

__global__ void sigmoid_kernel(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

void sigmoid_f32(const float* x, float* y, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(x, y, N);
}

// ---------------------------------------------------------------------------
// Softmax over last dimension (one warp per row)
// ---------------------------------------------------------------------------

__global__ void softmax_kernel(const float* x, float* y, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    // Find max
    float max_val = -1e30f;
    for (int d = threadIdx.x; d < D; d += 32) {
        float v = x_row[d];
        if (v > max_val) max_val = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        if (other > max_val) max_val = other;
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // Compute exp sum
    float exp_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32) {
        exp_sum += expf(x_row[d] - max_val);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        exp_sum += __shfl_down_sync(0xFFFFFFFF, exp_sum, offset);
    float inv_sum = 1.0f / __shfl_sync(0xFFFFFFFF, exp_sum, 0);

    // Write normalized values
    for (int d = threadIdx.x; d < D; d += 32) {
        y_row[d] = expf(x_row[d] - max_val) * inv_sum;
    }
}

void softmax_f32(const float* x, float* y, int N, int D, cudaStream_t stream) {
    softmax_kernel<<<N, 32, 0, stream>>>(x, y, N, D);
}

// ---------------------------------------------------------------------------
// Bias add: y[n, d] = x[n, d] + bias[d]
// ---------------------------------------------------------------------------

__global__ void bias_add_kernel(const float* x, const float* bias, float* y,
                                 int total, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int d = idx % D;
        y[idx] = x[idx] + bias[d];
    }
}

void bias_add_f32(const float* x, const float* bias, float* y, int N, int D,
                  cudaStream_t stream) {
    int total = N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads, 0, stream>>>(x, bias, y, total, D);
}

// ---------------------------------------------------------------------------
// Transpose 2D: y[j, i] = x[i, j]
// ---------------------------------------------------------------------------

static constexpr int TILE = 32;

__global__ void transpose_kernel(const float* x, float* y, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 to avoid bank conflicts

    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;

    int xi = bx + threadIdx.x;
    int yi = by + threadIdx.y;
    if (xi < N && yi < M)
        tile[threadIdx.y][threadIdx.x] = x[yi * N + xi];

    __syncthreads();

    int xo = by + threadIdx.x;
    int yo = bx + threadIdx.y;
    if (xo < M && yo < N)
        y[yo * M + xo] = tile[threadIdx.x][threadIdx.y];
}

void transpose_f32(const float* x, float* y, int M, int N,
                   cudaStream_t stream) {
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_kernel<<<blocks, threads, 0, stream>>>(x, y, M, N);
}

// ---------------------------------------------------------------------------
// Conv1d: y[t, c_out] = sum_{c_in, k} w[c_out, c_in, k] * x[t+k-pad, c_in] + b[c_out]
//   x: [T, C_in], y: [T, C_out], w: [C_out, C_in, K]
//   One thread block per (c_out, t) pair. Simple implementation for small T.
// ---------------------------------------------------------------------------

__global__ void conv1d_kernel(const float* x, const float* w, const float* bias,
                               float* y, int C_in, int C_out, int T, int K) {
    int co = blockIdx.x;  // output channel
    int t = blockIdx.y;   // time position
    if (co >= C_out || t >= T) return;

    int pad = K / 2;
    float sum = 0.0f;

    // Each thread handles a subset of the (c_in, k) pairs
    for (int idx = threadIdx.x; idx < C_in * K; idx += blockDim.x) {
        int ci = idx / K;
        int k = idx % K;
        int t_in = t + k - pad;
        if (t_in >= 0 && t_in < T) {
            sum += w[co * C_in * K + ci * K + k] * x[t_in * C_in + ci];
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (threadIdx.x == 0) {
        y[t * C_out + co] = sum + (bias ? bias[co] : 0.0f);
    }
}

void conv1d_f32(const float* x, const float* w, const float* bias,
                float* y, int C_in, int C_out, int T, int K,
                cudaStream_t stream) {
    // Use 32 threads (1 warp) per (c_out, t) — good for C_in*K < ~1024
    dim3 blocks(C_out, T);
    conv1d_kernel<<<blocks, 32, 0, stream>>>(x, w, bias, y, C_in, C_out, T, K);
}

// ---------------------------------------------------------------------------
// Weight norm: w[o, ...] = wg[o] * wv[o, ...] / ||wv[o]||_2
//   One warp per output channel.
// ---------------------------------------------------------------------------

__global__ void weight_norm_kernel(const float* wg, const float* wv, float* w,
                                    int C_out, int fan_in) {
    int o = blockIdx.x;
    if (o >= C_out) return;

    const float* v = wv + o * fan_in;

    // Compute L2 norm of v
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < fan_in; i += 32) {
        float val = v[i];
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    float inv_norm = rsqrtf(__shfl_sync(0xFFFFFFFF, sum_sq, 0));

    float g = wg[o];  // wg is [C_out, 1, 1], stride = 1 per output channel

    // Apply: w = g * v / ||v||
    float* w_o = w + o * fan_in;
    for (int i = threadIdx.x; i < fan_in; i += 32) {
        w_o[i] = g * v[i] * inv_norm;
    }
}

void weight_norm_f32(const float* wg, const float* wv, float* w,
                     int C_out, int fan_in, cudaStream_t stream) {
    weight_norm_kernel<<<C_out, 32, 0, stream>>>(wg, wv, w, C_out, fan_in);
}

// ---------------------------------------------------------------------------
// LayerNorm across channels: x[T, C] -> y[T, C]
//   For each t: normalize across the C dimension.
//   One warp per time position.
// ---------------------------------------------------------------------------

__global__ void layer_norm_cf_kernel(const float* x, const float* gamma,
                                      const float* beta, float* y,
                                      int C, int T, float eps) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float* x_row = x + t * C;
    float* y_row = y + t * C;

    // Compute mean across channels at position t
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += 32) {
        sum += x_row[c];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    float mean = __shfl_sync(0xFFFFFFFF, sum, 0) / C;

    // Compute variance
    float var_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += 32) {
        float diff = x_row[c] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    float inv_std = rsqrtf(__shfl_sync(0xFFFFFFFF, var_sum, 0) / C + eps);

    // Normalize
    for (int c = threadIdx.x; c < C; c += 32) {
        float val = (x_row[c] - mean) * inv_std;
        y_row[c] = val * gamma[c] + beta[c];
    }
}

void layer_norm_channels_first_f32(const float* x, const float* gamma,
                                    const float* beta, float* y,
                                    int C, int T, float eps,
                                    cudaStream_t stream) {
    layer_norm_cf_kernel<<<T, 32, 0, stream>>>(x, gamma, beta, y, C, T, eps);
}

// ---------------------------------------------------------------------------
// Cast int64 to int32
// ---------------------------------------------------------------------------

__global__ void cast_i64_to_i32_kernel(const int64_t* src, int* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = (int)src[i];
}

void cast_i64_to_i32(const int64_t* src, int* dst, int N,
                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cast_i64_to_i32_kernel<<<blocks, threads, 0, stream>>>(src, dst, N);
}

// ---------------------------------------------------------------------------
// Coalesced per-channel sum + sum_sq accumulation kernel
//   Used by both instance_norm_1d and instance_norm_style_affine.
//   Grid(blocks_x, blocks_y): x covers C, y tiles over T for parallelism.
//   Adjacent threads read adjacent channels → coalesced memory access.
// ---------------------------------------------------------------------------

__global__ void instnorm_sum_kernel(
        const float* __restrict__ x,
        float* __restrict__ red_sum,    // [C]
        float* __restrict__ red_sum_sq, // [C]
        int C, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float s = 0.0f, sq = 0.0f;
    for (int t = blockIdx.y; t < T; t += gridDim.y) {
        float v = x[t * C + c];
        s += v;
        sq += v * v;
    }
    atomicAdd(&red_sum[c], s);
    atomicAdd(&red_sum_sq[c], sq);
}

// ---------------------------------------------------------------------------
// Instance Normalization 1D: normalize each channel across time
//   x, y: [T, C], weight, bias: [C]
//   Two-pass coalesced: same approach as fused instance_norm_style_affine.
// ---------------------------------------------------------------------------

// Pass 1: reuses instnorm_sum_kernel (defined below in fused version)

// Pass 2 (standalone, without style affine)
__global__ void instnorm_norm_kernel(
        const float* __restrict__ x,
        const float* __restrict__ red_sum,
        const float* __restrict__ red_sum_sq,
        const float* __restrict__ weight, const float* __restrict__ bias,
        float* __restrict__ y,
        int C, int T, float eps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float mean = red_sum[c] / T;
    float var = red_sum_sq[c] / T - mean * mean;
    float inv_std = rsqrtf(var + eps);
    float w = weight[c], b = bias[c];

    for (int t = blockIdx.y; t < T; t += gridDim.y) {
        y[t * C + c] = w * (x[t * C + c] - mean) * inv_std + b;
    }
}

void instance_norm_1d_f32(const float* x, const float* weight, const float* bias,
                           float* y, float* workspace,
                           int C, int T, float eps,
                           cudaStream_t stream) {
    float* red_sum = workspace;
    float* red_sum_sq = workspace + C;
    cudaMemsetAsync(workspace, 0, 2 * C * sizeof(float), stream);

    int threads = (C < 256) ? C : 256;
    int blocks_x = (C + threads - 1) / threads;
    int blocks_y = (T < 256) ? T : 256;
    dim3 grid(blocks_x, blocks_y);

    // Forward declaration: instnorm_sum_kernel is defined below
    // but works identically for standalone instance norm
    instnorm_sum_kernel<<<grid, threads, 0, stream>>>(
        x, red_sum, red_sum_sq, C, T);

    instnorm_norm_kernel<<<grid, threads, 0, stream>>>(
        x, red_sum, red_sum_sq, weight, bias, y, C, T, eps);
}

// ---------------------------------------------------------------------------
// Style affine 1D: y[t,c] = (1 + gamma[c]) * x[t,c] + beta[c]
//   x, y: [T, C], gamma, beta: [C]
// ---------------------------------------------------------------------------

__global__ void style_affine_1d_kernel(const float* x, const float* gamma,
                                        const float* beta, float* y,
                                        int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T;
    if (idx < total) {
        int c = idx % C;
        y[idx] = (1.0f + gamma[c]) * x[idx] + beta[c];
    }
}

void style_affine_1d_f32(const float* x, const float* gamma, const float* beta,
                           float* y, int C, int T, cudaStream_t stream) {
    int total = C * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    style_affine_1d_kernel<<<blocks, threads, 0, stream>>>(x, gamma, beta, y, C, T);
}

// ---------------------------------------------------------------------------
// Fused InstanceNorm + StyleAffine: two-pass coalesced AdaIN
//   y[t,c] = (1+gamma[c]) * (norm_w[c] * (x[t,c]-mean)/sqrt(var+eps) + norm_b[c]) + beta[c]
//   x, y: [T, C].
//
//   Pass 1: coalesced reads along C, accumulate per-channel sum + sum_sq.
//   Pass 2: normalize + style affine with coalesced reads and writes.
//   All memory accesses are coalesced (adjacent threads access adjacent C positions).
// ---------------------------------------------------------------------------

// Pass 2: compute mean/var from reduction buffer, normalize + style affine
__global__ void instnorm_style_norm_kernel(
        const float* __restrict__ x,
        const float* __restrict__ red_sum,
        const float* __restrict__ red_sum_sq,
        const float* __restrict__ norm_w, const float* __restrict__ norm_b,
        const float* __restrict__ gamma, const float* __restrict__ beta,
        float* __restrict__ y,
        int C, int T, float eps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    // Compute mean and inv_std from reduction buffer
    float mean = red_sum[c] / T;
    float var = red_sum_sq[c] / T - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Precompute fused scale and bias
    float g = 1.0f + gamma[c];
    float combined_scale = g * norm_w[c] * inv_std;
    float combined_bias = g * (norm_b[c] - norm_w[c] * mean * inv_std) + beta[c];

    // Normalize with coalesced reads/writes
    for (int t = blockIdx.y; t < T; t += gridDim.y) {
        y[t * C + c] = combined_scale * x[t * C + c] + combined_bias;
    }
}

void instance_norm_style_affine_f32(const float* x, const float* norm_w,
                                      const float* norm_b, const float* gamma,
                                      const float* beta, float* y,
                                      float* workspace,
                                      int C, int T, float eps,
                                      cudaStream_t stream) {
    float* red_sum = workspace;
    float* red_sum_sq = workspace + C;
    cudaMemsetAsync(workspace, 0, 2 * C * sizeof(float), stream);

    // Grid: x-dim covers C, y-dim covers T tiles for parallelism
    int threads = (C < 256) ? C : 256;
    int blocks_x = (C + threads - 1) / threads;
    // Use enough y-blocks to saturate the GPU (target ~2048 total blocks)
    int blocks_y = (T < 256) ? T : 256;
    dim3 grid(blocks_x, blocks_y);

    instnorm_sum_kernel<<<grid, threads, 0, stream>>>(
        x, red_sum, red_sum_sq, C, T);

    instnorm_style_norm_kernel<<<grid, threads, 0, stream>>>(
        x, red_sum, red_sum_sq, norm_w, norm_b, gamma, beta, y, C, T, eps);
}

// ---------------------------------------------------------------------------
// Adaptive LayerNorm: normalize rows, then apply style conditioning
//   x, y: [N, D], gamma, beta: [D] (from style FC)
//   y[n,d] = (1 + gamma[d]) * (x[n,d] - mean) / sqrt(var + eps) + beta[d]
//   One warp per row.
// ---------------------------------------------------------------------------

__global__ void ada_layer_norm_kernel(const float* x, const float* gamma,
                                       const float* beta, float* y,
                                       int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32)
        sum += x_row[d];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    float mean = __shfl_sync(0xFFFFFFFF, sum, 0) / D;

    float var_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32) {
        float diff = x_row[d] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    float inv_std = rsqrtf(__shfl_sync(0xFFFFFFFF, var_sum, 0) / D + eps);

    for (int d = threadIdx.x; d < D; d += 32) {
        float val = (x_row[d] - mean) * inv_std;
        y_row[d] = (1.0f + gamma[d]) * val + beta[d];
    }
}

void ada_layer_norm_f32(const float* x, const float* gamma, const float* beta,
                         float* y, int N, int D, float eps,
                         cudaStream_t stream) {
    ada_layer_norm_kernel<<<N, 32, 0, stream>>>(x, gamma, beta, y, N, D, eps);
}

// ---------------------------------------------------------------------------
// Depthwise transposed Conv1d
//   x: [T_in, C], w: [C, 1, K], bias: [C], y: [T_out, C]
//   T_out = (T_in - 1) * stride - 2*pad + K + out_pad
//   groups = C (depthwise: each channel processed independently)
// ---------------------------------------------------------------------------

__global__ void conv_transpose1d_depthwise_kernel(const float* __restrict__ x,
                                                    const float* __restrict__ w,
                                                    const float* __restrict__ bias,
                                                    float* __restrict__ y,
                                                    int C, int T_in, int T_out,
                                                    int K, int stride, int pad) {
    int total = C * T_out;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int t_out = idx / C;
        int c = idx % C;

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            int t_up = t_out + pad - k;
            if (t_up >= 0 && t_up % stride == 0) {
                int t_in = t_up / stride;
                if (t_in >= 0 && t_in < T_in) {
                    sum += w[c * K + k] * x[t_in * C + c];
                }
            }
        }
        y[idx] = sum + bias[c];
    }
}

void conv_transpose1d_depthwise_f32(const float* x, const float* w, const float* bias,
                                      float* y, int C, int T_in, int K,
                                      int stride, int pad, int out_pad,
                                      cudaStream_t stream) {
    int T_out = (T_in - 1) * stride - 2 * pad + K + out_pad;
    int total = C * T_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_depthwise_kernel<<<blocks, threads, 0, stream>>>(
        x, w, bias, y, C, T_in, T_out, K, stride, pad);
}

// ---------------------------------------------------------------------------
// Nearest-neighbor 2x upsampling: x[T, C] -> y[2*T, C]
// ---------------------------------------------------------------------------

__global__ void upsample_nearest_1d_2x_kernel(const float* x, float* y,
                                                int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * 2 * T;
    if (idx < total) {
        int t_out = idx / C;
        int c = idx % C;
        int t_in = t_out / 2;
        y[idx] = x[t_in * C + c];
    }
}

void upsample_nearest_1d_2x_f32(const float* x, float* y, int C, int T,
                                  cudaStream_t stream) {
    int total = C * 2 * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_nearest_1d_2x_kernel<<<blocks, threads, 0, stream>>>(x, y, C, T);
}

// ---------------------------------------------------------------------------
// Scale: y = x * scalar
// ---------------------------------------------------------------------------

__global__ void scale_kernel(const float* x, float* y, int N, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = x[i] * scalar;
}

void scale_f32(const float* x, float* y, int N, float scalar,
               cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, stream>>>(x, y, N, scalar);
}

// ---------------------------------------------------------------------------
// Fused sigmoid + sum reduce over last dimension
//   x: [N, D], y: [N]
//   y[n] = sum_d sigmoid(x[n, d])
//   One warp per row.
// ---------------------------------------------------------------------------

__global__ void sigmoid_sum_kernel(const float* x, float* y, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * D;
    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32)
        sum += 1.0f / (1.0f + expf(-x_row[d]));

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (threadIdx.x == 0)
        y[row] = sum;
}

void sigmoid_sum_f32(const float* x, float* y, int N, int D,
                      cudaStream_t stream) {
    sigmoid_sum_kernel<<<N, 32, 0, stream>>>(x, y, N, D);
}

// ---------------------------------------------------------------------------
// Tile 1D: broadcast x[C] to y[T, C] (repeat each element T times)
// ---------------------------------------------------------------------------

__global__ void tile_1d_kernel(const float* x, float* y, int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T;
    if (idx < total) {
        int c = idx % C;
        y[idx] = x[c];
    }
}

void tile_1d_f32(const float* x, float* y, int C, int T,
                  cudaStream_t stream) {
    int total = C * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    tile_1d_kernel<<<blocks, threads, 0, stream>>>(x, y, C, T);
}

// ---------------------------------------------------------------------------
// Channel bias add: y[t, c] += bias[c]  for all t
//   y: [T, C], bias: [C]. Flat iteration, extract c = idx % C.
// ---------------------------------------------------------------------------

__global__ void channel_bias_add_kernel(float* y, const float* bias, int C, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx % C;
        y[idx] += bias[c];
    }
}

void channel_bias_add_f32(float* y, const float* bias, int C, int T,
                            cudaStream_t stream) {
    int total = C * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    channel_bias_add_kernel<<<blocks, threads, 0, stream>>>(y, bias, C, total);
}

// ---------------------------------------------------------------------------
// Generalized Conv1d: supports stride, dilation, explicit padding
//   x: [T_in, C_in], y: [T_out, C_out], w: [C_out, C_in, K]
//   One warp per (c_out, t_out) pair.
// ---------------------------------------------------------------------------

__global__ void conv1d_general_kernel(const float* x, const float* w, const float* bias,
                                       float* y, int C_in, int C_out, int T_in, int T_out,
                                       int K, int stride, int padding, int dilation) {
    int co = blockIdx.x;
    int t = blockIdx.y;
    if (co >= C_out || t >= T_out) return;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < C_in * K; idx += blockDim.x) {
        int ci = idx / K;
        int k = idx % K;
        int t_in = t * stride - padding + k * dilation;
        if (t_in >= 0 && t_in < T_in) {
            sum += w[co * C_in * K + ci * K + k] * x[t_in * C_in + ci];
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (threadIdx.x == 0) {
        y[t * C_out + co] = sum + (bias ? bias[co] : 0.0f);
    }
}

void conv1d_general_f32(const float* x, const float* w, const float* bias,
                        float* y, int C_in, int C_out, int T_in, int K,
                        int stride, int padding, int dilation,
                        cudaStream_t stream) {
    int T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    dim3 blocks(C_out, T_out);
    conv1d_general_kernel<<<blocks, 32, 0, stream>>>(
        x, w, bias, y, C_in, C_out, T_in, T_out, K, stride, padding, dilation);
}

// ---------------------------------------------------------------------------
// ConvTranspose1d (non-depthwise, groups=1)
//   x: [T_in, C_in], y: [T_out, C_out], w: [C_in, C_out, K]
//   For each output position, gather from valid input positions.
// ---------------------------------------------------------------------------

__global__ void conv_transpose1d_kernel(const float* x, const float* w, const float* bias,
                                         float* y, int C_in, int C_out, int T_in, int T_out,
                                         int K, int stride, int padding) {
    int co = blockIdx.x;
    int t_out = blockIdx.y;
    if (co >= C_out || t_out >= T_out) return;

    float sum = 0.0f;
    // For each input channel, sum contributions
    for (int idx = threadIdx.x; idx < C_in * K; idx += blockDim.x) {
        int ci = idx / K;
        int k = idx % K;
        int t_up = t_out + padding - k;
        if (t_up >= 0 && t_up % stride == 0) {
            int t_in = t_up / stride;
            if (t_in >= 0 && t_in < T_in) {
                sum += w[ci * C_out * K + co * K + k] * x[t_in * C_in + ci];
            }
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (threadIdx.x == 0) {
        y[t_out * C_out + co] = sum + (bias ? bias[co] : 0.0f);
    }
}

void conv_transpose1d_f32(const float* x, const float* w, const float* bias,
                          float* y, int C_in, int C_out, int T_in, int K,
                          int stride, int padding, int output_padding,
                          cudaStream_t stream) {
    int T_out = (T_in - 1) * stride - 2 * padding + K + output_padding;
    dim3 blocks(C_out, T_out);
    conv_transpose1d_kernel<<<blocks, 32, 0, stream>>>(
        x, w, bias, y, C_in, C_out, T_in, T_out, K, stride, padding);
}

// ---------------------------------------------------------------------------
// Snake activation: y = x + (1/alpha) * sin(alpha * x)^2
//   alpha: [C] (one per channel), x: [T, C]
// ---------------------------------------------------------------------------

__global__ void snake_kernel(const float* x, const float* alpha, float* y,
                              int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T;
    if (idx < total) {
        int c = idx % C;
        float a = alpha[c];
        float val = x[idx];
        float s = sinf(a * val);
        y[idx] = val + (1.0f / a) * s * s;
    }
}

void snake_f32(const float* x, const float* alpha, float* y,
               int C, int T, cudaStream_t stream) {
    int total = C * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    snake_kernel<<<blocks, threads, 0, stream>>>(x, alpha, y, C, T);
}

// ---------------------------------------------------------------------------
// Nearest-neighbor upsampling with arbitrary factor
//   x: [T_in, C], y: [T_in * factor, C]
// ---------------------------------------------------------------------------

__global__ void upsample_nearest_kernel(const float* x, float* y, int C,
                                          int T_in, int factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int T_out = T_in * factor;
    int total = C * T_out;
    if (idx < total) {
        int t_out = idx / C;
        int c = idx % C;
        int t_in = t_out / factor;
        y[idx] = x[t_in * C + c];
    }
}

void upsample_nearest_f32(const float* x, float* y, int C, int T_in,
                           int factor, cudaStream_t stream) {
    int total = C * T_in * factor;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_nearest_kernel<<<blocks, threads, 0, stream>>>(x, y, C, T_in, factor);
}

// ---------------------------------------------------------------------------
// Reflection pad 1D: x[T, C] -> y[T+pad_left+pad_right, C]
// ---------------------------------------------------------------------------

__global__ void reflection_pad_1d_kernel(const float* x, float* y, int C,
                                           int T, int T_out,
                                           int pad_left, int pad_right) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T_out;
    if (idx < total) {
        int t = idx / C;
        int c = idx % C;
        int t_src;
        if (t < pad_left) {
            t_src = pad_left - t;  // reflect from left
        } else if (t >= pad_left + T) {
            t_src = 2 * T + pad_left - t - 2;  // reflect from right
        } else {
            t_src = t - pad_left;
        }
        y[idx] = x[t_src * C + c];
    }
}

void reflection_pad_1d_f32(const float* x, float* y, int C, int T,
                            int pad_left, int pad_right,
                            cudaStream_t stream) {
    int T_out = T + pad_left + pad_right;
    int total = C * T_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reflection_pad_1d_kernel<<<blocks, threads, 0, stream>>>(
        x, y, C, T, T_out, pad_left, pad_right);
}

// ---------------------------------------------------------------------------
// Element-wise exp
// ---------------------------------------------------------------------------

__global__ void exp_kernel(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = expf(x[i]);
}

void exp_f32(const float* x, float* y, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    exp_kernel<<<blocks, threads, 0, stream>>>(x, y, N);
}

// ---------------------------------------------------------------------------
// Element-wise sin
// ---------------------------------------------------------------------------

__global__ void sin_kernel(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = sinf(x[i]);
}

void sin_f32(const float* x, float* y, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sin_kernel<<<blocks, threads, 0, stream>>>(x, y, N);
}

// ---------------------------------------------------------------------------
// Element-wise tanh
// ---------------------------------------------------------------------------

__global__ void tanh_kernel(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = tanhf(x[i]);
}

void tanh_f32(const float* x, float* y, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads, 0, stream>>>(x, y, N);
}

// ---------------------------------------------------------------------------
// STFT forward: magnitude and phase from time-domain signal
//   Uses center=True (reflect-pad by n_fft/2 on each side), Hann window.
//   One thread block per frame, one warp computes DFT for each freq bin.
// ---------------------------------------------------------------------------

__global__ void stft_kernel(const float* x_padded, float* mag, float* phase,
                             int T_padded, int n_fft, int hop_length,
                             int n_frames, int n_freqs) {
    int frame = blockIdx.x;
    int freq = blockIdx.y;
    if (frame >= n_frames || freq >= n_freqs) return;

    int start = frame * hop_length;

    // Compute DFT at this frequency using Hann window
    float real_sum = 0.0f, imag_sum = 0.0f;
    float inv_nfft = 1.0f / n_fft;

    for (int n = threadIdx.x; n < n_fft; n += blockDim.x) {
        // Hann window: 0.5 * (1 - cos(2*pi*n / n_fft))  [periodic]
        float w = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * n * inv_nfft));
        float sample = (start + n < T_padded) ? x_padded[start + n] : 0.0f;
        float windowed = sample * w;

        // DFT: X[k] = sum_n x[n] * exp(-2pi*j*k*n/N)
        float angle = -2.0f * 3.14159265358979323846f * freq * n * inv_nfft;
        real_sum += windowed * cosf(angle);
        imag_sum += windowed * sinf(angle);
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        real_sum += __shfl_down_sync(0xFFFFFFFF, real_sum, offset);
        imag_sum += __shfl_down_sync(0xFFFFFFFF, imag_sum, offset);
    }

    if (threadIdx.x == 0) {
        int idx = freq * n_frames + frame;
        float m = sqrtf(real_sum * real_sum + imag_sum * imag_sum);
        float p = atan2f(imag_sum, real_sum);
        mag[idx] = m;
        phase[idx] = p;
    }
}

void stft_f32(const float* x, float* mag, float* phase,
              int T_signal, int n_fft, int hop_length,
              cudaStream_t stream) {
    int pad = n_fft / 2;
    int T_padded = T_signal + 2 * pad;
    int n_frames = (T_padded - n_fft) / hop_length + 1;
    int n_freqs = n_fft / 2 + 1;

    // Create reflect-padded signal on GPU
    float* x_padded;
    cudaMalloc(&x_padded, T_padded * sizeof(float));

    // Pad with reflection: [pad-1, pad-2, ..., 0, 0, 1, ..., T-1, T-2, T-3, ...]
    reflection_pad_1d_f32(x, x_padded, 1, T_signal, pad, pad, stream);

    dim3 blocks(n_frames, n_freqs);
    stft_kernel<<<blocks, 32, 0, stream>>>(x_padded, mag, phase, T_padded,
                                             n_fft, hop_length, n_frames, n_freqs);

    cudaFree(x_padded);
}

// ---------------------------------------------------------------------------
// iSTFT inverse: reconstruct time-domain from magnitude and phase
//   Uses overlap-add with Hann window, center=True padding stripped.
//   mag, phase: [n_freqs, n_frames] where n_freqs = n_fft/2+1
// ---------------------------------------------------------------------------

__global__ void istft_kernel(const float* mag, const float* phase,
                              float* y_padded, float* window_sum,
                              int n_fft, int hop_length, int n_frames,
                              int n_freqs, int T_padded) {
    // Each thread block handles one frame
    int frame = blockIdx.x;
    if (frame >= n_frames) return;

    int start = frame * hop_length;
    float inv_nfft = 1.0f / n_fft;

    // For each sample in this frame's window
    for (int n = threadIdx.x; n < n_fft; n += blockDim.x) {
        // Hann window
        float w = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * n * inv_nfft));

        // Inverse DFT at position n: x[n] = (1/N) * sum_k X[k] * exp(2pi*j*k*n/N)
        // But for real-valued signals, we use the Hermitian symmetry
        float val = 0.0f;
        for (int k = 0; k < n_freqs; k++) {
            int idx = k * n_frames + frame;
            float m = mag[idx];
            float p = phase[idx];
            float angle = 2.0f * 3.14159265358979323846f * k * n * inv_nfft;
            float real_part = m * cosf(p);
            float imag_part = m * sinf(p);
            // Re(X[k] * exp(j*angle)) = real*cos - imag*sin
            float contribution = real_part * cosf(angle) - imag_part * sinf(angle);
            // Count both positive and negative frequencies
            if (k == 0 || k == n_freqs - 1) {
                val += contribution;
            } else {
                val += 2.0f * contribution;
            }
        }
        val *= inv_nfft;  // normalize

        int out_pos = start + n;
        if (out_pos < T_padded) {
            atomicAdd(&y_padded[out_pos], val * w);
            atomicAdd(&window_sum[out_pos], w * w);
        }
    }
}

__global__ void istft_normalize_kernel(const float* __restrict__ y_padded,
                                       const float* __restrict__ window_sum,
                                       float* __restrict__ y,
                                       int T_signal, int pad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_signal) return;
    float ws = window_sum[t + pad];
    y[t] = (ws > 1e-8f) ? y_padded[t + pad] / ws : 0.0f;
}

void istft_f32(const float* mag, const float* phase, float* y,
               int n_frames, int n_fft, int hop_length, int T_signal,
               cudaStream_t stream) {
    int pad = n_fft / 2;
    int T_padded = n_fft + hop_length * (n_frames - 1);

    float *y_padded, *window_sum;
    cudaMalloc(&y_padded, T_padded * sizeof(float));
    cudaMalloc(&window_sum, T_padded * sizeof(float));
    cudaMemsetAsync(y_padded, 0, T_padded * sizeof(float), stream);
    cudaMemsetAsync(window_sum, 0, T_padded * sizeof(float), stream);

    int n_freqs = n_fft / 2 + 1;
    istft_kernel<<<n_frames, 32, 0, stream>>>(
        mag, phase, y_padded, window_sum,
        n_fft, hop_length, n_frames, n_freqs, T_padded);

    // Normalize by window sum and strip center padding on GPU
    // y[t] = y_padded[t + pad] / window_sum[t + pad]
    {
        int threads = 256;
        int blocks = (T_signal + threads - 1) / threads;
        istft_normalize_kernel<<<blocks, threads, 0, stream>>>(
            y_padded, window_sum, y, T_signal, pad);
    }

    cudaFreeAsync(y_padded, stream);
    cudaFreeAsync(window_sum, stream);
}

// ---------------------------------------------------------------------------
// LSTM gate activation kernel (kept for backwards compatibility)
// ---------------------------------------------------------------------------

__global__ void lstm_gates_kernel(const float* gates, const float* c_prev,
                                   float* c_out, float* h_out, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;

    float ig = 1.0f / (1.0f + expf(-gates[i]));
    float fg = 1.0f / (1.0f + expf(-gates[H + i]));
    float gg = tanhf(gates[2 * H + i]);
    float og = 1.0f / (1.0f + expf(-gates[3 * H + i]));

    float c = fg * c_prev[i] + ig * gg;
    c_out[i] = c;
    h_out[i] = og * tanhf(c);
}

void lstm_gates_f32(const float* gates, const float* c_prev,
                     float* c_out, float* h_out, int H,
                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (H + threads - 1) / threads;
    lstm_gates_kernel<<<blocks, threads, 0, stream>>>(gates, c_prev, c_out, h_out, H);
}

// ---------------------------------------------------------------------------
// Fused LSTM: run ALL timesteps in a single kernel launch
//   Pre-computed input gates: ig[T, 4H] row-major (Wih@x + bih + bhh)
//   Whh: [4H, H] row-major
//   Output: h_all[T, H] row-major
//   Each thread handles one hidden unit j, loops over all timesteps.
//   The Whh @ h GEMV and gate nonlinearities are computed per-thread.
//   H threads per block (1 block total). H must be <= 1024.
// ---------------------------------------------------------------------------

__global__ void fused_lstm_kernel(
    const float* __restrict__ Whh,       // [4H, H]
    const float* __restrict__ ig_all,    // [T, 4H]
    float* __restrict__ h_all,           // [T, H]
    int T, int H, int reverse) {

    extern __shared__ float s_h[];  // [H]

    int j = threadIdx.x;
    if (j >= H) return;

    // Initialize h = 0, c = 0
    s_h[j] = 0.0f;
    float c_j = 0.0f;
    __syncthreads();

    for (int step = 0; step < T; step++) {
        int t = reverse ? (T - 1 - step) : step;

        // Compute 4 gates: gates[g] = ig[t, g*H+j] + Whh[(g*H+j), :] @ h
        const float* ig_t = ig_all + t * 4 * H;
        float gates[4];

        for (int g = 0; g < 4; g++) {
            float sum = ig_t[g * H + j];
            const float* w_row = Whh + (g * H + j) * H;
            for (int k = 0; k < H; k++)
                sum += w_row[k] * s_h[k];
            gates[g] = sum;
        }

        // Gate nonlinearities
        float i_g = 1.0f / (1.0f + expf(-gates[0]));
        float f_g = 1.0f / (1.0f + expf(-gates[1]));
        float g_g = tanhf(gates[2]);
        float o_g = 1.0f / (1.0f + expf(-gates[3]));

        // Update cell and hidden state
        c_j = f_g * c_j + i_g * g_g;
        float h_new = o_g * tanhf(c_j);

        // Write output
        h_all[t * H + j] = h_new;

        // Sync: ensure all threads finished reading s_h before any writes
        __syncthreads();
        s_h[j] = h_new;
        __syncthreads();
    }
}

void fused_lstm_f32(const float* Whh, const float* ig_all,
                     float* h_all, int T, int H, int reverse,
                     cudaStream_t stream) {
    // Launch H threads per block, 1 block
    fused_lstm_kernel<<<1, H, H * sizeof(float), stream>>>(
        Whh, ig_all, h_all, T, H, reverse);
}

// ---------------------------------------------------------------------------
// im2col for 1D convolution
//   x: [T_in, C_in] → col: [T_out, C_in*K]  (row-major)
//   T_out = (T_in + 2*padding - dilation*(K-1) - 1) / stride + 1
//   col[t, c*K+k] = x[t*stride - padding + k*dilation, c]  (0 if OOB)
// ---------------------------------------------------------------------------

__global__ void im2col_1d_kernel(const float* __restrict__ x,
                                   float* __restrict__ col,
                                   int C_in, int T_in, int K,
                                   int stride, int padding, int dilation,
                                   int T_out) {
    int CK = C_in * K;
    int total = T_out * CK;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int t = idx / CK;
        int ck = idx % CK;
        int c = ck / K;
        int k = ck % K;
        int t_in = t * stride - padding + k * dilation;
        col[idx] = (t_in >= 0 && t_in < T_in) ? x[t_in * C_in + c] : 0.0f;
    }
}

void im2col_1d_f32(const float* x, float* col,
                    int C_in, int T_in, int K,
                    int stride, int padding, int dilation, int T_out,
                    cudaStream_t stream) {
    int total = C_in * K * T_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    im2col_1d_kernel<<<blocks, threads, 0, stream>>>(
        x, col, C_in, T_in, K, stride, padding, dilation, T_out);
}

// ---------------------------------------------------------------------------
// col2im for 1D transposed convolution (with atomicAdd for overlapping)
//   col: [T_in, C_out*K] → y: [T_out, C_out]
//   T_out = (T_in - 1) * stride + K - 2*padding
//   For each (t_in, c, k): y[t_in*stride+k-padding, c] += col[t_in, c*K+k]
// ---------------------------------------------------------------------------

__global__ void col2im_1d_kernel(const float* __restrict__ col,
                                   float* __restrict__ y,
                                   int C_out, int K, int T_in,
                                   int stride, int padding, int T_out) {
    int COK = C_out * K;
    int total = T_in * COK;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int t_in = idx / COK;
        int ck = idx % COK;
        int c = ck / K;
        int k = ck % K;
        int t_out = t_in * stride + k - padding;
        if (t_out >= 0 && t_out < T_out) {
            atomicAdd(&y[t_out * C_out + c], col[idx]);
        }
    }
}

void col2im_1d_f32(const float* col, float* y,
                    int C_out, int K, int T_in,
                    int stride, int padding, int T_out,
                    cudaStream_t stream) {
    // NOTE: caller must initialize y before calling (memset 0 or tile bias).
    // This kernel uses atomicAdd to accumulate overlapping positions.
    int total = C_out * K * T_in;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    col2im_1d_kernel<<<blocks, threads, 0, stream>>>(
        col, y, C_out, K, T_in, stride, padding, T_out);
}

// ---------------------------------------------------------------------------
// Concatenate along channels for [T, C] layout
//   a: [T, Ca], b: [T, Cb] → y: [T, Ca+Cb]
//   y[t, 0..Ca-1] = a[t, :], y[t, Ca..Ca+Cb-1] = b[t, :]
// ---------------------------------------------------------------------------

__global__ void concat_channels_kernel(const float* a, const float* b, float* y,
                                         int T, int Ca, int Cb) {
    int Cy = Ca + Cb;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * Cy;
    if (idx < total) {
        int t = idx / Cy;
        int c = idx % Cy;
        y[idx] = (c < Ca) ? a[t * Ca + c] : b[t * Cb + (c - Ca)];
    }
}

void concat_channels_f32(const float* a, const float* b, float* y,
                           int T, int Ca, int Cb, cudaStream_t stream) {
    int total = T * (Ca + Cb);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat_channels_kernel<<<blocks, threads, 0, stream>>>(a, b, y, T, Ca, Cb);
}

// ---------------------------------------------------------------------------
// Concatenate 4 tensors along channels for [T, C] layout
//   a: [T, Ca], b: [T, Cb], c: [T, Cc], d: [T, Cd] → y: [T, Ca+Cb+Cc+Cd]
// ---------------------------------------------------------------------------

__global__ void concat4_channels_kernel(const float* a, const float* b,
                                          const float* c, const float* d,
                                          float* y,
                                          int T, int Ca, int Cb, int Cc, int Cd) {
    int Cy = Ca + Cb + Cc + Cd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * Cy;
    if (idx < total) {
        int t = idx / Cy;
        int ch = idx % Cy;
        float val;
        if (ch < Ca) {
            val = a[t * Ca + ch];
        } else if (ch < Ca + Cb) {
            val = b[t * Cb + (ch - Ca)];
        } else if (ch < Ca + Cb + Cc) {
            val = c[t * Cc + (ch - Ca - Cb)];
        } else {
            val = d[t * Cd + (ch - Ca - Cb - Cc)];
        }
        y[idx] = val;
    }
}

void concat4_channels_f32(const float* a, const float* b,
                            const float* c, const float* d,
                            float* y,
                            int T, int Ca, int Cb, int Cc, int Cd,
                            cudaStream_t stream) {
    int total = T * (Ca + Cb + Cc + Cd);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat4_channels_kernel<<<blocks, threads, 0, stream>>>(
        a, b, c, d, y, T, Ca, Cb, Cc, Cd);
}

// ---------------------------------------------------------------------------
// Concatenate 3 tensors along channels for [T, C] layout
//   a: [T, Ca], b: [T, Cb], c: [T, Cc] → y: [T, Ca+Cb+Cc]
// ---------------------------------------------------------------------------

__global__ void concat3_channels_kernel(const float* a, const float* b,
                                          const float* c, float* y,
                                          int T, int Ca, int Cb, int Cc) {
    int Cy = Ca + Cb + Cc;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * Cy;
    if (idx < total) {
        int t = idx / Cy;
        int ch = idx % Cy;
        float val;
        if (ch < Ca) {
            val = a[t * Ca + ch];
        } else if (ch < Ca + Cb) {
            val = b[t * Cb + (ch - Ca)];
        } else {
            val = c[t * Cc + (ch - Ca - Cb)];
        }
        y[idx] = val;
    }
}

void concat3_channels_f32(const float* a, const float* b, const float* c,
                            float* y, int T, int Ca, int Cb, int Cc,
                            cudaStream_t stream) {
    int total = T * (Ca + Cb + Cc);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat3_channels_kernel<<<blocks, threads, 0, stream>>>(
        a, b, c, y, T, Ca, Cb, Cc);
}

// ---------------------------------------------------------------------------
// SineGen phase computation: f0[L2] → phase_low[L2, 9]
//   9 threads (one per harmonic), each does sequential cumsum.
//   Matches PyTorch SineGen: NN upsample → normalize → downsample → cumsum → scale
// ---------------------------------------------------------------------------

__global__ void sinegen_phase_kernel(const float* __restrict__ f0,
                                      const float* __restrict__ rand_ini,
                                      float* __restrict__ phase_low,
                                      int L2) {
    int h = threadIdx.x;
    if (h >= 9) return;

    constexpr int HARMONICS = 9;
    constexpr int SR = 24000;
    constexpr int UPSAMPLE = 300;
    int T_audio = L2 * UPSAMPLE;

    // Compute rad_down[t, h] by simulating NN upsample → fmod → downsample
    for (int t = 0; t < L2; t++) {
        float src = (t + 0.5f) * T_audio / (float)L2 - 0.5f;
        src = fmaxf(0.0f, fminf(src, (float)(T_audio - 1)));
        int lo = (int)src;
        int hi = min(lo + 1, T_audio - 1);
        float frac = src - lo;

        float f0_lo = f0[lo / UPSAMPLE];
        float f0_hi = f0[hi / UPSAMPLE];
        float rad_lo = fmodf(f0_lo * (h + 1) / (float)SR, 1.0f);
        float rad_hi = fmodf(f0_hi * (h + 1) / (float)SR, 1.0f);
        if (rad_lo < 0.0f) rad_lo += 1.0f;
        if (rad_hi < 0.0f) rad_hi += 1.0f;

        // rand_ini only at upsampled position 0
        if (lo == 0) rad_lo += rand_ini[h];
        if (hi == 0) rad_hi += rand_ini[h];

        phase_low[t * HARMONICS + h] = (1.0f - frac) * rad_lo + frac * rad_hi;
    }

    // Cumulative sum
    for (int t = 1; t < L2; t++)
        phase_low[t * HARMONICS + h] += phase_low[(t - 1) * HARMONICS + h];

    // Scale: multiply by 2*pi*UPSAMPLE
    float scale = 2.0f * 3.14159265358979323846f * UPSAMPLE;
    for (int t = 0; t < L2; t++)
        phase_low[t * HARMONICS + h] *= scale;
}

void sinegen_phase_f32(const float* f0, const float* rand_ini,
                        float* phase_low, int L2, cudaStream_t stream) {
    sinegen_phase_kernel<<<1, 9, 0, stream>>>(f0, rand_ini, phase_low, L2);
}

// ---------------------------------------------------------------------------
// SineGen source: phase_low[L2, 9] + f0[L2] → har_source[T_audio]
//   Massively parallel: upsample phase, sin, UV mask, noise, linear combine, tanh
//   Uses hash-based PRNG for noise (different from CPU RNG, but same distribution)
// ---------------------------------------------------------------------------

__global__ void sinegen_source_kernel(const float* __restrict__ phase_low,
                                       const float* __restrict__ f0,
                                       const float* __restrict__ l_linear_w,
                                       const float* __restrict__ l_linear_b,
                                       float* __restrict__ har_source,
                                       int L2, int T_audio, unsigned int seed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_audio) return;

    constexpr int HARMONICS = 9;
    constexpr int UPSAMPLE = 300;
    constexpr float SINE_AMP = 0.1f;
    constexpr float NOISE_STD = 0.003f;
    constexpr float VOICED_THRESH = 10.0f;

    // Upsample phase from [L2, 9] → this position
    float src = (t + 0.5f) * L2 / (float)T_audio - 0.5f;
    src = fmaxf(0.0f, fminf(src, (float)(L2 - 1)));
    int lo = (int)src;
    int hi = min(lo + 1, L2 - 1);
    float frac = src - lo;

    // UV flag from f0 (NN upsampled)
    float f0_val = f0[t / UPSAMPLE];
    float uv = (f0_val > VOICED_THRESH) ? 1.0f : 0.0f;
    float noise_amp = uv * NOISE_STD + (1.0f - uv) * SINE_AMP / 3.0f;

    float sum = l_linear_b[0];
    for (int h = 0; h < HARMONICS; h++) {
        // Interpolate phase
        float phase = (1.0f - frac) * phase_low[lo * HARMONICS + h] +
                      frac * phase_low[hi * HARMONICS + h];
        float sine = __sinf(phase) * SINE_AMP;

        // Hash-based Gaussian noise (Box-Muller transform)
        unsigned int hash = (unsigned int)t * 2654435761u +
                           (unsigned int)h * 340573321u + seed;
        hash ^= hash >> 16; hash *= 0x85ebca6bu;
        hash ^= hash >> 13; hash *= 0xc2b2ae35u; hash ^= hash >> 16;
        float u1 = ((hash & 0xFFFFFFu) + 1u) / 16777217.0f;  // (0, 1)
        hash = hash * 1664525u + 1013904223u;
        float u2 = (hash & 0xFFFFFFu) / 16777216.0f;
        float noise = sqrtf(-2.0f * logf(u1)) *
                     cosf(2.0f * 3.14159265358979323846f * u2) * noise_amp;

        float val = sine * uv + noise;
        sum += val * l_linear_w[h];
    }
    har_source[t] = tanhf(sum);
}

void sinegen_source_f32(const float* phase_low, const float* f0,
                         const float* l_linear_w, const float* l_linear_b,
                         float* har_source, int L2, int T_audio,
                         unsigned int seed, cudaStream_t stream) {
    int threads = 256;
    int blocks = (T_audio + threads - 1) / threads;
    sinegen_source_kernel<<<blocks, threads, 0, stream>>>(
        phase_low, f0, l_linear_w, l_linear_b, har_source,
        L2, T_audio, seed);
}

// ---------------------------------------------------------------------------
// Round+clamp durations on GPU, compute total L
// ---------------------------------------------------------------------------
__global__ void round_clamp_durations_kernel(const float* __restrict__ durations,
                                              int* __restrict__ int_durations,
                                              int* __restrict__ L_out, int T) {
    int tid = threadIdx.x;
    if (tid >= T) return;

    int dur = max(1, __float2int_rn(durations[tid]));
    int_durations[tid] = dur;
    __syncthreads();

    // Thread 0 sums all durations (T is tiny, ~15)
    if (tid == 0) {
        int L = 0;
        for (int i = 0; i < T; i++) L += int_durations[i];
        *L_out = L;
    }
}

void round_clamp_durations_f32(const float* durations, int* int_durations,
                                int* L_out, int T, cudaStream_t stream) {
    round_clamp_durations_kernel<<<1, T, 0, stream>>>(
        durations, int_durations, L_out, T);
}

// ---------------------------------------------------------------------------
// Build alignment matrix on GPU: alignment[T, L] from int_durations[T]
// ---------------------------------------------------------------------------
__global__ void build_alignment_kernel(const int* __restrict__ int_durations,
                                        float* __restrict__ alignment,
                                        int T, int L) {
    int tid = threadIdx.x;
    if (tid >= T) return;

    // Compute frame offset via prefix sum (T is tiny)
    int offset = 0;
    for (int i = 0; i < tid; i++) offset += int_durations[i];
    int dur = int_durations[tid];

    // Fill row: 1.0 in [offset, offset+dur), 0.0 elsewhere
    for (int col = 0; col < L; col++) {
        alignment[tid * L + col] = (col >= offset && col < offset + dur) ? 1.0f : 0.0f;
    }
}

void build_alignment_f32(const int* int_durations, float* alignment,
                          int T, int L, cudaStream_t stream) {
    build_alignment_kernel<<<1, T, 0, stream>>>(
        int_durations, alignment, T, L);
}
