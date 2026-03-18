// rokoko_f16.cpp — FP16 TTS CUDA inference (pre-baked FP16 weights from v2 file)
//
// Provides rokoko_infer(), precompute_weight_norms().

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "rokoko_common.h"
#include "kernels.h"

// ---------------------------------------------------------------------------
// Cutlass GEMM — FP32 (activation-only NT, stays FP32)
// ---------------------------------------------------------------------------
extern "C" int cutlass_gemm_nt(int M, int N, int K,
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

// ---------------------------------------------------------------------------
// Cutlass GEMM — FP16 (extern, defined in cutlass_gemm_f16.cu)
// ---------------------------------------------------------------------------
extern "C" int cutlass_gemm_tn_f16(int M, int N, int K,
    const __half* A, int lda, const __half* B, int ldb,
    float* C, int ldc, float alpha, float beta,
    float* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" int cutlass_gemm_tn_bias_f16(int M, int N, int K,
    const __half* A, int lda, const __half* B, int ldb,
    float* D, int ldd, const float* bias,
    float* workspace, size_t workspace_bytes, cudaStream_t stream);
extern "C" int cutlass_gemm_nn_f16(int M, int N, int K,
    const __half* A, int lda, const __half* B, int ldb,
    float* C, int ldc, float alpha, float beta,
    float* workspace, size_t workspace_bytes, cudaStream_t stream);

// Workspace pointer + size, threaded through all functions
static float* s_workspace = nullptr;
static size_t s_workspace_bytes = 0;

// FP16 activation staging buffer (pre-allocated, reused across all GEMM/Conv calls)
static __half* s_fp16_buf = nullptr;
static size_t s_fp16_buf_size = 0;  // in bytes

// ---------------------------------------------------------------------------
// GEMM wrappers — FP16 direct (no map lookup, no fallback)
// ---------------------------------------------------------------------------

// C = alpha * A^T * B + beta * C  (A is FP16 weight, B is FP32 activation)
static void sgemm_tn(int m, int n, int k,
                      const __half* A, int lda,
                      const float* B, int ldb,
                      float* C, int ldc,
                      cudaStream_t stream,
                      float alpha = 1.0f, float beta = 0.0f) {
    if (n == 1) {
        gemv_tn_f16(A, lda, B, C, m, k, alpha, beta, stream);
        return;
    }
    cast_f32_to_f16(B, s_fp16_buf, n * k, stream);
    cutlass_gemm_tn_f16(m, n, k, A, lda, s_fp16_buf, ldb,
                         C, ldc, alpha, beta, s_workspace, s_workspace_bytes, stream);
}

// C = alpha * A * B^T + beta * C  (activation-only, stays FP32)
static void sgemm_nt(int m, int n, int k,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float* C, int ldc,
                      cudaStream_t stream,
                      float alpha = 1.0f, float beta = 0.0f) {
    cutlass_gemm_nt(m, n, k, A, lda, B, ldb, C, ldc,
                     alpha, beta, s_workspace, s_workspace_bytes, stream);
}

// C = alpha * A * B + beta * C  (A is FP16 weight, B is FP32 activation)
static void sgemm_nn(int m, int n, int k,
                      const __half* A, int lda,
                      const float* B, int ldb,
                      float* C, int ldc,
                      cudaStream_t stream,
                      float alpha = 1.0f, float beta = 0.0f) {
    cast_f32_to_f16(B, s_fp16_buf, n * k, stream);
    cutlass_gemm_nn_f16(m, n, k, A, lda, s_fp16_buf, ldb,
                         C, ldc, alpha, beta, s_workspace, s_workspace_bytes, stream);
}

// C = A^T * B + bias  (A is FP16 weight)
static void sgemm_bias(int m, int n, int k,
                        const __half* A, int lda,
                        const float* B, int ldb,
                        float* C, int ldc,
                        const float* bias,
                        cudaStream_t stream) {
    cast_f32_to_f16(B, s_fp16_buf, n * k, stream);
    cutlass_gemm_tn_bias_f16(m, n, k, A, lda, s_fp16_buf, ldb,
                              C, ldc, bias, s_workspace, s_workspace_bytes, stream);
}

// ---------------------------------------------------------------------------
// Cutlass implicit GEMM Conv1d — FP16 (extern, defined in cutlass_conv_f16.cu)
// ---------------------------------------------------------------------------
extern "C" int cutlass_conv1d_fprop_f16(const __half* x, const __half* w,
                                          const float* bias,
                                          float* y, const float* residual,
                                          float* workspace, size_t workspace_bytes,
                                          int C_in, int C_out, int T_in, int K,
                                          int stride, int padding, int dilation,
                                          cudaStream_t stream);

// ---------------------------------------------------------------------------
// Conv1d forward: FP16 Cutlass (NHWC weights), K=1 falls back to GEMM.
// ---------------------------------------------------------------------------

// gemm_conv1d: Conv1d forward.
//   When residual != nullptr: y = conv(x,w) + residual, then bias_add(y, bias).
//   When residual == nullptr: y = conv(x,w) + bias (fused in Cutlass epilogue).
static void gemm_conv1d(const float* x, const __half* w, const float* bias,
                         float* y, float* workspace, size_t workspace_bytes,
                         int C_in, int C_out, int T_in, int K,
                         int stride, int padding, int dilation,
                         cudaStream_t stream,
                         const float* residual = nullptr,
                         int C_in_pad = 0) {
    int T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    // Cutlass FP16 implicit GEMM conv (w is NHWC FP16, works for all K)
    int actual_cin = C_in_pad ? C_in_pad : C_in;
    if (C_in_pad) {
        cast_f32_to_f16_pad(x, s_fp16_buf, T_in, C_in, C_in_pad, stream);
    } else {
        cast_f32_to_f16(x, s_fp16_buf, T_in * C_in, stream);
    }
    const float* cutlass_bias = residual ? nullptr : bias;
    cutlass_conv1d_fprop_f16(s_fp16_buf, w, cutlass_bias, y, residual,
                              workspace, workspace_bytes,
                              actual_cin, C_out, T_in, K,
                              stride, padding, dilation, stream);
    if (residual && bias)
        channel_bias_add_f32(y, bias, C_out, T_out, stream);
}

// im2col + GEMM ConvTranspose1d: GEMM + col2im
//   x[T_in, C_in] * w[C_in, C_out, K] → y[T_out, C_out]
//   T_out = (T_in - 1) * stride - 2*padding + K + output_padding
static void gemm_conv_transpose1d(const float* x, const __half* w, const float* bias,
                                    float* y, float* workspace, size_t workspace_bytes,
                                    int C_in, int C_out, int T_in, int K,
                                    int stride, int padding, int output_padding,
                                    cudaStream_t stream) {
    int T_out = (T_in - 1) * stride - 2 * padding + K + output_padding;
    int COK = C_out * K;

    // GEMM: C[COK,T_in] = A[COK,C_in] * B[C_in,T_in]  (OP_N, OP_N)
    float* col_buf = workspace;
    sgemm_nn(COK, T_in, C_in, w, COK, x, C_in, col_buf, COK, stream);

    // Initialize y: broadcast bias or zero
    if (bias) {
        tile_1d_f32(bias, y, C_out, T_out, stream);
    } else {
        cudaMemsetAsync(y, 0, (size_t)C_out * T_out * sizeof(float), stream);
    }

    // col2im: scatter col[T_in, COK] → y[T_out, C_out] via atomicAdd
    col2im_1d_f32(col_buf, y, C_out, K, T_in, stride, padding, T_out, stream);
}

// ---------------------------------------------------------------------------
// ALBERT encoder forward pass
// ---------------------------------------------------------------------------

// Run ALBERT encoder: input_ids[T] -> hidden[T, 768]
static void albert_forward(const Weights& w, AlbertBuffers& buf,
                           cudaStream_t stream, int T) {
    // Step 1: Embeddings
    embedding_gather(w.bert_word_embed, buf.token_ids, buf.emb, T, 128, stream);
    add_f32(buf.emb, w.bert_pos_embed, buf.emb, T * 128, stream);
    bias_add_f32(buf.emb, w.bert_type_embed, buf.emb, T, 128, stream);
    layer_norm_f32(buf.emb, w.bert_embed_ln_w, w.bert_embed_ln_b,
                   buf.emb, T, 128, 1e-12f, stream);

    // Step 2: Project 128 -> 768 (GEMM + bias)
    sgemm_bias(768, T, 128,
               w.bert_proj_w_f16, 128, buf.emb, 128, buf.hidden, 768,
               w.bert_proj_b, stream);

    // Step 3: Shared ALBERT layer x12
    auto& a = w.albert;
    for (int layer = 0; layer < 12; layer++) {
        // --- Self-attention ---
        float* Q = buf.qkv;
        float* K = buf.qkv + T * 768;
        float* V = buf.qkv + T * 768 * 2;

        sgemm_bias(768, T, 768,
                   a.q_w_f16, 768, buf.hidden, 768, Q, 768, a.q_b, stream);
        sgemm_bias(768, T, 768,
                   a.k_w_f16, 768, buf.hidden, 768, K, 768, a.k_b, stream);
        sgemm_bias(768, T, 768,
                   a.v_w_f16, 768, buf.hidden, 768, V, 768, a.v_b, stream);

        // Multi-head attention: 12 heads, 64 dim each
        // scores[h] = Q_h @ K_h^T / sqrt(64)
        float scale = 1.0f / sqrtf(64.0f);
        cutlass_gemm_batched_tn(T, T, 64,
            K, 768, 64,
            Q, 768, 64,
            buf.attn_scores, T, (long long)T * T,
            12, scale, 0.0f,
            s_workspace, s_workspace_bytes, stream);

        // Softmax over last dim of scores[h, t, :]
        softmax_f32(buf.attn_scores, buf.attn_scores, 12 * T, T, stream);

        // Attn output: context = scores @ V per head
        cutlass_gemm_batched_nn(64, T, T,
            V, 768, 64,
            buf.attn_scores, T, (long long)T * T,
            buf.attn_out, 768, 64,
            12, 1.0f, 0.0f,
            s_workspace, s_workspace_bytes, stream);

        // Dense projection: attn_out @ dense_w^T + dense_b -> [T, 768]
        sgemm_bias(768, T, 768,
                   a.dense_w_f16, 768, buf.attn_out, 768, buf.temp, 768,
                   a.dense_b, stream);

        // Fused Residual + LayerNorm (attention)
        residual_layer_norm_f32(buf.hidden, buf.temp, a.attn_ln_w, a.attn_ln_b,
                                 buf.hidden, T, 768, 1e-12f, stream);

        // --- FFN ---
        sgemm_bias(2048, T, 768,
                   a.ffn_w_f16, 768, buf.hidden, 768, buf.ff_mid, 2048,
                   a.ffn_b, stream);
        gelu_f32(buf.ff_mid, buf.ff_mid, T * 2048, stream);

        sgemm_bias(768, T, 2048,
                   a.ffn_out_w_f16, 2048, buf.ff_mid, 2048, buf.ff_out, 768,
                   a.ffn_out_b, stream);

        // Fused Residual + LayerNorm (FFN)
        residual_layer_norm_f32(buf.hidden, buf.ff_out, a.ffn_ln_w, a.ffn_ln_b,
                                 buf.hidden, T, 768, 1e-12f, stream);
    }
}

// ---------------------------------------------------------------------------
// Text encoder forward pass
// ---------------------------------------------------------------------------

// Forward declaration
static void bilstm_gpu(const float* d_input, float* d_output,
                        const Weights::BiLSTMWeights& lstm_w,
                        int T, int input_size, int hidden_size,
                        cudaStream_t stream, GpuArena& arena);

// Run text encoder: input_ids[T] -> output[T, 512]
static void text_encoder_forward(const Weights& w, TextEncoderBuffers& buf,
                                  const int* token_ids_gpu,
                                  cudaStream_t stream,
                                  int T, GpuArena& arena,
                                  float* workspace = nullptr,
                                  size_t workspace_bytes = 0) {
    // Step 1: Embedding lookup -> [T, 512] — already in [T, C] layout
    float* te_embed_w = w.get("text_encoder.embedding.weight");
    embedding_gather(te_embed_w, token_ids_gpu, buf.emb, T, 512, stream);

    // Step 2: 3 Conv blocks — all kernels now operate on [T, C]
    float* x = buf.emb;  // current activation [T, 512]
    for (int i = 0; i < 3; i++) {
        auto& blk = w.text_conv[i];

        // Conv1d: [T, 512] -> [T, 512] — using pre-baked NHWC FP16 weights
        gemm_conv1d(x, blk.conv_wv_nhwc_f16, blk.conv_b, buf.conv_out,
                     workspace, workspace_bytes, 512, 512, T, 5, 1, 2, 1, stream);

        // LayerNorm across channels
        layer_norm_channels_first_f32(buf.conv_out, blk.ln_w, blk.ln_b,
                                       buf.conv_out, 512, T, 1e-5f, stream);

        // LeakyReLU(0.2)
        leaky_relu_f32(buf.conv_out, buf.conv_out, 512 * T, 0.2f, stream);

        // Swap buffers (output becomes next input)
        float* tmp = x;
        x = buf.conv_out;
        buf.conv_out = tmp;
    }

    // Step 3: Bidirectional LSTM — data is already [T, 512], no transpose needed
    Weights::BiLSTMWeights te_lstm_w;
    te_lstm_w.wih_fwd = w.text_lstm_wih_fwd;
    te_lstm_w.whh_fwd = w.text_lstm_whh_fwd;
    te_lstm_w.bih_fwd = w.text_lstm_bih_fwd;
    te_lstm_w.bhh_fwd = w.text_lstm_bhh_fwd;
    te_lstm_w.wih_rev = w.text_lstm_wih_rev;
    te_lstm_w.whh_rev = w.text_lstm_whh_rev;
    te_lstm_w.bih_rev = w.text_lstm_bih_rev;
    te_lstm_w.bhh_rev = w.text_lstm_bhh_rev;
    te_lstm_w.bias_fwd = w.text_lstm_bias_fwd;
    te_lstm_w.bias_rev = w.text_lstm_bias_rev;
    te_lstm_w.wih_fwd_f16 = w.text_lstm_wih_fwd_f16;
    te_lstm_w.whh_fwd_f16 = w.text_lstm_whh_fwd_f16;
    te_lstm_w.wih_rev_f16 = w.text_lstm_wih_rev_f16;
    te_lstm_w.whh_rev_f16 = w.text_lstm_whh_rev_f16;
    bilstm_gpu(x, buf.lstm_out, te_lstm_w, T, 512, 256, stream, arena);

    // Copy LSTM output back to emb for downstream use
    cudaMemcpyAsync(buf.emb, buf.lstm_out, T * 512 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    // Final result is in buf.emb [T, 512]
}

// ---------------------------------------------------------------------------
// GPU BiLSTM: Cutlass GEMM for gate pre-computation + kernel for nonlinearities
//   input: [T, input_size] on GPU, output: [T, 2*hidden_size] on GPU
//
//   1) Pre-compute input gates for all T at once:
//      ig[T, 4H] = input[T, inp] @ Wih^T[inp, 4H] + (bih + bhh)
//   2) For each timestep: ig[t] += Whh @ h_prev, apply sigmoid/tanh gates
//   3) Both directions, then interleave into output
// ---------------------------------------------------------------------------

static void bilstm_gpu(const float* d_input, float* d_output,
                        const Weights::BiLSTMWeights& lstm_w,
                        int T, int input_size, int hidden_size,
                        cudaStream_t stream, GpuArena& arena) {
    int G = 4 * hidden_size;  // gate_size
    int H = hidden_size;

    // Buffers from arena
    size_t arena_save = arena.save();
    float* d_ig      = arena.alloc<float>(T * G);
    float* d_h_fwd   = arena.alloc<float>(T * H);
    float* d_h_rev   = arena.alloc<float>(T * H);
    float* d_c       = arena.alloc<float>(H);
    float* d_h_zero  = arena.alloc<float>(H);

    // Helper: run one direction (FP16 weights directly)
    auto run_direction = [&](const __half* wih_f16, const __half* whh_f16,
                              const float* bias_combined,
                              float* d_h_all, bool reverse) {
        // Step 1: Pre-compute input gates for all timesteps (Cutlass GEMM)
        sgemm_tn(G, T, input_size,
                 wih_f16, input_size, d_input, input_size, d_ig, G, stream);

        // Add precomputed bias (bih + bhh)
        bias_add_f32(d_ig, bias_combined, d_ig, T, G, stream);

        // Step 2: Per-timestep LSTM using GEMV for Whh @ h
        CUDA_CHECK(cudaMemsetAsync(d_c, 0, H * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_h_zero, 0, H * sizeof(float), stream));

        for (int step = 0; step < T; step++) {
            int t = reverse ? (T - 1 - step) : step;
            int t_prev = reverse ? (t + 1) : (t - 1);
            const float* h_prev = (step == 0) ? d_h_zero : d_h_all + t_prev * H;

            // ig[t] += Whh^T * h_prev  (alpha=1, beta=1 to accumulate)
            gemv_tn_f16(whh_f16, H, h_prev, d_ig + t * G, G, H, 1.0f, 1.0f, stream);
            lstm_gates_f32(d_ig + t * G, d_c, d_c, d_h_all + t * H, H, stream);
        }
    };

    // Forward direction
    run_direction(lstm_w.wih_fwd_f16, lstm_w.whh_fwd_f16,
                  lstm_w.bias_fwd, d_h_fwd, false);

    // Reverse direction
    run_direction(lstm_w.wih_rev_f16, lstm_w.whh_rev_f16,
                  lstm_w.bias_rev, d_h_rev, true);

    // Interleave: d_output[t, 0:H] = h_fwd[t], d_output[t, H:2H] = h_rev[t]
    CUDA_CHECK(cudaMemcpy2DAsync(
        d_output, 2 * H * sizeof(float),
        d_h_fwd, H * sizeof(float),
        H * sizeof(float), T,
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(
        d_output + H, 2 * H * sizeof(float),
        d_h_rev, H * sizeof(float),
        H * sizeof(float), T,
        cudaMemcpyDeviceToDevice, stream));

    arena.restore(arena_save);
}

// ---------------------------------------------------------------------------
// AdaIN1d forward: InstanceNorm(affine) + style conditioning
//   x: [T, C] on GPU, s: [style_dim] on GPU
//   output written to y: [T, C] on GPU
//   Uses AdaIN1dWeights (norm.weight/bias, fc.weight/bias)
// ---------------------------------------------------------------------------

// AdaIN1d forward, optionally fusing snake activation into the norm output.
// When snake_alpha != nullptr: y = snake(norm(x, style), alpha)
static void adain_1d_forward(const float* x, const float* style,
                              const Weights::AdaIN1dWeights& ain_w,
                              float* y, float* fc_buf, float* norm_ws,
                              int C, int T, int style_dim,
                              cudaStream_t stream,
                              const float* snake_alpha = nullptr) {
    sgemm_tn(2*C, 1, style_dim,
             ain_w.fc_w_f16, style_dim, style, style_dim, fc_buf, 2*C, stream);
    bias_add_f32(fc_buf, ain_w.fc_b, fc_buf, 1, 2*C, stream);

    instance_norm_style_affine_f32(x, ain_w.norm_w, ain_w.norm_b,
                                    fc_buf, fc_buf + C, y,
                                    norm_ws, C, T, 1e-5f, stream,
                                    snake_alpha);
}

// ---------------------------------------------------------------------------
// AdainResBlk1d forward
//   x: [T, dim_in] -> y: [T_out, dim_out]
//   T_out = T (no upsample) or 2*T (upsample=True)
// ---------------------------------------------------------------------------

static void adain_resblk1d_forward(const float* x, const float* style,
                                     const Weights::AdainResBlk1dWeights& blk,
                                     float* residual_buf, float* shortcut_buf,
                                     float* fc_buf, float* y,
                                     int dim_in, int dim_out, int T,
                                     int style_dim,
                                     cudaStream_t stream,
                                     float* workspace = nullptr,
                                     size_t workspace_bytes = 0) {
    int T_out = blk.has_upsample ? 2 * T : T;

    // ---- Residual path ----
    // fc_buf layout: [0..2C) = gamma/beta from FC, [2C..4C) = norm reduction workspace
    float* norm_ws = fc_buf + 2 * dim_in;
    adain_1d_forward(x, style, blk.norm1, residual_buf, fc_buf, norm_ws,
                     dim_in, T, style_dim, stream);
    leaky_relu_f32(residual_buf, residual_buf, dim_in * T, 0.2f, stream);

    if (blk.has_upsample) {
        conv_transpose1d_depthwise_f32(residual_buf, blk.pool_wv, blk.pool_b,
                                         shortcut_buf, dim_in, T, 3, 2, 1, 1, stream);
        cudaMemcpyAsync(residual_buf, shortcut_buf, dim_in * T_out * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    gemm_conv1d(residual_buf, blk.conv1_wv_nhwc_f16, blk.conv1_b, y,
                 workspace, workspace_bytes, dim_in, dim_out, T_out, 3,
                 1, 1, 1, stream, nullptr, blk.conv1_c_in_pad);
    cudaMemcpyAsync(residual_buf, y, dim_out * T_out * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    norm_ws = fc_buf + 2 * dim_out;
    adain_1d_forward(residual_buf, style, blk.norm2, residual_buf, fc_buf, norm_ws,
                     dim_out, T_out, style_dim, stream);
    leaky_relu_f32(residual_buf, residual_buf, dim_out * T_out, 0.2f, stream);

    gemm_conv1d(residual_buf, blk.conv2_wv_nhwc_f16, blk.conv2_b, y,
                 workspace, workspace_bytes, dim_out, dim_out, T_out, 3,
                 1, 1, 1, stream);

    // ---- Shortcut path ----
    if (blk.has_upsample) {
        upsample_nearest_1d_2x_f32(x, shortcut_buf, dim_in, T, stream);
    } else {
        cudaMemcpyAsync(shortcut_buf, x, dim_in * T * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    if (blk.has_shortcut) {
        gemm_conv1d(shortcut_buf, blk.conv1x1_wv_nhwc_f16, nullptr, residual_buf,
                     workspace, workspace_bytes, dim_in, dim_out, T_out, 1,
                     1, 0, 1, stream, nullptr, blk.conv1x1_c_in_pad);
        cudaMemcpyAsync(shortcut_buf, residual_buf, dim_out * T_out * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // ---- Combine: (residual + shortcut) / sqrt(2) ----
    add_f32(y, shortcut_buf, y, dim_out * T_out, stream);
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    scale_f32(y, y, dim_out * T_out, inv_sqrt2, stream);
}

// ---------------------------------------------------------------------------
// SineGen (CPU, validation-only): F0 [L2] -> har_source [T_audio=L2*300]
//   Used by --validate path for deterministic comparison against PyTorch.
//   Inference uses GPU kernels sinegen_phase_f32 + sinegen_source_f32 instead.
//   rand_ini: if non-null, use these initial phases (for validation).
// ---------------------------------------------------------------------------

static void sinegen_cpu(const float* f0, int L2,
                        const float* l_linear_w, const float* l_linear_b,
                        float* har_source, int T_audio,
                        const float* rand_ini_override,
                        unsigned int seed) {
    constexpr int HARMONICS = 9;
    constexpr int SR = 24000;
    constexpr int UPSAMPLE = 300;
    constexpr float SINE_AMP = 0.1f;
    constexpr float NOISE_STD = 0.003f;
    constexpr float VOICED_THRESH = 10.0f;

    // Step 1: F0 nearest-neighbor upsample [L2] -> [T_audio]
    std::vector<float> f0_up(T_audio);
    for (int t = 0; t < T_audio; t++)
        f0_up[t] = f0[t / UPSAMPLE];

    // Step 2: Harmonic expansion fn[t, h] = f0_up[t] * (h+1)
    std::vector<float> fn(T_audio * HARMONICS);
    for (int t = 0; t < T_audio; t++)
        for (int h = 0; h < HARMONICS; h++)
            fn[t * HARMONICS + h] = f0_up[t] * (h + 1);

    // Step 3: Normalized frequency (cycles per sample)
    // Python-compatible modulo: always non-negative
    std::vector<float> rad(T_audio * HARMONICS);
    for (int i = 0; i < T_audio * HARMONICS; i++) {
        float val = fmodf(fn[i] / SR, 1.0f);
        if (val < 0) val += 1.0f;
        rad[i] = val;
    }

    // Random initial phase for overtones
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float rand_ini[HARMONICS];
    rand_ini[0] = 0.0f;
    if (rand_ini_override) {
        for (int h = 0; h < HARMONICS; h++)
            rand_ini[h] = rand_ini_override[h];
    } else {
        for (int h = 1; h < HARMONICS; h++)
            rand_ini[h] = uni(rng);
    }
    for (int h = 0; h < HARMONICS; h++)
        rad[0 * HARMONICS + h] += rand_ini[h];

    // Step 4: Downsample by UPSAMPLE (linear interpolation, align_corners=False)
    std::vector<float> rad_down(L2 * HARMONICS);
    for (int h = 0; h < HARMONICS; h++) {
        for (int t = 0; t < L2; t++) {
            float src = (t + 0.5f) * T_audio / L2 - 0.5f;
            src = std::max(0.0f, std::min(src, (float)(T_audio - 1)));
            int lo = (int)src;
            int hi = std::min(lo + 1, T_audio - 1);
            float frac = src - lo;
            rad_down[t * HARMONICS + h] =
                (1.0f - frac) * rad[lo * HARMONICS + h] +
                frac * rad[hi * HARMONICS + h];
        }
    }

    // Step 5: Cumulative sum -> phase [L2, 9]
    for (int h = 0; h < HARMONICS; h++)
        for (int t = 1; t < L2; t++)
            rad_down[t * HARMONICS + h] += rad_down[(t-1) * HARMONICS + h];

    // Multiply by 2*pi
    for (int i = 0; i < L2 * HARMONICS; i++)
        rad_down[i] *= 2.0f * (float)M_PI;

    // Step 6: Multiply by UPSAMPLE, then upsample by UPSAMPLE (linear, align_corners=False)
    for (int i = 0; i < L2 * HARMONICS; i++)
        rad_down[i] *= UPSAMPLE;

    std::vector<float> phase(T_audio * HARMONICS);
    for (int h = 0; h < HARMONICS; h++) {
        for (int t = 0; t < T_audio; t++) {
            float src = (t + 0.5f) * L2 / T_audio - 0.5f;
            src = std::max(0.0f, std::min(src, (float)(L2 - 1)));
            int lo = (int)src;
            int hi = std::min(lo + 1, L2 - 1);
            float frac = src - lo;
            phase[t * HARMONICS + h] =
                (1.0f - frac) * rad_down[lo * HARMONICS + h] +
                frac * rad_down[hi * HARMONICS + h];
        }
    }

    // Step 7: sin(phase) * SINE_AMP
    std::vector<float> sine_waves(T_audio * HARMONICS);
    for (int i = 0; i < T_audio * HARMONICS; i++)
        sine_waves[i] = sinf(phase[i]) * SINE_AMP;

    // Step 8: UV masking + noise
    std::normal_distribution<float> norm(0.0f, 1.0f);
    for (int t = 0; t < T_audio; t++) {
        float uv = (f0_up[t] > VOICED_THRESH) ? 1.0f : 0.0f;
        float noise_amp = uv * NOISE_STD + (1.0f - uv) * SINE_AMP / 3.0f;
        for (int h = 0; h < HARMONICS; h++) {
            float n = noise_amp * norm(rng);
            sine_waves[t * HARMONICS + h] =
                sine_waves[t * HARMONICS + h] * uv + n;
        }
    }

    // Step 9: Linear combination [T_audio, 9] -> [T_audio, 1] + tanh
    // l_linear: weight [1, 9], bias [1]
    for (int t = 0; t < T_audio; t++) {
        float sum = l_linear_b[0];
        for (int h = 0; h < HARMONICS; h++)
            sum += sine_waves[t * HARMONICS + h] * l_linear_w[h];
        har_source[t] = tanhf(sum);
    }
}

// ---------------------------------------------------------------------------
// AdaINResBlock1 forward (Generator resblocks with Snake activation)
//   3 rounds of: adain1 -> Snake -> dilated_conv -> adain2 -> Snake -> conv -> residual
//   x: [T, C] in-place, output: [T, C]
//   Working buffers: xt_buf [T, C], conv_out_buf [T, C], fc_buf [2*C]
// ---------------------------------------------------------------------------

static void adain_resblock1_forward(float* x, const float* style,
                                      const Weights::AdaINResBlock1Weights& rb,
                                      float* xt_buf, float* conv_out_buf,
                                      float* fc_buf,
                                      int T, int style_dim,
                                      cudaStream_t stream,
                                      float* workspace = nullptr,
                                      size_t workspace_bytes = 0) {
    int C = rb.channels;
    int K = rb.kernel_size;
    int dilations[] = {1, 3, 5};

    float* norm_ws = fc_buf + 2 * C;
    for (int j = 0; j < 3; j++) {
        adain_1d_forward(x, style, rb.adain1[j], xt_buf, fc_buf, norm_ws,
                         C, T, style_dim, stream, rb.alpha1[j]);

        int d = dilations[j];
        int pad = (K * d - d) / 2;
        gemm_conv1d(xt_buf, rb.convs1[j].wv_nhwc_f16, rb.convs1[j].b, conv_out_buf,
                     workspace, workspace_bytes, C, C, T, K, 1, pad, d, stream);

        adain_1d_forward(conv_out_buf, style, rb.adain2[j], xt_buf, fc_buf, norm_ws,
                         C, T, style_dim, stream, rb.alpha2[j]);

        int pad2 = (K - 1) / 2;
        gemm_conv1d(xt_buf, rb.convs2[j].wv_nhwc_f16, rb.convs2[j].b, x,
                     workspace, workspace_bytes, C, C, T, K, 1, pad2, 1, stream,
                     x);  // residual: x = conv(xt_buf) + x, then + bias
    }
}

// ---------------------------------------------------------------------------
// Encode-phase CUDA graph cache: keyed by T (token count).
// All arena allocations are deterministic per T, so cached graphs
// use the same device pointers on replay.
// ---------------------------------------------------------------------------

static std::unordered_map<int, cudaGraphExec_t> s_encode_graph_cache;

// Decode-phase CUDA graph cache: keyed by (T << 32 | L).
// Arena allocations are deterministic per (T,L), so cached graphs
// use the same device pointers on replay.
struct DecodeGraph {
    cudaGraphExec_t exec;
    size_t audio_offset;  // byte offset of d_audio from decode_arena.base
};
static std::unordered_map<int64_t, DecodeGraph> s_decode_graph_cache;
static float* s_d_rand_ini = nullptr;  // persistent rand_ini on GPU (seed=42)

// ---------------------------------------------------------------------------
// Rokoko inference: phoneme IDs + style vector -> audio
// ---------------------------------------------------------------------------

std::vector<float> rokoko_infer(const Weights& w,
                                const int* token_ids, int T,
                                const float* style_vec,  // [256] on host
                                cudaStream_t stream,
                                GpuArena& arena,
                                GpuArena& decode_arena,
                                float* d_workspace,
                                size_t workspace_bytes) {
    // Set workspace for Cutlass GEMM (used by all sgemm_* wrappers)
    s_workspace = d_workspace;
    s_workspace_bytes = workspace_bytes;
    arena.reset();

    // ---- All arena allocations upfront (deterministic per T) ----
    int* d_token_ids = arena.alloc<int>(T);
    float* d_ref_s = arena.alloc<float>(256);
    float* d_style_prosody = d_ref_s + 128;
    float* d_style_acoustic = d_ref_s;

    AlbertBuffers albert_buf;
    albert_buf.alloc(T, arena);

    float* d_en = arena.alloc<float>(T * 512);

    TextEncoderBuffers te_buf;
    te_buf.alloc(T, arena);

    float* d_style_tiled = arena.alloc<float>(T * 128);
    float* d_cat_buf  = arena.alloc<float>(T * 640);
    float* d_lstm_out = arena.alloc<float>(T * 512);
    float* d_x_buf    = arena.alloc<float>(T * 512);
    float* d_ada_fc   = arena.alloc<float>(1024);

    float* d_dur_lstm_out = arena.alloc<float>(T * 512);
    float* d_dur_proj     = arena.alloc<float>(T * 50);
    float* d_durations    = arena.alloc<float>(T);
    int*   d_int_durations = arena.alloc<int>(T);
    int*   d_L             = arena.alloc<int>(1);

    // ---- H2D uploads (async, stream-ordered before encode graph) ----
    CUDA_CHECK(cudaMemcpyAsync(d_token_ids, token_ids, T * sizeof(int),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ref_s, style_vec, 256 * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    // ---- Encode phase: CUDA graph capture/replay ----
    auto git = s_encode_graph_cache.find(T);
    if (git == s_encode_graph_cache.end()) {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // ALBERT encoder
        cudaMemcpyAsync(albert_buf.token_ids, d_token_ids, T * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);
        albert_forward(w, albert_buf, stream, T);

        // bert_encoder: Linear 768->512 -> d_en [T, 512]
        sgemm_bias(512, T, 768,
                   w.bert_enc_w_f16, 768, albert_buf.hidden, 768, d_en, 512,
                   w.bert_enc_b, stream);

        // Text encoder
        text_encoder_forward(w, te_buf, d_token_ids, stream, T, arena,
                             d_workspace, workspace_bytes);

        // Duration encoder: 3x (BiLSTM + AdaLN)
        tile_1d_f32(d_style_prosody, d_style_tiled, 128, T, stream);
        concat_channels_f32(d_en, d_style_tiled, d_cat_buf, T, 512, 128, stream);

        for (int layer_i = 0; layer_i < 3; layer_i++) {
            bilstm_gpu(d_cat_buf, d_lstm_out, w.dur_enc_lstm[layer_i],
                       T, 640, 256, stream, arena);

            auto& aln = w.dur_enc_aln[layer_i];
            sgemm_tn(1024, 1, 128,
                     aln.fc_w_f16, 128, d_style_prosody, 128, d_ada_fc, 1024, stream);
            bias_add_f32(d_ada_fc, aln.fc_b, d_ada_fc, 1, 1024, stream);
            ada_layer_norm_f32(d_lstm_out, d_ada_fc, d_ada_fc + 512,
                               d_x_buf, T, 512, 1e-5f, stream);

            concat_channels_f32(d_x_buf, d_style_tiled, d_cat_buf, T, 512, 128, stream);
        }

        // Duration LSTM + projection
        bilstm_gpu(d_cat_buf, d_dur_lstm_out, w.dur_lstm,
                   T, 640, 256, stream, arena);
        sgemm_bias(50, T, 512,
                   w.dur_proj_w_f16, 512, d_dur_lstm_out, 512, d_dur_proj, 50,
                   w.dur_proj_b, stream);
        sigmoid_sum_f32(d_dur_proj, d_durations, T, 50, stream);
        round_clamp_durations_f32(d_durations, d_int_durations, d_L, T, stream);

        cudaGraph_t graph;
        cudaStreamEndCapture(stream, &graph);
        cudaGraphExec_t exec;
        cudaGraphInstantiateWithFlags(&exec, graph, 0);
        cudaGraphDestroy(graph);
        s_encode_graph_cache[T] = exec;
        cudaGraphLaunch(exec, stream);
    } else {
        cudaGraphLaunch(git->second, stream);
    }

    // Sync for L only (4 bytes) -- needed for decode arena sizing
    int L;
    CUDA_CHECK(cudaMemcpyAsync(&L, d_L, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int L2 = 2 * L;
    int T_audio_actual = L2 * 300;  // actual audio length for download

    // Bucket L for decode graph cache hit rate: different L values
    // within the same bucket share a single cached graph.
    static constexpr int DECODE_L_BUCKET = 32;
    L = ((L + DECODE_L_BUCKET - 1) / DECODE_L_BUCKET) * DECODE_L_BUCKET;
    L2 = 2 * L;
    int T_audio = L2 * 300;  // padded -- used by decode operations

    // ===== DECODE PHASE: exact-sized arena =====
    size_t decode_bytes = compute_decode_bytes(T, L);
    if (decode_arena.capacity < decode_bytes) {
        // Arena grows -- invalidate all cached decode graphs
        for (auto& [k, dg] : s_decode_graph_cache)
            cudaGraphExecDestroy(dg.exec);
        s_decode_graph_cache.clear();
        decode_arena.destroy();
        decode_arena.init(decode_bytes);
    } else {
        decode_arena.reset();
    }

    // Lazy-init persistent rand_ini device buffer (SineGen, seed=42)
    if (!s_d_rand_ini) {
        float ri[9] = {0};
        std::mt19937 rng_ri(42);
        std::uniform_real_distribution<float> uni_ri(0.0f, 1.0f);
        for (int h = 1; h < 9; h++) ri[h] = uni_ri(rng_ri);
        CUDA_CHECK(cudaMalloc(&s_d_rand_ini, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(s_d_rand_ini, ri, 9 * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    // Decode graph cache: replay if available
    int64_t decode_key = ((int64_t)T << 32) | (int64_t)(unsigned int)L;
    auto dgit = s_decode_graph_cache.find(decode_key);
    if (dgit != s_decode_graph_cache.end()) {
        float* d_audio = (float*)(decode_arena.base + dgit->second.audio_offset);
        cudaGraphLaunch(dgit->second.exec, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> audio(T_audio_actual);
        cudaMemcpy(audio.data(), d_audio, T_audio_actual * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return audio;
    }

    // First time for (T, L): capture decode graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);

    // Decode workspace (first alloc, matches compute_decode_bytes order)
    int har_frames_ws = (L2 * 300) / 5 + 1;
    size_t dec_ws_floats = (size_t)128 * 11 * har_frames_ws;
    float* d_dec_workspace = decode_arena.alloc<float>(dec_ws_floats);
    size_t dec_ws_bytes = dec_ws_floats * sizeof(float);

    // Build alignment matrix on GPU
    float* d_alignment = decode_arena.alloc<float>(T * L);
    build_alignment_f32(d_int_durations, d_alignment, T, L, stream);

    // Expand encoder output: alignment^T [L, T] x d_cat_buf [T, 640] -> [L, 640]
    // d_cat_buf is [T, 640] row-major = col-major [640, T]
    // alignment is [T, L] row-major = col-major [L, T]
    // We want result [L, 640] row-major = col-major [640, L]
    // GEMM: C[640,L] = A[640,T] * B[T,L]
    //   A = d_cat_buf row-major [T,640] = col-major [640,T], lda=640
    //   B = d_alignment row-major [T,L] = col-major [L,T], transposed -> [T,L], ldb=L
    float* d_en_expanded = decode_arena.alloc<float>(L * 640);
    sgemm_nt(640, L, T,
             d_cat_buf, 640, d_alignment, L, d_en_expanded, 640, stream);

    // Shared LSTM: d_en_expanded [L, 640] already row-major -- no transpose
    float* d_shared_out = decode_arena.alloc<float>(L * 512);
    bilstm_gpu(d_en_expanded, d_shared_out, w.shared_lstm,
               L, 640, 256, stream, decode_arena);

    // F0 and N prediction chains
    // Both: [L, 512] -> 3 AdainResBlk1d -> Conv1d proj -> [2L, 1]
    float* d_res_buf = decode_arena.alloc<float>(512 * L2);
    float* d_sc_buf  = decode_arena.alloc<float>(512 * L2);
    float* d_fc_buf  = decode_arena.alloc<float>(2048);  // 2*C for gamma/beta + 2*C for norm reduction

    auto run_f0n_chain = [&](const Weights::AdainResBlk1dWeights blocks[3],
                              const float* proj_w, const float* proj_b,
                              float* d_pred) {
        size_t chain_save = decode_arena.save();
        float* d_fx = decode_arena.alloc<float>(512 * L2);
        float* d_fo = decode_arena.alloc<float>(512 * L2);
        // d_shared_out is [L, 512] -- already [T, C] layout, no transpose
        cudaMemcpyAsync(d_fx, d_shared_out, L * 512 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // block 0: [L, 512] -> [L, 512]
        adain_resblk1d_forward(d_fx, d_style_prosody, blocks[0],
                                d_res_buf, d_sc_buf, d_fc_buf, d_fo,
                                512, 512, L, 128, stream,
                                d_dec_workspace, dec_ws_bytes);
        // block 1: [L, 512] -> [2L, 256] (upsample)
        cudaMemcpyAsync(d_fx, d_fo, 512 * L * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        adain_resblk1d_forward(d_fx, d_style_prosody, blocks[1],
                                d_res_buf, d_sc_buf, d_fc_buf, d_fo,
                                512, 256, L, 128, stream,
                                d_dec_workspace, dec_ws_bytes);
        // block 2: [2L, 256] -> [2L, 256]
        cudaMemcpyAsync(d_fx, d_fo, 256 * L2 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        adain_resblk1d_forward(d_fx, d_style_prosody, blocks[2],
                                d_res_buf, d_sc_buf, d_fc_buf, d_fo,
                                256, 256, L2, 128, stream,
                                d_dec_workspace, dec_ws_bytes);
        // Projection: Conv1d(256->1, k=1) -> [2L, 1]
        conv1d_f32(d_fo, proj_w, proj_b, d_pred, 256, 1, L2, 1, stream);
        decode_arena.restore(chain_save);
    };

    float* d_f0_pred = decode_arena.alloc<float>(L2);
    float* d_n_pred  = decode_arena.alloc<float>(L2);
    run_f0n_chain(w.f0_blocks, w.f0_proj_w, w.f0_proj_b, d_f0_pred);
    run_f0n_chain(w.n_blocks, w.n_proj_w, w.n_proj_b, d_n_pred);

    // ---- Decoder ----
    // Build asr_aligned: alignment^T [L, T] x te_buf.emb [T, 512] -> [L, 512]
    float* d_asr_aligned = decode_arena.alloc<float>(L * 512);
    sgemm_nt(512, L, T,
             te_buf.emb, 512, d_alignment, L, d_asr_aligned, 512, stream);

    // F0/N downsampling: Conv1d(1,1,k=3,s=2,p=1) on [2L, 1] -> [L, 1]
    float* d_f0_down = decode_arena.alloc<float>(L);
    float* d_n_down  = decode_arena.alloc<float>(L);

    // F0/N downsample -- wv has precomputed weight
    conv1d_general_f32(d_f0_pred, w.dec_f0_conv_wv, w.dec_f0_conv_b,
                       d_f0_down, 1, 1, L2, 3, 2, 1, 1, stream);
    conv1d_general_f32(d_n_pred, w.dec_n_conv_wv, w.dec_n_conv_b,
                       d_n_down, 1, 1, L2, 3, 2, 1, 1, stream);

    // Concatenate [asr_aligned, F0_down, N_down] -> [L, 514]
    float* d_dec_cat = decode_arena.alloc<float>(L * 514);
    concat3_channels_f32(d_asr_aligned, d_f0_down, d_n_down,
                          d_dec_cat, L, 512, 1, 1, stream);

    // Decoder working buffers
    int max_ch = 1090;
    float* d_dec_res = decode_arena.alloc<float>(max_ch * L2);
    float* d_dec_sc  = decode_arena.alloc<float>(max_ch * L2);
    float* d_dec_fc  = decode_arena.alloc<float>(4 * max_ch);  // 2*C gamma/beta + 2*C norm reduction

    // Encode: AdainResBlk1d(514->1024) [L, 514] -> [L, 1024]
    float* d_dec_out = decode_arena.alloc<float>(L * 1024);
    adain_resblk1d_forward(d_dec_cat, d_style_acoustic, w.dec_encode,
                            d_dec_res, d_dec_sc, d_dec_fc, d_dec_out,
                            514, 1024, L, 128, stream,
                            d_dec_workspace, dec_ws_bytes);

    // asr_res: Conv1d(512->64, k=1) -- wv has precomputed weight
    float* d_asr_res = decode_arena.alloc<float>(L * 64);
    conv1d_f32(d_asr_aligned, w.dec_asr_res_wv, w.dec_asr_res_b, d_asr_res,
               512, 64, L, 1, stream);

    // Decode blocks [0-3] with skip connections
    float* d_dec_x = d_dec_out;
    float* d_dec_in = decode_arena.alloc<float>(L2 * 1090);

    int dim_in_sizes[] = {1090, 1090, 1090, 1090};
    int dim_out_sizes[] = {1024, 1024, 1024, 512};
    bool res_flag = true;

    for (int bi = 0; bi < 4; bi++) {
        int T_blk = L;

        if (res_flag) {
            // Concatenate: d_dec_x [T_blk, 1024] + d_asr_res [T_blk, 64]
            //            + d_f0_down [T_blk, 1] + d_n_down [T_blk, 1] -> [T_blk, 1090]
            concat4_channels_f32(d_dec_x, d_asr_res, d_f0_down, d_n_down,
                                  d_dec_in, T_blk, 1024, 64, 1, 1, stream);
        } else {
            cudaMemcpyAsync(d_dec_in, d_dec_x, dim_in_sizes[bi] * T_blk * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        int T_blk_out = w.dec_decode[bi].has_upsample ? 2 * T_blk : T_blk;
        float* d_blk_out = decode_arena.alloc<float>(T_blk_out * dim_out_sizes[bi]);
        adain_resblk1d_forward(d_dec_in, d_style_acoustic, w.dec_decode[bi],
                                d_dec_res, d_dec_sc, d_dec_fc, d_blk_out,
                                dim_in_sizes[bi], dim_out_sizes[bi], T_blk,
                                128, stream,
                                d_dec_workspace, dec_ws_bytes);

        d_dec_x = d_blk_out;

        if (w.dec_decode[bi].has_upsample) {
            res_flag = false;
            L = T_blk_out;  // update L after upsample
        }
    }
    // d_dec_x is now [L2, 512] -- generator input

    // ---- Generator ----
    // SineGen: F0 -> harmonic source -> STFT -> har [22, frames]
    // STFT output is inherently [n_fft/2+1, n_frames] = [C, T] layout.
    // gen_har stays in [C, T] layout; the generator noise_convs kernels
    // use [T, C] layout, so we'll transpose gen_har to [T, C] for them.
    int n_fft_gen = 20, hop_gen = 5;
    int har_frames = T_audio / hop_gen + 1;

    // gen_har stored as [22, har_frames] (STFT output, channels-first)
    float* d_gen_har_ct = decode_arena.alloc<float>(22 * har_frames);
    {
        size_t sinegen_save = decode_arena.save();
        float* d_phase_low = decode_arena.alloc<float>(L2 * 9);
        float* d_har_source = decode_arena.alloc<float>(T_audio);

        // Use persistent s_d_rand_ini (pre-allocated outside graph capture)
        sinegen_phase_f32(d_f0_pred, s_d_rand_ini, d_phase_low, L2, stream);
        sinegen_source_f32(d_phase_low, d_f0_pred, w.gen_source_w, w.gen_source_b,
                           d_har_source, L2, T_audio, 42, stream);

        float* d_har_spec = d_gen_har_ct;
        float* d_har_phase = d_gen_har_ct + 11 * har_frames;
        stft_f32(d_har_source, d_har_spec, d_har_phase,
                 T_audio, n_fft_gen, hop_gen, stream);
        decode_arena.restore(sinegen_save);
    }

    // Transpose gen_har from [22, har_frames] -> [har_frames, 22] for [T,C] layout
    // Reuse sinegen temps area (already restored)
    float* d_gen_har = decode_arena.alloc<float>(har_frames * 22);
    transpose_f32(d_gen_har_ct, d_gen_har, 22, har_frames, stream);

    // Generator working pool: 5 slots of 128*har_frames each
    size_t gen_slot = (size_t)128 * har_frames;
    float* gen_pool = decode_arena.alloc<float>(5 * gen_slot);
    float* d_gen_fc = decode_arena.alloc<float>(512);

    float* slot0 = gen_pool;
    float* slot1 = gen_pool + gen_slot;
    float* slot2 = gen_pool + 2 * gen_slot;
    float* slot3 = gen_pool + 3 * gen_slot;
    float* slot4 = gen_pool + 4 * gen_slot;

    // Copy generator input into slot0
    float* d_gen_x = slot0;
    int T_gen = L2;
    cudaMemcpyAsync(d_gen_x, d_dec_x, 512 * T_gen * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    int T_loop = T_gen;
    int gen_channels[] = {512, 256, 128};
    int upsample_rates[] = {10, 6};
    int upsample_kernels[] = {20, 12};

    for (int i = 0; i < 2; i++) {
        int C_in = gen_channels[i];
        int C_out = gen_channels[i + 1];
        int us = upsample_rates[i];
        int uk = upsample_kernels[i];
        int up = (uk - us) / 2;

        leaky_relu_f32(d_gen_x, d_gen_x, C_in * T_loop, 0.1f, stream);

        // noise_convs[i]: gen_har [har_frames, 22] -> slot1 [src_T, C_out]
        float* d_gen_src = slot1;
        if (i == 0) {
            gemm_conv1d(d_gen_har, w.gen_noise_convs[0].w_nhwc_f16,
                         w.gen_noise_convs[0].b, d_gen_src,
                         d_dec_workspace, dec_ws_bytes,
                         22, C_out, har_frames, 12, 6, 3, 1, stream,
                         nullptr, w.gen_noise_convs[0].c_in_pad);
        } else {
            gemm_conv1d(d_gen_har, w.gen_noise_convs[1].w_nhwc_f16,
                         w.gen_noise_convs[1].b, d_gen_src,
                         d_dec_workspace, dec_ws_bytes,
                         22, C_out, har_frames, 1, 1, 0, 1, stream,
                         nullptr, w.gen_noise_convs[1].c_in_pad);
        }

        // noise_res[i]: uses slot2 (xt), slot3 (co) as temporaries
        float* d_gen_xt = slot2;
        float* d_gen_co = slot3;
        int src_T = (i == 0) ? ((har_frames + 2*3 - 12) / 6 + 1) : har_frames;
        adain_resblock1_forward(d_gen_src, d_style_acoustic, w.gen_noise_res[i],
                                d_gen_xt, d_gen_co, d_gen_fc,
                                src_T, 128, stream,
                                d_dec_workspace, dec_ws_bytes);

        // ups[i]: gen_x (slot0) -> slot4 (gen_tmp)
        float* d_gen_tmp = slot4;
        int T_ups = (T_loop - 1) * us - 2 * up + uk;
        gemm_conv_transpose1d(d_gen_x, w.gen_ups[i].wv_f16, w.gen_ups[i].b,
                               d_gen_tmp, d_dec_workspace, dec_ws_bytes,
                               C_in, C_out, T_loop, uk, us, up, 0, stream);

        // Copy/pad tmp -> gen_x
        if (i == 1) {
            reflection_pad_1d_f32(d_gen_tmp, d_gen_x, C_out, T_ups, 1, 0, stream);
            T_ups += 1;
        } else {
            cudaMemcpyAsync(d_gen_x, d_gen_tmp, C_out * T_ups * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // Merge: x += src
        add_f32(d_gen_x, d_gen_src, d_gen_x, C_out * T_ups, stream);

        // ResBlocks: reuse slot1 as rb_avg
        float* d_rb_avg = slot1;
        CUDA_CHECK(cudaMemsetAsync(d_rb_avg, 0, C_out * T_ups * sizeof(float), stream));

        for (int j = 0; j < 3; j++) {
            int rb_idx = i * 3 + j;
            cudaMemcpyAsync(d_gen_tmp, d_gen_x, C_out * T_ups * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            adain_resblock1_forward(d_gen_tmp, d_style_acoustic, w.gen_resblocks[rb_idx],
                                    d_gen_xt, d_gen_co, d_gen_fc,
                                    T_ups, 128, stream,
                                    d_dec_workspace, dec_ws_bytes);
            add_f32(d_rb_avg, d_gen_tmp, d_rb_avg, C_out * T_ups, stream);
        }
        scale_f32(d_rb_avg, d_gen_x, C_out * T_ups, 1.0f / 3.0f, stream);

        T_loop = T_ups;
    }

    // Post: LeakyReLU(0.01) + conv_post(128->22, k=7)
    // conv_post: C_out=22 misaligned for Cutlass conv, use im2col + FP16 GEMM.
    // im2col output is large (128*7*T_loop), so cast into workspace directly
    // instead of s_fp16_buf which may be too small.
    float* d_gen_tmp = slot4;
    leaky_relu_f32(d_gen_x, d_gen_x, 128 * T_loop, 0.01f, stream);
    im2col_1d_f32(d_gen_x, d_dec_workspace, 128, T_loop, 7, 1, 3, 1, T_loop, stream);
    {
        int CK = 128 * 7;
        size_t col_floats = (size_t)CK * T_loop;
        // Cast im2col FP32 output to FP16 after the FP32 data in the workspace
        __half* col_f16 = (__half*)(d_dec_workspace + col_floats);
        cast_f32_to_f16(d_dec_workspace, col_f16, (int)col_floats, stream);
        cutlass_gemm_tn_f16(22, T_loop, CK, w.gen_conv_post_wv_f16, CK,
                             col_f16, CK, d_gen_tmp, 22, 1.0f, 0.0f,
                             s_workspace, s_workspace_bytes, stream);
    }
    channel_bias_add_f32(d_gen_tmp, w.gen_conv_post_b, 22, T_loop, stream);

    // Transpose [T_loop, 22] -> [22, T_loop] for STFT boundary (channels-first)
    float* d_gen_ct = d_gen_x;  // reuse slot0 (128*har_frames >> 22*T_loop)
    transpose_f32(d_gen_tmp, d_gen_ct, T_loop, 22, stream);

    // spec = exp(x[:11,:]), phase = sin(x[11:,:]) -- in [C,T] layout
    int n_freqs = 11;
    exp_f32(d_gen_ct, d_gen_tmp, n_freqs * T_loop, stream);
    sin_f32(d_gen_ct + n_freqs * T_loop, d_gen_tmp + n_freqs * T_loop,
            n_freqs * T_loop, stream);

    // iSTFT -> audio (operates on [n_freqs, T_loop] channels-first)
    float* d_audio = decode_arena.alloc<float>(T_audio);
    istft_f32(d_gen_tmp, d_gen_tmp + n_freqs * T_loop, d_audio,
              T_loop, 20, 5, T_audio, stream);
    // End decode graph capture
    cudaGraph_t decode_graph;
    cudaStreamEndCapture(stream, &decode_graph);
    cudaGraphExec_t decode_exec;
    cudaGraphInstantiateWithFlags(&decode_exec, decode_graph, 0);
    cudaGraphDestroy(decode_graph);

    size_t audio_offset = (size_t)((char*)d_audio - decode_arena.base);
    s_decode_graph_cache[decode_key] = {decode_exec, audio_offset};

    cudaGraphLaunch(decode_exec, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Download audio (actual length, not padded)
    std::vector<float> audio(T_audio_actual);
    cudaMemcpy(audio.data(), d_audio, T_audio_actual * sizeof(float),
               cudaMemcpyDeviceToHost);

    return audio;
}

// ---------------------------------------------------------------------------
// Precompute weight norms: v2 file has everything pre-baked.
// Just assign FP16 pointers and allocate staging buffer.
// ---------------------------------------------------------------------------

void precompute_weight_norms(Weights& w, cudaStream_t stream) {
    w.assign_v2_fp16_pointers();
    // Staging buffer sized for largest activation cast: generator resblock conv
    // at max ~90k frames * 128 channels * sizeof(half) ≈ 23 MB. Allocate 64 MB.
    s_fp16_buf_size = 64 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&s_fp16_buf, s_fp16_buf_size));

    // Verify FP16 pointers were populated
    int n_f16 = 0;
    auto chk = [&](const __half* p) { if (p) n_f16++; };
    chk(w.bert_proj_w_f16);
    chk(w.albert.q_w_f16); chk(w.albert.k_w_f16); chk(w.albert.v_w_f16);
    chk(w.albert.dense_w_f16); chk(w.albert.ffn_w_f16); chk(w.albert.ffn_out_w_f16);
    chk(w.bert_enc_w_f16);
    for (int i = 0; i < 3; i++) chk(w.text_conv[i].conv_wv_nhwc_f16);
    chk(w.text_lstm_wih_fwd_f16); chk(w.text_lstm_whh_fwd_f16);
    chk(w.dur_proj_w_f16);
    chk(w.dec_encode.conv1_wv_nhwc_f16);
    for (int i = 0; i < 6; i++) chk(w.gen_resblocks[i].convs1[0].wv_nhwc_f16);
    fprintf(stderr, "  FP16 binary: %d key pointers set, staging=64 MB\n", n_f16);
}

// FP16 binary: self-contained bundle with v2 weights + G2P + voices
const char* default_bundle_url() {
    return "https://github.com/lfrati/rokoko/releases/download/v1.0.0/rokoko.fp16.bundle";
}
const char* default_bundle_filename() { return "rokoko.fp16.bundle"; }
