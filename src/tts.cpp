// tts.cpp — TTS CUDA inference (FP32)
//
// Provides rokoko_infer(), write_wav(), precompute_weight_norms().

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
#include <cublas_v2.h>
#include <cublasLt.h>
#include "weights.h"
#include "kernels.h"

// ---------------------------------------------------------------------------
// cuBLAS SGEMM wrapper
// ---------------------------------------------------------------------------

// C = alpha * op(A) * op(B) + beta * C  (all FP32)
static void sgemm(cublasHandle_t cublas,
                  cublasOperation_t transa, cublasOperation_t transb,
                  int m, int n, int k,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float* C, int ldc,
                  float alpha = 1.0f, float beta = 0.0f) {
    CUBLAS_CHECK(cublasSgemm(cublas, transa, transb, m, n, k,
                              &alpha, A, lda, B, ldb, &beta, C, ldc));
}

// Strided batched SGEMM
static void sgemm_strided_batched(cublasHandle_t cublas,
                                   cublasOperation_t transa,
                                   cublasOperation_t transb,
                                   int m, int n, int k,
                                   const float* A, int lda, long long strideA,
                                   const float* B, int ldb, long long strideB,
                                   float* C, int ldc, long long strideC,
                                   int batch_count,
                                   float alpha = 1.0f, float beta = 0.0f) {
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        cublas, transa, transb, m, n, k,
        &alpha, A, lda, strideA, B, ldb, strideB,
        &beta, C, ldc, strideC, batch_count));
}

// ---------------------------------------------------------------------------
// cuBLASLt SGEMM with fused bias epilogue
//   C[m,n] = alpha * op(A)[m,k] * op(B)[k,n] + bias[m]
//   Matches the sgemm() signature but fuses bias_add into the GEMM.
//   Uses a pre-created cublasLtHandle — caller manages lifetime.
// ---------------------------------------------------------------------------

static void sgemm_bias(cublasLtHandle_t ltHandle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float* C, int ldc,
                        const float* bias,
                        cudaStream_t stream) {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                    &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &transb, sizeof(transb));

    // Set bias epilogue
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                    &epilogue, sizeof(epilogue));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                    &bias, sizeof(bias));

    // Matrix layouts (column-major)
    int rowsA = (transa == CUBLAS_OP_N) ? m : k;
    int colsA = (transa == CUBLAS_OP_N) ? k : m;
    int rowsB = (transb == CUBLAS_OP_N) ? k : n;
    int colsB = (transb == CUBLAS_OP_N) ? n : k;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, rowsA, colsA, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, rowsB, colsB, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc);

    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(ltHandle, matmulDesc,
                    &alpha, A, Adesc, B, Bdesc,
                    &beta, C, Cdesc, C, Cdesc,
                    nullptr, nullptr, 0, stream);

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(matmulDesc);
}

// ---------------------------------------------------------------------------
// Cutlass implicit GEMM Conv1d (extern, defined in cutlass_conv.cu)
// ---------------------------------------------------------------------------
extern "C" int cutlass_conv1d_fprop(const float* x, const float* w, const float* bias,
                                     float* y, const float* residual,
                                     float* workspace, size_t workspace_bytes,
                                     int C_in, int C_out, int T_in, int K,
                                     int stride, int padding, int dilation,
                                     cudaStream_t stream);
extern "C" void cutlass_reshape_weights(const float* src, float* dst,
                                         int C_out, int C_in, int K, cudaStream_t stream);

// NHWC weight map: wv pointer → w_nhwc pointer (populated at startup)
static std::unordered_map<const float*, float*> s_w_nhwc;

// ---------------------------------------------------------------------------
// Conv1d forward: tries Cutlass implicit GEMM first (w_nhwc), falls back to
// im2col + cuBLAS SGEMM (w in original [C_out, C_in, K] layout).
// ---------------------------------------------------------------------------

// gemm_conv1d: Conv1d forward.
//   When residual != nullptr: y = conv(x,w) + residual, then bias_add(y, bias).
//   When residual == nullptr: y = conv(x,w) + bias (fused in Cutlass epilogue).
static void gemm_conv1d(cublasHandle_t cublas,
                         const float* x, const float* w, const float* bias,
                         float* y, float* workspace, size_t workspace_bytes,
                         int C_in, int C_out, int T_in, int K,
                         int stride, int padding, int dilation,
                         cudaStream_t stream,
                         const float* residual = nullptr) {
    int T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int CK = C_in * K;

    // Try Cutlass implicit GEMM (needs NHWC weights, K>1)
    if (K > 1) {
        auto it = s_w_nhwc.find(w);
        if (it != s_w_nhwc.end()) {
            // With residual: Cutlass does y = conv + residual, bias added after
            // Without residual: Cutlass does y = conv + bias (fused epilogue)
            const float* cutlass_bias = residual ? nullptr : bias;
            int rc = cutlass_conv1d_fprop(x, it->second, cutlass_bias, y,
                                           residual, workspace, workspace_bytes,
                                           C_in, C_out, T_in, K,
                                           stride, padding, dilation, stream);
            if (rc == 0) {
                if (residual && bias)
                    channel_bias_add_f32(y, bias, C_out, T_out, stream);
                return;
            }
        }
    }

    // cuBLAS path: im2col + SGEMM
    const float* col = x;
    if (K > 1 || stride > 1 || padding > 0 || dilation > 1) {
        float* col_buf = workspace;
        im2col_1d_f32(x, col_buf, C_in, T_in, K, stride, padding, dilation,
                       T_out, stream);
        col = col_buf;
    }

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             C_out, T_out, CK,
                             &alpha,
                             w, CK,
                             col, CK,
                             &beta,
                             y, C_out));

    if (residual) {
        add_f32(y, residual, y, C_out * T_out, stream);
    }
    if (bias) {
        channel_bias_add_f32(y, bias, C_out, T_out, stream);
    }
}

// im2col + cuBLAS ConvTranspose1d: GEMM + col2im
//   x[T_in, C_in] * w[C_in, C_out, K] → y[T_out, C_out]
//   T_out = (T_in - 1) * stride - 2*padding + K + output_padding
static void gemm_conv_transpose1d(cublasHandle_t cublas,
                                    const float* x, const float* w, const float* bias,
                                    float* y, float* workspace, size_t workspace_bytes,
                                    int C_in, int C_out, int T_in, int K,
                                    int stride, int padding, int output_padding,
                                    cudaStream_t stream) {
    int T_out = (T_in - 1) * stride - 2 * padding + K + output_padding;
    int COK = C_out * K;

    // GEMM: col[T_in, COK] = x[T_in, C_in] × w[C_in, COK]
    // Row-major [T_in, COK] = col-major [COK, T_in]
    //   x row-major [T_in, C_in] = col-major [C_in, T_in], ldb=C_in
    //   w row-major [C_in, COK] = col-major [COK, C_in], lda=COK
    //   col = col-major [COK, T_in], ldc=COK
    // cuBLAS: C[COK,T_in] = A[COK,C_in] * B[C_in,T_in]
    //   A=w (col-major [COK,C_in]), transa=OP_N
    //   B=x (col-major [C_in,T_in]), transb=OP_N
    float* col_buf = workspace;
    float alpha = 1.0f, beta_zero = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             COK, T_in, C_in,
                             &alpha,
                             w, COK,           // A: col-major [COK, C_in]
                             x, C_in,          // B: col-major [C_in, T_in]
                             &beta_zero,
                             col_buf, COK));    // C: col-major [COK, T_in] = row-major [T_in, COK]

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

struct AlbertBuffers {
    float* emb = nullptr;       // [T, 128] embeddings sum
    float* hidden = nullptr;    // [T, 768] main activation
    float* qkv = nullptr;       // [T, 3*768] fused QKV
    float* attn_scores = nullptr; // [N_HEADS, T, T]
    float* attn_out = nullptr;  // [T, 768] attention output
    float* ff_mid = nullptr;    // [T, 2048] FFN intermediate
    float* ff_out = nullptr;    // [T, 768] FFN output
    float* temp = nullptr;      // [T, 768] temporary buffer
    int* token_ids = nullptr;   // [T] int32 token IDs

    void alloc(int T, GpuArena& arena) {
        emb        = arena.alloc<float>(T * 128);
        hidden     = arena.alloc<float>(T * 768);
        qkv        = arena.alloc<float>(T * 3 * 768);
        attn_scores= arena.alloc<float>(12 * T * T);
        attn_out   = arena.alloc<float>(T * 768);
        ff_mid     = arena.alloc<float>(T * 2048);
        ff_out     = arena.alloc<float>(T * 768);
        temp       = arena.alloc<float>(T * 768);
        token_ids  = arena.alloc<int>(T);
    }
};

// Run ALBERT encoder: input_ids[T] -> hidden[T, 768]
static void albert_forward(const Weights& w, AlbertBuffers& buf,
                           cublasHandle_t cublas, cublasLtHandle_t ltHandle,
                           cudaStream_t stream, int T) {
    // Step 1: Embeddings
    embedding_gather(w.bert_word_embed, buf.token_ids, buf.emb, T, 128, stream);
    add_f32(buf.emb, w.bert_pos_embed, buf.emb, T * 128, stream);
    bias_add_f32(buf.emb, w.bert_type_embed, buf.emb, T, 128, stream);
    layer_norm_f32(buf.emb, w.bert_embed_ln_w, w.bert_embed_ln_b,
                   buf.emb, T, 128, 1e-12f, stream);

    // Step 2: Project 128 -> 768 (GEMM + fused bias)
    sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 768, T, 128,
               w.bert_proj_w, 128, buf.emb, 128, buf.hidden, 768,
               w.bert_proj_b, stream);

    // Step 3: Shared ALBERT layer x12
    auto& a = w.albert;
    for (int layer = 0; layer < 12; layer++) {
        // --- Self-attention ---
        float* Q = buf.qkv;
        float* K = buf.qkv + T * 768;
        float* V = buf.qkv + T * 768 * 2;

        sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 768, T, 768,
                   a.q_w, 768, buf.hidden, 768, Q, 768, a.q_b, stream);
        sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 768, T, 768,
                   a.k_w, 768, buf.hidden, 768, K, 768, a.k_b, stream);
        sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 768, T, 768,
                   a.v_w, 768, buf.hidden, 768, V, 768, a.v_b, stream);

        // Multi-head attention: 12 heads, 64 dim each
        // scores[h] = Q_h @ K_h^T / sqrt(64)
        float scale = 1.0f / sqrtf(64.0f);
        sgemm_strided_batched(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, 64,
            K, 768, 64,
            Q, 768, 64,
            buf.attn_scores, T, (long long)T * T,
            12, scale);

        // Softmax over last dim of scores[h, t, :]
        softmax_f32(buf.attn_scores, buf.attn_scores, 12 * T, T, stream);

        // Attn output: context = scores @ V per head
        sgemm_strided_batched(cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            64, T, T,
            V, 768, 64,
            buf.attn_scores, T, (long long)T * T,
            buf.attn_out, 768, 64,
            12);

        // Dense projection: attn_out @ dense_w^T + dense_b -> [T, 768]
        sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 768, T, 768,
                   a.dense_w, 768, buf.attn_out, 768, buf.temp, 768,
                   a.dense_b, stream);

        // Fused Residual + LayerNorm (attention)
        residual_layer_norm_f32(buf.hidden, buf.temp, a.attn_ln_w, a.attn_ln_b,
                                 buf.hidden, T, 768, 1e-12f, stream);

        // --- FFN ---
        sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 2048, T, 768,
                   a.ffn_w, 768, buf.hidden, 768, buf.ff_mid, 2048,
                   a.ffn_b, stream);
        gelu_f32(buf.ff_mid, buf.ff_mid, T * 2048, stream);

        sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 768, T, 2048,
                   a.ffn_out_w, 2048, buf.ff_mid, 2048, buf.ff_out, 768,
                   a.ffn_out_b, stream);

        // Fused Residual + LayerNorm (FFN)
        residual_layer_norm_f32(buf.hidden, buf.ff_out, a.ffn_ln_w, a.ffn_ln_b,
                                 buf.hidden, T, 768, 1e-12f, stream);
    }
}

// ---------------------------------------------------------------------------
// Text encoder forward pass
// ---------------------------------------------------------------------------

struct TextEncoderBuffers {
    float* emb = nullptr;       // [T, 512] embedding (time-major)
    float* conv_out = nullptr;  // [T, 512] conv output / working buffer
    float* lstm_out = nullptr;  // [T, 512] LSTM output

    void alloc(int T, GpuArena& arena) {
        emb      = arena.alloc<float>(T * 512);
        conv_out = arena.alloc<float>(T * 512);
        lstm_out = arena.alloc<float>(T * 512);
    }
};

// Forward declaration
static void bilstm_gpu(const float* d_input, float* d_output,
                        const Weights::BiLSTMWeights& lstm_w,
                        int T, int input_size, int hidden_size,
                        cublasHandle_t cublas, cudaStream_t stream,
                        GpuArena& arena);

// Run text encoder: input_ids[T] -> output[T, 512]
static void text_encoder_forward(const Weights& w, TextEncoderBuffers& buf,
                                  const int* token_ids_gpu,
                                  cublasHandle_t cublas, cudaStream_t stream,
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

        // Conv1d: [T, 512] -> [T, 512] — wv already has precomputed weight
        gemm_conv1d(cublas, x, blk.conv_wv, blk.conv_b, buf.conv_out,
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
    bilstm_gpu(x, buf.lstm_out, te_lstm_w, T, 512, 256, cublas, stream, arena);

    // Copy LSTM output back to emb for downstream use
    cudaMemcpyAsync(buf.emb, buf.lstm_out, T * 512 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    // Final result is in buf.emb [T, 512]
}

// ---------------------------------------------------------------------------
// GPU BiLSTM: cuBLAS SGEMM for gate pre-computation + kernel for nonlinearities
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
                        cublasHandle_t cublas, cudaStream_t stream,
                        GpuArena& arena) {
    int G = 4 * hidden_size;  // gate_size
    int H = hidden_size;

    // Buffers from arena
    size_t arena_save = arena.save();
    float* d_ig      = arena.alloc<float>(T * G);
    float* d_h_fwd   = arena.alloc<float>(T * H);
    float* d_h_rev   = arena.alloc<float>(T * H);
    float* d_c       = arena.alloc<float>(H);
    float* d_h_zero  = arena.alloc<float>(H);

    // Helper: run one direction using cuBLAS SGEMM + per-timestep SGEMV
    auto run_direction = [&](const float* wih, const float* whh,
                              const float* bias_combined,
                              float* d_h_all, bool reverse) {
        // Step 1: Pre-compute input gates for all timesteps (batched GEMM)
        sgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, G, T, input_size,
              wih, input_size, d_input, input_size, d_ig, G);

        // Add precomputed bias (bih + bhh)
        bias_add_f32(d_ig, bias_combined, d_ig, T, G, stream);

        // Step 2: Per-timestep LSTM using cuBLAS SGEMV for Whh @ h
        CUDA_CHECK(cudaMemsetAsync(d_c, 0, H * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_h_zero, 0, H * sizeof(float), stream));

        float one = 1.0f;
        for (int step = 0; step < T; step++) {
            int t = reverse ? (T - 1 - step) : step;
            int t_prev = reverse ? (t + 1) : (t - 1);
            const float* h_prev = (step == 0) ? d_h_zero : d_h_all + t_prev * H;

            CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_T,
                                      H, G, &one,
                                      whh, H, h_prev, 1, &one,
                                      d_ig + t * G, 1));
            lstm_gates_f32(d_ig + t * G, d_c, d_c, d_h_all + t * H, H, stream);
        }
    };

    // Forward direction
    run_direction(lstm_w.wih_fwd, lstm_w.whh_fwd,
                  lstm_w.bias_fwd, d_h_fwd, false);

    // Reverse direction
    run_direction(lstm_w.wih_rev, lstm_w.whh_rev,
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

static void adain_1d_forward(const float* x, const float* style,
                              const Weights::AdaIN1dWeights& ain_w,
                              float* y, float* fc_buf, float* norm_ws,
                              int C, int T, int style_dim,
                              cublasHandle_t cublas, cudaStream_t stream) {
    // Step 1: fc(style) -> [2*C] = [gamma, beta]
    // fc_buf must be >= 2*C floats
    sgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2*C, 1, style_dim,
          ain_w.fc_w, style_dim, style, style_dim, fc_buf, 2*C);
    bias_add_f32(fc_buf, ain_w.fc_b, fc_buf, 1, 2*C, stream);

    // Step 2: Fused InstanceNorm + StyleAffine (two-pass coalesced)
    // gamma = fc_buf[0..C-1], beta = fc_buf[C..2C-1]
    // norm_ws must be >= 2*C floats (reduction buffer)
    instance_norm_style_affine_f32(x, ain_w.norm_w, ain_w.norm_b,
                                    fc_buf, fc_buf + C, y,
                                    norm_ws, C, T, 1e-5f, stream);
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
                                     cublasHandle_t cublas, cudaStream_t stream,
                                     float* workspace = nullptr,
                                     size_t workspace_bytes = 0) {
    int T_out = blk.has_upsample ? 2 * T : T;

    // ---- Residual path ----
    // fc_buf layout: [0..2C) = gamma/beta from FC, [2C..4C) = norm reduction workspace
    float* norm_ws = fc_buf + 2 * dim_in;
    adain_1d_forward(x, style, blk.norm1, residual_buf, fc_buf, norm_ws,
                     dim_in, T, style_dim, cublas, stream);
    leaky_relu_f32(residual_buf, residual_buf, dim_in * T, 0.2f, stream);

    if (blk.has_upsample) {
        conv_transpose1d_depthwise_f32(residual_buf, blk.pool_wv, blk.pool_b,
                                         shortcut_buf, dim_in, T, 3, 2, 1, 1, stream);
        cudaMemcpyAsync(residual_buf, shortcut_buf, dim_in * T_out * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    gemm_conv1d(cublas, residual_buf, blk.conv1_wv, blk.conv1_b, y,
                 workspace, workspace_bytes, dim_in, dim_out, T_out, 3,
                 1, 1, 1, stream);
    cudaMemcpyAsync(residual_buf, y, dim_out * T_out * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    norm_ws = fc_buf + 2 * dim_out;
    adain_1d_forward(residual_buf, style, blk.norm2, residual_buf, fc_buf, norm_ws,
                     dim_out, T_out, style_dim, cublas, stream);
    leaky_relu_f32(residual_buf, residual_buf, dim_out * T_out, 0.2f, stream);

    gemm_conv1d(cublas, residual_buf, blk.conv2_wv, blk.conv2_b, y,
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
        gemm_conv1d(cublas, shortcut_buf, blk.conv1x1_wv, nullptr, residual_buf,
                     workspace, workspace_bytes, dim_in, dim_out, T_out, 1,
                     1, 0, 1, stream);
        cudaMemcpyAsync(shortcut_buf, residual_buf, dim_out * T_out * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // ---- Combine: (residual + shortcut) / sqrt(2) ----
    add_f32(y, shortcut_buf, y, dim_out * T_out, stream);
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    scale_f32(y, y, dim_out * T_out, inv_sqrt2, stream);
}

// ---------------------------------------------------------------------------
// SineGen (CPU, validation-only): F0 [L2] → har_source [T_audio=L2*300]
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

    // Step 1: F0 nearest-neighbor upsample [L2] → [T_audio]
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

    // Step 5: Cumulative sum → phase [L2, 9]
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

    // Step 9: Linear combination [T_audio, 9] → [T_audio, 1] + tanh
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
//   3 rounds of: adain1 → Snake → dilated_conv → adain2 → Snake → conv → residual
//   x: [T, C] in-place, output: [T, C]
//   Working buffers: xt_buf [T, C], conv_out_buf [T, C], fc_buf [2*C]
// ---------------------------------------------------------------------------

static void adain_resblock1_forward(float* x, const float* style,
                                      const Weights::AdaINResBlock1Weights& rb,
                                      float* xt_buf, float* conv_out_buf,
                                      float* fc_buf,
                                      int T, int style_dim,
                                      cublasHandle_t cublas, cudaStream_t stream,
                                      float* workspace = nullptr,
                                      size_t workspace_bytes = 0) {
    int C = rb.channels;
    int K = rb.kernel_size;
    int dilations[] = {1, 3, 5};

    float* norm_ws = fc_buf + 2 * C;
    for (int j = 0; j < 3; j++) {
        adain_1d_forward(x, style, rb.adain1[j], xt_buf, fc_buf, norm_ws,
                         C, T, style_dim, cublas, stream);
        snake_f32(xt_buf, rb.alpha1[j], xt_buf, C, T, stream);

        int d = dilations[j];
        int pad = (K * d - d) / 2;
        gemm_conv1d(cublas, xt_buf, rb.convs1[j].wv, rb.convs1[j].b, conv_out_buf,
                     workspace, workspace_bytes, C, C, T, K, 1, pad, d, stream);

        adain_1d_forward(conv_out_buf, style, rb.adain2[j], xt_buf, fc_buf, norm_ws,
                         C, T, style_dim, cublas, stream);
        snake_f32(xt_buf, rb.alpha2[j], xt_buf, C, T, stream);

        int pad2 = (K - 1) / 2;
        gemm_conv1d(cublas, xt_buf, rb.convs2[j].wv, rb.convs2[j].b, x,
                     workspace, workspace_bytes, C, C, T, K, 1, pad2, 1, stream,
                     x);  // residual: x = conv(xt_buf) + x, then + bias
    }
}

// ---------------------------------------------------------------------------
// WAV file writer (16-bit PCM)
// ---------------------------------------------------------------------------

void write_wav_to_(std::ostream& f, const float* audio, int n_samples,
                   int sample_rate) {
    int16_t bits_per_sample = 16;
    int16_t num_channels = 1;
    int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int16_t block_align = num_channels * bits_per_sample / 8;
    int32_t data_size = n_samples * block_align;
    int32_t chunk_size = 36 + data_size;

    // RIFF header
    f.write("RIFF", 4);
    f.write(reinterpret_cast<char*>(&chunk_size), 4);
    f.write("WAVE", 4);

    // fmt subchunk
    f.write("fmt ", 4);
    int32_t fmt_size = 16;
    int16_t audio_format = 1;  // PCM
    f.write(reinterpret_cast<char*>(&fmt_size), 4);
    f.write(reinterpret_cast<char*>(&audio_format), 2);
    f.write(reinterpret_cast<char*>(&num_channels), 2);
    f.write(reinterpret_cast<char*>(&sample_rate), 4);
    f.write(reinterpret_cast<char*>(&byte_rate), 4);
    f.write(reinterpret_cast<char*>(&block_align), 2);
    f.write(reinterpret_cast<char*>(&bits_per_sample), 2);

    // data subchunk
    f.write("data", 4);
    f.write(reinterpret_cast<char*>(&data_size), 4);

    // Convert float [-1,1] to int16
    for (int i = 0; i < n_samples; i++) {
        float s = audio[i];
        s = std::max(-1.0f, std::min(1.0f, s));
        int16_t sample = (int16_t)(s * 32767.0f);
        f.write(reinterpret_cast<char*>(&sample), 2);
    }
}

bool write_wav(const std::string& path, const float* audio, int n_samples,
               int sample_rate) {
    if (path == "-") {
        write_wav_to_(std::cout, audio, n_samples, sample_rate);
        std::cout.flush();
        return true;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    write_wav_to_(f, audio, n_samples, sample_rate);
    return f.good();
}

// ---------------------------------------------------------------------------
// Compute exact decode-arena bytes for given T (tokens) and L (duration frames)
// Walks every arena.alloc() from the decode phase with 256-byte alignment.
// ---------------------------------------------------------------------------

size_t compute_decode_bytes(int T, int L) {
    int L2 = 2 * L;
    int T_audio = L2 * 300;
    int har_frames = T_audio / 5 + 1;  // 60*L2 + 1

    // Mimic GpuArena::alloc alignment: aligned = (off + 255) & ~255
    auto a = [](size_t off, size_t bytes) -> size_t {
        return ((off + 255) & ~(size_t)255) + bytes;
    };

    size_t off = 0;

    // Workspace for gemm_conv1d/gemm_conv_transpose1d (first allocation).
    // Max is generator resblocks: C=128, K=11, T=har_frames → 1408*har_frames floats.
    size_t max_ws_floats = (size_t)128 * 11 * har_frames;
    off = a(off, max_ws_floats * sizeof(float));

    // Alignment matrix + expanded encoder + shared LSTM output
    off = a(off, (size_t)T * L * sizeof(float));        // d_alignment
    off = a(off, (size_t)L * 640 * sizeof(float));       // d_en_expanded [L, 640]
    off = a(off, (size_t)L * 512 * sizeof(float));       // d_shared_out [L, 512]

    // F0/N working buffers (shared across both chains)
    off = a(off, (size_t)512 * L2 * sizeof(float));      // d_res_buf
    off = a(off, (size_t)512 * L2 * sizeof(float));      // d_sc_buf
    off = a(off, (size_t)2048 * sizeof(float));           // d_fc_buf (2*C + 2*C norm ws)

    // F0/N predictions
    off = a(off, (size_t)L2 * sizeof(float));             // d_f0_pred
    off = a(off, (size_t)L2 * sizeof(float));             // d_n_pred

    // Decoder inputs
    off = a(off, (size_t)L * 512 * sizeof(float));        // d_asr_aligned [L, 512]
    off = a(off, (size_t)L * sizeof(float));              // d_f0_down
    off = a(off, (size_t)L * sizeof(float));              // d_n_down
    off = a(off, (size_t)L * 514 * sizeof(float));        // d_dec_cat [L, 514]

    // Decoder working buffers
    int max_ch = 1090;
    off = a(off, (size_t)max_ch * L2 * sizeof(float));   // d_dec_res
    off = a(off, (size_t)max_ch * L2 * sizeof(float));   // d_dec_sc
    off = a(off, (size_t)4 * max_ch * sizeof(float));    // d_dec_fc (2*C + 2*C norm ws)

    // Decoder blocks
    off = a(off, (size_t)L * 1024 * sizeof(float));       // d_dec_out [L, 1024]
    off = a(off, (size_t)L * 64 * sizeof(float));         // d_asr_res [L, 64]
    off = a(off, (size_t)L2 * 1090 * sizeof(float));      // d_dec_in [L2, 1090]

    // Decode blocks 0-2: L*1024 each; block 3: L2*512 (has_upsample)
    for (int i = 0; i < 3; i++)
        off = a(off, (size_t)L * 1024 * sizeof(float));
    off = a(off, (size_t)L2 * 512 * sizeof(float));

    // Generator: d_gen_har_ct [22, har_frames] persists through STFT,
    // then d_gen_har [har_frames, 22] (transposed) persists for generator.
    // SineGen temps overlap with gen_har_ct area after restore.
    off = a(off, (size_t)22 * har_frames * sizeof(float)); // d_gen_har_ct
    size_t sinegen_off = off;
    // SineGen temporaries (reclaimed via save/restore)
    size_t sg = off;
    sg = a(sg, (size_t)L2 * 9 * sizeof(float));            // d_phase_low
    sg = a(sg, (size_t)T_audio * sizeof(float));            // d_har_source
    sg = a(sg, (size_t)9 * sizeof(float));                  // d_rand_ini

    // After sinegen restore, d_gen_har (transposed) is allocated
    off = a(off, (size_t)har_frames * 22 * sizeof(float));  // d_gen_har [har_frames, 22]

    // Generator working pool: 5 slots of 128*har_frames (buffers reused via liveness)
    // Must account for max of sinegen temps or gen pool
    size_t gen_pool_end = a(off, (size_t)5 * 128 * har_frames * sizeof(float));
    off = (sg > gen_pool_end) ? sg : gen_pool_end;
    off = a(off, (size_t)512 * sizeof(float));              // d_gen_fc

    // Final audio buffer
    off = a(off, (size_t)T_audio * sizeof(float));          // d_audio

    return off;
}

// ---------------------------------------------------------------------------
// Encode-phase CUDA graph cache: keyed by T (token count).
// All arena allocations are deterministic per T, so cached graphs
// use the same device pointers on replay.
// ---------------------------------------------------------------------------

static std::unordered_map<int, cudaGraphExec_t> s_encode_graph_cache;

// ---------------------------------------------------------------------------
// Rokoko inference: phoneme IDs + style vector → audio
// ---------------------------------------------------------------------------

std::vector<float> rokoko_infer(const Weights& w,
                                const int* token_ids, int T,
                                const float* style_vec,  // [256] on host
                                cublasHandle_t cublas,
                                cublasLtHandle_t ltHandle,
                                cudaStream_t stream,
                                GpuArena& arena,
                                GpuArena& decode_arena,
                                float* d_workspace,
                                size_t workspace_bytes) {
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

    // ---- H2D uploads (outside graph — blocking) ----
    CUDA_CHECK(cudaMemcpy(d_token_ids, token_ids, T * sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_s, style_vec, 256 * sizeof(float),
                           cudaMemcpyHostToDevice));

    // ---- Encode phase: CUDA graph capture/replay ----
    auto git = s_encode_graph_cache.find(T);
    if (git == s_encode_graph_cache.end()) {
        // Provide cuBLAS workspace to prevent cudaMalloc during capture
        cublasSetWorkspace(cublas, d_workspace, workspace_bytes);

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // ALBERT encoder
        cudaMemcpyAsync(albert_buf.token_ids, d_token_ids, T * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);
        albert_forward(w, albert_buf, cublas, ltHandle, stream, T);

        // bert_encoder: Linear 768→512 → d_en [T, 512]
        sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 512, T, 768,
                   w.bert_enc_w, 768, albert_buf.hidden, 768, d_en, 512,
                   w.bert_enc_b, stream);

        // Text encoder
        text_encoder_forward(w, te_buf, d_token_ids, cublas, stream, T, arena,
                             d_workspace, workspace_bytes);

        // Duration encoder: 3x (BiLSTM + AdaLN)
        tile_1d_f32(d_style_prosody, d_style_tiled, 128, T, stream);
        concat_channels_f32(d_en, d_style_tiled, d_cat_buf, T, 512, 128, stream);

        for (int layer_i = 0; layer_i < 3; layer_i++) {
            bilstm_gpu(d_cat_buf, d_lstm_out, w.dur_enc_lstm[layer_i],
                       T, 640, 256, cublas, stream, arena);

            auto& aln = w.dur_enc_aln[layer_i];
            sgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 1024, 1, 128,
                  aln.fc_w, 128, d_style_prosody, 128, d_ada_fc, 1024);
            bias_add_f32(d_ada_fc, aln.fc_b, d_ada_fc, 1, 1024, stream);
            ada_layer_norm_f32(d_lstm_out, d_ada_fc, d_ada_fc + 512,
                               d_x_buf, T, 512, 1e-5f, stream);

            concat_channels_f32(d_x_buf, d_style_tiled, d_cat_buf, T, 512, 128, stream);
        }

        // Duration LSTM + projection
        bilstm_gpu(d_cat_buf, d_dur_lstm_out, w.dur_lstm,
                   T, 640, 256, cublas, stream, arena);
        sgemm_bias(ltHandle, CUBLAS_OP_T, CUBLAS_OP_N, 50, T, 512,
                   w.dur_proj_w, 512, d_dur_lstm_out, 512, d_dur_proj, 50,
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

    // Sync for L only (4 bytes) — needed for decode arena sizing
    int L;
    CUDA_CHECK(cudaMemcpyAsync(&L, d_L, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int L2 = 2 * L;

    // ===== DECODE PHASE: exact-sized arena =====
    size_t decode_bytes = compute_decode_bytes(T, L);
    if (decode_arena.capacity < decode_bytes) {
        decode_arena.destroy();
        decode_arena.init(decode_bytes);
    } else {
        decode_arena.reset();
    }

    // Decode workspace (first alloc, matches compute_decode_bytes order)
    int har_frames_ws = (L2 * 300) / 5 + 1;
    size_t dec_ws_floats = (size_t)128 * 11 * har_frames_ws;
    float* d_dec_workspace = decode_arena.alloc<float>(dec_ws_floats);
    size_t dec_ws_bytes = dec_ws_floats * sizeof(float);

    // Build alignment matrix on GPU
    float* d_alignment = decode_arena.alloc<float>(T * L);
    build_alignment_f32(d_int_durations, d_alignment, T, L, stream);

    // Expand encoder output: alignment^T [L, T] × d_cat_buf [T, 640] → [L, 640]
    // d_cat_buf is [T, 640] row-major = col-major [640, T]
    // alignment is [T, L] row-major = col-major [L, T]
    // We want result [L, 640] row-major = col-major [640, L]
    // cuBLAS: C[640,L] = A[640,T] * B[T,L]
    //   A=d_cat_buf (col-major [640,T]), OP_N
    //   B=d_alignment (col-major [L,T]), OP_T → [T,L]
    // Wait — alignment col-major [L,T] transposed gives [T,L] which is wrong.
    // alignment row-major [T,L] = col-major [L,T], we want alignment^T = [L,T].
    // cuBLAS: C[640,L] = A[640,T] × B[T,L]
    //   B = alignment^T in row-major = [L,T], but alignment is stored col-major [L,T]
    //   So OP_N on alignment gives [L,T] → that's T columns, but we need [T,L].
    // Let me think in cuBLAS col-major terms:
    //   Result: row-major [L, 640] = col-major [640, L], so m=640, n=L
    //   C[640,L] = A[640,T] * B[T,L]
    //   A = d_cat_buf stored as col-major [640, T], lda=640 → BUT d_cat_buf is row-major [T,640]
    //     = col-major [640, T] with lda=640. YES.
    //   B = d_alignment stored as col-major [L, T]. We need [T, L] so transb=OP_T, ldb=L
    float* d_en_expanded = decode_arena.alloc<float>(L * 640);
    sgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, 640, L, T,
          d_cat_buf, 640, d_alignment, L, d_en_expanded, 640);

    // Shared LSTM: d_en_expanded [L, 640] already row-major — no transpose
    float* d_shared_out = decode_arena.alloc<float>(L * 512);
    bilstm_gpu(d_en_expanded, d_shared_out, w.shared_lstm,
               L, 640, 256, cublas, stream, decode_arena);

    // F0 and N prediction chains
    // Both: [L, 512] → 3 AdainResBlk1d → Conv1d proj → [2L, 1]
    float* d_res_buf = decode_arena.alloc<float>(512 * L2);
    float* d_sc_buf  = decode_arena.alloc<float>(512 * L2);
    float* d_fc_buf  = decode_arena.alloc<float>(2048);  // 2*C for gamma/beta + 2*C for norm reduction

    auto run_f0n_chain = [&](const Weights::AdainResBlk1dWeights blocks[3],
                              const float* proj_w, const float* proj_b,
                              float* d_pred) {
        size_t chain_save = decode_arena.save();
        float* d_fx = decode_arena.alloc<float>(512 * L2);
        float* d_fo = decode_arena.alloc<float>(512 * L2);
        // d_shared_out is [L, 512] — already [T, C] layout, no transpose
        cudaMemcpyAsync(d_fx, d_shared_out, L * 512 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // block 0: [L, 512] → [L, 512]
        adain_resblk1d_forward(d_fx, d_style_prosody, blocks[0],
                                d_res_buf, d_sc_buf, d_fc_buf, d_fo,
                                512, 512, L, 128, cublas, stream,
                                d_dec_workspace, dec_ws_bytes);
        // block 1: [L, 512] → [2L, 256] (upsample)
        cudaMemcpyAsync(d_fx, d_fo, 512 * L * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        adain_resblk1d_forward(d_fx, d_style_prosody, blocks[1],
                                d_res_buf, d_sc_buf, d_fc_buf, d_fo,
                                512, 256, L, 128, cublas, stream,
                                d_dec_workspace, dec_ws_bytes);
        // block 2: [2L, 256] → [2L, 256]
        cudaMemcpyAsync(d_fx, d_fo, 256 * L2 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        adain_resblk1d_forward(d_fx, d_style_prosody, blocks[2],
                                d_res_buf, d_sc_buf, d_fc_buf, d_fo,
                                256, 256, L2, 128, cublas, stream,
                                d_dec_workspace, dec_ws_bytes);
        // Projection: Conv1d(256→1, k=1) → [2L, 1]
        conv1d_f32(d_fo, proj_w, proj_b, d_pred, 256, 1, L2, 1, stream);
        decode_arena.restore(chain_save);
    };

    float* d_f0_pred = decode_arena.alloc<float>(L2);
    float* d_n_pred  = decode_arena.alloc<float>(L2);
    run_f0n_chain(w.f0_blocks, w.f0_proj_w, w.f0_proj_b, d_f0_pred);
    run_f0n_chain(w.n_blocks, w.n_proj_w, w.n_proj_b, d_n_pred);

    // ---- Decoder ----
    // Build asr_aligned: alignment^T [L, T] × te_buf.emb [T, 512] → [L, 512]
    float* d_asr_aligned = decode_arena.alloc<float>(L * 512);
    sgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, 512, L, T,
          te_buf.emb, 512, d_alignment, L, d_asr_aligned, 512);

    // F0/N downsampling: Conv1d(1,1,k=3,s=2,p=1) on [2L, 1] → [L, 1]
    float* d_f0_down = decode_arena.alloc<float>(L);
    float* d_n_down  = decode_arena.alloc<float>(L);

    // F0/N downsample — wv has precomputed weight
    conv1d_general_f32(d_f0_pred, w.dec_f0_conv_wv, w.dec_f0_conv_b,
                       d_f0_down, 1, 1, L2, 3, 2, 1, 1, stream);
    conv1d_general_f32(d_n_pred, w.dec_n_conv_wv, w.dec_n_conv_b,
                       d_n_down, 1, 1, L2, 3, 2, 1, 1, stream);

    // Concatenate [asr_aligned, F0_down, N_down] → [L, 514]
    float* d_dec_cat = decode_arena.alloc<float>(L * 514);
    concat3_channels_f32(d_asr_aligned, d_f0_down, d_n_down,
                          d_dec_cat, L, 512, 1, 1, stream);

    // Decoder working buffers
    int max_ch = 1090;
    float* d_dec_res = decode_arena.alloc<float>(max_ch * L2);
    float* d_dec_sc  = decode_arena.alloc<float>(max_ch * L2);
    float* d_dec_fc  = decode_arena.alloc<float>(4 * max_ch);  // 2*C gamma/beta + 2*C norm reduction

    // Encode: AdainResBlk1d(514→1024) [L, 514] → [L, 1024]
    float* d_dec_out = decode_arena.alloc<float>(L * 1024);
    adain_resblk1d_forward(d_dec_cat, d_style_acoustic, w.dec_encode,
                            d_dec_res, d_dec_sc, d_dec_fc, d_dec_out,
                            514, 1024, L, 128, cublas, stream,
                            d_dec_workspace, dec_ws_bytes);

    // asr_res: Conv1d(512→64, k=1) — wv has precomputed weight
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
            //            + d_f0_down [T_blk, 1] + d_n_down [T_blk, 1] → [T_blk, 1090]
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
                                128, cublas, stream,
                                d_dec_workspace, dec_ws_bytes);

        d_dec_x = d_blk_out;

        if (w.dec_decode[bi].has_upsample) {
            res_flag = false;
            L = T_blk_out;  // update L after upsample
        }
    }
    // d_dec_x is now [L2, 512] — generator input

    // ---- Generator ----
    // SineGen: F0 → harmonic source → STFT → har [22, frames]
    // STFT output is inherently [n_fft/2+1, n_frames] = [C, T] layout.
    // gen_har stays in [C, T] layout; the generator noise_convs kernels
    // use [T, C] layout, so we'll transpose gen_har to [T, C] for them.
    int T_audio = L2 * 300;
    int n_fft_gen = 20, hop_gen = 5;
    int har_frames = T_audio / hop_gen + 1;

    // gen_har stored as [22, har_frames] (STFT output, channels-first)
    float* d_gen_har_ct = decode_arena.alloc<float>(22 * har_frames);
    {
        size_t sinegen_save = decode_arena.save();
        float* d_phase_low = decode_arena.alloc<float>(L2 * 9);
        float* d_har_source = decode_arena.alloc<float>(T_audio);

        float rand_ini[9];
        rand_ini[0] = 0.0f;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        for (int h = 1; h < 9; h++) rand_ini[h] = uni(rng);
        float* d_rand_ini = decode_arena.alloc<float>(9);
        cudaMemcpyAsync(d_rand_ini, rand_ini, 9 * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        sinegen_phase_f32(d_f0_pred, d_rand_ini, d_phase_low, L2, stream);
        sinegen_source_f32(d_phase_low, d_f0_pred, w.gen_source_w, w.gen_source_b,
                           d_har_source, L2, T_audio, 42, stream);

        float* d_har_spec = d_gen_har_ct;
        float* d_har_phase = d_gen_har_ct + 11 * har_frames;
        stft_f32(d_har_source, d_har_spec, d_har_phase,
                 T_audio, n_fft_gen, hop_gen, stream);
        decode_arena.restore(sinegen_save);
    }

    // Transpose gen_har from [22, har_frames] → [har_frames, 22] for [T,C] layout
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

        // noise_convs[i]: gen_har [har_frames, 22] → slot1 [src_T, C_out]
        float* d_gen_src = slot1;
        if (i == 0) {
            gemm_conv1d(cublas, d_gen_har, w.gen_noise_convs[i].w,
                         w.gen_noise_convs[i].b, d_gen_src,
                         d_dec_workspace, dec_ws_bytes,
                         22, C_out, har_frames, 12, 6, 3, 1, stream);
        } else {
            gemm_conv1d(cublas, d_gen_har, w.gen_noise_convs[i].w,
                         w.gen_noise_convs[i].b, d_gen_src,
                         d_dec_workspace, dec_ws_bytes,
                         22, C_out, har_frames, 1, 1, 0, 1, stream);
        }

        // noise_res[i]: uses slot2 (xt), slot3 (co) as temporaries
        float* d_gen_xt = slot2;
        float* d_gen_co = slot3;
        int src_T = (i == 0) ? ((har_frames + 2*3 - 12) / 6 + 1) : har_frames;
        adain_resblock1_forward(d_gen_src, d_style_acoustic, w.gen_noise_res[i],
                                d_gen_xt, d_gen_co, d_gen_fc,
                                src_T, 128, cublas, stream,
                                d_dec_workspace, dec_ws_bytes);

        // ups[i]: gen_x (slot0) → slot4 (gen_tmp)
        float* d_gen_tmp = slot4;
        int T_ups = (T_loop - 1) * us - 2 * up + uk;
        gemm_conv_transpose1d(cublas, d_gen_x, w.gen_ups[i].wv, w.gen_ups[i].b,
                               d_gen_tmp, d_dec_workspace, dec_ws_bytes,
                               C_in, C_out, T_loop, uk, us, up, 0, stream);

        // Copy/pad tmp → gen_x
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
                                    T_ups, 128, cublas, stream,
                                    d_dec_workspace, dec_ws_bytes);
            add_f32(d_rb_avg, d_gen_tmp, d_rb_avg, C_out * T_ups, stream);
        }
        scale_f32(d_rb_avg, d_gen_x, C_out * T_ups, 1.0f / 3.0f, stream);

        T_loop = T_ups;
    }

    // Post: LeakyReLU(0.01) + conv_post(128→22, k=7)
    // Result is [T_loop, 22] in [T,C] layout
    float* d_gen_tmp = slot4;
    leaky_relu_f32(d_gen_x, d_gen_x, 128 * T_loop, 0.01f, stream);
    gemm_conv1d(cublas, d_gen_x, w.gen_conv_post_wv, w.gen_conv_post_b, d_gen_tmp,
                 d_dec_workspace, dec_ws_bytes, 128, 22, T_loop, 7, 1, 3, 1, stream);

    // Transpose [T_loop, 22] → [22, T_loop] for STFT boundary (channels-first)
    float* d_gen_ct = d_gen_x;  // reuse slot0 (128*har_frames >> 22*T_loop)
    transpose_f32(d_gen_tmp, d_gen_ct, T_loop, 22, stream);

    // spec = exp(x[:11,:]), phase = sin(x[11:,:]) — in [C,T] layout
    int n_freqs = 11;
    exp_f32(d_gen_ct, d_gen_tmp, n_freqs * T_loop, stream);
    sin_f32(d_gen_ct + n_freqs * T_loop, d_gen_tmp + n_freqs * T_loop,
            n_freqs * T_loop, stream);

    // iSTFT → audio (operates on [n_freqs, T_loop] channels-first)
    float* d_audio = decode_arena.alloc<float>(T_audio);
    istft_f32(d_gen_tmp, d_gen_tmp + n_freqs * T_loop, d_audio,
              T_loop, 20, 5, T_audio, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Download audio
    std::vector<float> audio(T_audio);
    cudaMemcpy(audio.data(), d_audio, T_audio * sizeof(float),
               cudaMemcpyDeviceToHost);

    return audio;
}

// ---------------------------------------------------------------------------
// Precompute all weight norms: wv[o,:] = wg[o] * wv[o,:] / ||wv[o,:]||
// Call once after upload. After this, wv contains the materialized weight
// and weight_norm_f32 calls in inference can be skipped.
// ---------------------------------------------------------------------------

void precompute_weight_norms(Weights& w, cudaStream_t stream) {
    auto wn = [&](float* wg, float* wv, int C_out, int fan_in) {
        weight_norm_f32(wg, wv, wv, C_out, fan_in, stream);  // in-place
    };

    // Text encoder conv blocks (3x)
    for (int i = 0; i < 3; i++)
        wn(w.text_conv[i].conv_wg, w.text_conv[i].conv_wv, 512, 512 * 5);

    // F0/N blocks (3x AdainResBlk1d each):
    //   [0] 512→512, [1] 512→256 (upsample+shortcut), [2] 256→256
    {
        int dim_in[]  = {512, 512, 256};
        int dim_out[] = {512, 256, 256};
        for (auto* blocks : {w.f0_blocks, w.n_blocks}) {
            for (int i = 0; i < 3; i++) {
                auto& b = blocks[i];
                wn(b.conv1_wg, b.conv1_wv, dim_out[i], dim_in[i] * 3);
                wn(b.conv2_wg, b.conv2_wv, dim_out[i], dim_out[i] * 3);
                if (b.has_shortcut)
                    wn(b.conv1x1_wg, b.conv1x1_wv, dim_out[i], dim_in[i] * 1);
                if (b.has_upsample)
                    wn(b.pool_wg, b.pool_wv, dim_in[i], 3);
            }
        }
    }

    // Decoder: F0_conv, N_conv, asr_res
    wn(w.dec_f0_conv_wg, w.dec_f0_conv_wv, 1, 1 * 3);
    wn(w.dec_n_conv_wg, w.dec_n_conv_wv, 1, 1 * 3);
    wn(w.dec_asr_res_wg, w.dec_asr_res_wv, 64, 512 * 1);

    // Decoder: encode block (AdainResBlk1d 514→1024)
    {
        auto& b = w.dec_encode;
        wn(b.conv1_wg, b.conv1_wv, 1024, 514 * 3);
        wn(b.conv2_wg, b.conv2_wv, 1024, 1024 * 3);
        if (b.has_shortcut)
            wn(b.conv1x1_wg, b.conv1x1_wv, 1024, 514 * 1);
        if (b.has_upsample)
            wn(b.pool_wg, b.pool_wv, 514, 3);
    }

    // Decoder: decode blocks (4x AdainResBlk1d)
    for (int i = 0; i < Weights::DEC_N_DECODE; i++) {
        auto& b = w.dec_decode[i];
        int dim_in = 1090;  // all blocks take 1090 (1024+64+2)
        int dim_out = (i < 3) ? 1024 : 512;
        wn(b.conv1_wg, b.conv1_wv, dim_out, dim_in * 3);
        wn(b.conv2_wg, b.conv2_wv, dim_out, dim_out * 3);
        if (b.has_shortcut)
            wn(b.conv1x1_wg, b.conv1x1_wv, dim_out, dim_in * 1);
        if (b.has_upsample)
            wn(b.pool_wg, b.pool_wv, dim_in, 3);
    }

    // Generator: ups (ConvTranspose1d with weight_norm)
    // ups[0]: 512→256 k=20, ups[1]: 256→128 k=12
    // weight: [C_in, C_out, K], so C_out_wn = C_in, fan_in = C_out * K
    int up_cin[]  = {512, 256};
    int up_cout[] = {256, 128};
    int up_k[]    = {20, 12};
    for (int i = 0; i < Weights::GEN_N_UPS; i++)
        wn(w.gen_ups[i].wg, w.gen_ups[i].wv, up_cin[i], up_cout[i] * up_k[i]);

    // Generator: resblocks (6x AdaINResBlock1) — use stored channels/kernel_size
    for (int i = 0; i < Weights::GEN_N_RESBLOCKS; i++) {
        auto& rb = w.gen_resblocks[i];
        int C = rb.channels, K = rb.kernel_size;
        for (int j = 0; j < 3; j++) {
            wn(rb.convs1[j].wg, rb.convs1[j].wv, C, C * K);
            wn(rb.convs2[j].wg, rb.convs2[j].wv, C, C * K);
        }
    }

    // Generator: noise_res (2x AdaINResBlock1) — use stored channels/kernel_size
    for (int i = 0; i < Weights::GEN_N_UPS; i++) {
        auto& rb = w.gen_noise_res[i];
        int C = rb.channels, K = rb.kernel_size;
        for (int j = 0; j < 3; j++) {
            wn(rb.convs1[j].wg, rb.convs1[j].wv, C, C * K);
            wn(rb.convs2[j].wg, rb.convs2[j].wv, C, C * K);
        }
    }

    // Generator: conv_post (128→22, k=7)
    wn(w.gen_conv_post_wg, w.gen_conv_post_wv, 22, 128 * 7);

    // Precompute LSTM biases: bias = bih + bhh for each direction
    auto precompute_lstm_bias = [&](Weights::BiLSTMWeights& lw, int H) {
        int G = 4 * H;
        CUDA_CHECK(cudaMalloc(&lw.bias_fwd, G * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lw.bias_rev, G * sizeof(float)));
        add_f32(lw.bih_fwd, lw.bhh_fwd, lw.bias_fwd, G, stream);
        add_f32(lw.bih_rev, lw.bhh_rev, lw.bias_rev, G, stream);
    };

    // Text encoder LSTM (H=256)
    // Build temporary BiLSTMWeights for text encoder since it uses flat fields
    {
        w.text_lstm_bias_fwd = nullptr;
        w.text_lstm_bias_rev = nullptr;
        CUDA_CHECK(cudaMalloc(&w.text_lstm_bias_fwd, 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w.text_lstm_bias_rev, 1024 * sizeof(float)));
        add_f32(w.text_lstm_bih_fwd, w.text_lstm_bhh_fwd, w.text_lstm_bias_fwd, 1024, stream);
        add_f32(w.text_lstm_bih_rev, w.text_lstm_bhh_rev, w.text_lstm_bias_rev, 1024, stream);
    }

    // Prosody predictor LSTMs
    for (int i = 0; i < 3; i++)
        precompute_lstm_bias(w.dur_enc_lstm[i], 256);
    precompute_lstm_bias(w.dur_lstm, 256);
    precompute_lstm_bias(w.shared_lstm, 256);

    // ---- Precompute NHWC weights for Cutlass implicit GEMM ----
    // Reshape [C_out, C_in, K] → [C_out, K, C_in] for Conv1d weights with K>1.
    // K=1 convolutions are just GEMMs and don't benefit from implicit GEMM.
    auto make_nhwc = [&](float* wv, int C_out, int C_in, int K) {
        if (K <= 1) return;
        float* w_nhwc = nullptr;
        CUDA_CHECK(cudaMalloc(&w_nhwc, (size_t)C_out * C_in * K * sizeof(float)));
        cutlass_reshape_weights(wv, w_nhwc, C_out, C_in, K, stream);
        s_w_nhwc[wv] = w_nhwc;
    };

    // Text encoder conv blocks (K=5)
    for (int i = 0; i < 3; i++)
        make_nhwc(w.text_conv[i].conv_wv, 512, 512, 5);

    // F0/N AdainResBlk1d conv weights (K=3)
    {
        int dim_in[]  = {512, 512, 256};
        int dim_out[] = {512, 256, 256};
        for (auto* blocks : {w.f0_blocks, w.n_blocks}) {
            for (int i = 0; i < 3; i++) {
                make_nhwc(blocks[i].conv1_wv, dim_out[i], dim_in[i], 3);
                make_nhwc(blocks[i].conv2_wv, dim_out[i], dim_out[i], 3);
            }
        }
    }

    // Decoder F0/N downsample (K=3)
    make_nhwc(w.dec_f0_conv_wv, 1, 1, 3);
    make_nhwc(w.dec_n_conv_wv, 1, 1, 3);

    // Decoder encode block (K=3)
    make_nhwc(w.dec_encode.conv1_wv, 1024, 514, 3);
    make_nhwc(w.dec_encode.conv2_wv, 1024, 1024, 3);

    // Decoder decode blocks (K=3)
    for (int i = 0; i < Weights::DEC_N_DECODE; i++) {
        int dim_in = 1090;
        int dim_out = (i < 3) ? 1024 : 512;
        make_nhwc(w.dec_decode[i].conv1_wv, dim_out, dim_in, 3);
        make_nhwc(w.dec_decode[i].conv2_wv, dim_out, dim_out, 3);
    }

    // Generator resblocks (K varies: 3, 7, 11)
    for (int i = 0; i < Weights::GEN_N_RESBLOCKS; i++) {
        auto& rb = w.gen_resblocks[i];
        int C = rb.channels, K = rb.kernel_size;
        for (int j = 0; j < 3; j++) {
            make_nhwc(rb.convs1[j].wv, C, C, K);
            make_nhwc(rb.convs2[j].wv, C, C, K);
        }
    }

    // Generator noise_res (K varies)
    for (int i = 0; i < Weights::GEN_N_UPS; i++) {
        auto& rb = w.gen_noise_res[i];
        int C = rb.channels, K = rb.kernel_size;
        for (int j = 0; j < 3; j++) {
            make_nhwc(rb.convs1[j].wv, C, C, K);
            make_nhwc(rb.convs2[j].wv, C, C, K);
        }
    }

    // Generator conv_post (K=7)
    make_nhwc(w.gen_conv_post_wv, 22, 128, 7);

    // Generator noise_convs (not weight-normed, use .w not .wv)
    make_nhwc(w.gen_noise_convs[0].w, 128, 22, 12);  // K=12, stride=6
    // noise_convs[1] has K=1, skip

    cudaStreamSynchronize(stream);
}
