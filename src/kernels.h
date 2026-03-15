// kernels.h — Custom CUDA kernel launch wrappers for Rokoko TTS
//
// All kernels operate on FP32 data.
// Signal tensors use [T, C] layout (time-major, channels last).

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Embedding gather: y[i, :] = table[ids[i], :]
//   table: [V, D], ids: [N], y: [N, D]
// ---------------------------------------------------------------------------
void embedding_gather(const float* table, const int* ids, float* y,
                      int N, int D, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Element-wise add: y = a + b  (N elements)
// ---------------------------------------------------------------------------
void add_f32(const float* a, const float* b, float* y, int N,
             cudaStream_t stream);

// ---------------------------------------------------------------------------
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
//   x, y:     [N, D]
//   gamma, beta: [D]
// ---------------------------------------------------------------------------
void layer_norm_f32(const float* x, const float* gamma, const float* beta,
                    float* y, int N, int D, float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused Residual + LayerNorm: y = LayerNorm(a + b; gamma, beta, eps)
//   Computes residual addition and layer normalization in a single kernel.
// ---------------------------------------------------------------------------
void residual_layer_norm_f32(const float* a, const float* b,
                              const float* gamma, const float* beta,
                              float* y, int N, int D, float eps,
                              cudaStream_t stream);

// ---------------------------------------------------------------------------
// GELU (tanh approx): y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   Matches PyTorch's gelu_new / approximate='tanh'.
// ---------------------------------------------------------------------------
void gelu_f32(const float* x, float* y, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// LeakyReLU: y = x > 0 ? x : alpha * x
// ---------------------------------------------------------------------------
void leaky_relu_f32(const float* x, float* y, int N, float alpha,
                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Sigmoid: y = 1 / (1 + exp(-x))
// ---------------------------------------------------------------------------
void sigmoid_f32(const float* x, float* y, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Softmax over last dimension: y[n, :] = softmax(x[n, :])
//   x, y: [N, D]
// ---------------------------------------------------------------------------
void softmax_f32(const float* x, float* y, int N, int D, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Bias add: y[n, d] = x[n, d] + bias[d]
//   x, y: [N, D], bias: [D]
// ---------------------------------------------------------------------------
void bias_add_f32(const float* x, const float* bias, float* y, int N, int D,
                  cudaStream_t stream);

// ---------------------------------------------------------------------------
// Transpose 2D: y[j, i] = x[i, j]
//   x: [M, N], y: [N, M]
// ---------------------------------------------------------------------------
void transpose_f32(const float* x, float* y, int M, int N,
                   cudaStream_t stream);

// ---------------------------------------------------------------------------
// Conv1d: y[t, c_out] = sum_{c_in, k} w[c_out, c_in, k] * x[t+k-pad, c_in] + b[c_out]
//   x: [T, C_in], w: [C_out, C_in, K], b: [C_out], y: [T, C_out]
//   Padding = (K-1)/2 for same-padding.
// ---------------------------------------------------------------------------
void conv1d_f32(const float* x, const float* w, const float* bias,
                float* y, int C_in, int C_out, int T, int K,
                cudaStream_t stream);

// ---------------------------------------------------------------------------
// Weight norm: w[o, i, k] = wg[o] * wv[o, i, k] / ||wv[o]||_2
//   wg: [C_out, 1, 1], wv: [C_out, C_in, K], w: [C_out, C_in, K]
// ---------------------------------------------------------------------------
void weight_norm_f32(const float* wg, const float* wv, float* w,
                     int C_out, int fan_in, cudaStream_t stream);

// ---------------------------------------------------------------------------
// LayerNorm across channels: normalize across C at each time position
//   x, y: [T, C], gamma, beta: [C]
//   For each t: y[t, :] = gamma * (x[t, :] - mean) / sqrt(var + eps) + beta
// ---------------------------------------------------------------------------
void layer_norm_channels_first_f32(const float* x, const float* gamma,
                                    const float* beta, float* y,
                                    int C, int T, float eps,
                                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Convert int64 array to int32 (for phoneme IDs from Python)
// ---------------------------------------------------------------------------
void cast_i64_to_i32(const int64_t* src, int* dst, int N,
                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// Instance normalization 1D: normalize each channel across time
//   x, y: [T, C], weight, bias: [C]
//   For each c: y[t,c] = weight[c] * (x[t,c] - mean_c) / sqrt(var_c + eps) + bias[c]
// ---------------------------------------------------------------------------
void instance_norm_1d_f32(const float* x, const float* weight, const float* bias,
                           float* y, int C, int T, float eps,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// Style affine 1D: y[t,c] = (1 + gamma[c]) * x[t,c] + beta[c]
//   x, y: [T, C], gamma, beta: [C]
// ---------------------------------------------------------------------------
void style_affine_1d_f32(const float* x, const float* gamma, const float* beta,
                           float* y, int C, int T, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused InstanceNorm + StyleAffine: single-pass AdaIN
//   y[t,c] = (1+gamma[c]) * (norm_w[c] * (x[t,c]-mean)/sqrt(var+eps) + norm_b[c]) + beta[c]
//   x, y: [T, C], norm_w, norm_b: [C], gamma, beta: [C]
// ---------------------------------------------------------------------------
void instance_norm_style_affine_f32(const float* x, const float* norm_w,
                                      const float* norm_b, const float* gamma,
                                      const float* beta, float* y,
                                      int C, int T, float eps,
                                      cudaStream_t stream);

// ---------------------------------------------------------------------------
// Adaptive LayerNorm: normalize rows, then style conditioning
//   x, y: [N, D], gamma, beta: [D] (style-derived)
//   y[n,d] = (1 + gamma[d]) * (x[n,d] - mean) / sqrt(var + eps) + beta[d]
// ---------------------------------------------------------------------------
void ada_layer_norm_f32(const float* x, const float* gamma, const float* beta,
                         float* y, int N, int D, float eps,
                         cudaStream_t stream);

// ---------------------------------------------------------------------------
// Depthwise transposed Conv1d: x[T_in, C] -> y[T_out, C]
//   T_out = (T_in - 1) * stride - 2*pad + K + out_pad
//   w: [C, 1, K], bias: [C], groups = C
// ---------------------------------------------------------------------------
void conv_transpose1d_depthwise_f32(const float* x, const float* w, const float* bias,
                                      float* y, int C, int T_in, int K,
                                      int stride, int pad, int out_pad,
                                      cudaStream_t stream);

// ---------------------------------------------------------------------------
// Nearest-neighbor 2x upsampling: x[T, C] -> y[2*T, C]
// ---------------------------------------------------------------------------
void upsample_nearest_1d_2x_f32(const float* x, float* y, int C, int T,
                                  cudaStream_t stream);

// ---------------------------------------------------------------------------
// Scale: y = x * scalar
// ---------------------------------------------------------------------------
void scale_f32(const float* x, float* y, int N, float scalar,
               cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused sigmoid + sum reduce over last dim: y[n] = sum_d sigmoid(x[n,d])
//   x: [N, D], y: [N]
// ---------------------------------------------------------------------------
void sigmoid_sum_f32(const float* x, float* y, int N, int D,
                      cudaStream_t stream);

// ---------------------------------------------------------------------------
// Tile 1D: broadcast x[C] to y[T, C]  (y[t,c] = x[c] for all t)
// ---------------------------------------------------------------------------
void tile_1d_f32(const float* x, float* y, int C, int T,
                  cudaStream_t stream);

// ---------------------------------------------------------------------------
// Channel bias add: y[t, c] += bias[c]  for all t
//   y: [T, C], bias: [C]
// ---------------------------------------------------------------------------
void channel_bias_add_f32(float* y, const float* bias, int C, int T,
                            cudaStream_t stream);

// ---------------------------------------------------------------------------
// Generalized Conv1d with stride, dilation, explicit padding
//   x: [T_in, C_in], w: [C_out, C_in, K], b: [C_out] or nullptr
//   y: [T_out, C_out]
//   T_out = (T_in + 2*padding - dilation*(K-1) - 1) / stride + 1
// ---------------------------------------------------------------------------
void conv1d_general_f32(const float* x, const float* w, const float* bias,
                        float* y, int C_in, int C_out, int T_in, int K,
                        int stride, int padding, int dilation,
                        cudaStream_t stream);

// ---------------------------------------------------------------------------
// ConvTranspose1d (non-depthwise, groups=1)
//   x: [T_in, C_in], w: [C_in, C_out, K], b: [C_out] or nullptr
//   y: [T_out, C_out]
//   T_out = (T_in - 1) * stride - 2*padding + K + output_padding
// ---------------------------------------------------------------------------
void conv_transpose1d_f32(const float* x, const float* w, const float* bias,
                          float* y, int C_in, int C_out, int T_in, int K,
                          int stride, int padding, int output_padding,
                          cudaStream_t stream);

// ---------------------------------------------------------------------------
// Snake activation: y = x + (1/alpha) * sin(alpha * x)^2
//   x, y: [T, C], alpha: [C] (one per channel)
// ---------------------------------------------------------------------------
void snake_f32(const float* x, const float* alpha, float* y,
               int C, int T, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Nearest-neighbor upsampling with arbitrary integer factor
//   x: [T_in, C], y: [T_in * factor, C]
// ---------------------------------------------------------------------------
void upsample_nearest_f32(const float* x, float* y, int C, int T_in,
                           int factor, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Reflection pad 1D: pad left by pad_left, right by pad_right
//   x: [T, C], y: [T + pad_left + pad_right, C]
// ---------------------------------------------------------------------------
void reflection_pad_1d_f32(const float* x, float* y, int C, int T,
                            int pad_left, int pad_right,
                            cudaStream_t stream);

// ---------------------------------------------------------------------------
// Element-wise exp: y = exp(x)
// ---------------------------------------------------------------------------
void exp_f32(const float* x, float* y, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Element-wise sin: y = sin(x)
// ---------------------------------------------------------------------------
void sin_f32(const float* x, float* y, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Element-wise tanh: y = tanh(x)
// ---------------------------------------------------------------------------
void tanh_f32(const float* x, float* y, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// STFT forward: compute magnitude and phase spectrograms
//   x: [T_signal], hann_window precomputed: [n_fft]
//   mag: [n_fft/2+1, n_frames], phase: [n_fft/2+1, n_frames]
//   Uses center=True (reflect-pad by n_fft/2 on each side)
// ---------------------------------------------------------------------------
void stft_f32(const float* x, float* mag, float* phase,
              int T_signal, int n_fft, int hop_length,
              cudaStream_t stream);

// ---------------------------------------------------------------------------
// iSTFT inverse: reconstruct time-domain signal from magnitude and phase
//   mag, phase: [n_fft/2+1, n_frames]
//   y: [T_signal] where T_signal = n_frames * hop_length (after center crop)
//   Uses overlap-add with Hann window
// ---------------------------------------------------------------------------
void istft_f32(const float* mag, const float* phase, float* y,
               int n_frames, int n_fft, int hop_length, int T_signal,
               cudaStream_t stream);

// ---------------------------------------------------------------------------
// LSTM gate activation: apply sigmoid/tanh to pre-computed gates
//   gates: [4*H] (i,f,g,o), c_prev: [H] → c_out: [H], h_out: [H]
// ---------------------------------------------------------------------------
void lstm_gates_f32(const float* gates, const float* c_prev,
                     float* c_out, float* h_out, int H,
                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused LSTM: run ALL timesteps in a single kernel launch
//   Whh: [4H, H], ig_all: [T, 4H] (pre-computed input gates)
//   h_all: [T, H] output. reverse=1 for backward direction.
// ---------------------------------------------------------------------------
void fused_lstm_f32(const float* Whh, const float* ig_all,
                     float* h_all, int T, int H, int reverse,
                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// im2col for 1D convolution: x[T_in, C_in] → col[T_out, C_in*K]
//   T_out = (T_in + 2*padding - dilation*(K-1) - 1) / stride + 1
// ---------------------------------------------------------------------------
void im2col_1d_f32(const float* x, float* col,
                    int C_in, int T_in, int K,
                    int stride, int padding, int dilation, int T_out,
                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// col2im for 1D transposed convolution: col[T_in, C_out*K] → y[T_out, C_out]
//   T_out = (T_in - 1) * stride + K - 2*padding
//   Uses atomicAdd for overlapping positions.
// ---------------------------------------------------------------------------
void col2im_1d_f32(const float* col, float* y,
                    int C_out, int K, int T_in,
                    int stride, int padding, int T_out,
                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Concatenate along channels for [T, C] layout
//   a: [T, Ca], b: [T, Cb] → y: [T, Ca+Cb]
// ---------------------------------------------------------------------------
void concat_channels_f32(const float* a, const float* b, float* y,
                           int T, int Ca, int Cb, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Concatenate 3 tensors along channels: a[T,Ca] + b[T,Cb] + c[T,Cc] → y[T,Ca+Cb+Cc]
// ---------------------------------------------------------------------------
void concat3_channels_f32(const float* a, const float* b, const float* c,
                            float* y, int T, int Ca, int Cb, int Cc,
                            cudaStream_t stream);

// ---------------------------------------------------------------------------
// Concatenate 4 tensors along channels: a[T,Ca]+b[T,Cb]+c[T,Cc]+d[T,Cd] → y[T,Ca+Cb+Cc+Cd]
// ---------------------------------------------------------------------------
void concat4_channels_f32(const float* a, const float* b,
                            const float* c, const float* d,
                            float* y,
                            int T, int Ca, int Cb, int Cc, int Cd,
                            cudaStream_t stream);

// ---------------------------------------------------------------------------
// SineGen phase computation: f0[L2] → phase_low[L2 * 9]
//   Sequential cumulative sum per harmonic (9 threads)
// ---------------------------------------------------------------------------
void sinegen_phase_f32(const float* f0, const float* rand_ini,
                        float* phase_low, int L2, cudaStream_t stream);

// ---------------------------------------------------------------------------
// SineGen source generation: phase_low[L2*9] + f0[L2] → har_source[T_audio]
//   Parallel: upsample phase, sin, UV mask, noise, linear combine, tanh
//   T_audio = L2 * 300
// ---------------------------------------------------------------------------
void sinegen_source_f32(const float* phase_low, const float* f0,
                         const float* l_linear_w, const float* l_linear_b,
                         float* har_source, int L2, int T_audio,
                         unsigned int seed, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Round+clamp durations on GPU, compute total L
//   durations: [T] (float, from sigmoid_sum), int_durations: [T] (output int)
//   L_out: [1] (output, total frames = sum of int_durations)
// ---------------------------------------------------------------------------
void round_clamp_durations_f32(const float* durations, int* int_durations,
                                int* L_out, int T, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Build alignment matrix on GPU from int durations
//   int_durations: [T], alignment: [T, L] (output, zeroed + filled)
// ---------------------------------------------------------------------------
void build_alignment_f32(const int* int_durations, float* alignment,
                          int T, int L, cudaStream_t stream);
