// weights.h — Weight loading + CUDA inference for Rokoko-82M TTS
//
// Defines:
//   Weights  — pointers into GPU weight allocation (loaded from weights.bin)

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Utility macros
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// GPU scratch arena: single allocation, bump pointer, reset per inference
// ---------------------------------------------------------------------------

struct GpuArena {
    char* base = nullptr;
    size_t capacity = 0;
    size_t offset = 0;

    void init(size_t bytes) {
        CUDA_CHECK(cudaMalloc(&base, bytes));
        capacity = bytes;
        offset = 0;
    }

    void destroy() {
        if (base) { cudaFree(base); base = nullptr; }
        capacity = 0;
        offset = 0;
    }

    void reset() { offset = 0; }

    // Allocate n bytes, 256-byte aligned
    void* alloc(size_t bytes) {
        size_t aligned = (offset + 255) & ~(size_t)255;
        if (aligned + bytes > capacity) {
            fprintf(stderr, "GpuArena OOM: need %zu + %zu, have %zu\n",
                    aligned, bytes, capacity);
            abort();
        }
        void* ptr = base + aligned;
        offset = aligned + bytes;
        return ptr;
    }

    // Typed helper
    template<typename T>
    T* alloc(size_t count) {
        return static_cast<T*>(alloc(count * sizeof(T)));
    }

    // Save/restore for nested scopes
    size_t save() const { return offset; }
    void restore(size_t saved) { offset = saved; }
};

// ---------------------------------------------------------------------------
// Weight file format constants
// ---------------------------------------------------------------------------

static constexpr uint32_t KOKO_MAGIC   = 0x4F4B4F4B;  // "KOKO" little-endian
static constexpr uint32_t KOKO_VERSION = 1;
static constexpr size_t   HEADER_ALIGN = 4096;

// ---------------------------------------------------------------------------
// Tensor descriptor (parsed from header)
// ---------------------------------------------------------------------------

struct TensorDesc {
    std::string name;
    size_t offset;       // offset from data start
    size_t size_bytes;
    std::string dtype;
    std::vector<int> shape;
};

// ---------------------------------------------------------------------------
// Model dimensions
// ---------------------------------------------------------------------------

// ALBERT encoder
static constexpr int BERT_VOCAB      = 178;
static constexpr int BERT_EMBED_DIM  = 128;
static constexpr int BERT_HIDDEN     = 768;
static constexpr int BERT_MAX_POS    = 512;
static constexpr int BERT_FF_DIM     = 2048;  // intermediate_size
static constexpr int BERT_N_HEADS    = 12;
static constexpr int BERT_HEAD_DIM   = BERT_HIDDEN / BERT_N_HEADS;  // 64
static constexpr int BERT_N_LAYERS   = 6;     // looped through 1 shared layer

// bert_encoder: Linear 768 → 512
static constexpr int STYLE_DIM       = 128;   // style embedding dim
static constexpr int ENCODER_DIM     = 512;   // text encoder hidden dim

// Text encoder
static constexpr int TEXT_N_CONV     = 3;      // 3 conv blocks
static constexpr int TEXT_CONV_K     = 5;      // conv kernel size
static constexpr int TEXT_LSTM_H     = 256;    // LSTM hidden (bidir → 512)

// Prosody predictor
static constexpr int PRED_N_LAYERS   = 3;      // DurationEncoder LSTM+AdaLN pairs
static constexpr int PRED_LSTM_IN    = 640;    // LSTM input: 512 + 128 (style)
static constexpr int PRED_LSTM_H     = 256;    // LSTM hidden (bidir → 512)
static constexpr int PRED_MAX_DUR    = 50;     // duration projection output dim
static constexpr int PRED_F0N_BLOCKS = 3;      // F0/N AdainResBlk1d blocks

// Decoder constants
static constexpr int SAMPLE_RATE     = 24000;

// ---------------------------------------------------------------------------
// Weights struct — all model weights on GPU
// ---------------------------------------------------------------------------

struct Weights {
    void* gpu_data = nullptr;        // single contiguous GPU allocation
    size_t gpu_data_size = 0;

    // Prefetch state (temporary — cleared after upload)
    const void* prefetch_base = nullptr;  // points to KOKO data start
    void* mmap_ptr = nullptr;             // non-null only if we own the mmap
    size_t mmap_size = 0;
    size_t data_offset = 0;

    // Parsed index
    std::vector<TensorDesc> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;

    // -----------------------------------------------------------------------
    // ALBERT (bert) — shared parameters, looped 6 times
    //   Embeddings: word [178,128], position [512,128], token_type [2,128]
    //   Projection: embedding_hidden_mapping_in [768,128]
    //   1 shared layer with attention + ffn
    // -----------------------------------------------------------------------

    // Embeddings
    float *bert_word_embed = nullptr;       // [178, 128]
    float *bert_pos_embed = nullptr;        // [512, 128]
    float *bert_type_embed = nullptr;       // [2, 128]
    float *bert_embed_ln_w = nullptr;       // [128]
    float *bert_embed_ln_b = nullptr;       // [128]

    // Embedding → hidden projection
    float *bert_proj_w = nullptr;           // [768, 128]
    float *bert_proj_b = nullptr;           // [768]
    __half *bert_proj_w_f16 = nullptr;

    // Shared attention layer (looped 6 times)
    struct AlbertLayer {
        float *q_w = nullptr, *q_b = nullptr;     // [768, 768] + [768]
        float *k_w = nullptr, *k_b = nullptr;     // [768, 768] + [768]
        float *v_w = nullptr, *v_b = nullptr;     // [768, 768] + [768]
        float *dense_w = nullptr, *dense_b = nullptr;  // [768, 768] + [768]
        float *attn_ln_w = nullptr, *attn_ln_b = nullptr;  // [768]

        // FFN
        float *ffn_w = nullptr, *ffn_b = nullptr;       // [2048, 768] + [2048]
        float *ffn_out_w = nullptr, *ffn_out_b = nullptr; // [768, 2048] + [768]
        float *ffn_ln_w = nullptr, *ffn_ln_b = nullptr;   // [768]

        // FP16 companions (v2)
        __half *q_w_f16 = nullptr, *k_w_f16 = nullptr, *v_w_f16 = nullptr;
        __half *dense_w_f16 = nullptr;
        __half *ffn_w_f16 = nullptr, *ffn_out_w_f16 = nullptr;
    } albert;

    // Pooler (may not be needed for inference, but load anyway)
    float *bert_pooler_w = nullptr;         // [768, 768]
    float *bert_pooler_b = nullptr;         // [768]

    // -----------------------------------------------------------------------
    // bert_encoder: Linear(768, 512)
    // -----------------------------------------------------------------------
    float *bert_enc_w = nullptr;            // [512, 768]
    float *bert_enc_b = nullptr;            // [512]
    __half *bert_enc_w_f16 = nullptr;

    // -----------------------------------------------------------------------
    // Text encoder: 3 conv blocks + bidirectional LSTM
    //   Each conv block: Conv1d(512, 512, k=5, pad=2) + LayerNorm + GELU + Dropout
    // -----------------------------------------------------------------------
    struct TextConvBlock {
        float *conv_wg = nullptr;           // [512, 1, 1] weight_g (scale)
        float *conv_wv = nullptr;           // [512, 512, 5] weight_v (direction)
        float *conv_b = nullptr;            // [512]
        float *ln_w = nullptr;              // [512] (gamma)
        float *ln_b = nullptr;              // [512] (beta)
        __half *conv_wv_nhwc_f16 = nullptr; // v2: NHWC FP16 for Cutlass conv
    } text_conv[TEXT_N_CONV];

    // Bidirectional LSTM (1 layer)
    float *text_lstm_wih_fwd = nullptr;     // [1024, 512]
    float *text_lstm_whh_fwd = nullptr;     // [1024, 256]
    float *text_lstm_bih_fwd = nullptr;     // [1024]
    float *text_lstm_bhh_fwd = nullptr;     // [1024]
    float *text_lstm_wih_rev = nullptr;     // [1024, 512]
    float *text_lstm_whh_rev = nullptr;     // [1024, 256]
    float *text_lstm_bih_rev = nullptr;     // [1024]
    float *text_lstm_bhh_rev = nullptr;     // [1024]
    float *text_lstm_bias_fwd = nullptr;    // precomputed bih+bhh [1024]
    float *text_lstm_bias_rev = nullptr;    // precomputed bih+bhh [1024]
    __half *text_lstm_wih_fwd_f16 = nullptr;
    __half *text_lstm_whh_fwd_f16 = nullptr;
    __half *text_lstm_wih_rev_f16 = nullptr;
    __half *text_lstm_whh_rev_f16 = nullptr;

    // -----------------------------------------------------------------------
    // Prosody predictor
    // -----------------------------------------------------------------------

    // BiLSTM weight set (reused for all 5 BiLSTMs in predictor)
    struct BiLSTMWeights {
        float *wih_fwd = nullptr, *whh_fwd = nullptr;  // [4H, input], [4H, H]
        float *bih_fwd = nullptr, *bhh_fwd = nullptr;  // [4H]
        float *wih_rev = nullptr, *whh_rev = nullptr;
        float *bih_rev = nullptr, *bhh_rev = nullptr;
        float *bias_fwd = nullptr, *bias_rev = nullptr; // precomputed bih+bhh [4H]
        __half *wih_fwd_f16 = nullptr, *whh_fwd_f16 = nullptr;
        __half *wih_rev_f16 = nullptr, *whh_rev_f16 = nullptr;
    };

    // AdaLayerNorm: fc(style) -> gamma, beta; then (1+gamma)*LN(x)+beta
    struct AdaLayerNormWeights {
        float *fc_w = nullptr;   // [2*D, style_dim=128]
        float *fc_b = nullptr;   // [2*D]
        __half *fc_w_f16 = nullptr;
    };

    // AdaIN1d: InstanceNorm(affine=True) + style FC
    struct AdaIN1dWeights {
        float *norm_w = nullptr;  // [C] InstanceNorm weight
        float *norm_b = nullptr;  // [C] InstanceNorm bias
        float *fc_w = nullptr;    // [2*C, style_dim=128]
        float *fc_b = nullptr;    // [2*C]
        __half *fc_w_f16 = nullptr;
    };

    // AdainResBlk1d: 2 conv + 2 AdaIN + optional shortcut + optional upsample pool
    struct AdainResBlk1dWeights {
        float *conv1_wg = nullptr, *conv1_wv = nullptr, *conv1_b = nullptr;
        float *conv2_wg = nullptr, *conv2_wv = nullptr, *conv2_b = nullptr;
        AdaIN1dWeights norm1, norm2;
        // Optional shortcut (when dim_in != dim_out)
        float *conv1x1_wg = nullptr, *conv1x1_wv = nullptr;  // no bias
        // Optional upsample pool (ConvTranspose1d depthwise)
        float *pool_wg = nullptr, *pool_wv = nullptr, *pool_b = nullptr;
        bool has_shortcut = false;
        bool has_upsample = false;
        // FP16 companions (v2)
        __half *conv1_wv_nhwc_f16 = nullptr;  // NHWC FP16 for K=3 conv
        __half *conv2_wv_nhwc_f16 = nullptr;
        __half *conv1x1_wv_nhwc_f16 = nullptr; // K=1 NHWC FP16 (identity layout)
        int conv1_c_in_pad = 0;               // padded C_in for conv1 (0 = unpadded)
        int conv1x1_c_in_pad = 0;             // padded C_in for conv1x1 (0 = unpadded)
    };

    // DurationEncoder: 3x (BiLSTM + AdaLayerNorm)
    BiLSTMWeights dur_enc_lstm[PRED_N_LAYERS];
    AdaLayerNormWeights dur_enc_aln[PRED_N_LAYERS];

    // Duration LSTM + projection
    BiLSTMWeights dur_lstm;
    float *dur_proj_w = nullptr;   // [50, 512]
    float *dur_proj_b = nullptr;   // [50]
    __half *dur_proj_w_f16 = nullptr;

    // Shared LSTM (F0/N prediction)
    BiLSTMWeights shared_lstm;

    // F0 chain: 3 AdainResBlk1d + Conv1d projection
    AdainResBlk1dWeights f0_blocks[PRED_F0N_BLOCKS];
    float *f0_proj_w = nullptr;    // [1, 256, 1] Conv1d
    float *f0_proj_b = nullptr;    // [1]

    // N (noise) chain: 3 AdainResBlk1d + Conv1d projection
    AdainResBlk1dWeights n_blocks[PRED_F0N_BLOCKS];
    float *n_proj_w = nullptr;     // [1, 256, 1] Conv1d
    float *n_proj_b = nullptr;     // [1]

    // -----------------------------------------------------------------------
    // Decoder (ISTFTNet) — 491 tensors
    // -----------------------------------------------------------------------

    // AdaINResBlock1: Generator resblock with Snake activation
    // 3 rounds of: adain1 → Snake → dilated conv → adain2 → Snake → conv → residual add
    struct AdaINResBlock1Weights {
        // convs1[0..2]: dilated conv (d=1,3,5), weight_norm
        struct { float *wg, *wv, *b; __half *wv_nhwc_f16 = nullptr; } convs1[3];
        // convs2[0..2]: conv (d=1), weight_norm
        struct { float *wg, *wv, *b; __half *wv_nhwc_f16 = nullptr; } convs2[3];
        // adain1[0..2], adain2[0..2]: AdaIN1d
        AdaIN1dWeights adain1[3], adain2[3];
        // alpha1[0..2], alpha2[0..2]: Snake learnable params [C]
        float *alpha1[3], *alpha2[3];
        int channels;
        int kernel_size;
    };

    // F0_conv, N_conv: Conv1d(1,1,k=3,s=2,p=1) with weight_norm
    float *dec_f0_conv_wg = nullptr, *dec_f0_conv_wv = nullptr, *dec_f0_conv_b = nullptr;
    float *dec_n_conv_wg = nullptr, *dec_n_conv_wv = nullptr, *dec_n_conv_b = nullptr;

    // asr_res: Conv1d(512,64,k=1) with weight_norm
    float *dec_asr_res_wg = nullptr, *dec_asr_res_wv = nullptr, *dec_asr_res_b = nullptr;
    __half *dec_asr_res_wv_f16 = nullptr;

    // encode: AdainResBlk1d(514→1024)
    AdainResBlk1dWeights dec_encode;

    // decode[0..3]: AdainResBlk1d blocks
    // [0-2]: 1090→1024, no upsample; [3]: 1090→512, upsample=True
    static constexpr int DEC_N_DECODE = 4;
    AdainResBlk1dWeights dec_decode[DEC_N_DECODE];

    // Generator: ups, noise_convs, noise_res, resblocks, conv_post, m_source
    static constexpr int GEN_N_UPS = 2;
    static constexpr int GEN_N_KERNELS = 3;  // resblock kernel sizes: 3, 7, 11
    static constexpr int GEN_N_RESBLOCKS = GEN_N_UPS * GEN_N_KERNELS;  // 6
    static constexpr int GEN_N_FFT = 20;
    static constexpr int GEN_HOP = 5;
    static constexpr int GEN_FREQ_BINS = GEN_N_FFT / 2 + 1;  // 11
    static constexpr int GEN_HAR_CH = GEN_N_FFT + 2;  // 22

    // ups[0..1]: ConvTranspose1d (non-depthwise) with weight_norm
    struct { float *wg, *wv, *b; __half *wv_f16 = nullptr; } gen_ups[GEN_N_UPS];

    // noise_convs[0..1]: regular Conv1d (no weight_norm)
    struct { float *w, *b; __half *w_f16 = nullptr; __half *w_nhwc_f16 = nullptr; int c_in_pad = 0; } gen_noise_convs[GEN_N_UPS];

    // noise_res[0..1]: AdaINResBlock1
    AdaINResBlock1Weights gen_noise_res[GEN_N_UPS];

    // resblocks[0..5]: AdaINResBlock1
    AdaINResBlock1Weights gen_resblocks[GEN_N_RESBLOCKS];

    // conv_post: Conv1d(128→22,k=7,p=3) with weight_norm
    float *gen_conv_post_wg = nullptr, *gen_conv_post_wv = nullptr, *gen_conv_post_b = nullptr;
    __half *gen_conv_post_wv_f16 = nullptr;  // for im2col+GEMM (C_out=22 misaligned)

    // m_source.l_linear: Linear(9→1)
    float *gen_source_w = nullptr, *gen_source_b = nullptr;

    // -----------------------------------------------------------------------
    // Methods
    // -----------------------------------------------------------------------

    static Weights prefetch(const std::string& path);
    static Weights prefetch(const void* data, size_t size);
    void upload(cudaStream_t stream = nullptr);
    static Weights load(const std::string& path, cudaStream_t stream = nullptr);
    void free();

    /// Get a GPU pointer for a named tensor (nullptr if not found).
    float* get(const std::string& name) const;

    /// Get shape for a named tensor.
    const std::vector<int>* get_shape(const std::string& name) const;

    /// Populate __half* fields from v2 tensor names (.f16, .nhwc_f16, etc.)
    void assign_v2_fp16_pointers();

    void print_info() const;
};

// ---------------------------------------------------------------------------
// Inference API (defined in rokoko.cpp / rokoko_f16.cpp)
// ---------------------------------------------------------------------------

/// Compute exact decode-arena bytes needed for given T (tokens) and L (frames).
size_t compute_decode_bytes(int T, int L);

/// Run Kokoro inference: phoneme token IDs + style vector → audio samples.
std::vector<float> rokoko_infer(const Weights& w,
    const int* token_ids, int T, const float* style_vec,
    cudaStream_t stream, GpuArena& arena, GpuArena& decode_arena,
    float* d_workspace, size_t workspace_bytes);

/// Write audio samples to a WAV file (16-bit PCM). path="-" writes to stdout.
bool write_wav(const std::string& path, const float* audio,
    int n_samples, int sample_rate);

/// Write WAV data to an output stream (16-bit PCM).
void write_wav_to_(std::ostream& f, const float* audio,
    int n_samples, int sample_rate);

/// Precompute weight norms (called once after weight upload).
void precompute_weight_norms(Weights& w, cudaStream_t stream);
