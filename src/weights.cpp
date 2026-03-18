// weights.cpp — Weight loading for Rokoko-82M CUDA backend
//
// Loads a flat binary weight file (produced by scripts/export_weights.py)
// into a single contiguous GPU allocation, then assigns struct field pointers
// by matching tensor names from the file header.

#include "weights.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helper: align up
// ---------------------------------------------------------------------------

static size_t align_up(size_t x, size_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Parse header
// ---------------------------------------------------------------------------

static std::vector<TensorDesc> parse_header(const char* header_text, size_t header_len) {
    std::vector<TensorDesc> tensors;
    std::string text(header_text, header_len);
    std::istringstream iss(text);
    std::string line;

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        TensorDesc td;
        ls >> td.name >> td.offset >> td.size_bytes >> td.dtype;
        int d;
        while (ls >> d) td.shape.push_back(d);
        tensors.push_back(std::move(td));
    }
    return tensors;
}

// ---------------------------------------------------------------------------
// Weights: pointer assignment
// ---------------------------------------------------------------------------

static void assign_weight_pointers(Weights& w) {
    uint8_t* gpu_base = (uint8_t*)w.gpu_data;

    auto lookup = [&](const std::string& name) -> float* {
        auto it = w.name_to_idx.find(name);
        if (it == w.name_to_idx.end()) return nullptr;
        return (float*)(gpu_base + w.tensors[it->second].offset);
    };

    // -----------------------------------------------------------------------
    // ALBERT (bert)
    // -----------------------------------------------------------------------
    w.bert_word_embed = lookup("bert.embeddings.word_embeddings.weight");
    w.bert_pos_embed  = lookup("bert.embeddings.position_embeddings.weight");
    w.bert_type_embed = lookup("bert.embeddings.token_type_embeddings.weight");
    w.bert_embed_ln_w = lookup("bert.embeddings.LayerNorm.weight");
    w.bert_embed_ln_b = lookup("bert.embeddings.LayerNorm.bias");

    w.bert_proj_w = lookup("bert.encoder.embedding_hidden_mapping_in.weight");
    w.bert_proj_b = lookup("bert.encoder.embedding_hidden_mapping_in.bias");

    // Shared ALBERT layer
    auto& a = w.albert;
    const char* pfx = "bert.encoder.albert_layer_groups.0.albert_layers.0";
    auto apfx = [&](const char* suffix) -> std::string {
        return std::string(pfx) + suffix;
    };

    a.q_w      = lookup(apfx(".attention.query.weight"));
    a.q_b      = lookup(apfx(".attention.query.bias"));
    a.k_w      = lookup(apfx(".attention.key.weight"));
    a.k_b      = lookup(apfx(".attention.key.bias"));
    a.v_w      = lookup(apfx(".attention.value.weight"));
    a.v_b      = lookup(apfx(".attention.value.bias"));
    a.dense_w  = lookup(apfx(".attention.dense.weight"));
    a.dense_b  = lookup(apfx(".attention.dense.bias"));
    a.attn_ln_w = lookup(apfx(".attention.LayerNorm.weight"));
    a.attn_ln_b = lookup(apfx(".attention.LayerNorm.bias"));

    a.ffn_w     = lookup(apfx(".ffn.weight"));
    a.ffn_b     = lookup(apfx(".ffn.bias"));
    a.ffn_out_w = lookup(apfx(".ffn_output.weight"));
    a.ffn_out_b = lookup(apfx(".ffn_output.bias"));
    a.ffn_ln_w  = lookup(apfx(".full_layer_layer_norm.weight"));
    a.ffn_ln_b  = lookup(apfx(".full_layer_layer_norm.bias"));

    w.bert_pooler_w = lookup("bert.pooler.weight");
    w.bert_pooler_b = lookup("bert.pooler.bias");

    // -----------------------------------------------------------------------
    // bert_encoder: Linear(768, 512)
    // -----------------------------------------------------------------------
    w.bert_enc_w = lookup("bert_encoder.weight");
    w.bert_enc_b = lookup("bert_encoder.bias");

    // -----------------------------------------------------------------------
    // Text encoder: 3 conv blocks + bidirectional LSTM
    // -----------------------------------------------------------------------
    for (int i = 0; i < TEXT_N_CONV; i++) {
        auto& blk = w.text_conv[i];
        std::string pre = "text_encoder.cnn." + std::to_string(i) + ".";
        // Weight-normed Conv1d: weight_g [out, 1, 1], weight_v [out, in, k], bias [out]
        blk.conv_wg = lookup(pre + "0.weight_g");
        blk.conv_wv = lookup(pre + "0.weight_v");
        blk.conv_b  = lookup(pre + "0.bias");
        // LayerNorm uses gamma/beta naming
        blk.ln_w   = lookup(pre + "1.gamma");
        blk.ln_b   = lookup(pre + "1.beta");
    }

    w.text_lstm_wih_fwd = lookup("text_encoder.lstm.weight_ih_l0");
    w.text_lstm_whh_fwd = lookup("text_encoder.lstm.weight_hh_l0");
    w.text_lstm_bih_fwd = lookup("text_encoder.lstm.bias_ih_l0");
    w.text_lstm_bhh_fwd = lookup("text_encoder.lstm.bias_hh_l0");
    w.text_lstm_wih_rev = lookup("text_encoder.lstm.weight_ih_l0_reverse");
    w.text_lstm_whh_rev = lookup("text_encoder.lstm.weight_hh_l0_reverse");
    w.text_lstm_bih_rev = lookup("text_encoder.lstm.bias_ih_l0_reverse");
    w.text_lstm_bhh_rev = lookup("text_encoder.lstm.bias_hh_l0_reverse");

    // -----------------------------------------------------------------------
    // Prosody predictor
    // -----------------------------------------------------------------------

    // Helper to assign BiLSTM weights
    auto assign_bilstm = [&](Weights::BiLSTMWeights& lstm, const std::string& prefix) {
        lstm.wih_fwd = lookup(prefix + ".weight_ih_l0");
        lstm.whh_fwd = lookup(prefix + ".weight_hh_l0");
        lstm.bih_fwd = lookup(prefix + ".bias_ih_l0");
        lstm.bhh_fwd = lookup(prefix + ".bias_hh_l0");
        lstm.wih_rev = lookup(prefix + ".weight_ih_l0_reverse");
        lstm.whh_rev = lookup(prefix + ".weight_hh_l0_reverse");
        lstm.bih_rev = lookup(prefix + ".bias_ih_l0_reverse");
        lstm.bhh_rev = lookup(prefix + ".bias_hh_l0_reverse");
    };

    // Helper to assign AdaIN1d weights
    auto assign_adain1d = [&](Weights::AdaIN1dWeights& ain, const std::string& prefix) {
        ain.norm_w = lookup(prefix + ".norm.weight");
        ain.norm_b = lookup(prefix + ".norm.bias");
        ain.fc_w   = lookup(prefix + ".fc.weight");
        ain.fc_b   = lookup(prefix + ".fc.bias");
    };

    // Helper to assign AdainResBlk1d weights
    auto assign_adain_resblk = [&](Weights::AdainResBlk1dWeights& blk,
                                    const std::string& prefix,
                                    bool has_shortcut, bool has_upsample) {
        blk.conv1_wg = lookup(prefix + ".conv1.weight_g");
        blk.conv1_wv = lookup(prefix + ".conv1.weight_v");
        blk.conv1_b  = lookup(prefix + ".conv1.bias");
        blk.conv2_wg = lookup(prefix + ".conv2.weight_g");
        blk.conv2_wv = lookup(prefix + ".conv2.weight_v");
        blk.conv2_b  = lookup(prefix + ".conv2.bias");
        assign_adain1d(blk.norm1, prefix + ".norm1");
        assign_adain1d(blk.norm2, prefix + ".norm2");
        blk.has_shortcut = has_shortcut;
        blk.has_upsample = has_upsample;
        if (has_shortcut) {
            blk.conv1x1_wg = lookup(prefix + ".conv1x1.weight_g");
            blk.conv1x1_wv = lookup(prefix + ".conv1x1.weight_v");
        }
        if (has_upsample) {
            blk.pool_wg = lookup(prefix + ".pool.weight_g");
            blk.pool_wv = lookup(prefix + ".pool.weight_v");
            blk.pool_b  = lookup(prefix + ".pool.bias");
        }
    };

    // DurationEncoder: lstms[0,2,4] = BiLSTM, lstms[1,3,5] = AdaLayerNorm
    for (int i = 0; i < PRED_N_LAYERS; i++) {
        assign_bilstm(w.dur_enc_lstm[i],
            "predictor.text_encoder.lstms." + std::to_string(i * 2));
        auto& aln = w.dur_enc_aln[i];
        std::string aln_pfx = "predictor.text_encoder.lstms." + std::to_string(i * 2 + 1);
        aln.fc_w = lookup(aln_pfx + ".fc.weight");
        aln.fc_b = lookup(aln_pfx + ".fc.bias");
    }

    // Duration LSTM + projection
    assign_bilstm(w.dur_lstm, "predictor.lstm");
    w.dur_proj_w = lookup("predictor.duration_proj.linear_layer.weight");
    w.dur_proj_b = lookup("predictor.duration_proj.linear_layer.bias");

    // Shared LSTM
    assign_bilstm(w.shared_lstm, "predictor.shared");

    // F0 chain: 3 AdainResBlk1d + projection
    // F0[0]: 512->512, no upsample, no shortcut
    assign_adain_resblk(w.f0_blocks[0], "predictor.F0.0", false, false);
    // F0[1]: 512->256, upsample=True, has shortcut (dim_in != dim_out)
    assign_adain_resblk(w.f0_blocks[1], "predictor.F0.1", true, true);
    // F0[2]: 256->256, no upsample, no shortcut
    assign_adain_resblk(w.f0_blocks[2], "predictor.F0.2", false, false);
    w.f0_proj_w = lookup("predictor.F0_proj.weight");
    w.f0_proj_b = lookup("predictor.F0_proj.bias");

    // N chain: identical structure to F0
    assign_adain_resblk(w.n_blocks[0], "predictor.N.0", false, false);
    assign_adain_resblk(w.n_blocks[1], "predictor.N.1", true, true);
    assign_adain_resblk(w.n_blocks[2], "predictor.N.2", false, false);
    w.n_proj_w = lookup("predictor.N_proj.weight");
    w.n_proj_b = lookup("predictor.N_proj.bias");

    // -----------------------------------------------------------------------
    // Decoder
    // -----------------------------------------------------------------------

    // F0_conv, N_conv
    w.dec_f0_conv_wg = lookup("decoder.F0_conv.weight_g");
    w.dec_f0_conv_wv = lookup("decoder.F0_conv.weight_v");
    w.dec_f0_conv_b  = lookup("decoder.F0_conv.bias");
    w.dec_n_conv_wg  = lookup("decoder.N_conv.weight_g");
    w.dec_n_conv_wv  = lookup("decoder.N_conv.weight_v");
    w.dec_n_conv_b   = lookup("decoder.N_conv.bias");

    // asr_res
    w.dec_asr_res_wg = lookup("decoder.asr_res.0.weight_g");
    w.dec_asr_res_wv = lookup("decoder.asr_res.0.weight_v");
    w.dec_asr_res_b  = lookup("decoder.asr_res.0.bias");

    // encode: AdainResBlk1d(514→1024, shortcut=True, upsample=False)
    assign_adain_resblk(w.dec_encode, "decoder.encode", true, false);

    // decode[0..3]
    for (int i = 0; i < 4; i++) {
        std::string pfx = "decoder.decode." + std::to_string(i);
        bool has_shortcut = true;  // all decode blocks have dim_in != dim_out (1090 != 1024 or 512)
        bool has_upsample = (i == 3);
        assign_adain_resblk(w.dec_decode[i], pfx, has_shortcut, has_upsample);
    }

    // Helper to assign AdaINResBlock1 weights (generator resblocks)
    auto assign_adain_resblock1 = [&](Weights::AdaINResBlock1Weights& rb,
                                       const std::string& prefix,
                                       int channels, int kernel_size) {
        rb.channels = channels;
        rb.kernel_size = kernel_size;
        for (int j = 0; j < 3; j++) {
            std::string js = std::to_string(j);
            rb.convs1[j].wg = lookup(prefix + ".convs1." + js + ".weight_g");
            rb.convs1[j].wv = lookup(prefix + ".convs1." + js + ".weight_v");
            rb.convs1[j].b  = lookup(prefix + ".convs1." + js + ".bias");
            rb.convs2[j].wg = lookup(prefix + ".convs2." + js + ".weight_g");
            rb.convs2[j].wv = lookup(prefix + ".convs2." + js + ".weight_v");
            rb.convs2[j].b  = lookup(prefix + ".convs2." + js + ".bias");
            assign_adain1d(rb.adain1[j], prefix + ".adain1." + js);
            assign_adain1d(rb.adain2[j], prefix + ".adain2." + js);
            rb.alpha1[j] = lookup(prefix + ".alpha1." + js);
            rb.alpha2[j] = lookup(prefix + ".alpha2." + js);
        }
    };

    // Generator: ups
    for (int i = 0; i < 2; i++) {
        std::string pfx = "decoder.generator.ups." + std::to_string(i);
        w.gen_ups[i].wg = lookup(pfx + ".weight_g");
        w.gen_ups[i].wv = lookup(pfx + ".weight_v");
        w.gen_ups[i].b  = lookup(pfx + ".bias");
    }

    // Generator: noise_convs (regular Conv1d, NOT weight_norm)
    for (int i = 0; i < 2; i++) {
        std::string pfx = "decoder.generator.noise_convs." + std::to_string(i);
        w.gen_noise_convs[i].w = lookup(pfx + ".weight");
        w.gen_noise_convs[i].b = lookup(pfx + ".bias");
    }

    // Generator: noise_res
    // noise_res[0]: ch=256, kernel=7; noise_res[1]: ch=128, kernel=11
    int noise_res_ch[] = {256, 128};
    int noise_res_k[] = {7, 11};
    for (int i = 0; i < 2; i++) {
        assign_adain_resblock1(w.gen_noise_res[i],
            "decoder.generator.noise_res." + std::to_string(i),
            noise_res_ch[i], noise_res_k[i]);
    }

    // Generator: resblocks
    // resblocks[0..2]: ch=256, kernels=[3,7,11]
    // resblocks[3..5]: ch=128, kernels=[3,7,11]
    int rb_kernels[] = {3, 7, 11};
    for (int i = 0; i < 6; i++) {
        int ch = (i < 3) ? 256 : 128;
        int k = rb_kernels[i % 3];
        assign_adain_resblock1(w.gen_resblocks[i],
            "decoder.generator.resblocks." + std::to_string(i),
            ch, k);
    }

    // Generator: conv_post
    w.gen_conv_post_wg = lookup("decoder.generator.conv_post.weight_g");
    w.gen_conv_post_wv = lookup("decoder.generator.conv_post.weight_v");
    w.gen_conv_post_b  = lookup("decoder.generator.conv_post.bias");

    // Generator: m_source.l_linear
    w.gen_source_w = lookup("decoder.generator.m_source.l_linear.weight");
    w.gen_source_b = lookup("decoder.generator.m_source.l_linear.bias");
}

// ---------------------------------------------------------------------------
// Weights: v2 FP16 pointer assignment
// ---------------------------------------------------------------------------

void Weights::assign_v2_fp16_pointers() {
    uint8_t* gpu_base = (uint8_t*)gpu_data;

    // Build suffix map: base tensor name → {f16, nhwc_f16, nhwc_f16_pad, c_in_pad, bias_combined}
    struct V2Info {
        __half* f16 = nullptr;
        __half* nhwc_f16 = nullptr;
        __half* nhwc_f16_pad = nullptr;
        int c_in_pad = 0;
        float* bias_combined_fwd = nullptr;
        float* bias_combined_rev = nullptr;
    };
    std::unordered_map<std::string, V2Info> v2;

    for (const auto& td : tensors) {
        void* ptr = gpu_base + td.offset;
        const std::string& n = td.name;

        auto ends_with = [&](const char* suf) -> bool {
            size_t slen = strlen(suf);
            return n.size() >= slen && n.compare(n.size() - slen, slen, suf) == 0;
        };

        if (ends_with(".f16")) {
            v2[n.substr(0, n.size() - 4)].f16 = (__half*)ptr;
        } else if (ends_with(".nhwc_f16")) {
            v2[n.substr(0, n.size() - 9)].nhwc_f16 = (__half*)ptr;
        } else if (auto pos = n.find(".nhwc_f16_pad"); pos != std::string::npos) {
            auto& info = v2[n.substr(0, pos)];
            info.nhwc_f16_pad = (__half*)ptr;
            info.c_in_pad = std::stoi(n.substr(pos + 13));
        } else if (ends_with(".bias_combined_fwd")) {
            v2[n.substr(0, n.size() - 18)].bias_combined_fwd = (float*)ptr;
        } else if (ends_with(".bias_combined_rev")) {
            v2[n.substr(0, n.size() - 18)].bias_combined_rev = (float*)ptr;
        }
    }

    // Lookup helpers
    auto f16 = [&](const std::string& base) -> __half* {
        auto it = v2.find(base); return it != v2.end() ? it->second.f16 : nullptr;
    };
    auto nhwc16 = [&](const std::string& base) -> __half* {
        auto it = v2.find(base);
        if (it == v2.end()) return nullptr;
        if (it->second.nhwc_f16) return it->second.nhwc_f16;
        return it->second.nhwc_f16_pad;  // fall back to padded
    };
    auto pad_cin = [&](const std::string& base) -> int {
        auto it = v2.find(base); return it != v2.end() ? it->second.c_in_pad : 0;
    };

    // --- ALBERT ---
    bert_proj_w_f16 = f16("bert.encoder.embedding_hidden_mapping_in.weight");

    const char* apfx = "bert.encoder.albert_layer_groups.0.albert_layers.0";
    auto aname = [&](const char* s) { return std::string(apfx) + s; };
    albert.q_w_f16      = f16(aname(".attention.query.weight"));
    albert.k_w_f16      = f16(aname(".attention.key.weight"));
    albert.v_w_f16      = f16(aname(".attention.value.weight"));
    albert.dense_w_f16  = f16(aname(".attention.dense.weight"));
    albert.ffn_w_f16    = f16(aname(".ffn.weight"));
    albert.ffn_out_w_f16 = f16(aname(".ffn_output.weight"));

    // --- bert_encoder ---
    bert_enc_w_f16 = f16("bert_encoder.weight");

    // --- Text encoder conv blocks ---
    for (int i = 0; i < TEXT_N_CONV; i++) {
        std::string wv = "text_encoder.cnn." + std::to_string(i) + ".0.weight_v";
        text_conv[i].conv_wv_nhwc_f16 = nhwc16(wv);
    }

    // --- Text encoder LSTM ---
    text_lstm_wih_fwd_f16 = f16("text_encoder.lstm.weight_ih_l0");
    text_lstm_whh_fwd_f16 = f16("text_encoder.lstm.weight_hh_l0");
    text_lstm_wih_rev_f16 = f16("text_encoder.lstm.weight_ih_l0_reverse");
    text_lstm_whh_rev_f16 = f16("text_encoder.lstm.weight_hh_l0_reverse");
    {
        auto it = v2.find("text_encoder.lstm");
        if (it != v2.end()) {
            if (it->second.bias_combined_fwd) text_lstm_bias_fwd = it->second.bias_combined_fwd;
            if (it->second.bias_combined_rev) text_lstm_bias_rev = it->second.bias_combined_rev;
        }
    }

    // --- Prosody predictor ---
    auto assign_bilstm_v2 = [&](BiLSTMWeights& lstm, const std::string& prefix) {
        lstm.wih_fwd_f16 = f16(prefix + ".weight_ih_l0");
        lstm.whh_fwd_f16 = f16(prefix + ".weight_hh_l0");
        lstm.wih_rev_f16 = f16(prefix + ".weight_ih_l0_reverse");
        lstm.whh_rev_f16 = f16(prefix + ".weight_hh_l0_reverse");
        auto it = v2.find(prefix);
        if (it != v2.end()) {
            if (it->second.bias_combined_fwd) lstm.bias_fwd = it->second.bias_combined_fwd;
            if (it->second.bias_combined_rev) lstm.bias_rev = it->second.bias_combined_rev;
        }
    };

    auto assign_adain1d_v2 = [&](AdaIN1dWeights& ain, const std::string& prefix) {
        ain.fc_w_f16 = f16(prefix + ".fc.weight");
    };

    auto assign_resblk_v2 = [&](AdainResBlk1dWeights& blk, const std::string& prefix) {
        std::string c1 = prefix + ".conv1.weight_v";
        std::string c2 = prefix + ".conv2.weight_v";
        blk.conv1_wv_nhwc_f16 = nhwc16(c1);
        blk.conv1_c_in_pad = pad_cin(c1);
        blk.conv2_wv_nhwc_f16 = nhwc16(c2);
        if (blk.has_shortcut) {
            blk.conv1x1_wv_nhwc_f16 = nhwc16(prefix + ".conv1x1.weight_v");
            blk.conv1x1_c_in_pad = pad_cin(prefix + ".conv1x1.weight_v");
        }
        assign_adain1d_v2(blk.norm1, prefix + ".norm1");
        assign_adain1d_v2(blk.norm2, prefix + ".norm2");
    };

    // DurationEncoder
    for (int i = 0; i < PRED_N_LAYERS; i++) {
        assign_bilstm_v2(dur_enc_lstm[i],
            "predictor.text_encoder.lstms." + std::to_string(i * 2));
        dur_enc_aln[i].fc_w_f16 = f16(
            "predictor.text_encoder.lstms." + std::to_string(i * 2 + 1) + ".fc.weight");
    }

    assign_bilstm_v2(dur_lstm, "predictor.lstm");
    dur_proj_w_f16 = f16("predictor.duration_proj.linear_layer.weight");

    assign_bilstm_v2(shared_lstm, "predictor.shared");

    // F0/N blocks
    for (int i = 0; i < PRED_F0N_BLOCKS; i++) {
        assign_resblk_v2(f0_blocks[i], "predictor.F0." + std::to_string(i));
        assign_resblk_v2(n_blocks[i], "predictor.N." + std::to_string(i));
    }

    // --- Decoder ---
    dec_asr_res_wv_f16 = f16("decoder.asr_res.0.weight_v");
    assign_resblk_v2(dec_encode, "decoder.encode");
    for (int i = 0; i < DEC_N_DECODE; i++)
        assign_resblk_v2(dec_decode[i], "decoder.decode." + std::to_string(i));

    // --- Generator ---
    auto assign_resblock1_v2 = [&](AdaINResBlock1Weights& rb, const std::string& prefix) {
        for (int j = 0; j < 3; j++) {
            std::string js = std::to_string(j);
            rb.convs1[j].wv_nhwc_f16 = nhwc16(prefix + ".convs1." + js + ".weight_v");
            rb.convs2[j].wv_nhwc_f16 = nhwc16(prefix + ".convs2." + js + ".weight_v");
            assign_adain1d_v2(rb.adain1[j], prefix + ".adain1." + js);
            assign_adain1d_v2(rb.adain2[j], prefix + ".adain2." + js);
        }
    };

    for (int i = 0; i < GEN_N_UPS; i++) {
        std::string pfx = "decoder.generator.ups." + std::to_string(i);
        gen_ups[i].wv_f16 = f16(pfx + ".weight_v");
    }

    for (int i = 0; i < GEN_N_UPS; i++) {
        std::string pfx = "decoder.generator.noise_convs." + std::to_string(i);
        gen_noise_convs[i].w_f16 = f16(pfx + ".weight");
        gen_noise_convs[i].w_nhwc_f16 = nhwc16(pfx + ".weight");
        gen_noise_convs[i].c_in_pad = pad_cin(pfx + ".weight");
    }

    for (int i = 0; i < GEN_N_UPS; i++)
        assign_resblock1_v2(gen_noise_res[i],
            "decoder.generator.noise_res." + std::to_string(i));

    for (int i = 0; i < GEN_N_RESBLOCKS; i++)
        assign_resblock1_v2(gen_resblocks[i],
            "decoder.generator.resblocks." + std::to_string(i));

    gen_conv_post_wv_f16 = f16("decoder.generator.conv_post.weight_v");
}

// ---------------------------------------------------------------------------
// Weights::prefetch — CPU only, no CUDA. mmap + populate pages + parse header.
// ---------------------------------------------------------------------------

Weights Weights::prefetch(const std::string& path) {
    Weights w;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open weights: %s\n", path.c_str());
        std::exit(1);
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", path.c_str());
        std::exit(1);
    }
    madvise(mapped, file_size, MADV_SEQUENTIAL);
    const uint8_t* base = (const uint8_t*)mapped;

    // Parse file header
    uint32_t magic;
    memcpy(&magic, base, 4);
    if (magic != KOKO_MAGIC) {
        fprintf(stderr, "Bad magic in %s: expected KOKO\n", path.c_str());
        munmap(mapped, file_size);
        std::exit(1);
    }

    uint32_t version;
    memcpy(&version, base + 4, 4);
    if (version != 1 && version != 2) {
        fprintf(stderr, "Unsupported weight file version %u (expected 1 or 2)\n", version);
        munmap(mapped, file_size);
        std::exit(1);
    }

    uint64_t header_len;
    memcpy(&header_len, base + 8, 8);
    w.tensors = parse_header((const char*)(base + 16), header_len);

    for (size_t i = 0; i < w.tensors.size(); i++)
        w.name_to_idx[w.tensors[i].name] = i;

    size_t header_end = 16 + header_len;
    size_t data_start = align_up(header_end, HEADER_ALIGN);

    size_t total_data = 0;
    for (auto& td : w.tensors) {
        size_t end = td.offset + td.size_bytes;
        if (end > total_data) total_data = end;
    }
    if (!w.tensors.empty()) {
        auto& last = w.tensors.back();
        total_data = std::max(total_data, align_up(last.offset + last.size_bytes, 256));
    }

    w.gpu_data_size = total_data;
    w.prefetch_base = (const uint8_t*)mapped + data_start;
    w.mmap_ptr = mapped;
    w.mmap_size = file_size;
    w.data_offset = data_start;

    return w;
}

// ---------------------------------------------------------------------------
// Weights::prefetch (from memory) — parse KOKO data already in memory.
// Caller owns the buffer (e.g. bundle mmap); we just store a pointer.
// ---------------------------------------------------------------------------

Weights Weights::prefetch(const void* data, size_t size) {
    Weights w;
    const uint8_t* base = (const uint8_t*)data;

    uint32_t magic;
    memcpy(&magic, base, 4);
    if (magic != KOKO_MAGIC) {
        fprintf(stderr, "Bad magic: expected KOKO\n");
        std::exit(1);
    }

    uint32_t version;
    memcpy(&version, base + 4, 4);
    if (version != 1 && version != 2) {
        fprintf(stderr, "Unsupported weight file version %u (expected 1 or 2)\n", version);
        std::exit(1);
    }

    uint64_t header_len;
    memcpy(&header_len, base + 8, 8);
    w.tensors = parse_header((const char*)(base + 16), header_len);

    for (size_t i = 0; i < w.tensors.size(); i++)
        w.name_to_idx[w.tensors[i].name] = i;

    size_t header_end = 16 + header_len;
    size_t data_start = align_up(header_end, HEADER_ALIGN);

    size_t total_data = 0;
    for (auto& td : w.tensors) {
        size_t end = td.offset + td.size_bytes;
        if (end > total_data) total_data = end;
    }
    if (!w.tensors.empty()) {
        auto& last = w.tensors.back();
        total_data = std::max(total_data, align_up(last.offset + last.size_bytes, 256));
    }

    w.gpu_data_size = total_data;
    w.prefetch_base = base + data_start;
    // mmap_ptr stays nullptr — we don't own the mapping
    return w;
}

// ---------------------------------------------------------------------------
// Weights::upload — cudaMalloc + cudaMemcpy from prefetched data, assign ptrs.
// ---------------------------------------------------------------------------

void Weights::upload(cudaStream_t stream) {
    CUDA_CHECK(cudaMalloc(&gpu_data, gpu_data_size));
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_data, prefetch_base, gpu_data_size,
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(gpu_data, prefetch_base, gpu_data_size,
                               cudaMemcpyHostToDevice));
    }

    // Free mmap only if we own it (file-based prefetch)
    if (mmap_ptr) {
        munmap(mmap_ptr, mmap_size);
        mmap_ptr = nullptr;
        mmap_size = 0;
    }
    prefetch_base = nullptr;
    data_offset = 0;

    assign_weight_pointers(*this);
}

// ---------------------------------------------------------------------------
// Weights::load — convenience: prefetch + upload in one call.
// ---------------------------------------------------------------------------

Weights Weights::load(const std::string& path, cudaStream_t stream) {
    Weights w = prefetch(path);
    w.upload(stream);
    return w;
}

// ---------------------------------------------------------------------------
// Weights::free
// ---------------------------------------------------------------------------

void Weights::free() {
    if (gpu_data) {
        cudaFree(gpu_data);
        gpu_data = nullptr;
        gpu_data_size = 0;
    }
}

// ---------------------------------------------------------------------------
// Weights::get
// ---------------------------------------------------------------------------

float* Weights::get(const std::string& name) const {
    auto it = name_to_idx.find(name);
    if (it == name_to_idx.end()) return nullptr;
    return (float*)((uint8_t*)gpu_data + tensors[it->second].offset);
}

// ---------------------------------------------------------------------------
// Weights::get_shape
// ---------------------------------------------------------------------------

const std::vector<int>* Weights::get_shape(const std::string& name) const {
    auto it = name_to_idx.find(name);
    if (it == name_to_idx.end()) return nullptr;
    return &tensors[it->second].shape;
}

// ---------------------------------------------------------------------------
// Weights::print_info
// ---------------------------------------------------------------------------

void Weights::print_info() const {
    fprintf(stderr, "weights: %zu tensors, %.1f MB GPU\n",
            tensors.size(), gpu_data_size / (1024.0 * 1024.0));

    int missing = 0;
    auto check = [&](const char* label, const float* ptr) {
        if (!ptr) {
            fprintf(stderr, "  WARNING: %s not found\n", label);
            missing++;
        }
    };

    // ALBERT
    check("bert_word_embed", bert_word_embed);
    check("bert_pos_embed", bert_pos_embed);
    check("bert_proj_w", bert_proj_w);
    check("albert.q_w", albert.q_w);
    check("albert.k_w", albert.k_w);
    check("albert.v_w", albert.v_w);
    check("albert.ffn_w", albert.ffn_w);
    check("albert.ffn_ln_w", albert.ffn_ln_w);

    // bert_encoder
    check("bert_enc_w", bert_enc_w);
    check("bert_enc_b", bert_enc_b);

    // Text encoder
    check("text_conv[0].conv_wv", text_conv[0].conv_wv);
    check("text_conv[2].conv_wv", text_conv[2].conv_wv);
    check("text_lstm_wih_fwd", text_lstm_wih_fwd);
    check("text_lstm_wih_rev", text_lstm_wih_rev);

    // Predictor
    check("dur_enc_lstm[0].wih_fwd", dur_enc_lstm[0].wih_fwd);
    check("dur_enc_aln[0].fc_w", dur_enc_aln[0].fc_w);
    check("dur_lstm.wih_fwd", dur_lstm.wih_fwd);
    check("dur_proj_w", dur_proj_w);
    check("shared_lstm.wih_fwd", shared_lstm.wih_fwd);
    check("f0_blocks[0].conv1_wg", f0_blocks[0].conv1_wg);
    check("f0_blocks[1].pool_wg", f0_blocks[1].pool_wg);
    check("n_blocks[0].conv1_wg", n_blocks[0].conv1_wg);
    check("f0_proj_w", f0_proj_w);
    check("n_proj_w", n_proj_w);

    // Decoder
    check("dec_f0_conv_wg", dec_f0_conv_wg);
    check("dec_n_conv_wg", dec_n_conv_wg);
    check("dec_asr_res_wg", dec_asr_res_wg);
    check("dec_encode.conv1_wg", dec_encode.conv1_wg);
    check("dec_decode[0].conv1_wg", dec_decode[0].conv1_wg);
    check("dec_decode[3].pool_wg", dec_decode[3].pool_wg);
    check("gen_ups[0].wg", gen_ups[0].wg);
    check("gen_ups[1].wg", gen_ups[1].wg);
    check("gen_noise_convs[0].w", gen_noise_convs[0].w);
    check("gen_resblocks[0].convs1[0].wg", gen_resblocks[0].convs1[0].wg);
    check("gen_resblocks[0].alpha1[0]", gen_resblocks[0].alpha1[0]);
    check("gen_resblocks[5].convs1[0].wg", gen_resblocks[5].convs1[0].wg);
    check("gen_conv_post_wg", gen_conv_post_wg);
    check("gen_source_w", gen_source_w);

    if (missing) {
        fprintf(stderr, "  %d weight(s) missing!\n", missing);
    } else {
        fprintf(stderr, "  all key weights found\n");
    }
}
