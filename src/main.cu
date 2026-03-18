// main.cu — Neural G2P + TTS in one binary
//
// Pipeline: text → preprocess → G2P infer → tokenize → chunk → TTS infer → WAV
//
// Build: make rokoko
// Usage: ./rokoko --text "Hello world." -o output.wav --voice af_heart
//        ./rokoko --serve 8080

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "rokoko_common.h"
#include "kernels.h"
#include "normalize.h"
#include "g2p.h"
#include "bundle.h"
#include "server.h"

// Backend-specific: each binary provides its own bundle URL + filename
extern const char* default_bundle_url();
extern const char* default_bundle_filename();

// ---------------------------------------------------------------------------
// Voice map (populated from bundle)
// ---------------------------------------------------------------------------

struct VoicePack { const char* start; const char* end; };
using VoiceMap = std::unordered_map<std::string, VoicePack>;

static VoiceMap build_voice_map(const Bundle& bundle) {
    VoiceMap voices;
    for (auto& [name, span] : bundle.entries) {
        if (name.rfind("voice/", 0) == 0) {
            std::string vname = name.substr(6); // strip "voice/"
            voices[vname] = {span.data, span.data + span.size};
        }
    }
    return voices;
}

// ---------------------------------------------------------------------------
// Vocab table (phoneme codepoint → Kokoro token ID)
// Copied from phonemize.cpp to avoid linking the full phonemizer.
// ---------------------------------------------------------------------------

namespace {

struct VocabEntry { uint32_t codepoint; int32_t token_id; };

static const VocabEntry VOCAB[] = {
    {0x3B, 1},    // ;
    {0x3A, 2},    // :
    {0x2C, 3},    // ,
    {0x2E, 4},    // .
    {0x21, 5},    // !
    {0x3F, 6},    // ?
    {0x2014, 9},  // —
    {0x2026, 10}, // …
    {0x22, 11},   // "
    {0x28, 12},   // (
    {0x29, 13},   // )
    {0x201C, 14}, // "
    {0x201D, 15}, // "
    {0x20, 16},   // space
    {0x303, 17},  // ̃
    {0x2A3, 18},  // ʣ
    {0x2A5, 19},  // ʥ
    {0x2A6, 20},  // ʦ
    {0x2A8, 21},  // ʨ
    {0x1D5D, 22}, // ᵝ
    {0xAB67, 23}, // ꭧ
    {'A', 24}, {'I', 25}, {'O', 31}, {'Q', 33}, {'S', 35}, {'T', 36},
    {'W', 39}, {'Y', 41},
    {0x1D4A, 42}, // ᵊ
    {'a', 43}, {'b', 44}, {'c', 45}, {'d', 46}, {'e', 47}, {'f', 48},
    {'h', 50}, {'i', 51}, {'j', 52}, {'k', 53}, {'l', 54}, {'m', 55},
    {'n', 56}, {'o', 57}, {'p', 58}, {'q', 59}, {'r', 60}, {'s', 61},
    {'t', 62}, {'u', 63}, {'v', 64}, {'w', 65}, {'x', 66}, {'y', 67},
    {'z', 68},
    {0x251, 69},  // ɑ
    {0x250, 70},  // ɐ
    {0x252, 71},  // ɒ
    {0xE6, 72},   // æ
    {0x3B2, 75},  // β
    {0x254, 76},  // ɔ
    {0x255, 77},  // ɕ
    {0xE7, 78},   // ç
    {0x256, 80},  // ɖ
    {0xF0, 81},   // ð
    {0x2A4, 82},  // ʤ
    {0x259, 83},  // ə
    {0x25A, 85},  // ɚ
    {0x25B, 86},  // ɛ
    {0x25C, 87},  // ɜ
    {0x25F, 90},  // ɟ
    {0x261, 92},  // ɡ
    {0x265, 99},  // ɥ
    {0x268, 101}, // ɨ
    {0x26A, 102}, // ɪ
    {0x29D, 103}, // ʝ
    {0x26F, 110}, // ɯ
    {0x270, 111}, // ɰ
    {0x14B, 112}, // ŋ
    {0x273, 113}, // ɳ
    {0x272, 114}, // ɲ
    {0x274, 115}, // ɴ
    {0xF8, 116},  // ø
    {0x278, 118}, // ɸ
    {0x3B8, 119}, // θ
    {0x153, 120}, // œ
    {0x279, 123}, // ɹ
    {0x27E, 125}, // ɾ
    {0x27B, 126}, // ɻ
    {0x281, 128}, // ʁ
    {0x27D, 129}, // ɽ
    {0x282, 130}, // ʂ
    {0x283, 131}, // ʃ
    {0x288, 132}, // ʈ
    {0x2A7, 133}, // ʧ
    {0x28A, 135}, // ʊ
    {0x28B, 136}, // ʋ
    {0x28C, 138}, // ʌ
    {0x263, 139}, // ɣ
    {0x264, 140}, // ɤ
    {0x3C7, 142}, // χ
    {0x28E, 143}, // ʎ
    {0x292, 147}, // ʒ
    {0x294, 148}, // ʔ
    {0x2C8, 156}, // ˈ (primary stress)
    {0x2CC, 157}, // ˌ (secondary stress)
    {0x2D0, 158}, // ː
    {0x2B0, 162}, // ʰ
    {0x2B2, 164}, // ʲ
    {0x2193, 169}, // ↓
    {0x2192, 171}, // →
    {0x2197, 172}, // ↗
    {0x2198, 173}, // ↘
    {0x1D7B, 177}, // ᵻ
};

static const std::unordered_map<uint32_t, int32_t>& vocab_map() {
    static std::unordered_map<uint32_t, int32_t> m = []() {
        std::unordered_map<uint32_t, int32_t> m;
        for (auto& e : VOCAB) m[e.codepoint] = e.token_id;
        return m;
    }();
    return m;
}

// UTF-8 decode one codepoint, advance i
static uint32_t utf8_decode(const std::string& s, size_t& i) {
    uint8_t c = s[i];
    if (c < 0x80) { i++; return c; }
    uint32_t cp; int extra;
    if ((c & 0xE0) == 0xC0)      { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else                          { cp = c & 0x07; extra = 3; }
    i++;
    for (int j = 0; j < extra && i < s.size(); j++, i++)
        cp = (cp << 6) | (s[i] & 0x3F);
    return cp;
}

// Count UTF-8 codepoints in string
static size_t utf8_len(const std::string& s) {
    size_t n = 0, i = 0;
    while (i < s.size()) { utf8_decode(s, i); n++; }
    return n;
}

// Convert IPA string to Kokoro token IDs with BOS(0)/EOS(0)
static std::vector<int32_t> to_tokens(const std::string& phonemes) {
    std::vector<int32_t> ids;
    ids.push_back(0); // BOS
    const auto& vm = vocab_map();
    size_t i = 0;
    while (i < phonemes.size()) {
        uint32_t cp = utf8_decode(phonemes, i);
        auto it = vm.find(cp);
        if (it != vm.end())
            ids.push_back(it->second);
    }
    ids.push_back(0); // EOS
    return ids;
}

// ---------------------------------------------------------------------------
// Chunking: split IPA string at natural boundaries if > max_tokens codepoints
// ---------------------------------------------------------------------------

static constexpr int MAX_IPA_CHARS = 510;

struct Chunk {
    std::string phonemes;
    std::vector<int32_t> tokens;
};

// Find byte offset of the N-th UTF-8 codepoint
static size_t utf8_byte_offset(const std::string& s, size_t n_codepoints) {
    size_t i = 0, count = 0;
    while (i < s.size() && count < n_codepoints) {
        utf8_decode(s, i);
        count++;
    }
    return i;
}

static std::vector<Chunk> chunk_ipa(const std::string& ipa) {
    std::vector<Chunk> chunks;
    size_t len = utf8_len(ipa);

    if (len <= MAX_IPA_CHARS) {
        chunks.push_back({ipa, to_tokens(ipa)});
        return chunks;
    }

    // Need to split
    size_t pos = 0; // byte position in ipa
    while (pos < ipa.size()) {
        // How many codepoints remain?
        std::string rest = ipa.substr(pos);
        size_t rest_len = utf8_len(rest);
        if (rest_len <= MAX_IPA_CHARS) {
            chunks.push_back({rest, to_tokens(rest)});
            break;
        }

        // Find byte offset for MAX_IPA_CHARS codepoints
        size_t limit_byte = utf8_byte_offset(rest, MAX_IPA_CHARS);

        // Scan backwards for split point
        size_t split = std::string::npos;

        // Priority 1: sentence-end punctuation (.!?)
        for (size_t j = limit_byte; j > 0; j--) {
            char c = rest[j - 1];
            if (c == '.' || c == '!' || c == '?') { split = j; break; }
        }

        // Priority 2: clause punctuation (,:;)
        if (split == std::string::npos) {
            for (size_t j = limit_byte; j > 0; j--) {
                char c = rest[j - 1];
                if (c == ',' || c == ':' || c == ';') { split = j; break; }
            }
        }

        // Priority 3: space
        if (split == std::string::npos) {
            for (size_t j = limit_byte; j > 0; j--) {
                if (rest[j - 1] == ' ') { split = j; break; }
            }
        }

        // Fallback: hard split at limit
        if (split == std::string::npos || split == 0)
            split = limit_byte;

        std::string piece = rest.substr(0, split);
        if (!piece.empty())
            chunks.push_back({piece, to_tokens(piece)});
        pos += split;
    }

    return chunks;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// TTS Pipeline — wraps all persistent state for server mode
// ---------------------------------------------------------------------------

struct TtsPipeline {
    Weights& weights;
    G2PModelCuda& g2p;
    cudaStream_t stream;
    GpuArena& encode_arena;
    GpuArena& decode_arena;
    float* d_workspace;
    size_t ws_bytes;
    VoiceMap& voices;

    // Timing (set after each synthesize call)
    double last_preprocess_ms = 0;
    double last_g2p_ms = 0;
    double last_tts_ms = 0;

    // Streaming synthesis: calls on_chunk(float_data, n_samples) per chunk.
    // Returns "" on success, error string on failure.
    template<typename F>
    std::string synthesize_streaming(const std::string& text, const std::string& voice,
                                      F on_chunk) {
        using clk = std::chrono::high_resolution_clock;
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        // Preprocess
        auto t0 = clk::now();
        std::string preprocessed = text_norm::preprocess_text(text);
        auto t1 = clk::now();
        last_preprocess_ms = ms(t0, t1);
        fprintf(stderr, "Preprocess: \"%s\" (%.1f ms)\n", preprocessed.c_str(), last_preprocess_ms);

        // G2P
        t0 = clk::now();
        std::string ipa = g2p.infer(preprocessed, stream);
        t1 = clk::now();
        last_g2p_ms = ms(t0, t1);
        fprintf(stderr, "G2P: \"%s\" (%.1f ms)\n", ipa.c_str(), last_g2p_ms);

        if (ipa.empty())
            return "G2P produced no output";

        auto chunks = chunk_ipa(ipa);

        auto vit = voices.find(voice);
        if (vit == voices.end())
            return "unknown voice '" + voice + "'";

        const float* voice_data = reinterpret_cast<const float*>(vit->second.start);
        size_t voice_size = vit->second.end - vit->second.start;
        int voice_rows = voice_size / (256 * sizeof(float));

        auto t_tts0 = clk::now();
        size_t total_samples = 0;
        for (size_t c = 0; c < chunks.size(); c++) {
            auto& chunk = chunks[c];
            int T = (int)chunk.tokens.size();

            int phoneme_count = T - 2;
            if (phoneme_count < 0) phoneme_count = 0;
            if (phoneme_count >= voice_rows) phoneme_count = voice_rows - 1;
            const float* style = voice_data + phoneme_count * 256;

            auto audio = rokoko_infer(weights, chunk.tokens.data(), T, style,
                                       stream, encode_arena,
                                       decode_arena, d_workspace, ws_bytes);
            fprintf(stderr, "  chunk %zu: T=%d, %zu samples (%.2fs), decode arena=%.1f MB\n",
                    c, T, audio.size(), audio.size()/24000.0, decode_arena.offset/1e6);
            total_samples += audio.size();
            if (!on_chunk(audio.data(), audio.size()))
                return "";
        }
        auto t_tts1 = clk::now();
        last_tts_ms = ms(t_tts0, t_tts1);

        double audio_sec = total_samples / 24000.0;
        double rtfx = audio_sec / (last_tts_ms / 1000.0);
        fprintf(stderr, "TTS: %.3f sec in %.1f ms (%.0fx realtime, %zu chunks)\n",
                audio_sec, last_tts_ms, rtfx, chunks.size());

        return "";
    }

    std::string synthesize(const std::string& text, const std::string& voice,
                           std::vector<float>& audio_out) {
        audio_out.clear();
        return synthesize_streaming(text, voice,
            [&audio_out](const float* data, size_t n) -> bool {
                audio_out.insert(audio_out.end(), data, data + n);
                return true;
            });
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    auto print_usage = [&]() {
        fprintf(stderr,
            "Usage: %s <text> [options]\n"
            "       %s --serve [port] [options]\n"
            "\n"
            "Options:\n"
            "  --voice <name>      Voice (default: af_heart)\n"
            "  -o <file>           Output WAV (default: output.wav)\n"
            "  --stdout            Write WAV to stdout\n"
            "  --serve [port]      HTTP server with web UI (default: 8080)\n"
            "  --host <addr>       Server bind address (default: 0.0.0.0)\n"
            "  --bundle <file>     Model bundle (default: ~/.cache/rokoko/rokoko.bundle)\n"
            "  --weights <file>    Standalone .koko weight file (overrides bundle weights)\n"
            "  --help              Show this help\n"
            "\n"
            "Examples:\n"
            "  %s \"Hello world.\" -o hello.wav\n"
            "  %s \"Hello world.\" --stdout | aplay\n"
            "  %s --serve 8080\n",
            argv[0], argv[0], argv[0], argv[0], argv[0]);
    };

    if (argc < 2) { print_usage(); return 1; }

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            print_usage(); return 0;
        }
    }

    std::string home = std::getenv("HOME") ? std::getenv("HOME") : ".";
    std::string bundle_path = home + "/.cache/rokoko/" + default_bundle_filename();
    std::string weights_path;  // standalone .koko file (overrides bundle weights)
    std::string text_input;
    std::string voice_name = "af_heart";
    std::string output_path = "output.wav";
    bool serve_mode = false;
    int serve_port = 8080;
    std::string serve_host = "0.0.0.0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--bundle" && i + 1 < argc)       bundle_path = argv[++i];
        else if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--voice" && i + 1 < argc)   voice_name = argv[++i];
        else if (arg == "-o" && i + 1 < argc)        output_path = argv[++i];
        else if (arg == "--stdout")                   output_path = "-";
        else if (arg == "--host" && i + 1 < argc)    serve_host = argv[++i];
        else if (arg == "--serve") {
            serve_mode = true;
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
                serve_port = std::atoi(argv[++i]);
        }
        else if (arg[0] != '-') {
            if (!text_input.empty()) {
                fprintf(stderr, "Error: unexpected argument '%s' (text already set)\n", arg.c_str());
                return 1;
            }
            text_input = arg;
        }
    }

    if (!serve_mode && text_input.empty()) {
        fprintf(stderr, "Error: provide text or use --serve for server mode\n");
        return 1;
    }

    // --- Auto-download bundle if missing ---
    {
        struct stat st;
        if (stat(bundle_path.c_str(), &st) != 0) {
            // Create parent directory
            std::string dir = bundle_path.substr(0, bundle_path.rfind('/'));
            for (size_t p = 1; p < dir.size(); p++) {
                if (dir[p] == '/') {
                    dir[p] = '\0';
                    mkdir(dir.c_str(), 0755);
                    dir[p] = '/';
                }
            }
            mkdir(dir.c_str(), 0755);

            fprintf(stderr, "Bundle not found at %s — downloading...\n", bundle_path.c_str());

            const char* url = default_bundle_url();

            // Download to temp file, then rename atomically
            std::string tmp_path = bundle_path + ".tmp";
            pid_t pid = fork();
            if (pid == 0) {
                execlp("curl", "curl", "-L", "-#", "-o", tmp_path.c_str(), url, nullptr);
                _exit(127);
            }
            int status;
            waitpid(pid, &status, 0);
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                fprintf(stderr, "Error: download failed\n");
                unlink(tmp_path.c_str());
                return 1;
            }
            if (rename(tmp_path.c_str(), bundle_path.c_str()) != 0) {
                fprintf(stderr, "Error: failed to move downloaded bundle\n");
                unlink(tmp_path.c_str());
                return 1;
            }
            fprintf(stderr, "Download complete.\n");
        }
    }

    using clk = std::chrono::high_resolution_clock;
    auto t_start = clk::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // --- Load bundle ---
    Bundle bundle = Bundle::load(bundle_path);

    auto weights_span = bundle.get("weights");
    auto g2p_span = bundle.get("g2p");
    if (!weights_span.data || !g2p_span.data) {
        fprintf(stderr, "Error: bundle missing 'weights' or 'g2p' entry\n");
        return 1;
    }

    // --- Load TTS weights (prefetch in background) + init CUDA ---
    Weights prefetched;
    std::thread prefetch_thread([&]() {
        if (!weights_path.empty()) {
            prefetched = Weights::prefetch(weights_path);
        } else {
            prefetched = Weights::prefetch(weights_span.data, weights_span.size);
        }
    });
    cudaFree(0); // lazy CUDA init
    prefetch_thread.join();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    prefetched.upload(stream);
    precompute_weight_norms(prefetched, stream);

    // --- Load G2P model ---
    G2PModelCuda g2p;
    if (!g2p.load(g2p_span.data, g2p_span.size, stream)) {
        fprintf(stderr, "Error: failed to load G2P model from bundle\n");
        return 1;
    }

    auto t_init = clk::now();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "GPU: %s | Init: %.0f ms | TTS: %.0f MB | G2P: %.1f MB\n",
            prop.name, ms(t_start, t_init),
            prefetched.gpu_data_size / 1e6, g2p.param_bytes() / 1e6);

    // --- Voice map ---
    VoiceMap voices = build_voice_map(bundle);

    // --- Pre-allocate arenas + workspace ---
    static constexpr size_t ENCODE_ARENA_BYTES = 64 * 1024 * 1024;  // 64 MB
    static constexpr size_t WORKSPACE_BYTES    = 128 * 1024 * 1024; // 128 MB

    GpuArena encode_arena;
    encode_arena.init(ENCODE_ARENA_BYTES);
    GpuArena decode_arena;  // starts empty, grows on first use
    float* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_BYTES));

    // --- Pre-warm: populate Cutlass operator caches + CUDA graphs ---
    {
        auto vit = voices.find("af_heart");
        if (vit != voices.end()) {
            const float* voice_data = reinterpret_cast<const float*>(vit->second.start);
            const float* style = voice_data;  // row 0
            auto warmup_ipa = g2p.infer("Warmup.", stream);
            auto warmup_tokens = to_tokens(warmup_ipa.empty() ? "." : warmup_ipa);
            rokoko_infer(prefetched, warmup_tokens.data(), (int)warmup_tokens.size(),
                         style, stream, encode_arena, decode_arena,
                         d_workspace, WORKSPACE_BYTES);
            cudaStreamSynchronize(stream);
        }
        auto t_warm = clk::now();
        fprintf(stderr, "Pre-warm: %.0f ms\n", ms(t_init, t_warm));
    }

    fprintf(stderr, "Encode arena: %.0f MB | Workspace: %.0f MB | Decode arena: on demand\n",
            ENCODE_ARENA_BYTES / 1e6, WORKSPACE_BYTES / 1e6);

    // =======================================================================
    // Server mode
    // =======================================================================
    if (serve_mode) {
        TtsPipeline pipeline{prefetched, g2p, stream,
                             encode_arena, decode_arena, d_workspace, WORKSPACE_BYTES, voices};

        run_server(pipeline, serve_host, serve_port);

        // Cleanup (unreachable unless server stops)
        cudaFree(d_workspace);
        decode_arena.destroy();
        encode_arena.destroy();
        g2p.free();
        prefetched.free();
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    }

    // =======================================================================
    // CLI mode (original behavior)
    // =======================================================================

    // --- Preprocess text ---
    auto t_pre0 = clk::now();
    std::string preprocessed = text_norm::preprocess_text(text_input);
    auto t_pre1 = clk::now();
    fprintf(stderr, "Preprocess: \"%s\" (%.1f ms)\n", preprocessed.c_str(), ms(t_pre0, t_pre1));

    // --- G2P infer ---
    auto t_g2p0 = clk::now();
    std::string ipa = g2p.infer(preprocessed, stream);
    auto t_g2p1 = clk::now();
    fprintf(stderr, "G2P: \"%s\" (%.1f ms)\n", ipa.c_str(), ms(t_g2p0, t_g2p1));

    if (ipa.empty()) {
        fprintf(stderr, "Error: G2P produced no output\n");
        return 1;
    }

    // --- Tokenize + chunk ---
    auto chunks = chunk_ipa(ipa);
    fprintf(stderr, "Chunks: %zu (total %zu IPA codepoints)\n", chunks.size(), utf8_len(ipa));

    // --- Voice pack ---
    auto vit = voices.find(voice_name);
    if (vit == voices.end()) {
        fprintf(stderr, "Error: unknown voice '%s'\n", voice_name.c_str());
        return 1;
    }
    const float* voice_data = reinterpret_cast<const float*>(vit->second.start);
    size_t voice_size = vit->second.end - vit->second.start;
    int voice_rows = voice_size / (256 * sizeof(float));

    // --- TTS infer per chunk ---
    std::vector<float> all_audio;

    for (size_t c = 0; c < chunks.size(); c++) {
        auto& chunk = chunks[c];
        int T = (int)chunk.tokens.size();

        fprintf(stderr, "Chunk %zu: \"%s\" (%d tokens)\n", c, chunk.phonemes.c_str(), T);

        // Style vector: index by phoneme count (T - 2 excludes BOS/EOS)
        int phoneme_count = T - 2;
        if (phoneme_count < 0) phoneme_count = 0;
        if (phoneme_count >= voice_rows) phoneme_count = voice_rows - 1;
        const float* style = voice_data + phoneme_count * 256;

        auto t0 = clk::now();
        auto audio = rokoko_infer(prefetched, chunk.tokens.data(), T, style,
                                   stream, encode_arena,
                                   decode_arena, d_workspace, WORKSPACE_BYTES);
        auto t1 = clk::now();

        double gen_ms = ms(t0, t1);
        double audio_sec = audio.size() / 24000.0;
        double rtfx = audio_sec / (gen_ms / 1000.0);
        fprintf(stderr, "  Generated %.3f sec in %.1f ms (%.0fx realtime), decode arena=%.1f MB\n",
                audio_sec, gen_ms, rtfx, decode_arena.offset/1e6);

        all_audio.insert(all_audio.end(), audio.begin(), audio.end());
    }

    cudaFree(d_workspace);
    decode_arena.destroy();
    encode_arena.destroy();

    // --- Write output ---
    write_wav(output_path, all_audio.data(), all_audio.size(), 24000);
    fprintf(stderr, "Output: %s (%.3f sec, %zu samples)\n",
            output_path.c_str(), all_audio.size() / 24000.0, all_audio.size());

    // Cleanup
    g2p.free();
    prefetched.free();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
