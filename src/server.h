// server.h — HTTP server for Rokoko TTS (header-only, templated)
//
// PipelineT must expose:
//   std::string synthesize(const std::string& text, const std::string& voice,
//                          std::vector<float>& audio_out)
//   double last_preprocess_ms, last_g2p_ms, last_tts_ms
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "cpp-httplib/httplib.h"
#include "weights.h"

static inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

static thread_local std::string t_log_detail;

static inline void log_request(const httplib::Request& req, const httplib::Response& res) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&tt, &tm);
    char ts[20];
    strftime(ts, sizeof(ts), "%H:%M:%S", &tm);

    fprintf(stderr, "%s  %s %s  %d\n", ts, req.method.c_str(), req.path.c_str(), res.status);

    if (!t_log_detail.empty()) {
        fprintf(stderr, "         %s\n", t_log_detail.c_str());
        t_log_detail.clear();
    }
}

// Minimal JSON string value extractor (no dependency on a JSON library).
// Finds "key":"value" and returns value. Returns fallback if not found.
static inline std::string json_get_string(const std::string& body,
                                           const std::string& key,
                                           const std::string& fallback = "") {
    std::string needle = "\"" + key + "\"";
    auto pos = body.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = body.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    pos = body.find('"', pos + 1);
    if (pos == std::string::npos) return fallback;
    pos++; // skip opening quote
    std::string result;
    while (pos < body.size() && body[pos] != '"') {
        if (body[pos] == '\\' && pos + 1 < body.size()) {
            pos++;
            switch (body[pos]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += body[pos]; break;
            }
        } else {
            result += body[pos];
        }
        pos++;
    }
    return result;
}

template<typename PipelineT>
static void run_server(PipelineT& pipeline, const std::string& host, int port) {
    httplib::Server svr;
    std::mutex mtx;

    svr.set_logger(log_request);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rokoko TTS</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0;
         display: flex; justify-content: center; padding: 2rem; min-height: 100vh; }
  .container { width: 100%; max-width: 600px; }
  h1 { font-size: 1.3rem; font-weight: 600; margin-bottom: 1.5rem; color: #fff; }
  textarea { width: 100%; height: 120px; background: #1a1a1a; color: #e0e0e0;
             border: 1px solid #333; border-radius: 8px; padding: 12px; font-size: 15px;
             font-family: inherit; resize: vertical; outline: none; }
  textarea:focus { border-color: #555; }
  .controls { display: flex; gap: 10px; margin-top: 12px; align-items: center; }
  select { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333;
           border-radius: 6px; padding: 8px 12px; font-size: 14px; outline: none; }
  button { background: #2563eb; color: #fff; border: none; border-radius: 6px;
           padding: 8px 20px; font-size: 14px; font-weight: 500; cursor: pointer; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #333; color: #666; cursor: default; }
  .status { font-size: 13px; color: #888; margin-left: auto; white-space: nowrap; }
  audio { width: 100%; margin-top: 16px; outline: none; }
  .timing { font-size: 12px; color: #666; margin-top: 8px; font-variant-numeric: tabular-nums; }
  kbd { display: inline-block; font-size: 11px; color: #666; margin-top: 6px; }
</style>
</head>
<body>
<div class="container">
  <h1>Rokoko TTS</h1>
  <textarea id="text" placeholder="Type something..." autofocus>The quick brown fox jumps over the lazy dog.</textarea>
  <div class="controls">
    <select id="voice">
      <option value="af_heart">af_heart</option>
      <option value="af_bella">af_bella</option>
      <option value="af_sky">af_sky</option>
      <option value="af_nicole">af_nicole</option>
    </select>
    <button id="btn" onclick="speak()">Speak</button>
    <span class="status" id="status"></span>
  </div>
  <audio id="audio" controls style="display:none"></audio>
  <div class="timing" id="timing"></div>
  <kbd>Ctrl+Enter to speak</kbd>
</div>
<script>
const $ = id => document.getElementById(id);
let curCtx = null;
async function speak() {
  const text = $('text').value.trim();
  if (!text) return;
  $('btn').disabled = true;
  $('status').textContent = 'generating...';
  $('timing').textContent = '';
  $('audio').style.display = 'none';
  if (curCtx) { curCtx.close(); curCtx = null; }
  const t0 = performance.now();
  let tfirst = 0;
  try {
    const r = await fetch('/synthesize/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, voice: $('voice').value})
    });
    if (!r.ok) {
      $('status').textContent = (await r.json()).error || 'error';
      return;
    }
    const ctx = new AudioContext({sampleRate: 24000});
    curCtx = ctx;
    let nt = 0, ns = 0;
    const pcm = [];
    let rem = new Uint8Array(0);
    const rd = r.body.getReader();
    for (;;) {
      const {done, value} = await rd.read();
      if (done) break;
      if (!tfirst) {
        tfirst = performance.now();
        $('status').textContent = 'streaming...';
        nt = ctx.currentTime + 0.05;
      }
      const c = new Uint8Array(rem.length + value.length);
      c.set(rem); c.set(value, rem.length);
      const u = c.length - c.length % 4;
      rem = u < c.length ? c.slice(u) : new Uint8Array(0);
      if (!u) continue;
      const ab = new ArrayBuffer(u);
      new Uint8Array(ab).set(c.subarray(0, u));
      const f = new Float32Array(ab);
      pcm.push(f); ns += f.length;
      const b = ctx.createBuffer(1, f.length, 24000);
      b.getChannelData(0).set(f);
      const s = ctx.createBufferSource();
      s.buffer = b; s.connect(ctx.destination);
      if (nt < ctx.currentTime) nt = ctx.currentTime;
      s.start(nt); nt += b.duration;
    }
    if (!ns) { $('status').textContent = 'no audio'; return; }
    $('audio').src = URL.createObjectURL(makeWav(pcm, ns));
    $('audio').style.display = 'block';
    $('status').textContent = '';
    const dur = (ns / 24000).toFixed(1);
    const first = tfirst ? (tfirst - t0).toFixed(0) : '?';
    const total = ((performance.now() - t0) / 1000).toFixed(2);
    $('timing').textContent = dur + 's audio  \u00b7  first chunk ' + first + 'ms  \u00b7  total ' + total + 's';
  } catch(e) {
    $('status').textContent = 'error';
  } finally {
    $('btn').disabled = false;
  }
}
function makeWav(chunks, n) {
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0,'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8,'WAVE');
  w(12,'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 24000, true); v.setUint32(28, 48000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36,'data'); v.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of chunks) for (let i = 0; i < c.length; i++) {
    const x = Math.max(-1, Math.min(1, c[i]));
    v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7FFF, true); o += 2;
  }
  return new Blob([buf], {type: 'audio/wav'});
}
$('text').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); speak(); }
});
</script>
</body>
</html>)html", "text/html");
    });

    svr.Post("/synthesize", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse JSON body
        std::string text = json_get_string(req.body, "text");
        std::string voice = json_get_string(req.body, "voice", "af_heart");

        if (text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'text' field\"}", "application/json");
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<float> audio;
        std::string err;
        {
            std::lock_guard<std::mutex> lock(mtx);
            err = pipeline.synthesize(text, voice, audio);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        if (!err.empty()) {
            res.status = 500;
            std::string body = "{\"error\":\"" + json_escape(err) + "\"}";
            res.set_content(body, "application/json");
            return;
        }

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double audio_dur = (double)audio.size() / SAMPLE_RATE;
        double rtfx = audio_dur / (elapsed_ms / 1000.0);

        // Timing headers
        res.set_header("X-Preprocess-Ms", std::to_string(pipeline.last_preprocess_ms));
        res.set_header("X-G2P-Ms", std::to_string(pipeline.last_g2p_ms));
        res.set_header("X-TTS-Ms", std::to_string(pipeline.last_tts_ms));
        res.set_header("X-Audio-Duration", std::to_string(audio_dur));

        // Write WAV to string via ostringstream
        std::ostringstream wav_stream(std::ios::binary);
        write_wav_to_(wav_stream, audio.data(), (int)audio.size(), SAMPLE_RATE);
        res.set_content(wav_stream.str(), "audio/wav");

        // Log detail
        std::string preview = text.substr(0, 80);
        if (text.size() > 80) preview += "...";
        char detail[256];
        snprintf(detail, sizeof(detail), "audio=%.1fs  inference=%.0fms  RTFx=%.0fx  \"%s\"",
                 audio_dur, elapsed_ms, rtfx, preview.c_str());
        t_log_detail = detail;
    });

    // Streaming endpoint: sends raw float32 PCM via chunked transfer encoding.
    // Each TTS chunk is flushed as it's synthesized, enabling low-latency playback.
    svr.Post("/synthesize/stream", [&](const httplib::Request& req, httplib::Response& res) {
        std::string text = json_get_string(req.body, "text");
        std::string voice = json_get_string(req.body, "voice", "af_heart");

        if (text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'text' field\"}", "application/json");
            return;
        }

        res.set_header("X-Sample-Rate", std::to_string(SAMPLE_RATE));

        res.set_chunked_content_provider(
            "audio/pcm",
            [&pipeline, &mtx, text, voice](size_t /*offset*/, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lock(mtx);
                auto t0 = std::chrono::high_resolution_clock::now();
                size_t total_samples = 0;

                pipeline.synthesize_streaming(text, voice,
                    [&sink, &total_samples](const float* data, size_t n) -> bool {
                        total_samples += n;
                        return sink.write(reinterpret_cast<const char*>(data),
                                          n * sizeof(float));
                    });

                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                double audio_dur = (double)total_samples / SAMPLE_RATE;
                double rtfx = (elapsed_ms > 0) ? audio_dur / (elapsed_ms / 1000.0) : 0;

                std::string preview = text.substr(0, 80);
                if (text.size() > 80) preview += "...";
                char detail[256];
                snprintf(detail, sizeof(detail),
                         "[stream] audio=%.1fs  inference=%.0fms  RTFx=%.0fx  \"%s\"",
                         audio_dur, elapsed_ms, rtfx, preview.c_str());
                t_log_detail = detail;

                sink.done();
                return true;
            }
        );
    });

    const char* display_host = (host == "0.0.0.0") ? "localhost" : host.c_str();
    fprintf(stderr, "listening on http://%s:%d\n", display_host, port);
    fprintf(stderr, "\n");
    if (!svr.listen(host, port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        std::exit(1);
    }
}
