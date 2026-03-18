#!/bin/bash
# bench.sh — Benchmark RTFx with warmup, repeated runs, and STT verification.
#
# Starts the server, warms up graph caches, runs N timed requests per text,
# reports median/p95 per-component timing, then verifies audio via paraketto STT.
#
# Usage: ./bench.sh [binary] [warmup_runs] [timed_runs]

set -e

BINARY="${1:-./rokoko}"
WARMUP="${2:-10}"
RUNS="${3:-30}"
shift 3 2>/dev/null || true
EXTRA_ARGS="$@"
PORT=8097
PARAKETTO="$HOME/git/LokalOptima/paraketto/paraketto.fp8"

# Test texts: short (~1.5s audio), medium (~3-5s), long (~15-20s)
TEXTS=(
    "Hello world."
    "The quick brown fox jumps over the lazy dog and then runs back home again through the meadow."
    "In the heart of every great city there lies a park, a green refuge from the concrete jungle that surrounds it. People come from all walks of life to sit beneath the ancient oak trees and watch the world go by. Children play on the swings while their parents chat on nearby benches, sharing stories of their day."
)
LABELS=("short" "medium" "long")

cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Kill any leftover rokoko servers
pkill -f "rokoko --serve" 2>/dev/null || true
sleep 0.5

# Start server
"$BINARY" --serve "$PORT" $EXTRA_ARGS 2>/dev/null &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 30); do
    if curl -s http://localhost:$PORT/health -o /dev/null 2>/dev/null; then break; fi
    sleep 0.2
done

hit() {
    # Returns: g2p_ms tts_ms total_ms audio_sec
    # Optional $2: path to save audio (otherwise discarded)
    local text="$1"
    local out="${2:-/dev/null}"
    local headers
    headers=$(mktemp)
    local json
    json=$(printf '%s' "$text" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

    local total_s
    total_s=$(curl -s -X POST "http://localhost:$PORT/synthesize" \
        -H "Content-Type: application/json" \
        -d "{\"text\":$json}" \
        -o "$out" \
        -D "$headers" \
        -w '%{time_total}')

    local g2p tts audio total_ms
    g2p=$(grep -i 'x-g2p-ms' "$headers" | tr -d '\r' | awk '{print $2}')
    tts=$(grep -i 'x-tts-ms' "$headers" | tr -d '\r' | awk '{print $2}')
    audio=$(grep -i 'x-audio-duration' "$headers" | tr -d '\r' | awk '{print $2}')
    total_ms=$(echo "$total_s * 1000" | bc -l)
    rm -f "$headers"
    echo "$g2p $tts $total_ms $audio"
}

echo "=== Rokoko Benchmark ==="
echo "Binary: $BINARY"
echo "Warmup: $WARMUP | Timed runs: $RUNS"
echo ""

for idx in "${!TEXTS[@]}"; do
    text="${TEXTS[$idx]}"
    label="${LABELS[$idx]}"

    # Warmup (populate graph caches, JIT, etc.)
    # Save audio from the last warmup run for STT verification
    for i in $(seq 1 "$WARMUP"); do
        if [ "$i" -eq "$WARMUP" ]; then
            hit "$text" "/tmp/bench_${label}.wav" >/dev/null
        else
            hit "$text" >/dev/null
        fi
    done

    # Timed runs — collect data
    g2p_vals=()
    tts_vals=()
    total_vals=()
    audio_dur=""

    for i in $(seq 1 "$RUNS"); do
        read -r g tts tot aud <<< "$(hit "$text")"
        g2p_vals+=("$g")
        tts_vals+=("$tts")
        total_vals+=("$tot")
        audio_dur="$aud"
    done

    # Compute stats with python
    python3 -c "
import math

def stats(vals):
    vals.sort()
    n = len(vals)
    med = vals[n // 2]
    p95 = vals[min(math.ceil(n * 0.95) - 1, n - 1)]
    p99 = vals[min(math.ceil(n * 0.99) - 1, n - 1)]
    return med, p95, p99, min(vals), max(vals)

g2p = [float(x) for x in '${g2p_vals[*]}'.split()]
tts = [float(x) for x in '${tts_vals[*]}'.split()]
total = [float(x) for x in '${total_vals[*]}'.split()]
audio = float('$audio_dur')

g_med, g_p95, g_p99, g_min, g_max = stats(g2p)
t_med, t_p95, t_p99, t_min, t_max = stats(tts)
e_med, e_p95, e_p99, e_min, e_max = stats(total)

rtfx_med = audio / (e_med / 1000)
rtfx_p95 = audio / (e_p95 / 1000)

print(f'--- {\"$label\"} ({audio:.2f}s audio, n={len(total)}) ---')
print(f'         {\"median\":>8s}  {\"p95\":>8s}  {\"min\":>8s}  {\"max\":>8s}')
print(f'  G2P:   {g_med:8.2f}  {g_p95:8.2f}  {g_min:8.2f}  {g_max:8.2f}  ms')
print(f'  TTS:   {t_med:8.2f}  {t_p95:8.2f}  {t_min:8.2f}  {t_max:8.2f}  ms')
print(f'  Total: {e_med:8.2f}  {e_p95:8.2f}  {e_min:8.2f}  {e_max:8.2f}  ms')
print(f'  RTFx:  {rtfx_med:8.0f}x  {rtfx_p95:8.0f}x  (median / p95)')
print()
"
done

# STT verification via paraketto
if [ -x "$PARAKETTO" ]; then
    echo "=== STT Verification (paraketto) ==="
    stt_fail=0
    for idx in "${!TEXTS[@]}"; do
        text="${TEXTS[$idx]}"
        label="${LABELS[$idx]}"
        wav="/tmp/bench_${label}.wav"

        if [ ! -f "$wav" ]; then
            echo "  $label: SKIP (no audio)"
            continue
        fi

        # Normalize: lowercase, strip punctuation
        normalize() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | xargs; }

        expected=$(normalize "$text")
        actual=$(normalize "$("$PARAKETTO" "$wav" 2>/dev/null | tail -1)")

        if [ "$expected" = "$actual" ]; then
            echo "  $label: PASS"
        else
            echo "  $label: FAIL"
            echo "    expected: $expected"
            echo "    got:      $actual"
            stt_fail=1
        fi
        rm -f "$wav"
    done
    echo ""
    if [ "$stt_fail" -eq 1 ]; then
        echo "STT verification FAILED"
        exit 1
    fi
else
    echo "(skipping STT verification — paraketto not found at $PARAKETTO)"
    echo ""
fi
