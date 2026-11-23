#!/usr/bin/env bash
# GPU smoke test for a fresh NVIDIA box (the intended machine is a RunPod RTX 4090).
#
# Installs build tools, Rust, and nvcc when they are missing, builds Flint with
# `--features cuda`, runs the ignored GPU-vs-CPU tests, serves Qwen2.5-0.5B on
# the GPU, and prints tokens/s and time-to-first-token at concurrency 1, 8, and 32.
#
# Usage (on the GPU machine, from anywhere):
#   bash scripts/gpu_smoke.sh
# Optional:
#   FLINT_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
#   FLINT_PORT=8000
#   FLINT_MODEL=Qwen/Qwen2.5-0.5B-Instruct
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script needs an NVIDIA driver and a GPU." >&2
  exit 1
fi
nvidia-smi

if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "== packages =="
$SUDO apt-get update
$SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential pkg-config libssl-dev curl ca-certificates python3

if ! command -v nvcc >/dev/null 2>&1; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
  else
    echo "nvcc not on PATH; installing nvidia-cuda-toolkit (or use a RunPod image that already has the CUDA toolkit)."
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit
  fi
fi
nvcc --version

if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
# Driver and cuBLAS sonames (libcuda.so.1, libcublas.so.12) must be visible
# to cudarc's dynamic loader.
for libdir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/lib64; do
  if [[ -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
  fi
done

if ! command -v cargo >/dev/null 2>&1; then
  echo "== rustup =="
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.100.0
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi
rustc --version

echo "== GPU correctness tests (release) =="
cargo test --release -p flint-kernels --features cuda -- --ignored --test-threads=1
cargo test --release -p flint-core --features cuda -- --ignored --test-threads=1

echo "== server and bench =="
cargo build --release -p flint-server --features cuda
cargo build --release -p flint-bench

PORT="${FLINT_PORT:-8000}"
MODEL_ARGS=()
if [[ -n "${FLINT_MODEL_DIR:-}" ]]; then
  MODEL_ARGS=(--model-dir "${FLINT_MODEL_DIR}")
fi

./target/release/flint serve \
  --device cuda \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --max-num-seqs 32 \
  --max-batched-tokens 4096 \
  --num-blocks 512 \
  "${MODEL_ARGS[@]}" \
  >/tmp/flint-gpu-serve.log 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "waiting for http://127.0.0.1:${PORT}/health (model download and GPU upload can take a few minutes)"
ready=0
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "flint serve exited. Log:" >&2
    cat /tmp/flint-gpu-serve.log >&2
    exit 1
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "server did not become healthy. Log:" >&2
  cat /tmp/flint-gpu-serve.log >&2
  exit 1
fi

mkdir -p bench-results
MODEL_NAME="${FLINT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
for c in 1 8 32; do
  reqs=8
  if [[ "${c}" -gt 8 ]]; then
    reqs="${c}"
  fi
  echo "== bench concurrency ${c} =="
  ./target/release/flint-bench \
    --url "http://127.0.0.1:${PORT}" \
    --model "${MODEL_NAME}" \
    --endpoint completions \
    --concurrency "${c}" \
    --requests "${reqs}" \
    --warmup 1 \
    --max-tokens 32 \
    --temperature 0 \
    --prompt "Hello, my name is" \
    --prompt "The capital of France is" \
    --json-out "bench-results/gpu-c${c}.json"
  python3 - "${c}" <<'PY'
import json, sys
c = sys.argv[1]
report = json.load(open(f"bench-results/gpu-c{c}.json"))
ttft = report["ttft_ms"]
print(
    f"concurrency {c}: tok/s={report['throughput_tok_s']:.2f}  "
    f"TTFT p50={ttft['p50']:.1f} ms  TTFT p99={ttft['p99']:.1f} ms"
)
PY
done
