#!/usr/bin/env bash
# GPU smoke test for a fresh NVIDIA box (the intended machine is a RunPod RTX 4090).
#
# Installs build tools, Rust, and nvcc when they are missing, downloads
# Qwen2.5-0.5B-Instruct, builds Flint with `--features cuda`, runs the ignored
# GPU-vs-CPU tests whose environment is present, serves the model twice
# (`--dtype bf16` and `--dtype bf16 --quant int8`), and prints tok/s, TTFT,
# and inter-token latency at concurrency 1, 8, and 32 for both.
#
# Usage (on the GPU machine, from anywhere):
#   bash scripts/gpu_smoke.sh
# Optional:
#   FLINT_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
#   FLINT_HF_LOGITS=/path/to/hf-logits.json
#   FLINT_PORT=8000
#   FLINT_MODEL=Qwen/Qwen2.5-0.5B-Instruct
set -euo pipefail

cd "$(dirname "$0")/.."

KERNEL_STATUS="not run"
CORE_STATUS="not run"
BENCH_ROWS=()

summary() {
  echo
  echo "================ flint gpu smoke ================"
  printf "%-12s %s\n" "kernels" "${KERNEL_STATUS}"
  printf "%-12s %s\n" "core" "${CORE_STATUS}"
  printf "%-8s %-6s %12s %12s %12s %12s\n" "weights" "conc" "tok/s" "TTFT p50" "TTFT p99" "ITL p50"
  if [[ ${#BENCH_ROWS[@]} -eq 0 ]]; then
    printf "%-8s %-6s %12s %12s %12s %12s\n" "-" "-" "-" "-" "-" "-"
  else
    local row
    for row in "${BENCH_ROWS[@]}"; do
      # shellcheck disable=SC2086
      printf "%-8s %-6s %12s %12s %12s %12s\n" ${row}
    done
  fi
  echo "bench JSON: bench-results/gpu-{bf16,int8}-c{1,8,32}.json"
  echo "==============================================="
}
trap summary EXIT

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
# `env` keeps DEBIAN_FRONTEND set when SUDO is empty (running as root).
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential pkg-config libssl-dev curl ca-certificates python3

if ! command -v nvcc >/dev/null 2>&1; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
  else
    echo "nvcc not on PATH; installing nvidia-cuda-toolkit (or use a RunPod image that already has the CUDA toolkit)."
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit
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

echo "== model snapshot =="
MODEL_ID="${FLINT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
if [[ -z "${FLINT_MODEL_DIR:-}" ]]; then
  FLINT_MODEL_DIR="${PWD}/models/${MODEL_ID//\//--}"
  export FLINT_MODEL_DIR
fi
mkdir -p "${FLINT_MODEL_DIR}"
HF_BASE="https://huggingface.co/${MODEL_ID}/resolve/main"
download_one() {
  local name="$1"
  local required="$2"
  if [[ -s "${FLINT_MODEL_DIR}/${name}" ]]; then
    return 0
  fi
  if curl -fL --retry 3 --retry-delay 2 -o "${FLINT_MODEL_DIR}/${name}" "${HF_BASE}/${name}"; then
    return 0
  fi
  rm -f "${FLINT_MODEL_DIR}/${name}"
  if [[ "${required}" == "required" ]]; then
    echo "failed to download ${name} from ${HF_BASE}" >&2
    return 1
  fi
}
download_one config.json required
download_one tokenizer.json required
download_one model.safetensors required
download_one tokenizer_config.json optional
download_one generation_config.json optional
echo "FLINT_MODEL_DIR=${FLINT_MODEL_DIR}"

echo "== GPU correctness tests (release) =="
if cargo test --release -p flint-kernels --features cuda -- --ignored --test-threads=1; then
  KERNEL_STATUS="PASS"
else
  KERNEL_STATUS="FAIL"
fi

CORE_SKIP=()
if [[ -z "${FLINT_HF_LOGITS:-}" ]]; then
  CORE_SKIP+=(--skip compare_against_hf_logits_file)
fi
if cargo test --release -p flint-core --features cuda -- --ignored --test-threads=1 "${CORE_SKIP[@]}"; then
  CORE_STATUS="PASS"
else
  CORE_STATUS="FAIL"
fi

echo "== server and bench =="
cargo build --release -p flint-server --features cuda
cargo build --release -p flint-bench

PORT="${FLINT_PORT:-8000}"
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}
trap 'cleanup; summary' EXIT

wait_healthy() {
  local port="$1"
  echo "waiting for http://127.0.0.1:${port}/health (GPU weight upload can take a minute)"
  local ready=0
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
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
}

# label is the summary-table name. quant is the server and bench flag.
bench_mode() {
  local label="$1"
  local quant="$2"
  cleanup
  echo "== serve ${label} (dtype bf16, quant ${quant}) =="
  ./target/release/flint serve \
    --device cuda \
    --dtype bf16 \
    --quant "${quant}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --model-dir "${FLINT_MODEL_DIR}" \
    --max-num-seqs 32 \
    --max-batched-tokens 4096 \
    --num-blocks 512 \
    >/tmp/flint-gpu-serve.log 2>&1 &
  SERVER_PID=$!
  wait_healthy "${PORT}"
  local c reqs row
  for c in 1 8 32; do
    reqs=8
    if [[ "${c}" -gt 8 ]]; then
      reqs="${c}"
    fi
    echo "== bench ${label} concurrency ${c} =="
    ./target/release/flint-bench \
      --url "http://127.0.0.1:${PORT}" \
      --model "${MODEL_ID}" \
      --endpoint completions \
      --concurrency "${c}" \
      --requests "${reqs}" \
      --warmup 1 \
      --max-tokens 32 \
      --temperature 0 \
      --quant "${quant}" \
      --prompt "Hello, my name is" \
      --prompt "The capital of France is" \
      --json-out "bench-results/gpu-${label}-c${c}.json"
    row="$(python3 - "${label}" "${c}" <<'PY'
import json, sys
label, c = sys.argv[1], sys.argv[2]
report = json.load(open(f"bench-results/gpu-{label}-c{c}.json"))
ttft = report["ttft_ms"]
itl = report.get("itl_ms") or {}
itl_p50 = itl.get("p50")
itl_s = "-" if itl_p50 is None else f"{itl_p50:.1f}"
print(f"{report['throughput_tok_s']:.2f} {ttft['p50']:.1f} {ttft['p99']:.1f} {itl_s}")
print(
    f"{label} concurrency {c}: tok/s={report['throughput_tok_s']:.2f}  "
    f"TTFT p50={ttft['p50']:.1f} ms  TTFT p99={ttft['p99']:.1f} ms  "
    f"ITL p50={itl_s} ms",
    file=sys.stderr,
)
PY
)"
    BENCH_ROWS+=("${label} ${c} ${row}")
  done
  cleanup
}

mkdir -p bench-results
bench_mode bf16 none
bench_mode int8 int8

if [[ "${KERNEL_STATUS}" != "PASS" || "${CORE_STATUS}" != "PASS" ]]; then
  exit 1
fi
