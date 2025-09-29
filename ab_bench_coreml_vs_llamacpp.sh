#!/usr/bin/env bash
set -euo pipefail

# USAGE:
#   ./ab_bench_coreml_vs_llamacpp.sh \
#     --hf-model microsoft/phi-4 \
#     --gguf-url https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf \
#     [--prompts-file my_prompts.txt]

HF_MODEL=""
GGUF_URL=""
PROMPTS_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-model)     HF_MODEL="$2"; shift 2 ;;
    --gguf-url)     GGUF_URL="$2"; shift 2 ;;
    --prompts-file) PROMPTS_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$HF_MODEL" || -z "$GGUF_URL" ]]; then
  echo "Usage: $0 --hf-model <hf_repo_id> --gguf-url <direct .gguf url> [--prompts-file <file>]"
  exit 1
fi

echo "[deps] Ensuring Xcode tools and Homebrew build deps..."
xcode-select --install >/dev/null 2>&1 || true
brew install cmake >/dev/null 2>&1 || true

WORKDIR="$(mktemp -d -t abbench-XXXXXXXX)"
echo "[workdir] $WORKDIR"
pushd "$WORKDIR" >/dev/null

echo "[venv] Creating ephemeral virtualenv..."
/usr/bin/env python3 -m venv .venv
source .venv/bin/activate
VENV_PY="$(/usr/bin/env python3 -c 'import sys,os; print(sys.executable)')"
python3 -m pip install -U pip setuptools wheel

echo "[pip] Installing Python deps..."
python3 -m pip install \
  coremltools \
  "huggingface_hub>=0.23" \
  "requests>=2.31" \
  "psutil>=5.9.8" \
  "tabulate>=0.9.0" \
  "rich>=13.7.0" \
  "git+https://github.com/huggingface/exporters.git"

# -------- 1) Export Core ML (.mlpackage) --------
echo "[export] Exporting Core ML package for: $HF_MODEL"
OUT_DIR="$WORKDIR/coreml_out"
mkdir -p "$OUT_DIR"

# Pass args to Python (no env reliance, no pyenv)
python3 - "$HF_MODEL" "$OUT_DIR" <<'PY'
import sys, subprocess
model = sys.argv[1]
out   = sys.argv[2]
cmd = [sys.executable, "-m", "exporters.coreml",
       "--model", model,
       "--feature", "causal-lm-with-past",
       "--quantize", "float16",
       out]
print("[export] Running:", " ".join(cmd))
subprocess.check_call(cmd)
PY

MLPKG=$(ls -d "$OUT_DIR"/*.mlpackage "$OUT_DIR"/*.mlmodel 2>/dev/null | head -n 1 || true)
if [[ -z "$MLPKG" ]]; then
  echo "[export] ERROR: No Core ML package produced."
  exit 1
fi
echo "[export] Core ML model: $MLPKG"

# -------- 2) Start Core ML OpenAI-compatible server (Swift) --------
echo "[server-coreml] Cloning Swift server..."
git clone --depth 1 https://github.com/gety-ai/apple-on-device-openai.git server_coreml >/dev/null

export MODEL_MLPACKAGE="$MLPKG"
export COREML_PORT=8000

echo "[server-coreml] Building & running (Release preferred)…"
pushd server_coreml >/dev/null
( swift build -c release >/dev/null 2>&1 || swift build >/dev/null 2>&1 )
( swift run -c release || swift run ) >/dev/null 2>&1 &
COREML_PID=$!
popd >/dev/null
sleep 5
echo "[server-coreml] PID: $COREML_PID (http://127.0.0.1:8000/v1)"

# -------- 3) Start llama.cpp OpenAI server with your GGUF --------
echo "[server-llama] Cloning & building llama.cpp..."
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git llama.cpp >/dev/null
pushd llama.cpp >/dev/null
make -j >/dev/null
popd >/dev/null

echo "[server-llama] Downloading GGUF…"
GGUF_FILE="$WORKDIR/model.gguf"
curl -L "$GGUF_URL" -o "$GGUF_FILE"

LLAMA_PORT=8080
echo "[server-llama] Starting server on :$LLAMA_PORT"
( "$WORKDIR/llama.cpp/server" -m "$GGUF_FILE" --port "$LLAMA_PORT" --host 127.0.0.1 -c 8192 -fa ) >/dev/null 2>&1 &
LLAMA_PID=$!
sleep 5
echo "[server-llama] PID: $LLAMA_PID (http://127.0.0.1:8080/v1)"

# -------- 4) Write the Python harness --------
cat > bench_harness.py <<'PY'
import os, sys, time, json, psutil, subprocess, requests
from tabulate import tabulate

def find_pid_on_port(port:int):
    try:
        out = subprocess.check_output(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"], text=True)
        for ln in out.splitlines():
            if ln.startswith("p"):
                return int(ln[1:])
    except Exception:
        return None

def rss_mib(pid):
    try:
        p = psutil.Process(pid)
        return round(p.memory_info().rss / (1024*1024), 1)
    except Exception:
        return None

def openai_stream(base, apikey, model, prompt, timeout=600):
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {apikey}", "Content-Type": "application/json"}
    payload = {"model": model, "messages":[{"role":"user","content":prompt}], "temperature":0.2, "stream":True}
    t0 = time.perf_counter()
    ttft = None
    gen_tokens_est = 0
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            delta = obj.get("choices",[{}])[0].get("delta",{}).get("content","")
            gen_tokens_est += max(len(delta.split()), 0)
    t1 = time.perf_counter()

    payload2 = {"model": model, "messages":[{"role":"user","content":prompt}], "temperature":0.2, "stream":False}
    try:
        r2 = requests.post(url, headers=headers, json=payload2, timeout=timeout)
        usage = r2.json().get("usage", {})
        exact = usage.get("completion_tokens")
    except Exception:
        exact = None
    tokens = exact or gen_tokens_est or 1
    total = t1 - t0
    decode_time = total - (ttft or 0.0)
    tps = tokens / max(decode_time, 1e-6)
    return dict(
        time_to_first_token_seconds=round(ttft or total, 4),
        tokens_generated=int(tokens),
        generation_tokens_per_second=round(tps, 2),
        total_response_time_seconds=round(total, 4),
    )

def run_suite(tag, base, apikey, model, port, prompts):
    pid = find_pid_on_port(port)
    before = rss_mib(pid) if pid else None
    rows = []
    for p in prompts:
        m = openai_stream(base, apikey, model, p)
        after = rss_mib(pid) if pid else None
        rows.append({
            "target_engine": tag,
            "prompt_length_characters": len(p),
            "time_to_first_token_seconds": m["time_to_first_token_seconds"],
            "generation_tokens_per_second": m["generation_tokens_per_second"],
            "total_response_time_seconds": m["total_response_time_seconds"],
            "tokens_generated": m["tokens_generated"],
            "process_memory_before_mebibytes": before,
            "process_memory_after_mebibytes": after,
        })
    return rows

def main():
    coreml_base = "http://127.0.0.1:8000/v1"
    llamacpp_base = "http://127.0.0.1:8080/v1"
    coreml_name = os.getenv("COREML_MODEL_NAME","CoreMLModel")
    llamacpp_name = os.getenv("LLAMACPP_MODEL_NAME","GGUFModel")

    prompts = [
        "Summarize in one sentence: Finland is a Nordic country with strong education and a vibrant tech sector.",
        "Explain, step by step, how to implement a token bucket rate limiter in Python.",
        "Translate to Finnish: 'The quick brown fox jumps over the lazy dog.'",
        "Return a valid JSON object with keys: id (integer), name (string), email (string). Keep it minimal."
    ]
    pf = os.getenv("PROMPTS_FILE")
    if pf and os.path.exists(pf):
        with open(pf,"r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            if lines: prompts = lines

    rows = []
    rows += run_suite("coreml_server",  coreml_base, "local", coreml_name, 8000, prompts)
    rows += run_suite("llama_cpp_metal", llamacpp_base,"local", llamacpp_name,8080, prompts)

    from tabulate import tabulate
    print(tabulate(rows, headers="keys", tablefmt="github"))

    import csv
    with open("benchmark_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    with open("benchmark_results.md","w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(tabulate(rows, headers="keys", tablefmt="github"))
        f.write("\n")

if __name__=="__main__":
    main()
PY

export COREML_MODEL_NAME="$(basename "$MLPKG")"
export LLAMACPP_MODEL_NAME="local-gguf"
if [[ -n "${PROMPTS_FILE}" ]]; then export PROMPTS_FILE="$PROMPTS_FILE"; fi

python3 bench_harness.py

echo
echo "Saved results:"
echo "- $WORKDIR/benchmark_results.csv"
echo "- $WORKDIR/benchmark_results.md"
echo

echo "[cleanup] Stopping servers and removing temp files..."
kill "$COREML_PID" >/dev/null 2>&1 || true
kill "$LLAMA_PID"  >/dev/null 2>&1 || true
deactivate || true
popd >/dev/null
rm -rf "$WORKDIR"
echo "[done] A/B complete."
