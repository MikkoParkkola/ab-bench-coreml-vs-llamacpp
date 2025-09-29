#!/usr/bin/env python3
"""
A/B benchmark: Core ML–accelerated Apple path (via MLC LLM) vs llama.cpp (Metal/CPU) on macOS.

This script:
  1) Takes a Hugging Face model id and a GGUF quantization target (e.g., Q4_K_M).
  2) Builds an Apple-targeted package with MLC LLM (uses ANE + GPU + CPU under the hood) and launches an OpenAI-compatible server.
  3) Prepares a llama.cpp server by either downloading a provided GGUF or converting from the HF model to GGUF + quantizing, then launches an OpenAI-compatible server.
  4) Runs a standardized benchmark (multiple prompts) to measure:
       - Time to first token (seconds)
       - Tokens per second during generation
       - Total response time (seconds)
       - Process memory usage before and after (MiB)
     Results are saved to CSV and Markdown, and printed to the console.
  5) Cleans up all temporary files, folders, and downloaded models.

USAGE (examples):
  python ab_bench_coreml_vs_llamacpp.py       --hf-model meta-llama/Llama-3.1-8B-Instruct       --quant Q4_K_M

  python ab_bench_coreml_vs_llamacpp.py       --hf-model Qwen/Qwen2.5-3B-Instruct       --quant Q5_K_M       --gguf-url https://huggingface.co/.../qwen-2.5-3b-instruct.Q5_K_M.gguf

NOTES
- Requires: macOS with Apple Silicon, Xcode command line tools, Python 3.10+, Homebrew recommended.
- This script will try to install Python deps via pip (mlc-ai, huggingface_hub, requests, psutil, tabulate, rich) in the current environment.
- It will also build llama.cpp from source (via make) if not present in the temp workdir.
- MLC LLM uses the Apple backend ("--target apple") which can schedule ops to ANE, GPU (Metal), and CPU for best performance.
- llama.cpp runs with Metal GPU (if available) and CPU; ANE is not used by llama.cpp.

DISCLAIMER
- The ANE/GPU/CPU scheduling details are controlled by the underlying runtimes (MLC/Apple Core ML path). Not all models/ops will hit ANE.
- Conversion of HF models to GGUF may fail for some architectures. If so, provide a direct --gguf-url for the same model family.
"""

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# Runtime deps we install/require at run-time
PY_DEPS = [
    "mlc-ai>=0.14.0",          # if unavailable, try 'mlc-ai-nightly' manually
    "huggingface_hub>=0.23.0",
    "requests>=2.31.0",
    "psutil>=5.9.8",
    "tabulate>=0.9.0",
    "rich>=13.7.0",
]

# Simple prompts for a balanced read of prefill vs decode
DEFAULT_PROMPTS = [
    "Summarize in one sentence: Finland is a Nordic country with strong education and a vibrant tech sector.",
    "Explain, step by step, how to implement a token bucket rate limiter in Python.",
    "Translate to Finnish: 'The quick brown fox jumps over the lazy dog.'",
    "Return a valid JSON object with keys: id (integer), name (string), email (string). Keep it minimal."
]

def sh(cmd, cwd=None, env=None, capture=False, check=True):
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, shell=shell,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True, check=False
    )
    if check and proc.returncode != 0:
        out = proc.stdout if capture else ""
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{out}")
    return proc.stdout if capture else ""

def ensure_pip_deps():
    """Install Python dependencies into the current environment if missing."""
    import importlib
    to_install = []
    for spec in PY_DEPS:
        base = spec.split(">=")[0].split("==")[0].split("[")[0]
        try:
            importlib.import_module(base.replace("-", "_"))
        except Exception:
            to_install.append(spec)
    if to_install:
        print(f"[setup] Installing Python deps: {to_install}")
        sh([sys.executable, "-m", "pip", "install", "-U"] + to_install, check=True)

def find_free_port(preferred):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

def start_mlc_openai_server(work, hf_model_id, mlc_model_alias, port):
    """Use MLC LLM to download/build Apple-targeted package and launch an OpenAI-compatible server."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Download
    print(f"[mlc] Downloading: {hf_model_id}")
    sh(f"mlc_llm download --model {hf_model_id}", cwd=work, env=env, check=True)
    # Build for Apple (uses Core ML / Metal backends under the hood)
    print("[mlc] Building Apple-targeted model package (this can take a while)…")
    sh(f"mlc_llm build --model {hf_model_id} --target apple --model-alias {mlc_model_alias}", cwd=work, env=env, check=True)
    # Serve (OpenAI-compatible API)
    print("[mlc] Launching OpenAI-compatible server…")
    log = open(Path(work)/"mlc_server.log", "w")
    proc = subprocess.Popen(
        ["mlc_llm", "serve", "--model", mlc_model_alias, "--api", "openai", "--host", "127.0.0.1", "--port", str(port)],
        cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
    )
    time.sleep(5.0)
    if proc.poll() is not None:
        raise RuntimeError("MLC server failed to start; see mlc_server.log")
    return proc

def prepare_llamacpp(work):
    """Clone and build llama.cpp if not already done in work dir."""
    repo = Path(work)/"llama.cpp"
    if not repo.exists():
        print("[llama.cpp] Cloning…")
        sh(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(repo)], check=True)
    print("[llama.cpp] Building…")
    sh("make -j", cwd=repo, check=True)
    return repo

def get_or_make_gguf(repo, work, hf_model_id, gguf_url, quant, ctx_len):
    """Return path to a GGUF file (downloaded or converted + quantized)."""
    gguf_dir = Path(work)/"gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    if gguf_url:
        target = gguf_dir / Path(gguf_url.split("?")[0]).name
        if not target.exists():
            print(f"[gguf] Downloading provided GGUF: {gguf_url}")
            sh(["curl", "-L", gguf_url, "-o", str(target)], check=True)
        return str(target)

    # Convert HF → GGUF
    print("[gguf] Converting HF weights to GGUF…")
    # Pull weights via huggingface_hub
    from huggingface_hub import snapshot_download
    hf_cache = snapshot_download(hf_model_id, allow_patterns=["*.json","*.safetensors","*.model","*.bin","*tokenizer*","*merges*","*vocab*"])
    py = Path(repo)/"convert-hf-to-gguf.py"
    out_base = gguf_dir/"model-f16.gguf"
    cmd = [sys.executable, str(py), "--outtype", "f16", "--outfile", str(out_base), hf_cache]
    sh(cmd, check=True)

    # Quantize
    print(f"[gguf] Quantizing to {quant}…")
    quant_bin = Path(repo)/"quantize"
    q_out = gguf_dir/f"model.{quant}.gguf"
    sh([str(quant_bin), str(out_base), str(q_out), quant], check=True)

    return str(q_out)

def start_llamacpp_server(repo, gguf_path, port, ctx_len):
    """Start llama.cpp OpenAI-compatible server on given port."""
    print("[llama.cpp] Starting OpenAI-compatible server…")
    log = open(Path(repo).parent/"llama_server.log", "w")
    cmd = [
        str(Path(repo)/"server"),
        "-m", gguf_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "-c", str(ctx_len),
        "-fa"  # try flash-attn path if supported; harmless otherwise
    ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    time.sleep(5.0)
    if proc.poll() is not None:
        raise RuntimeError("llama.cpp server failed to start; see llama_server.log")
    return proc

def find_pid_on_port(port:int):
    try:
        out = sh(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"], capture=True, check=False)
        for line in out.splitlines():
            if line.startswith("p"):
                return int(line[1:])
    except Exception:
        pass
    return None

def rss_mib(pid):
    try:
        import psutil
        p = psutil.Process(pid)
        return round(p.memory_info().rss / (1024*1024), 1)
    except Exception:
        return None

def openai_stream_chat(base, apikey, model, prompt, timeout=600):
    import requests, json
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {apikey}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": True
    }
    t0 = time.perf_counter()
    ttft = None
    gen_tokens_est = 0
    last_obj = None
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
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
            gen_tokens_est += max(len(delta.split()), 0)
            last_obj = obj
    t_end = time.perf_counter()

    # non-stream call to retrieve accurate usage if available
    payload2 = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False
    }
    try:
        import requests, json
        r2 = requests.post(url, headers=headers, json=payload2, timeout=timeout)
        usage = r2.json().get("usage", {})
        out_tokens = usage.get("completion_tokens")
    except Exception:
        out_tokens = None
    gen_tokens = out_tokens or gen_tokens_est or 1
    total_time = t_end - t0
    decode_time = total_time - (ttft or 0.0)
    tokens_per_second = gen_tokens / max(decode_time, 1e-6)
    return {
        "time_to_first_token_seconds": round(ttft or total_time, 4),
        "tokens_generated": int(gen_tokens),
        "generation_tokens_per_second": round(tokens_per_second, 2),
        "total_response_time_seconds": round(total_time, 4)
    }

def run_suite(tag, base, apikey, model, port, prompts):
    pid = find_pid_on_port(port)
    before = rss_mib(pid) if pid else None
    rows = []
    for p in prompts:
        m = openai_stream_chat(base, apikey, model, p)
        after = rss_mib(pid) if pid else None
        rows.append({
            "target_engine": tag,
            "prompt_length_characters": len(p),
            "time_to_first_token_seconds": m["time_to_first_token_seconds"],
            "generation_tokens_per_second": m["generation_tokens_per_second"],
            "total_response_time_seconds": m["total_response_time_seconds"],
            "tokens_generated": m["tokens_generated"],
            "process_memory_before_mebibytes": before,
            "process_memory_after_mebibytes": after
        })
    return rows

def print_table(rows):
    from tabulate import tabulate
    print("\n=== Benchmark Results ===\n")
    print(tabulate(rows, headers="keys", tablefmt="github"))

def write_outputs(rows, outdir):
    csv_path = Path(outdir)/"benchmark_results.csv"
    md_path = Path(outdir)/"benchmark_results.md"
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from tabulate import tabulate
    with open(md_path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(tabulate(rows, headers="keys", tablefmt="github"))
        f.write("\n")
    return str(csv_path), str(md_path)

def main():
    parser = argparse.ArgumentParser(
        description="A/B benchmark Core ML (via MLC) vs llama.cpp on macOS with Apple Silicon."
    )
    parser.add_argument("--hf-model", required=True, help="Hugging Face model id (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--quant", required=True, help="GGUF quantization (e.g., Q4_K_M, Q5_K_M)")
    parser.add_argument("--context-length", type=int, default=8192, help="Context window for llama.cpp")
    parser.add_argument("--mlc-port", type=int, default=8000, help="Port for MLC OpenAI server")
    parser.add_argument("--llama-port", type=int, default=8080, help="Port for llama.cpp server")
    parser.add_argument("--gguf-url", default=None, help="Optional direct URL to a prebuilt GGUF for the same model family")
    parser.add_argument("--prompts-file", default=None, help="Optional path to a text file with one prompt per line")
    parser.add_argument("--keep-artifacts", action="store_true", help="Do NOT delete temp files at the end")
    args = parser.parse_args()

    # Basic dependency check / install
    ensure_pip_deps()

    # Prepare prompts
    prompts = DEFAULT_PROMPTS
    if args.prompts_file and Path(args.prompts_file).exists():
        with open(args.prompts_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if lines:
                prompts = lines

    # Work directory
    work = tempfile.mkdtemp(prefix="ab-bench-")
    print(f"[workdir] {work}")

    mlc_proc = None
    llama_proc = None

    try:
        # 1) Start MLC OpenAI server (Apple backend -> ANE/GPU/CPU scheduling under the hood)
        mlc_alias = "ab_bench_model"
        mlc_port = find_free_port(args.mlc_port)
        mlc_proc = start_mlc_openai_server(work, args.hf_model, mlc_alias, mlc_port)
        coreml_target = dict(
            tag="coreml_via_mlc",
            base=f"http://127.0.0.1:{mlc_port}/v1",
            apikey="local",
            model=mlc_alias,
            port=mlc_port
        )

        # 2) Prepare llama.cpp (Metal/CPU) server
        repo = prepare_llamacpp(work)
        gguf_path = get_or_make_gguf(repo, work, args.hf_model, args.gguf_url, args.quant, args.context_length)
        llama_port = find_free_port(args.llama_port)
        llama_proc = start_llamacpp_server(repo, gguf_path, llama_port, args.context_length)
        llamacpp_target = dict(
            tag="llama_cpp_metal_cpu",
            base=f"http://127.0.0.1:{llama_port}/v1",
            apikey="local",
            model="local-gguf",
            port=llama_port
        )

        # 3) Run benchmark suites
        rows = []
        rows += run_suite(**coreml_target, prompts=prompts)
        rows += run_suite(**llamacpp_target, prompts=prompts)

        # 4) Output
        print_table(rows)
        csv_path, md_path = write_outputs(rows, work)
        print(f"\nSaved results:\n- CSV: {csv_path}\n- Markdown: {md_path}\n")

    finally:
        # Shutdown servers
        for proc, name in [(mlc_proc, "MLC"), (llama_proc, "llama.cpp")]:
            if proc and proc.poll() is None:
                print(f"[cleanup] Stopping {name} server…")
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        if args.keep_artifacts:
            print(f"[info] Keeping workdir: {work}")
        else:
            print("[cleanup] Removing temporary files and downloaded models…")
            try:
                shutil.rmtree(work, ignore_errors=True)
            except Exception as e:
                print(f"[warn] Failed to remove workdir: {e}")

if __name__ == "__main__":
    main()
