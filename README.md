# ab-bench-coreml-vs-llamacpp
test tool on macos to compare the real performance difference between core lm acceleration vs llama.cpp on apple hardware

How to run:

chmod +x ab_bench_coreml_vs_llamacpp.py

# Case A: you have a direct GGUF URL (recommended)
./ab_bench_coreml_vs_llamacpp.py \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --quant Q4_K_M \
  --gguf-url https://huggingface.co/ORG/MODEL/resolve/main/MODEL.Q4_K_M.gguf

# Case B: you only know the HF *page*; build a raw URL like this:
# https://huggingface.co/ORG/MODEL/resolve/main/<filename>.gguf
./ab_bench_coreml_vs_llamacpp.py \
  --hf-model Qwen/Qwen2.5-3B-Instruct \
  --quant Q5_K_M \
  --gguf-url https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct.Q5_K_M.gguf

# Optional: custom prompts file (one prompt per line)
./ab_bench_coreml_vs_llamacpp.py \
  --hf-model mistralai/Mistral-7B-Instruct-v0.3 \
  --quant Q4_K_S \
  --gguf-url https://huggingface.co/.../mistral-7b-instruct.Q4_K_S.gguf \
  --prompts-file my_prompts.txt
