import modal
import subprocess

MODEL_NAME = "Qwen/Qwen3-32B"

app = modal.App("qwen3-32b")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="H200:1",
    timeout=20 * 60,
    scaledown_window=5 * 60,
    min_containers=0,
    max_containers=2,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=20 * 60, requires_proxy_auth=True)
def serve():
    cmd = [
        "vllm", "serve", MODEL_NAME,

        "--served-model-name", "qwen/qwen3-32b",
        "--tensor-parallel-size", "1",

        "--dtype", "float16",
        "--quantization", "fp8",

        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.92",

        "--attention-backend", "flashinfer",

        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "8192",
        "--max-num-seqs", "16",

        "--no-enforce-eager",
        "--async-scheduling",

        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(cmd)


