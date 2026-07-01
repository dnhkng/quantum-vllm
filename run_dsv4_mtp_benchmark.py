#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/home/grace/vllm-build/vllm")
OUT = Path("/home/grace/dsv4-full-mtp-benchmark-20260607")
PORT = 8012
PROMPT_LEN = 8192
OUTPUT_LEN = 1024
NUM_PROMPTS = 3
MAX_CONCURRENCY = 1

MODELS = [
    {
        "name": "official",
        "path": "/mnt/storage8tb_2/deepseek-ai/DeepSeek-V4-Flash",
        "moe_backend": "triton_unfused",
        "gpu_memory_utilization": "0.95",
    },
    {
        "name": "canada",
        "path": "/home/grace/dsv4-full-mtp-benchmark-20260607/canada_model_view",
        "moe_backend": "auto",
        "gpu_memory_utilization": "0.97",
    },
]

MTP_LEVELS = [0, 1, 2, 3, 4]


def run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=True, **kwargs)


def health_url():
    return f"http://127.0.0.1:{PORT}/health"


def wait_healthy(proc, log_path, timeout=900):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            tail = (
                log_path.read_text(errors="replace")[-8000:]
                if log_path.exists()
                else ""
            )
            raise RuntimeError(f"server exited with {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(health_url(), timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"server did not become healthy after {timeout}s")


def stop_server(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def wait_gpus_clear(timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if not out:
            return
        time.sleep(5)
    raise TimeoutError("GPU compute processes did not exit")


def serve_cmd(model, mtp_level, run_dir):
    cmd = [
        str(ROOT / ".venv/bin/vllm"),
        "serve",
        model["path"],
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--served-model-name",
        "dsv4",
        "--trust-remote-code",
        "--tokenizer-mode",
        "deepseek_v4",
        "--distributed-executor-backend",
        "mp",
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "1",
        "--block-size",
        "256",
        "--gpu-memory-utilization",
        model["gpu_memory_utilization"],
        "--kv-cache-dtype",
        "fp8",
        "--disable-custom-all-reduce",
        "--safetensors-load-strategy",
        "prefetch",
        "--safetensors-prefetch-num-threads",
        "4",
        "--safetensors-prefetch-block-size",
        "67108864",
        "--generation-config",
        "vllm",
        "--moe-backend",
        model["moe_backend"],
    ]
    if mtp_level:
        cmd += ["--spec-method", "mtp", "--spec-tokens", str(mtp_level)]
    if "hf_config_path" in model:
        cmd += ["--hf-config-path", model["hf_config_path"]]
    (run_dir / "serve_cmd.json").write_text(json.dumps(cmd, indent=2) + "\n")
    return cmd


def bench_cmd(model, mtp_level, run_dir):
    filename = (
        f"bench_p{PROMPT_LEN}_o{OUTPUT_LEN}_n{NUM_PROMPTS}_"
        f"c{MAX_CONCURRENCY}.json"
    )
    cmd = [
        str(ROOT / ".venv/bin/vllm"),
        "bench",
        "serve",
        "--base-url",
        f"http://127.0.0.1:{PORT}",
        "--model",
        "dsv4",
        "--tokenizer",
        model["path"],
        "--trust-remote-code",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(PROMPT_LEN),
        "--random-output-len",
        str(OUTPUT_LEN),
        "--num-prompts",
        str(NUM_PROMPTS),
        "--max-concurrency",
        str(MAX_CONCURRENCY),
        "--temperature",
        "0",
        "--save-result",
        "--result-dir",
        str(run_dir),
        "--result-filename",
        filename,
    ]
    (run_dir / "bench_cmd.json").write_text(json.dumps(cmd, indent=2) + "\n")
    return cmd


def summarize():
    rows = []
    for path in sorted(OUT.glob("*/bench_p*.json")):
        data = json.loads(path.read_text())
        parts = path.parent.name.split("_")
        rows.append(
            {
                "run": path.parent.name,
                "model": parts[0],
                "mtp": parts[1].replace("mtp", ""),
                "completed": data.get("completed"),
                "failed": data.get("failed"),
                "output_tps": data.get("output_throughput"),
                "total_tps": data.get("total_token_throughput"),
                "mean_ttft_ms": data.get("mean_ttft_ms"),
                "mean_tpot_ms": data.get("mean_tpot_ms"),
                "mean_itl_ms": data.get("mean_itl_ms"),
                "duration": data.get("duration"),
            }
        )
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NCCL_P2P_DISABLE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "PYTHONPATH": f"{ROOT / 'vllm/third_party'}:{env.get('PYTHONPATH', '')}",
        }
    )
    for model in MODELS:
        for mtp in MTP_LEVELS:
            run_name = f"{model['name']}_mtp{mtp}"
            run_dir = OUT / run_name
            done_marker = run_dir / "DONE"
            if done_marker.exists():
                print(f"Skipping completed {run_name}", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "mtp_level": mtp,
                        "prompt_len": PROMPT_LEN,
                        "output_len": OUTPUT_LEN,
                        "num_prompts": NUM_PROMPTS,
                        "max_concurrency": MAX_CONCURRENCY,
                    },
                    indent=2,
                )
                + "\n"
            )
            wait_gpus_clear()
            log_path = run_dir / "server.log"
            with log_path.open("w") as log:
                proc = subprocess.Popen(
                    serve_cmd(model, mtp, run_dir),
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            try:
                wait_healthy(proc, log_path)
                with (run_dir / "bench.log").open("w") as bench_log:
                    run(
                        bench_cmd(model, mtp, run_dir),
                        env=env,
                        stdout=bench_log,
                        stderr=subprocess.STDOUT,
                    )
                done_marker.write_text("ok\n")
            except Exception as exc:
                (run_dir / "FAILED").write_text(str(exc) + "\n")
                print(f"FAILED {run_name}: {exc}", file=sys.stderr, flush=True)
            finally:
                stop_server(proc)
                wait_gpus_clear()
                summarize()


if __name__ == "__main__":
    main()
