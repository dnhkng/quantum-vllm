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

ROOT = Path("/home/grace/vllm-build")
VLLM = ROOT / ".venv/bin/vllm"
OUT = Path("/home/grace/dsv4-full-mtp-benchmark-20260607")
MODEL = "/mnt/storage8tb_2/deepseek-ai/DeepSeek-V4-Flash-W4A16-FP8-MTP"
PORT = 8013
PROMPT_LEN = 8192
OUTPUT_LEN = 1024
NUM_PROMPTS = 3
MAX_CONCURRENCY = 1


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


def serve_cmd(mtp):
    cmd = [
        str(VLLM),
        "serve",
        MODEL,
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
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "1",
        "--block-size",
        "256",
        "--gpu-memory-utilization",
        "0.97",
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
    ]
    if mtp:
        cmd += ["--spec-method", "mtp", "--spec-tokens", str(mtp)]
    return cmd


def bench_cmd(run_dir):
    return [
        str(VLLM),
        "bench",
        "serve",
        "--base-url",
        f"http://127.0.0.1:{PORT}",
        "--model",
        "dsv4",
        "--tokenizer",
        MODEL,
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
        f"bench_p{PROMPT_LEN}_o{OUTPUT_LEN}_n{NUM_PROMPTS}_c{MAX_CONCURRENCY}.json",
    ]


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
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NCCL_P2P_DISABLE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    for mtp in [0, 1, 2, 3, 4]:
        run_dir = OUT / f"canada_mtp{mtp}"
        done_marker = run_dir / "DONE"
        if done_marker.exists():
            print(f"Skipping completed canada_mtp{mtp}", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "serve_cmd_old_vllm.json").write_text(
            json.dumps(serve_cmd(mtp), indent=2) + "\n"
        )
        (run_dir / "bench_cmd_old_vllm.json").write_text(
            json.dumps(bench_cmd(run_dir), indent=2) + "\n"
        )
        wait_gpus_clear()
        log_path = run_dir / "server_old_vllm.log"
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                serve_cmd(mtp),
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        try:
            wait_healthy(proc, log_path)
            with (run_dir / "bench_old_vllm.log").open("w") as bench_log:
                subprocess.run(
                    bench_cmd(run_dir),
                    cwd=ROOT,
                    env=env,
                    check=True,
                    stdout=bench_log,
                    stderr=subprocess.STDOUT,
                )
            done_marker.write_text("ok\n")
        except Exception as exc:
            (run_dir / "FAILED").write_text(str(exc) + "\n")
            print(f"FAILED canada_mtp{mtp}: {exc}", file=sys.stderr, flush=True)
        finally:
            stop_server(proc)
            wait_gpus_clear()
            summarize()


if __name__ == "__main__":
    main()
