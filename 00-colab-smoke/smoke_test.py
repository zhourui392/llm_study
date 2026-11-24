"""Minimal PyTorch smoke test for verifying local (MPS/CPU) and Colab (CUDA) execution."""

from __future__ import annotations

import argparse
import platform
import time

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def print_env_info(device: torch.device) -> None:
    print("=== Environment ===", flush=True)
    print(f"Host: {platform.node()} ({platform.system()} {platform.machine()})", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Selected device: {device}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU: {props.name} ({vram_gb:.2f} GiB VRAM)", flush=True)
        print(f"CUDA version: {torch.version.cuda}", flush=True)
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}", flush=True)


def run_matmul_benchmark(device: torch.device, size: int, iters: int) -> None:
    print("\n=== Matmul Smoke Benchmark ===", flush=True)
    a = torch.randn((size, size), device=device, dtype=torch.float32)
    b = torch.randn((size, size), device=device, dtype=torch.float32)

    # Warmup
    _ = a @ b
    sync_device(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    sync_device(device)
    elapsed_s = (time.perf_counter() - t0) / iters

    flops = 2.0 * (size ** 3)
    tflops = (flops / elapsed_s) / 1e12
    checksum = c[0, 0].item()
    print(
        f"FP32 {size}x{size} matmul: {elapsed_s * 1000:.2f} ms/iter "
        f"({tflops:.2f} TFLOPS, sample={checksum:.4f})",
        flush=True,
    )


def run_autograd_check(device: torch.device) -> None:
    print("\n=== Autograd & Optimizer Check ===", flush=True)
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, 128),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(32, 128, device=device)
    y = torch.randn(32, 128, device=device)

    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        sync_device(device)
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(v)) for v in losses), f"Non-finite loss: {losses}"
    assert losses[-1] < losses[0], f"Expected loss to decrease, got {losses}"
    print(f"MSE loss over 5 steps: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab / Local PyTorch GPU smoke test")
    parser.add_argument("--size", type=int, default=2048, help="Square matrix dimension for matmul")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    device = get_device()
    print_env_info(device)
    run_matmul_benchmark(device, size=args.size, iters=args.iters)
    run_autograd_check(device)
    print("\nSMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
