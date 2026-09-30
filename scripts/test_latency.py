"""
scripts/test_latency.py
Profiles end-to-end policy inference latency and per-submodule timing on GPU/CPU.
Validates the <= 18.0 ms (50 Hz) real-time control constraint in bfloat16 or float32.
"""
import time
import argparse
from typing import Dict, List
import numpy as np
import torch
from configs.poc_config import CONFIG
from models.vga_policy import VGAPolicy

def benchmark_policy_latency(
    device_str: str,
    dtype_str: str = "bfloat16",
    load_pretrained: bool = False,
    num_warmup: int = 25,
    num_trials: int = 100
):
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    
    # Select precision: bfloat16 for RTX 4090 deployment, float32 for CPU
    if device.type == "cuda" and dtype_str == "bfloat16":
        exec_dtype = torch.bfloat16
    elif device.type == "cuda" and dtype_str == "float16":
        exec_dtype = torch.float16
    else:
        exec_dtype = torch.float32

    print("=" * 70)
    print(f">> PROFILING VGA POLICY LATENCY ON: {device} ({exec_dtype})")
    if device.type == "cuda":
        print(f">> Device Name: {torch.cuda.get_device_name(0)}")
    print(f">> Mode: {'Full Pretrained Backbones' if load_pretrained else 'Lightweight Architecture Shell'}")
    print("=" * 70)

    # Initialize policy
    try:
        policy = VGAPolicy(load_pretrained=load_pretrained).to(device=device, dtype=exec_dtype)
    except Exception as e:
        if load_pretrained:
            print(f">> [!] Warning: Failed loading pretrained weights ({e}). Falling back to shell.")
            policy = VGAPolicy(load_pretrained=False).to(device=device, dtype=exec_dtype)
        else:
            raise e

    policy.eval()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        policy.optimize_for_inference()

    # Synthetic observation inputs
    dummy_rgb = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    dummy_t_cam = np.array([0.25, -0.35, 0.45], dtype=np.float32)
    dummy_a_prev = torch.zeros(1, CONFIG.buffer_prefix_len, 6, device=device, dtype=exec_dtype)
    dummy_prompt = "pick up the black bowl and place it on the plate"

    # Pre-calculate rays to simulate production deployment caching
    K = torch.tensor([
        [CONFIG.default_intrinsics["fx"], 0.0, CONFIG.default_intrinsics["cx"]],
        [0.0, CONFIG.default_intrinsics["fy"], CONFIG.default_intrinsics["cy"]],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=exec_dtype)
    R_ident = torch.eye(3, device=device, dtype=exec_dtype)
    cached_rays = policy.ray_rope.compute_centroid_rays(K, R_ident, device=device).to(dtype=exec_dtype)

    print(f"\n>> Running {num_warmup} warm-up iterations...")
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = policy.sample_actions(
                rgb=dummy_rgb,
                t_base_cam=dummy_t_cam,
                a_prev=dummy_a_prev,
                rays=cached_rays,
                prompt=dummy_prompt
            )
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 1. Profile Submodule Breakdown
    print(f">> Profiling submodules over {num_trials} trials...")
    t_prep_list, t_proj_list, t_rope_list, t_dit_list = [], [], [], []

    with torch.inference_mode():
        for _ in range(num_trials):
            # A. Visual Preparation & SigLIP
            t0 = time.perf_counter()
            x_rgb = policy._prepare_rgb(dummy_rgb).to(dtype=exec_dtype)
            if policy.vision_encoder is not None:
                vis_feats = policy.vision_encoder(pixel_values=x_rgb).last_hidden_state
            else:
                vis_feats = torch.randn(1, CONFIG.raw_patches, CONFIG.vis_dim, device=device, dtype=exec_dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_prep_list.append((time.perf_counter() - t0) * 1000.0)

            # B. Space-to-Depth Projector
            t0 = time.perf_counter()
            compressed = policy.projector(vis_feats)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_proj_list.append((time.perf_counter() - t0) * 1000.0)

            # C. Centroid Ray-RoPE
            t0 = time.perf_counter()
            grounded = policy.ray_rope(compressed, torch.from_numpy(dummy_t_cam).to(device=device, dtype=exec_dtype), rays=cached_rays)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_rope_list.append((time.perf_counter() - t0) * 1000.0)

            # D. DiT Action Expert (4-Step Euler ODE Solver)
            t0 = time.perf_counter()
            _ = policy.dit.sample_4step_euler(grounded, dummy_a_prev)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_dit_list.append((time.perf_counter() - t0) * 1000.0)

    # 2. End-to-End Latency Measurement
    print(f">> Benchmarking {num_trials} end-to-end forward passes...")
    e2e_latencies = []

    with torch.inference_mode():
        for _ in range(num_trials):
            t0 = time.perf_counter()
            _ = policy.sample_actions(
                rgb=dummy_rgb,
                t_base_cam=dummy_t_cam,
                a_prev=dummy_a_prev,
                rays=cached_rays,
                prompt=dummy_prompt
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            e2e_latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_e2e = np.mean(e2e_latencies)
    median_e2e = np.median(e2e_latencies)
    p95_e2e = np.percentile(e2e_latencies, 95)
    hz = 1000.0 / mean_e2e

    print("\n" + "-" * 50)
    print(">> SUBMODULE LATENCY BREAKDOWN (Mean)")
    print("-" * 50)
    print(f"Vision Prep + SigLIP:       {np.mean(t_prep_list):.2f} ms")
    print(f"Space-to-Depth Projector:   {np.mean(t_proj_list):.2f} ms")
    print(f"Centroid Ray-RoPE:          {np.mean(t_rope_list):.2f} ms")
    print(f"DiT Action Expert (4-Step): {np.mean(t_dit_list):.2f} ms")
    print("-" * 50)
    print(f"End-to-End Mean Latency:    {mean_e2e:.2f} ms")
    print(f"Median Latency:             {median_e2e:.2f} ms")
    print(f"95th Percentile Latency:    {p95_e2e:.2f} ms")
    print(f"Control Frequency:          {hz:.1f} Hz")
    print("-" * 50)

    target_ceiling = 18.0
    if mean_e2e <= target_ceiling:
        print(f"[✓] PASSED: Mean latency ({mean_e2e:.2f} ms) satisfies the <= {target_ceiling} ms budget (>= 50 Hz).")
    else:
        print(f"[!] WARNING: Mean latency ({mean_e2e:.2f} ms) exceeds the {target_ceiling} ms ceiling.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--load_pretrained", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    benchmark_policy_latency(args.device, args.dtype, args.load_pretrained, args.warmup, args.trials)