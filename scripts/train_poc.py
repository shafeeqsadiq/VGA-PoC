"""
scripts/train_poc.py
Main 20,000-step training loop for VGA PoC policy.
Supports 5-shot and 10-shot demonstration regimes on single GPU with bfloat16 AMP,
gradient clipping, unified optimization over policy + auxiliary VQ-depth heads,
and programmatic ablation overrides for kinematic loss, VQ supervision, and Ray-RoPE.
"""
import os
import sys

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import shutil
import inspect
import torch
from torch.utils.data import DataLoader
from configs.poc_config import CONFIG
from data.dataset import LiberoSpatialDataset
from models.vga_policy import VGAPolicy
from losses.drift_injection import SyntheticDriftInjector
from losses.composite_loss import CompositeVGALoss


def train_vga(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("=" * 70)
    print(f">> INITIALIZING VGA TRAINING REGIME: {args.split.upper()}")
    print(f">> Target Device: {device}")
    print(f">> Steps: {args.steps} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print(f">> Ablation Config: lambda_kin={args.lambda_kin} | lambda_vq={args.lambda_vq} | disable_ray_rope={args.disable_ray_rope}")
    print("=" * 70)

    # 1. Resolve and Validate Paths
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        # Support both '5_shot' and '5shot' directory conventions
        alt_dir = data_dir.replace("_shot", "shot") if "_shot" in data_dir else data_dir.replace("shot", "_shot")
        if os.path.exists(alt_dir):
            data_dir = alt_dir
        else:
            raise FileNotFoundError(f"Demonstration data not found at '{data_dir}' or '{alt_dir}'.")

    if not os.path.exists(args.stats_path):
        raise FileNotFoundError(f"Missing action stats at '{args.stats_path}'. Run compute_action_stats.py first.")

    with open(args.stats_path, "r", encoding="utf-8") as f:
        action_stats = json.load(f)

    # 2. Build Dataset & DataLoader
    print(f">> Ingesting demonstration data from: {data_dir}")
    dataset = LiberoSpatialDataset.from_disk(
        data_path=data_dir,
        chunk_horizon=CONFIG.chunk_horizon,
        prefix_len=CONFIG.buffer_prefix_len,
        action_stats=action_stats
    )

    batch_size = min(args.batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(dataset) >= batch_size)
    )
    print(f">> Total training trajectory chunks: {len(dataset)} (Batch Size: {batch_size})")

    # 3. Model & Loss Setup (Honoring Ablation Flags)
    load_pretrained = not args.no_pretrained

    # Instantiate VGAPolicy with fallback handling for disable_ray_rope argument
    policy_kwargs = {"load_pretrained": load_pretrained}
    sig = inspect.signature(VGAPolicy.__init__)
    if "disable_ray_rope" in sig.parameters:
        policy_kwargs["disable_ray_rope"] = args.disable_ray_rope

    policy = VGAPolicy(**policy_kwargs).to(device)

    # If policy has direct module attributes for ray_rope ablation, set toggle
    if hasattr(policy, "disable_ray_rope"):
        policy.disable_ray_rope = args.disable_ray_rope
    elif hasattr(policy, "ray_rope") and hasattr(policy.ray_rope, "disable_ray_rope"):
        policy.ray_rope.disable_ray_rope = args.disable_ray_rope

    policy.train()

    # Pass command-line ablation overrides into composite loss
    loss_fn = CompositeVGALoss(
        sigma_pos_sq=action_stats.get("sigma_pos_sq", 1.0),
        sigma_rot_sq=action_stats.get("sigma_rot_sq", 1.0),
        w_rot=CONFIG.w_rot,
        beta_jerk=CONFIG.beta_jerk,
        lambda_kin_max=args.lambda_kin,
        kinematic_warmup_steps=CONFIG.kinematic_warmup_steps,
        lambda_vq=args.lambda_vq
    ).to(device)

    injector = SyntheticDriftInjector(
        max_trans_drift=CONFIG.max_trans_drift,
        max_rot_drift_deg=CONFIG.max_rot_drift_deg,
        ramp_steps=CONFIG.drift_ramp_steps
    ).to(device)

    # 4. Joint Optimization Parameters (Policy LoRA + Projector + DiT + Auxiliary Loss Head)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    trainable_params += [p for p in loss_fn.parameters() if p.requires_grad]

    print(f">> Active trainable parameter tensors: {len(trainable_params)}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    # 5. Mixed Precision Configuration
    use_amp = (device.type == "cuda" and not args.no_amp)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    sigma_d = torch.as_tensor(action_stats["sigma"], device=device, dtype=torch.float32)
    mu_d = torch.as_tensor(action_stats["mu"], device=device, dtype=torch.float32)

    os.makedirs(args.output_dir, exist_ok=True)
    # Save a copy of action normalizers in checkpoint directory
    shutil.copy(args.stats_path, os.path.join(args.output_dir, "action_stats.json"))

    step = 0
    data_iter = iter(loader)

    print(f"\n>> Commencing Optimization (AMP: {use_amp}, Dtype: {amp_dtype})...")
    while step < args.steps:
        step += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        rgb = batch["rgb"].to(device)
        t_base_cam = batch["t_base_cam"].to(device)
        x_1_norm = batch["actions_norm"].to(device)
        a_prev = batch["a_prev"].to(device)
        z_star = batch["z_star"].to(device)
        prompt = batch["prompt"]

        # 1. Apply synthetic drift to simulate deployment error accumulation
        drifted_chunk, drift_norm = injector(x_1_norm, return_drift_norm=True)

        # 2. Linear Flow Matching interpolation: x_tau = (1 - tau) * x_0 + tau * x_1
        B = rgb.shape[0]
        tau = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1_norm)
        tau_bc = tau.view(-1, 1, 1)
        x_tau = (1.0 - tau_bc) * x_0 + tau_bc * drifted_chunk

        # 3. Forward Pass & Loss Calculation under Autocast
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=use_amp):
            v_pred, vis_tokens = policy(
                rgb=rgb,
                t_base_cam=t_base_cam,
                x_t=x_tau,
                tau=tau,
                a_prev=a_prev,
                prompt=prompt
            )

            # Reconstruct clean action estimate: a_hat_norm = x_tau + (1 - tau) * v_pred
            a_hat_norm = policy.predict_clean_action(x_tau, tau, v_pred)

            # Multi-objective composite loss (Flow + Kinematics + Auxiliary Depth)
            loss, l_flow, l_acc, l_jerk, l_vq = loss_fn(
                v_pred=v_pred,
                x_0=x_0,
                x_1_norm=drifted_chunk,
                a_prev_phys=a_prev,
                a_hat_norm=a_hat_norm,
                vis_tokens=vis_tokens,
                z_star=z_star,
                step=step,
                sigma_d=sigma_d,
                mu_d=mu_d,
                proprio_err=drift_norm
            )

        # 4. Backward Pass & Step
        optimizer.zero_grad()
        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

        lr_scheduler.step()

        # Console logging
        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            print(
                f"Step [{step:05d}/{args.steps:05d}] | "
                f"Total: {loss.item():.4f} | "
                f"Flow: {l_flow.item():.4f} | "
                f"Acc: {l_acc.item():.4f} | "
                f"Jerk: {l_jerk.item():.4f} | "
                f"VQ: {l_vq.item():.4f}"
            )

        # Periodic Checkpoints
        if step % args.save_interval == 0 or step == args.steps:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "loss_state_dict": loss_fn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "action_stats": action_stats
            }, ckpt_path)
            print(f">> Saved checkpoint to: {ckpt_path}")

    # Export best inference checkpoint
    final_path = os.path.join(args.output_dir, "best.pt")
    torch.save(policy.state_dict(), final_path)
    print(f">> Training complete. Exported best model weights to: {final_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="5_shot", choices=["5_shot", "10_shot"])
    parser.add_argument("--data_dir", type=str, default="data/libero_spatial_5_shot")
    parser.add_argument("--stats_path", type=str, default="configs/action_stats.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="checkpoints/vga_5shot")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")

    # Ablation hyperparameter toggles
    parser.add_argument("--lambda_kin", type=float, default=CONFIG.lambda_kin_max, help="Max kinematic loss weight")
    parser.add_argument("--lambda_vq", type=float, default=CONFIG.lambda_vq, help="Auxiliary VQ-depth loss weight")
    parser.add_argument("--disable_ray_rope", action="store_true", help="Fallback to flat 2D positional embeddings")

    args = parser.parse_args()

    train_vga(args)