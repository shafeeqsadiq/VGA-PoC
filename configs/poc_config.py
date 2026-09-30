"""
configs/poc_config.py
Architectural hyperparameters, dimensional constants, and runtime configurations for VGA PoC.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple
import torch

@dataclass(frozen=True)
class VGAConfig:
    # --- Image Resolutions & Patch Geometry ---
    sim_img_size: Tuple[int, int] = (256, 256)      # Raw camera frame from MuJoCo/LIBERO
    vis_input_size: Tuple[int, int] = (384, 384)    # Resized input for SigLIP (24x24 patches)
    patch_size: int = 16                            # ViT patch resolution
    raw_patches: int = 576                          # (384 // 16) ** 2 = 24 * 24
    spatial_factor: int = 3                         # Downsample factor for pixel_unshuffle
    compressed_tokens: int = 64                     # (24 // 3) ** 2 = 8 * 8 grid

    # --- Perception & Language Dimensions ---
    vis_dim: int = 768                              # SigLIP-B/16 embedding dimension
    lm_dim: int = 960                               # SmolLM2-360M hidden state dimension
    unshuffled_channels: int = 6912                 # vis_dim * (spatial_factor ** 2) = 768 * 9
    rope_heads: int = 8                             # Ray-RoPE pre-pass attention heads

    # --- LoRA Fine-Tuning Hyperparameters ---
    lora_r: int = 16                                # Rank for parameter-efficient adaptation
    lora_alpha: int = 32                            # Scaling factor for LoRA updates

    # --- Action Trajectory & Chunking ---
    action_dim: int = 7                             # [dx, dy, dz, rx, ry, rz, gripper]
    chunk_horizon: int = 16                         # Prediction horizon H
    buffer_prefix_len: int = 4                      # Tail waypoints P (steps 13-16)

    # --- Loss Weights & Scheduling ---
    w_rot: float = 0.01                             # Relative rotation loss weight
    beta_jerk: float = 0.5                          # Ratio of jerk to acceleration loss
    lambda_kin_max: float = 0.05                    # Peak kinematic loss scalar
    kinematic_warmup_steps: int = 15000             # Steps to linearly ramp kinematic loss
    lambda_vq: float = 0.1                          # Auxiliary VQ-depth cross-entropy scalar
    vq_codebook_size: int = 256                     # Number of discrete surface depth codes K

    # --- Synthetic Drift Injection ---
    max_trans_drift: float = 0.015                  # 15 mm maximum spatial perturbation
    max_rot_drift_deg: float = 3.0                  # 3 degrees maximum orientation perturbation
    drift_ramp_steps: int = 3                       # Perturbation decay horizon (steps)

    # --- Execution & Controller Thresholds ---
    schmitt_low: float = 0.35                       # Gripper open threshold
    schmitt_high: float = 0.65                      # Gripper close threshold
    async_trigger_step: int = 11                    # Trigger step for 50 Hz background thread

    # --- Evaluation & Test-Time Runtime Flags ---
    max_eval_steps: int = 350                       # Maximum rollout duration per episode
    use_depth_input: bool = False                   # Monocular RGB at test time
    drop_depth_head_at_eval: bool = True            # Detach VQ-depth head during rollout

    # --- Camera Intrinsics (LIBERO Simulation Default) ---
    default_intrinsics: Dict[str, float] = field(default_factory=lambda: {
        "fx": 200.0,
        "fy": 200.0,
        "cx": 128.0,
        "cy": 128.0
    })

CONFIG = VGAConfig()