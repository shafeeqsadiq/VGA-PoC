"""
losses/drift_injection.py
Bounded Gaussian noise injector for closed-loop recovery training.
Applies truncated Gaussian perturbations with a decaying temporal ramp
and returns perturbation metrics for downstream error weighting.
"""
from typing import Tuple, Union
import torch
import torch.nn as nn
from configs.poc_config import CONFIG

class SyntheticDriftInjector(nn.Module):
    def __init__(
        self,
        max_trans_drift: float = CONFIG.max_trans_drift,
        max_rot_drift_deg: float = CONFIG.max_rot_drift_deg,
        ramp_steps: int = CONFIG.drift_ramp_steps,
        p_inject: float = 0.5):
        super().__init__()
        self.max_trans = max_trans_drift
        self.max_rot_rad = max_rot_drift_deg * (torch.pi / 180.0)
        self.ramp_steps = ramp_steps
        self.p_inject = p_inject

    def forward(
        self,
        action_chunk: torch.Tensor,
        return_drift_norm: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Injects bounded Gaussian drift into the action chunk during training.

        Args:
            action_chunk: Trajectory chunk of shape [B, H, 7]
            return_drift_norm: If True, also returns L2 spatial drift [B]
        Returns:
            Perturbed action chunk [B, H, 7] (and optionally drift norm [B])
        """
        B, H, _ = action_chunk.shape
        device = action_chunk.device
        dtype = action_chunk.dtype

        # Bypass during evaluation or when random roll skips perturbation
        if not self.training or torch.rand(1).item() > self.p_inject:
            if return_drift_norm:
                return action_chunk, torch.zeros(B, device=device, dtype=dtype)
            return action_chunk

        # Sample 2-sigma truncated Gaussian noise bounded strictly within [-1.0, 1.0]
        noise_p = torch.randn(B, 1, 3, device=device, dtype=dtype)
        noise_r = torch.randn(B, 1, 3, device=device, dtype=dtype)

        delta_p = torch.clamp(noise_p / 2.0, -1.0, 1.0) * self.max_trans
        delta_r = torch.clamp(noise_r / 2.0, -1.0, 1.0) * self.max_rot_rad

        # Decaying temporal ramp over ramp_steps (e.g. [1.0, 0.667, 0.333, 0.0, ...])
        ramp = torch.zeros(1, H, 1, device=device, dtype=dtype)
        decay = torch.linspace(1.0, 0.0, self.ramp_steps + 1, device=device, dtype=dtype)[:-1]
        ramp[:, :self.ramp_steps, 0] = decay

        # Construct 7-DoF perturbation tensor
        drift = torch.zeros(B, H, 7, device=device, dtype=dtype)
        drift[:, :, :3] = delta_p * ramp
        drift[:, :, 3:6] = delta_r * ramp

        perturbed_chunk = action_chunk + drift

        if return_drift_norm:
            drift_norm = torch.norm(delta_p.squeeze(1), p=2, dim=-1)  # [B]
            return perturbed_chunk, drift_norm

        return perturbed_chunk