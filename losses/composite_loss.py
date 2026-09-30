"""
losses/composite_loss.py
Multi-objective composite loss: Error-Weighted Flow Matching + Kinematic Acc/Jerk +
Auxiliary VQ-Depth Cross-Entropy.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.poc_config import CONFIG

class CompositeVGALoss(nn.Module):
    def __init__(
        self,
        sigma_pos_sq: float = 1.0,
        sigma_rot_sq: float = 1.0,
        w_rot: float = CONFIG.w_rot,
        beta_jerk: float = CONFIG.beta_jerk,
        lambda_kin_max: float = CONFIG.lambda_kin_max,
        kinematic_warmup_steps: int = CONFIG.kinematic_warmup_steps,
        lambda_vq: float = CONFIG.lambda_vq):
        super().__init__()
        self.sigma_pos_sq = sigma_pos_sq
        self.sigma_rot_sq = sigma_rot_sq
        self.w_rot = w_rot
        self.beta_jerk = beta_jerk
        self.lambda_kin_max = lambda_kin_max
        self.kinematic_warmup_steps = kinematic_warmup_steps
        self.lambda_vq = lambda_vq

        # Linear classifier projecting SmolLM2 visual tokens to discrete depth codebook K=256
        self.vq_classifier = nn.Linear(CONFIG.lm_dim, CONFIG.vq_codebook_size)

    def forward(
        self,
        v_pred: torch.Tensor,
        x_0: torch.Tensor,
        x_1_norm: torch.Tensor,
        a_prev_phys: torch.Tensor,
        a_hat_norm: torch.Tensor,
        vis_tokens: torch.Tensor,
        z_star: torch.Tensor,
        step: int,
        sigma_d: torch.Tensor,
        mu_d: torch.Tensor,
        proprio_err: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            v_pred: Predicted velocity field from DiT [B, H, 7]
            x_0: Sampled Gaussian noise trajectory [B, H, 7]
            x_1_norm: Normalized ground truth trajectory chunk [B, H, 7]
            a_prev_phys: Unnormalized previous buffer tail waypoints [B, P, 6]
            a_hat_norm: Predicted clean action chunk [B, H, 7]
            vis_tokens: Multimodal visual tokens [B, 64, 960]
            z_star: Target discrete surface depth tokens [B, 64]
            step: Active global training step index
            sigma_d: Action standard deviation normalizer [7]
            mu_d: Action mean normalizer [7]
            proprio_err: Injected or measured tracking error [B]
        """
        device = v_pred.device
        dtype = v_pred.dtype

        # -------------------------------------------------------------
        # 1. Error-Weighted Flow Matching Loss
        # -------------------------------------------------------------
        # Correctly unsqueeze weight [B, 1] to broadcast across horizon [B, H]
        weight = 1.0 + 0.5 * torch.tanh(proprio_err.view(-1, 1).to(device=device, dtype=dtype) / 0.05)
        target_v = x_1_norm - x_0
        flow_sq_err = torch.norm(v_pred - target_v, p=2, dim=-1)**2  # [B, H]
        l_flow = torch.mean(weight * flow_sq_err)

        # -------------------------------------------------------------
        # 2. Kinematic Regularization (Acceleration & Jerk)
        # -------------------------------------------------------------
        # Safely align normalizers to current device and precision
        sig = sigma_d.to(device=device, dtype=dtype)
        mu = mu_d.to(device=device, dtype=dtype)

        # De-normalize translational and rotational components
        a_hat_phys = a_hat_norm[..., :6] * sig[:6] + mu[:6]
        a_prev_aligned = a_prev_phys.to(device=device, dtype=dtype)

        # Concatenate history prefix with predicted horizon: [B, P + H, 6]
        a_stitched = torch.cat([a_prev_aligned, a_hat_phys], dim=1)

        delta_p = a_stitched[..., :3]
        r = a_stitched[..., 3:6]

        # 1st-order differences (acceleration profile)
        acc_p = delta_p[:, 1:] - delta_p[:, :-1]
        acc_r = r[:, 1:] - r[:, :-1]

        l_acc = torch.mean(
            (torch.norm(acc_p, dim=-1)**2) / self.sigma_pos_sq +
            self.w_rot * (torch.norm(acc_r, dim=-1)**2) / self.sigma_rot_sq
        )

        # 2nd-order differences (jerk profile)
        jerk_p = acc_p[:, 1:] - acc_p[:, :-1]
        jerk_r = acc_r[:, 1:] - acc_r[:, :-1]

        l_jerk = torch.mean(
            (torch.norm(jerk_p, dim=-1)**2) / self.sigma_pos_sq +
            self.w_rot * (torch.norm(jerk_r, dim=-1)**2) / self.sigma_rot_sq
        )

        # Linear warmup schedule for kinematics
        warmup_factor = min(1.0, float(step) / float(self.kinematic_warmup_steps)) if step > 0 else 0.0
        lambda_kin = self.lambda_kin_max * warmup_factor

        # -------------------------------------------------------------
        # 3. Auxiliary VQ-Depth Token Classification Loss
        # -------------------------------------------------------------
        # Temperature-scaled logits: [B, 64, 256]
        vq_logits = self.vq_classifier(vis_tokens) / 0.07

        # Enforce long targets for cross_entropy
        z_star_long = z_star.to(device=device).long()
        l_vq = F.cross_entropy(vq_logits.view(-1, CONFIG.vq_codebook_size), z_star_long.view(-1))

        # Composite objective
        l_unified = l_flow + lambda_kin * (l_acc + self.beta_jerk * l_jerk) + self.lambda_vq * l_vq

        return l_unified, l_flow, l_acc, l_jerk, l_vq