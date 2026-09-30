"""
models/ray_rope.py
Decoupled Centroid Ray-RoPE generator and camera origin embedding pre-pass.
Computes metric 3D directional vectors, projects rays and origins, and modulates
visual tokens via residual self-attention.
"""
from typing import Optional
import torch
import torch.nn as nn
from configs.poc_config import CONFIG

class CentroidRayRoPE(nn.Module):
    def __init__(
        self,
        lm_dim: int = CONFIG.lm_dim,
        num_heads: int = CONFIG.rope_heads,
        dropout: float = 0.05
    ):
        super().__init__()
        self.lm_dim = lm_dim
        
        # Camera translation MLP: R^3 -> R^960
        self.origin_mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.SiLU(),
            nn.Linear(256, lm_dim)
        )
        
        # 3D Ray directional projection MLP: S^2 (R^3) -> R^960
        self.ray_mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.SiLU(),
            nn.Linear(256, lm_dim)
        )
        
        # Geometric pre-pass self-attention layer
        self.pre_pass_attn = nn.MultiheadAttention(
            embed_dim=lm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for residual preservation of visual priors
        self.norm = nn.LayerNorm(lm_dim)
        self.dropout = nn.Dropout(dropout)

    def compute_centroid_rays(
        self,
        K: torch.Tensor,
        R_base_cam: torch.Tensor,
        img_w: int = CONFIG.sim_img_size[0],
        img_h: int = CONFIG.sim_img_size[1],
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Computes metric unit ray directions for the 8x8 centroid grid.
        Args:
            K: Camera intrinsics [3, 3] or [B, 3, 3]
            R_base_cam: Camera-to-base rotation matrix [3, 3] or [B, 3, 3]
        Returns:
            Unit rays on S^2 in robot base frame: [64, 3] or [B, 64, 3]
        """
        # Centroid coordinates at centers of each 32x32 block (indices 0..7)
        coords = torch.arange(8, device=device, dtype=torch.float32) + 0.5
        u = coords * (img_w / 8.0)
        v = coords * (img_h / 8.0)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')

        # Homogeneous pixel coordinates [64, 3]
        pixels = torch.stack(
            [grid_u.flatten(), grid_v.flatten(), torch.ones(64, device=device, dtype=torch.float32)],
            dim=1
        )

        # Optical backprojection: rays_cam = K^-1 * pixels
        K_mat = K.to(device).float()
        if K_mat.ndim == 2:
            K_inv = torch.inverse(K_mat)
            rays_cam = (K_inv @ pixels.T).T  # [64, 3]
        else:
            K_inv = torch.inverse(K_mat)  # [B, 3, 3]
            rays_cam = torch.matmul(pixels.unsqueeze(0), K_inv.transpose(-1, -2))  # [B, 64, 3]

        # Normalize to unit vectors on S^2
        rays_unit = rays_cam / torch.norm(rays_cam, dim=-1, keepdim=True).clamp_min(1e-8)

        # Transform to base coordinate frame: d_base = R_base_cam * d_cam
        R_mat = R_base_cam.to(device).float()
        if R_mat.ndim == 2:
            rays_base = (R_mat @ rays_unit.T).T  # [64, 3]
        else:
            rays_base = torch.matmul(rays_unit, R_mat.transpose(-1, -2))  # [B, 64, 3]

        return rays_base

    def forward(
        self,
        visual_tokens: torch.Tensor,
        t_base_cam: torch.Tensor,
        rays: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: Compressed tokens from projector [B, 64, 960]
            t_base_cam: Camera translation vector [B, 3]
            rays: Precomputed unit rays in base frame [64, 3] or [B, 64, 3]
        Returns:
            Geometrically grounded visual tokens [B, 64, 960]
        """
        B, N, D = visual_tokens.shape
        dtype = visual_tokens.dtype
        device = visual_tokens.device

        # 1. Embed camera translation: [B, 3] -> [B, 1, 960]
        t_embed = self.origin_mlp(t_base_cam.to(device, dtype=dtype)).unsqueeze(1)

        # 2. Embed 3D ray directions if supplied
        if rays is not None:
            rays_input = rays.to(device, dtype=dtype)
            if rays_input.ndim == 2:
                # [64, 3] -> [1, 64, 960] -> broadcast to [B, 64, 960]
                ray_embed = self.ray_mlp(rays_input).unsqueeze(0).expand(B, -1, -1)
            else:
                # [B, 64, 3] -> [B, 64, 960]
                ray_embed = self.ray_mlp(rays_input)
            tokens_augmented = visual_tokens + ray_embed + t_embed
        else:
            tokens_augmented = visual_tokens + t_embed

        # 3. Geometric self-attention pre-pass
        attn_out, _ = self.pre_pass_attn(
            tokens_augmented, tokens_augmented, tokens_augmented
        )

        # 4. Residual connection & LayerNorm to preserve visual priors
        return self.norm(visual_tokens + self.dropout(attn_out))