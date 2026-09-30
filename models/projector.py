"""
models/projector.py
Unified space-to-depth token compressor for SigLIP visual features.
Maps [B, 576, 768] -> [B, 64, 960] via 3x pixel unshuffle and linear projection.
"""
import math
import torch
import torch.nn as nn
from configs.poc_config import CONFIG

class UnifiedSpaceToDepthProjector(nn.Module):
    def __init__(
        self,
        vis_dim: int = CONFIG.vis_dim,
        lm_dim: int = CONFIG.lm_dim,
        spatial_factor: int = CONFIG.spatial_factor
    ):
        super().__init__()
        self.vis_dim = vis_dim
        self.lm_dim = lm_dim
        self.spatial_factor = spatial_factor
        self.in_features = vis_dim * (spatial_factor ** 2)  # 768 * 9 = 6912
        self.proj = nn.Linear(self.in_features, lm_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input patch tensor from SigLIP [B, 576, 768]
        Returns:
            Projected tokens aligned to SmolLM2 dimension [B, 64, 960]
        """
        B, N, D = x.shape
        assert D == self.vis_dim, f"Input feature dimension {D} does not match expected {self.vis_dim}"
        
        grid_dim = int(math.isqrt(N))
        assert grid_dim * grid_dim == N, f"Visual token count {N} is not a square grid"
        assert grid_dim % self.spatial_factor == 0, (
            f"Grid dimension {grid_dim} is not divisible by spatial factor {self.spatial_factor}"
        )

        # 1. Unflatten token sequence to 2D feature map: [B, 24, 24, 768] -> [B, 768, 24, 24]
        x_grid = x.view(B, grid_dim, grid_dim, D).permute(0, 3, 1, 2).contiguous()

        # 2. Space-to-depth pixel unshuffle: [B, 768, 24, 24] -> [B, 6912, 8, 8]
        x_unshuffled = nn.functional.pixel_unshuffle(
            x_grid, downscale_factor=self.spatial_factor
        )

        # 3. Reshape back into token sequence: [B, 8, 8, 6912] -> [B, 64, 6912]
        x_tokens = x_unshuffled.permute(0, 2, 3, 1).contiguous().reshape(B, -1, self.in_features)

        # 4. Dense linear projection: [B, 64, 6912] -> [B, 64, 960]
        return self.proj(x_tokens)