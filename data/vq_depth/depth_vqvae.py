"""
data/vq_depth/depth_vqvae.py
Lightweight f=32 Vector Quantized Autoencoder (VQ-VAE) for 256x256 metric depth maps.
Compresses continuous depth elevation into an 8x8 discrete codebook grid (K=256)
and reconstructs continuous depth via a symmetric 5-stage decoder.
"""
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.poc_config import CONFIG

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = CONFIG.vq_codebook_size,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_(mean=0.0, std=1.0 / self.embedding_dim)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e_perm.view(-1, self.embedding_dim)

        # L2 pairwise squared distance: ||x - e_k||^2
        distances = (
            torch.sum(flat_z_e ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_z_e, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = self.embedding(encoding_indices.view(-1)).view(z_e_perm.shape)

        loss_codebook = F.mse_loss(z_q, z_e_perm.detach())
        loss_commitment = F.mse_loss(z_q.detach(), z_e_perm)
        loss = loss_codebook + self.commitment_cost * loss_commitment

        # Straight-through gradient estimator
        z_q = z_e_perm + (z_q - z_e_perm).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Empirical codebook perplexity
        encodings = F.one_hot(encoding_indices.view(-1), self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        B, _, H, W = z_e.shape
        flat_indices = encoding_indices.view(B, H * W)

        return z_q, loss, flat_indices, perplexity

class DepthVQVAE(nn.Module):
    def __init__(
        self,
        codebook_size: int = CONFIG.vq_codebook_size,
        embedding_dim: int = 64
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        # Encoder: 256x256 -> 8x8 (Downsampling factor f = 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1)
        )

        self.quantizer = VectorQuantizer(codebook_size, embedding_dim)

        # Decoder: 8x8 -> 256x256 (Upsampling factor f = 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def _prepare_input(self, depth: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Sanitizes dimensions, formats to [B, 1, 256, 256], and syncs device/dtype."""
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)

        ref_param = next(self.parameters())
        depth = depth.to(device=ref_param.device, dtype=ref_param.dtype)

        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.ndim == 3:
            if depth.shape[-1] == 1:
                depth = depth.permute(2, 0, 1).unsqueeze(0)
            else:
                depth = depth.unsqueeze(1)
        elif depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)

        # Enforce exact nominal resolution (256, 256)
        target_res = (CONFIG.sim_img_size[0], CONFIG.sim_img_size[1])
        if depth.shape[-2:] != target_res:
            depth = F.interpolate(depth, size=target_res, mode="bilinear", align_corners=False)

        return depth

    def forward(
        self,
        depth_map: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._prepare_input(depth_map)
        z_e = self.encoder(x)
        z_q, vq_loss, indices, perplexity = self.quantizer(z_e)
        recon_depth = self.decoder(z_q)
        return recon_depth, vq_loss, indices, perplexity

    def encode_to_indices(self, depth_map: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Fast offline tokenization mapping [B, 1, 256, 256] depth to discrete codes [B, 64]."""
        x = self._prepare_input(depth_map)
        z_e = self.encoder(x)
        _, _, indices, _ = self.quantizer(z_e)
        return indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstructs continuous depth from codebook indices [B, 64]."""
        ref_weight = self.quantizer.embedding.weight
        indices = indices.to(device=ref_weight.device).long()
        B = indices.shape[0]

        z_q = self.quantizer.embedding(indices.view(-1)).view(B, 8, 8, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)