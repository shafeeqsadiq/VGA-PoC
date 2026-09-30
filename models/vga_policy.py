"""
models/vga_policy.py
Top-level Vision-Geometry-Action (VGA) Policy Module.
Connects SigLIP-LoRA -> Space-to-Depth Projector -> Ray-RoPE -> SmolLM2 -> DiT Action Expert.
Handles automatic image resizing to 384x384, text tokenization, bidirectional attention,
and cached ray generation.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from configs.poc_config import CONFIG
from models.projector import UnifiedSpaceToDepthProjector
from models.ray_rope import CentroidRayRoPE
from models.dit_expert import DiTActionExpert
from models.backbones import load_siglip_lora, load_pruned_smollm2_lora

class VGAPolicy(nn.Module):
    def __init__(
        self,
        load_pretrained: bool = False,
        siglip_name: str = "google/siglip-base-patch16-384",
        smollm_name: str = "HuggingFaceTB/SmolLM2-360M"
    ):
        super().__init__()
        self.config = CONFIG
        
        # 1. Vision Backbone & Language Spine
        if load_pretrained:
            self.vision_encoder = load_siglip_lora(siglip_name)
            self.lm_backbone, self.tokenizer = load_pruned_smollm2_lora(smollm_name, num_layers=12)
        else:
            self.vision_encoder = None
            self.lm_backbone = None
            self.tokenizer = None

        # 2. 3x Space-to-Depth Token Compressor (576 -> 64 tokens)
        self.projector = UnifiedSpaceToDepthProjector(
            vis_dim=CONFIG.vis_dim,
            lm_dim=CONFIG.lm_dim,
            spatial_factor=CONFIG.spatial_factor
        )

        # 3. 3D Ray-RoPE Metric Grounder
        self.ray_rope = CentroidRayRoPE(
            lm_dim=CONFIG.lm_dim,
            num_heads=CONFIG.rope_heads
        )

        # 4. Continuous Flow Matching DiT Action Expert
        self.dit = DiTActionExpert(
            action_dim=CONFIG.action_dim,
            chunk_horizon=CONFIG.chunk_horizon,
            hidden_dim=512,
            num_layers=12,
            num_heads=8,
            context_dim=CONFIG.lm_dim,
            prefix_len=CONFIG.buffer_prefix_len
        )

        # Cached nominal rays buffer
        self.register_buffer("cached_rays", None, persistent=False)

    def optimize_for_inference(self):
        """Compiles backbones and arms CUDA graphs for <=18ms deployment."""
        torch.set_float32_matmul_precision("high")
        if self.lm_backbone is not None:
            self.lm_backbone.config.use_cache = False
            try:
                self.lm_backbone = torch.compile(self.lm_backbone, mode="reduce-overhead")
            except Exception:
                self.lm_backbone = torch.compile(self.lm_backbone, mode="default")
        if self.vision_encoder is not None:
            try:
                self.vision_encoder = torch.compile(self.vision_encoder, mode="reduce-overhead")
            except Exception:
                self.vision_encoder = torch.compile(self.vision_encoder, mode="default")
        self.dit.enable_cuda_graphs = True
        return self

    def _prepare_rgb(self, rgb: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Sanitizes image shape to [B, 3, 384, 384] and normalizes float values."""
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)

        # Move to model's execution device and float32/bfloat16
        ref_param = next(self.projector.parameters())
        rgb = rgb.to(device=ref_param.device, dtype=ref_param.dtype)

        # Ensure batch dimension: [H, W, C] -> [1, H, W, C]
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)

        # Permute channels to [B, C, H, W]
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)

        # Direct GPU tensor normalization without scalar reduction sync
        is_uint8 = (rgb.dtype == torch.uint8)
        rgb = rgb.to(device=ref_param.device, dtype=ref_param.dtype, non_blocking=True)
        if is_uint8:
            rgb = rgb / 255.0

        # Enforce exact SigLIP input resolution (384, 384)
        if rgb.shape[-2:] != tuple(CONFIG.vis_input_size):
            rgb = TF.resize(
                rgb,
                list(CONFIG.vis_input_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=False
            )

        return rgb

    def _get_default_rays(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Retrieves or precomputes nominal centroid rays on device."""
        if self.cached_rays is None or self.cached_rays.device != device:
            K = torch.tensor([
                [CONFIG.default_intrinsics["fx"], 0.0, CONFIG.default_intrinsics["cx"]],
                [0.0, CONFIG.default_intrinsics["fy"], CONFIG.default_intrinsics["cy"]],
                [0.0, 0.0, 1.0]
            ], device=device, dtype=dtype)
            R_ident = torch.eye(3, device=device, dtype=dtype)
            self.cached_rays = self.ray_rope.compute_centroid_rays(K, R_ident, device=device).to(dtype=dtype)
        return self.cached_rays

    def extract_multimodal_context(
        self,
        rgb: Union[torch.Tensor, np.ndarray],
        t_base_cam: Union[torch.Tensor, np.ndarray],
        rays: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, List[str], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Extracts visual patches, compresses via Space-to-Depth, modulates with Ray-RoPE,
        and conditions through SmolLM2 with bidirectional cross-attention.
        """
        x_rgb = self._prepare_rgb(rgb)
        B = x_rgb.shape[0]
        device = x_rgb.device
        dtype = x_rgb.dtype

        # 1. Vision feature extraction: [B, 3, 384, 384] -> [B, 576, 768]
        if self.vision_encoder is not None:
            vis_features = self.vision_encoder(pixel_values=x_rgb).last_hidden_state
        else:
            vis_features = torch.randn(B, CONFIG.raw_patches, CONFIG.vis_dim, device=device, dtype=dtype)

        # 2. Space-to-depth compression: [B, 576, 768] -> [B, 64, 960]
        compressed_tokens = self.projector(vis_features)

        # 3. 3D Metric Ray-RoPE modulation
        if isinstance(t_base_cam, np.ndarray):
            t_base_cam = torch.from_numpy(t_base_cam)
        t_base_cam = t_base_cam.to(device=device, dtype=dtype)
        if t_base_cam.ndim == 1:
            t_base_cam = t_base_cam.unsqueeze(0).expand(B, -1)

        if rays is None:
            rays = self._get_default_rays(device, dtype)

        grounded_vis_tokens = self.ray_rope(compressed_tokens, t_base_cam, rays=rays)

        # 4. Multimodal fusion through SmolLM2
        if self.lm_backbone is not None and prompt is not None:
            if isinstance(prompt, (str, list)):
                if isinstance(prompt, str):
                    prompt = [prompt]
                tokens = self.tokenizer(
                    prompt,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )
                prompt_ids = tokens.input_ids.to(device=device)
            else:
                prompt_ids = prompt.to(device=device)

            text_embeds = self.lm_backbone.get_input_embeddings()(prompt_ids)
            fused_tokens = torch.cat([grounded_vis_tokens, text_embeds], dim=1)  # [B, 64 + L, 960]

            # Construct 4D bidirectional attention mask to bypass causal masking
            seq_len = fused_tokens.shape[1]
            bidirectional_mask = torch.zeros(
                (B, 1, seq_len, seq_len),
                device=device,
                dtype=dtype
            )
            context = self.lm_backbone(
                inputs_embeds=fused_tokens,
                attention_mask=bidirectional_mask
            ).last_hidden_state
        else:
            context = grounded_vis_tokens

        return context

    def forward(
        self,
        rgb: Union[torch.Tensor, np.ndarray],
        t_base_cam: Union[torch.Tensor, np.ndarray],
        x_t: torch.Tensor,
        tau: torch.Tensor,
        a_prev: torch.Tensor,
        rays: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, List[str], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass returning predicted flow vector field v_pred and visual tokens.
        """
        context = self.extract_multimodal_context(rgb, t_base_cam, rays, prompt)
        v_pred = self.dit(x_t, tau, context, a_prev)
        vis_tokens = context[:, :CONFIG.compressed_tokens, :]
        return v_pred, vis_tokens

    def predict_clean_action(
        self,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        v_pred: torch.Tensor
    ) -> torch.Tensor:
        """Computes clean action estimate a_hat_norm = x_t + (1 - tau) * v_pred."""
        return self.dit.predict_clean_action(x_t, tau, v_pred)

    @torch.inference_mode()
    def sample_actions(
        self,
        rgb: Union[torch.Tensor, np.ndarray],
        t_base_cam: Union[torch.Tensor, np.ndarray],
        a_prev: torch.Tensor,
        rays: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, List[str], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Inference execution: predicts a continuous 16-step action chunk via 4-step Euler ODE integration.
        """
        context = self.extract_multimodal_context(rgb, t_base_cam, rays, prompt)
        return self.dit.sample_4step_euler(context, a_prev)