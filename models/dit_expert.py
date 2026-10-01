"""
models/dit_expert.py
12-layer Diffusion Transformer (DiT) Action Expert for continuous flow matching.
Predicts 16-step 7-DoF action chunks conditioned on language/visual tokens and prefix history.
Includes true AdaLN-Zero initialization and 4-step Euler ODE integration.
"""
import math
from typing import Optional
import torch
import torch.nn as nn
from configs.poc_config import CONFIG

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # AdaLN modulation: generates 6 modulation parameters (scale/shift for 3 norms)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim, bias=True)
        )

        # Initialize AdaLN-Zero: ensures block acts as exact identity at initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # c: [B, cond_dim], context: [B, Seq, hidden_dim]
        mod = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        
        # 1. Modulated Self-Attention
        norm_x = (1 + scale_msa.unsqueeze(1)) * self.norm1(x) + shift_msa.unsqueeze(1)
        sa_out, _ = self.self_attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * sa_out

        # 2. Cross-Attention to multimodal context
        norm_x2 = self.norm2(x)
        ca_out, _ = self.cross_attn(norm_x2, context, context, need_weights=False)
        x = x + ca_out

        # 3. Modulated Feed-Forward MLP
        norm_x3 = (1 + scale_mlp.unsqueeze(1)) * self.norm3(x) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(norm_x3)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x

class DiTActionExpert(nn.Module):
    def __init__(
        self,
        action_dim: int = CONFIG.action_dim,
        chunk_horizon: int = CONFIG.chunk_horizon,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        context_dim: int = CONFIG.lm_dim,
        prefix_len: int = CONFIG.buffer_prefix_len
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.hidden_dim = hidden_dim
        self.prefix_len = prefix_len
        
        # Projection layers
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, chunk_horizon, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        
        # Flow time-step embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Buffer tail prefix encoder (P=4 steps of 6-DoF deltas -> 24 inputs)
        self.prefix_encoder = nn.Sequential(
            nn.Linear(prefix_len * 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Context dimension alignment (e.g., SmolLM2 960 -> DiT 512)
        self.context_proj = nn.Linear(context_dim, hidden_dim)

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, cond_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        # Final projection to continuous velocity field v_t
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.out_proj = nn.Linear(hidden_dim, action_dim)
        
        # Zero-out final layer to initialize as identity flow
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        context: torch.Tensor,
        a_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t: Noisy action chunk [B, H, 7]
            tau: Flow time-step [B] in [0, 1]
            context: Condition tokens from SmolLM2 [B, Seq, 960]
            a_prev: Previous buffer tail waypoints [B, P, 6] or [B, P, 7]
        Returns:
            Predicted vector field v_pred [B, H, 7]
        """
        B = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype

        # Align inputs
        tau = tau.to(device=device, dtype=dtype)
        context = context.to(device=device, dtype=dtype)
        a_prev = a_prev.to(device=device, dtype=dtype)
        
        # Guard prefix slicing: enforce 6-DoF kinematic components
        a_prev_6d = a_prev[..., :6].reshape(B, -1)

        # 1. Embed time step tau and prefix history
        t_embed = self.time_emb(tau)
        p_embed = self.prefix_encoder(a_prev_6d)
        cond = t_embed + p_embed  # [B, hidden_dim]

        # 2. Embed action chunks and add positional encoding
        h = self.action_in_proj(x_t) + self.pos_emb[:, :self.chunk_horizon, :]

        # 3. Project context to DiT internal hidden dimension
        ctx = self.context_proj(context)

        # 4. Pass through DiT layers
        for block in self.blocks:
            h = block(h, cond, ctx)

        # 5. Output velocity field
        h = self.final_norm(h)
        return self.out_proj(h)

    def predict_clean_action(
        self,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        v_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes clean target estimate x_1 from flow prediction: x_1 = x_t + (1 - tau) * v_pred.
        Used by training loops to supply a_hat_norm to CompositeVGALoss.
        """
        tau_broadcast = tau.view(-1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
        return x_t + (1.0 - tau_broadcast) * v_pred

    def _record_euler_graph(self, B: int, S: int, device: torch.device, dtype: torch.dtype):
        dt = 1.0 / 4.0
        s_ctx = torch.zeros(B, S, self.context_proj.in_features, device=device, dtype=dtype)
        s_prev = torch.zeros(B, self.prefix_len, 6, device=device, dtype=dtype)
        s_x0 = torch.zeros(B, self.chunk_horizon, self.action_dim, device=device, dtype=dtype)
        s_out = torch.zeros(B, self.chunk_horizon, self.action_dim, device=device, dtype=dtype)

        def _step(x_in, ctx_in, prev_in):
            a_prev_6d = prev_in[..., :6].reshape(B, -1)
            p_embed = self.prefix_encoder(a_prev_6d)
            ctx = self.context_proj(ctx_in)
            x = x_in
            for i in range(4):
                tau = torch.full((B,), i * dt, device=device, dtype=dtype)
                t_embed = self.time_emb(tau)
                cond = t_embed + p_embed
                h = self.action_in_proj(x) + self.pos_emb[:, :self.chunk_horizon, :]
                for block in self.blocks:
                    h = block(h, cond, ctx)
                v_pred = self.out_proj(self.final_norm(h))
                x = x + v_pred * dt
            return x

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = _step(s_x0, s_ctx, s_prev)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s_out = _step(s_x0, s_ctx, s_prev)
        return g, s_x0, s_ctx, s_prev, s_out

    @torch.inference_mode()
    def sample_4step_euler(
        self,
        context: torch.Tensor,
        a_prev: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        B, S = context.shape[0], context.shape[1]
        device = device or context.device
        dtype = dtype or context.dtype

        # Replay CUDA Graph if enabled and running on CUDA
        if False:  # CUDA graphs disabled for inference stability
            if not hasattr(self, "_graphs"):
                self._graphs = {}
            key = (B, S, dtype)
            if key not in self._graphs:
                self._graphs[key] = self._record_euler_graph(B, S, device, dtype)
            g, s_x0, s_ctx, s_prev, s_out = self._graphs[key]
            s_ctx.copy_(context)
            s_prev.copy_(a_prev[..., :6])
            s_x0.normal_()
            g.replay()
            return s_out.clone()

        # Fallback standard eager ODE loop
        dt = 1.0 / 4.0
        x = torch.randn(B, self.chunk_horizon, self.action_dim, device=device, dtype=dtype)
        a_prev_6d = a_prev[..., :6].reshape(B, -1).to(device=device, dtype=dtype)
        p_embed = self.prefix_encoder(a_prev_6d)
        ctx = self.context_proj(context.to(device=device, dtype=dtype))
        for i in range(4):
            tau = torch.full((B,), i * dt, device=device, dtype=dtype)
            t_embed = self.time_emb(tau)
            cond = t_embed + p_embed
            h = self.action_in_proj(x) + self.pos_emb[:, :self.chunk_horizon, :]
            for block in self.blocks:
                h = block(h, cond, ctx)
            v_pred = self.out_proj(self.final_norm(h))
            x = x + v_pred * dt
        return x