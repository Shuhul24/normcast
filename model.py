import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rot_to_6dof(poses, ref_poses):
    """
    poses:     [B, T, 4, 4]
    ref_poses: [B, 1, 4, 4]
    returns:   [B, T, 6]  (dx, dy, dz, roll, pitch, yaw) relative to ref
    """
    rel = torch.linalg.inv(ref_poses) @ poses
    dx, dy, dz = rel[..., 0, 3], rel[..., 1, 3], rel[..., 2, 3]
    R = rel[..., :3, :3]
    pitch = torch.asin((-R[..., 2, 0]).clamp(-1.0, 1.0))
    yaw   = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll  = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    return torch.stack([dx, dy, dz, roll, pitch, yaw], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, head_dim: int, context_dim: int | None = None):
        super().__init__()
        assert dim % head_dim == 0
        ctx_dim = context_dim or dim
        self.num_heads = dim // head_dim
        self.head_dim  = head_dim
        self.is_cross  = context_dim is not None

        self.norm_x   = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(ctx_dim)
        self.q   = nn.Linear(dim,     dim, bias=False)
        self.k   = nn.Linear(ctx_dim, dim, bias=False)
        self.v   = nn.Linear(ctx_dim, dim, bias=False)
        self.out = nn.Linear(dim,     dim, bias=False)

        self.sample: bool = False
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        src      = context if context is not None else x
        x_n      = self.norm_x(x)
        src_n    = self.norm_ctx(src) if self.is_cross else self.norm_x(src)

        q = self.q(x_n  ).reshape(B, T,  self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(src_n).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(src_n).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.sample and not self.is_cross:
            self.k_cache.append(k)
            self.v_cache.append(v)
            k = torch.cat(self.k_cache, dim=2)
            v = torch.cat(self.v_cache, dim=2)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           scale=1.0 / math.sqrt(self.head_dim))
        return self.out(o.transpose(1, 2).reshape(B, T, -1))

    def reset_cache(self):
        self.k_cache, self.v_cache = [], []


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, dim * expansion), nn.GELU(), nn.Linear(dim * expansion, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, head_dim: int, context_dim: int | None = None):
        super().__init__()
        self.self_attn  = Attention(dim, head_dim)
        self.cross_attn = Attention(dim, head_dim, context_dim) if context_dim else None
        self.mlp        = MLP(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(x, mask=mask)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(x, context=context)
        return x + self.mlp(x)

    def enable_sample_mode(self):
        self.self_attn.sample = True
        self.self_attn.reset_cache()

    def disable_sample_mode(self):
        self.self_attn.sample = False


class ContextEncoder(nn.Module):
    """Encodes past range-view frames + relative poses into context tokens."""

    def __init__(self, patch_dim: int, channels: int, num_layers: int, head_dim: int):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, channels)
        self.pose_proj  = nn.Sequential(
            nn.Linear(6, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.layers = nn.ModuleList(
            [TransformerLayer(channels, head_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, patches: torch.Tensor, poses_6dof: torch.Tensor) -> torch.Tensor:
        # patches:   [B, T*N, patch_dim]
        # poses_6dof:[B, T,   6]
        B, TN, _ = patches.shape
        T = poses_6dof.size(1)
        N = TN // T

        tokens   = self.patch_proj(patches)                            # [B, T*N, C]
        pose_emb = self.pose_proj(poses_6dof)                         # [B, T, C]
        pose_emb = pose_emb.unsqueeze(2).expand(-1, -1, N, -1)        # [B, T, N, C]
        tokens   = tokens + pose_emb.reshape(B, TN, -1)

        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens)


class FlowBlock(nn.Module):
    """
    Single NVP autoregressive coupling block.
    Causal self-attention over current-frame patches + cross-attention to context.
    Alternating flip permutation between blocks.
    """

    def __init__(self, patch_dim: int, channels: int, num_patches: int,
                 num_layers: int, head_dim: int, context_dim: int, flip: bool = False):
        super().__init__()
        self.flip = flip
        self.proj_in  = nn.Linear(patch_dim, channels)
        self.pos_embed = nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        self.layers   = nn.ModuleList(
            [TransformerLayer(channels, head_dim, context_dim) for _ in range(num_layers)]
        )
        self.proj_out = nn.Linear(channels, patch_dim * 2)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.register_buffer('causal_mask',
                             torch.tril(torch.ones(num_patches, num_patches)).bool())

    def _flip(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return x.flip(dims=[dim]) if self.flip else x

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x   = self._flip(x, 1)
        x_in = x
        pos  = self._flip(self.pos_embed, 0)
        h    = self.proj_in(x) + pos

        for layer in self.layers:
            h = layer(h, context=context, mask=self.causal_mask)

        out = self.proj_out(h)
        out = torch.cat([torch.zeros_like(out[:, :1]), out[:, :-1]], dim=1)

        scale, shift = out.chunk(2, dim=-1)
        z      = (x_in - shift) * (-scale).exp()
        logdet = -scale.mean(dim=[1, 2])
        return self._flip(z, 1), logdet

    def _set_sample(self, flag: bool):
        for layer in self.layers:
            if flag:
                layer.enable_sample_mode()
            else:
                layer.disable_sample_mode()

    def reverse(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        x   = self._flip(x, 1)
        pos = self._flip(self.pos_embed, 0)
        self._set_sample(True)

        for i in range(x.size(1) - 1):
            xi = x[:, i:i+1]
            h  = self.proj_in(xi) + pos[i:i+1]
            for layer in self.layers:
                h = layer(h, context=context)      # self-attn with KV cache, cross-attn full
            out         = self.proj_out(h)
            scale, shift = out.chunk(2, dim=-1)
            x[:, i + 1] = x[:, i + 1] * scale[:, 0].exp() + shift[:, 0]

        self._set_sample(False)
        return self._flip(x, 1)


class RangeViewFlowModel(nn.Module):
    """
    Conditional autoregressive normalizing flow for LiDAR range-view prediction.

    Forward pass: maps a future frame to latent z, conditioned on past frames + poses.
    Reverse pass: samples a future frame from z ~ N(0, I), conditioned on past context.
    Autoregressive future prediction: slide the conditioning window frame by frame.
    """

    def __init__(
        self,
        in_channels:      int = 2,
        img_h:            int = 64,
        img_w:            int = 2048,
        patch_h:          int = 4,
        patch_w:          int = 32,
        channels:         int = 512,
        num_flow_blocks:  int = 4,
        layers_per_block: int = 4,
        context_layers:   int = 2,
        head_dim:         int = 64,
    ):
        super().__init__()
        self.patch_h, self.patch_w = patch_h, patch_w
        self.img_h,   self.img_w   = img_h,   img_w

        patch_dim   = in_channels * patch_h * patch_w
        num_patches = (img_h // patch_h) * (img_w // patch_w)

        self.context_encoder = ContextEncoder(patch_dim, channels, context_layers, head_dim)

        self.flow_blocks = nn.ModuleList([
            FlowBlock(patch_dim, channels, num_patches, layers_per_block,
                      head_dim, context_dim=channels, flip=(i % 2 == 1))
            for i in range(num_flow_blocks)
        ])
        self.register_buffer('var', torch.ones(num_patches, patch_dim))

    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        x = x.reshape(B, C, H // ph, ph, W // pw, pw)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * ph * pw)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, _, D = x.shape
        ph, pw  = self.patch_h, self.patch_w
        C       = D // (ph * pw)
        nh, nw  = self.img_h // ph, self.img_w // pw
        x = x.reshape(B, nh, nw, C, ph, pw)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, self.img_h, self.img_w)

    # ------------------------------------------------------------------
    # Context encoding
    # ------------------------------------------------------------------

    def _encode_context(self, past_frames: torch.Tensor,
                        past_poses: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = past_frames.shape
        patches = self.patchify(past_frames.reshape(B * T, C, H, W))   # [B*T, N, D]
        N = patches.size(1)
        patches = patches.reshape(B, T * N, -1)

        ref  = past_poses[:, -1:, :, :]              # last past frame as reference
        p6   = rot_to_6dof(past_poses, ref)          # [B, T, 6]
        return self.context_encoder(patches, p6)     # [B, T*N, channels]

    # ------------------------------------------------------------------
    # Forward (training): x → z  +  log |det J|
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, past_frames: torch.Tensor,
                past_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context  = self._encode_context(past_frames, past_poses)
        z        = self.patchify(x)
        logdets  = torch.zeros(x.size(0), device=x.device)
        for block in self.flow_blocks:
            z, ld = block(z, context)
            logdets = logdets + ld
        return z, logdets

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor) -> torch.Tensor:
        return 0.5 * z.pow(2).mean() - logdets.mean()

    # ------------------------------------------------------------------
    # Reverse (sampling): z → x̂
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, past_frames: torch.Tensor, past_poses: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
        B        = past_frames.size(0)
        context  = self._encode_context(past_frames, past_poses)
        N        = (self.img_h // self.patch_h) * (self.img_w // self.patch_w)
        D        = past_frames.size(2) * self.patch_h * self.patch_w
        z        = torch.randn(B, N, D, device=past_frames.device) * temperature
        z        = z * self.var.sqrt()
        for block in reversed(self.flow_blocks):
            z = block.reverse(z, context)
        return self.unpatchify(z)

    # ------------------------------------------------------------------
    # Autoregressive multi-step prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_sequence(self, past_frames: torch.Tensor, past_poses: torch.Tensor,
                         num_future: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Slide the conditioning window forward, generating one frame at a time.
        Poses for future steps are estimated by extrapolating the last inter-frame delta.
        """
        frames = past_frames.clone()
        poses  = past_poses.clone()
        preds  = []

        for _ in range(num_future):
            pred = self.sample(frames, poses, temperature)
            preds.append(pred)

            last_delta = torch.linalg.inv(poses[:, -2:-1]) @ poses[:, -1:]
            next_pose  = poses[:, -1:] @ last_delta
            frames     = torch.cat([frames[:, 1:], pred.unsqueeze(1)], dim=1)
            poses      = torch.cat([poses[:, 1:],  next_pose],         dim=1)

        return torch.stack(preds, dim=1)   # [B, num_future, C, H, W]
