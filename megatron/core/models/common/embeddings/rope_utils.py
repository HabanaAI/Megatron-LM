# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

import logging

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.utils import is_real_cuda_device_available, is_te_min_version

logger = logging.getLogger(__name__)

# Prefer fused RoPE from Apex as we need the `transpose_output_memory` argument for the bshd trick.
# See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/2469.
try:
    from apex.transformer.functional import fused_apply_rotary_pos_emb
except ImportError:
    try:
        from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
    except:
        fused_apply_rotary_pos_emb = None


try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
except ImportError:
    try:
        from apex.transformer.functional import fused_apply_rotary_pos_emb_thd
    except ImportError:
        fused_apply_rotary_pos_emb_thd = None


try:
    from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_flash
except ImportError:
    apply_rotary_emb_flash = None


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV1
except:
    RotaryPosEmbeddingHelperV1 = None


__all__ = ['apply_rotary_emb_flash']


def get_pos_emb_on_this_cp_rank(pos_emb: Tensor, seq_dim: int) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    cos_cached: Tensor = None,
    sin_cached: Tensor = None,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    t_pass = None
    if t.shape[-1] != rot_dim:
        # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    if RotaryPosEmbeddingHelperV1 is None:
        # first part is cosine component
        # second part is sine component, need to change signs with _rotate_half method
        cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
        sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    else:
        if cos_cached is None or sin_cached is None or t.shape[0] != cos_cached.shape[0]:
            freqs_ = freqs[: t.shape[0]]
            cos_cached = (freqs_.cos() * mscale).to(t.dtype)
            sin_cached = (freqs_.sin() * mscale).to(t.dtype)
        t = RotaryPosEmbeddingHelperV1.apply(
            t, cos_cached, sin_cached, 0
        )  # offset already used in RotaryEmbedding.forward

    if t_pass is None:
        return t
    return torch.cat((t, t_pass), dim=-1)


def _get_thd_freqs_on_this_cp_rank(cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor) -> Tensor:
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )
    else:
        return freqs[: x.size(0)]


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    cos_cached: Tensor = None,
    sin_cached: Tensor = None,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cu_seqlens = cu_seqlens // cp_size
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    return torch.cat(
        [
            _apply_rotary_pos_emb_bshd(
                x.unsqueeze(1),
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs),
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                cos_cached=cos_cached,
                sin_cached=sin_cached,
            )
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
    cos_cached: Tensor = None,
    sin_cached: Tensor = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """

    if config.apply_rope_fusion and is_real_cuda_device_available():
        if cu_seqlens is None:
            assert fused_apply_rotary_pos_emb is not None, "apply_rope_fusion is not available."
            return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
        else:
            assert fused_apply_rotary_pos_emb_thd is not None, "apply_rope_fusion is not available."
            cp_size = parallel_state.get_context_parallel_world_size()
            if cp_size > 1:
                if not is_te_min_version("1.11.0", check_equality=False):
                    raise ValueError("Only TE >= 1.12 supports RoPE fusion for THD format with CP.")
                return fused_apply_rotary_pos_emb_thd(
                    t,
                    cu_seqlens,
                    freqs,
                    cp_size=cp_size,
                    cp_rank=parallel_state.get_context_parallel_rank(),
                )
            else:
                return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
    else:
        if not config.apply_rope_fusion:
            global RotaryPosEmbeddingHelperV1
            RotaryPosEmbeddingHelperV1 = None
        if cu_seqlens is None:
            return _apply_rotary_pos_emb_bshd(
                t,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
                cos_cached=cos_cached,
                sin_cached=sin_cached,
            )
        else:
            return _apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
            )


def apply_rotary_pos_emb_with_cos_sin(
    t: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_interleaved: bool = False,
    cos_cached: Tensor = None,
    sin_cached: Tensor = None,
) -> Tensor:
    """
    This function applies rotary positional embedding to the target tensor t
    using precomputed cos and sin of size (seq_len, d_rot / 2)
    """
    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    if apply_rotary_emb_flash is None:
        # Combine cos and sin into freqs
        freqs = torch.stack([cos, sin], dim=-1).flatten(start_dim=-2)

        # Expand freqs to match t's shape
        while freqs.dim() < t.dim():
            freqs = freqs.unsqueeze(1)
        freqs = freqs.expand(t.shape[:-1] + (-1,))

        y = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=False,
            mscale=1.0,
            cos_cached=cos_cached,
            sin_cached=sin_cached,
        )
    else:
        # Use Flash Attention's optimized kernel for rotary embedding
        t = t.permute(1, 0, 2, 3)
        y = apply_rotary_emb_flash(t, cos, sin, rotary_interleaved)
        y = y.permute(1, 0, 2, 3)

    return y
