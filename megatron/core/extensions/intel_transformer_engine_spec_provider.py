# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional, Tuple

from megatron.core.extensions.intel_transformer_engine import (
    IntelTEColumnParallelGroupedLinear,
    IntelTEColumnParallelLinear,
    IntelTEDotProductAttention,
    IntelTELayerNormColumnParallelLinear,
    IntelTENorm,
    IntelTERowParallelGroupedLinear,
    IntelTERowParallelLinear,
)
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP

try:
    import intel_transformer_engine as te
except:
    print("Could not import Intel TE package")


class IntelTESpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""
    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return IntelTEColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module TE backend uses"""
        return IntelTERowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return None

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module to use for layer norm"""
        return IntelTENorm

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return IntelTEDotProductAttention

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        if (
            moe_use_grouped_gemm
            and IntelTEColumnParallelGroupedLinear is not None
            and not moe_use_legacy_grouped_gemm
        ):
            return TEGroupedMLP, MLPSubmodules(
                linear_fc1=IntelTEColumnParallelGroupedLinear, linear_fc2=IntelTERowParallelGroupedLinear
            )
        elif moe_use_grouped_gemm:
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
            )
            return GroupedMLP, None
        else:
            return SequentialMLP, MLPSubmodules(
                linear_fc1=IntelTEColumnParallelLinear, linear_fc2=IntelTERowParallelLinear
            )
