# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except:
    HAVE_TE = False

try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEColumnParallelLinear,
        IntelTEDotProductAttention,
        IntelTENorm,
        IntelTERowParallelLinear,
    )
except:
    pass

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_dot_product_attention import FusedDotProductAttention
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_vit_layer_with_transformer_engine_spec() -> ModuleSpec:
    '''
    Returns ViT layer spec with Transformer Engine layers
    '''
    mlp = _get_mlp_module_spec(use_te=True)
    if HAVE_TE:
        core_attention_class = TEDotProductAttention
        linear_qkv = TELayerNormColumnParallelLinear
        linear_proj = TERowParallelLinear
    else:
        enable_fsdpa = False
        from intel_transformer_engine.utils import is_gaudi3

        if is_gaudi3() and enable_fsdpa:
            core_attention_class = IntelTEDotProductAttention
        elif enable_fsdpa:
            core_attention_class = FusedDotProductAttention
        else:
            core_attention_class = DotProductAttention
        linear_qkv = IntelTEColumnParallelLinear
        linear_proj = IntelTERowParallelLinear
        normalization_class = IntelTENorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=normalization_class if not HAVE_TE else IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params=(
                    {"attn_mask_type": AttnMaskType.no_mask}
                    if HAVE_TE
                    else {"attn_mask_type": AttnMaskType.causal}
                ),
                submodules=SelfAttentionSubmodules(
                    linear_qkv=linear_qkv,
                    core_attention=core_attention_class,
                    linear_proj=linear_proj,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=normalization_class if not HAVE_TE else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_vit_layer_with_local_spec() -> ModuleSpec:
    '''
    Returns ViT layer spec with Mcore local layers
    '''
    mlp = _get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    if HAVE_TE:
        linear_fc1 = TELayerNormColumnParallelLinear
        linear_fc2 = TERowParallelLinear
    else:
        linear_fc1 = IntelTEColumnParallelLinear
        linear_fc2 = IntelTERowParallelLinear
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=linear_fc1 if use_te else ColumnParallelLinear,
            linear_fc2=linear_fc2 if use_te else RowParallelLinear,
        ),
    )
