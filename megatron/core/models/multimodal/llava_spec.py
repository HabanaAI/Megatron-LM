# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
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
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
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


def decoder_model_with_transformer_engine_default_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    """LLava decoder TE spec (uses Transformer Engine components)."""
    mlp = _get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    if HAVE_TE:
        core_attention_class = TEDotProductAttention
        linear_proj = TERowParallelLinear
        linear_qkv = TELayerNormColumnParallelLinear
        normalization_class = TENorm
    else:
        from intel_transformer_engine.utils import is_gaudi3

        enable_fsdpa = False
        if is_gaudi3() and enable_fsdpa:
            core_attention_class = IntelTEDotProductAttention
        elif enable_fsdpa:
            core_attention_class = FusedDotProductAttention
        else:
            core_attention_class = DotProductAttention
        linear_proj = IntelTERowParallelLinear
        linear_qkv = IntelTEColumnParallelLinear
        normalization_class = IntelTENorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp if HAVE_TE else normalization_class,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=linear_qkv,
                    core_attention=core_attention_class,
                    linear_proj=linear_proj,
                    q_layernorm=normalization_class if qk_layernorm else IdentityOp,
                    k_layernorm=normalization_class if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp if HAVE_TE else normalization_class,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def decoder_model_with_local_default_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    """LLava decoder local spec."""
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
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
