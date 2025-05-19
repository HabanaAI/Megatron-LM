# © 2024-2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings

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
from megatron.core.utils import is_real_cuda_device_available

try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEColumnParallelLinear,
        IntelTEDotProductAttention,
        IntelTENorm,
        IntelTERowParallelLinear,
    )
except:
    pass

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def get_bert_layer_with_transformer_engine_spec():
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if not HAVE_TE and is_real_cuda_device_available():
        raise ImportError(
            "Transformer Engine is not installed. Please use local Bert layer spec instead."
        )
    if HAVE_TE:
        core_attention_class = TEDotProductAttention
        linear_fc1 = TELayerNormColumnParallelLinear
        linear_fc2 = TERowParallelLinear
        linear_proj = TERowParallelLinear
        linear_qkv = TELayerNormColumnParallelLinear
    else:
        enable_fsdpa = False
        from intel_transformer_engine.utils import is_gaudi3

        if is_gaudi3() and enable_fsdpa:
            core_attention_class = IntelTEDotProductAttention
        elif enable_fsdpa:
            core_attention_class = FusedDotProductAttention
        else:
            core_attention_class = DotProductAttention
        linear_fc1 = IntelTEColumnParallelLinear
        linear_fc2 = IntelTERowParallelLinear
        linear_proj = IntelTERowParallelLinear
        linear_qkv = IntelTEColumnParallelLinear

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp if HAVE_TE else IntelTENorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=linear_qkv,
                    core_attention=core_attention_class,
                    linear_proj=linear_proj,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp if HAVE_TE else IntelTENorm,
            mlp=ModuleSpec(
                module=MLP, submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def __getattr__(name):
    if name == 'bert_layer_with_transformer_engine_spec':
        warnings.warn(
            """Attribute bert_layer_specs.bert_layer_with_transformer_engine_spec is on a
            deprecation track and will be removed in future releases. Please migrate to
            bert_layer_specs.get_bert_layer_with_transformer_engine_spec()."""
        )

        return get_bert_layer_with_transformer_engine_spec()


# Use this spec for an implementation using only modules in megatron core
bert_layer_local_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=LNImpl,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=LNImpl,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
        ),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        },
    ),
)
