# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEDotProductAttention,
        IntelTENorm,
    )
except:
    pass

try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention, TENorm

    HAVE_TE = True
except:
    HAVE_TE = False
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


# Use this spec for ModelOpt PTQ and TensorRT-LLM export
def get_gpt_layer_modelopt_spec(
    remap_te_layernorm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    """Mix the native spec with TENorm.

    This is essentially the native local spec except for the layernorm implementation
    is using TENorm from Transformer-Engine. The issue is that FusedLayerNorm from apex
    has stopped supporting RMSNorm needed by llama.
    """
    sharded_state_dict_keys_map = {}
    if remap_te_layernorm:
        sharded_state_dict_keys_map = {
            'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        }
    norm_class = TENorm if HAVE_TE else IntelTENorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=norm_class,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=TEDotProductAttention if HAVE_TE else IntelTEDotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=norm_class if qk_layernorm else IdentityOp,
                    k_layernorm=norm_class if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=norm_class,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            # Map TE-layernorm-fusion keys back
            sharded_state_dict_keys_map=sharded_state_dict_keys_map,
        ),
    )
