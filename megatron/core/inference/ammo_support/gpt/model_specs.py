# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_dot_product_attention import FusedDotProductAttention
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEDotProductAttention,
        TENorm,
    )

    HAVE_TE = True
except:
    HAVE_TE = False
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    from megatron.core.transformer.custom_layers.intel_transformer_engine import (
        IntelTEDotProductAttention,
        IntelTENorm,
    )
except:
    pass


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
    if HAVE_TE:
        core_attention_class = TEDotProductAttention
        normalization_class = TENorm
    else:
        enable_fsdpa = False
        try:
            from intel_transformer_engine.utils import is_gaudi3
        except:
            from habana_transformer_engine.utils import is_gaudi3
        if is_gaudi3() and enable_fsdpa:
            core_attention_class = IntelTEDotProductAttention
        elif enable_fsdpa:
            core_attention_class = FusedDotProductAttention
        else:
            core_attention_class = DotProductAttention
        normalization_class = IntelTENorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=normalization_class,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=core_attention_class,
                    linear_proj=RowParallelLinear,
                    q_layernorm=normalization_class if qk_layernorm else IdentityOp,
                    k_layernorm=normalization_class if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=normalization_class,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            # Map TE-layernorm-fusion keys back
            sharded_state_dict_keys_map=sharded_state_dict_keys_map,
        ),
    )
