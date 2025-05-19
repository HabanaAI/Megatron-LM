# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

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
except:
    HAVE_TE = False

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_dot_product_attention import FusedDotProductAttention
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

if HAVE_TE:
    core_attention_class = TEDotProductAttention
else:
    from intel_transformer_engine.utils import is_gaudi3

    enable_fsdpa = False
    cp_enabled = False
    if (is_gaudi3() or cp_enabled) and enable_fsdpa:
        core_attention_class = IntelTEDotProductAttention
    elif enable_fsdpa:
        core_attention_class = FusedDotProductAttention
    else:
        core_attention_class = DotProductAttention

mamba_stack_spec = ModuleSpec(
    module=MambaStack,
    submodules=MambaStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                norm=IdentityOp if HAVE_TE else IntelTENorm,
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=(
                            TELayerNormColumnParallelLinear
                            if HAVE_TE
                            else IntelTEColumnParallelLinear
                        ),
                        out_proj=TERowParallelLinear if HAVE_TE else IntelTERowParallelLinear,
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=IdentityOp if HAVE_TE else IntelTENorm,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=(
                            TELayerNormColumnParallelLinear
                            if HAVE_TE
                            else IntelTEColumnParallelLinear
                        ),
                        core_attention=core_attention_class,
                        linear_proj=TERowParallelLinear if HAVE_TE else IntelTERowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=IdentityOp if HAVE_TE else IntelTENorm,
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=(
                            TELayerNormColumnParallelLinear
                            if HAVE_TE
                            else IntelTEColumnParallelLinear
                        ),
                        linear_fc2=TERowParallelLinear if HAVE_TE else IntelTERowParallelLinear,
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
    ),
)
