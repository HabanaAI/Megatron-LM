# © 2024-2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Specs for Retro encoder."""

from megatron.core.fusions.fused_dot_product_attention import FusedDotProductAttention
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.retro.config import RetroConfig
from megatron.core.models.retro.encoder_attention import (
    RetroEncoderBiasDropoutAdd,
    RetroEncoderCrossAttention,
    RetroEncoderLayerNorm,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import CrossAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
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
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm

try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEColumnParallelLinear,
        IntelTEDotProductAttention,
        IntelTENorm,
        IntelTERowParallelLinear,
    )
except:
    pass


def get_retro_encoder_layer_te_spec() -> ModuleSpec:
    """Retro encoder TE spec (uses Transformer Engine components).

    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.

    Returns:
        A module spec if Transformer Engine modules.
    """
    spec = get_gpt_layer_with_transformer_engine_spec()
    if HAVE_TE:
        core_attention_class = TEDotProductAttention
        linear_fc1 = TEColumnParallelLinear
        linear_fc2 = TERowParallelLinear
        linear_kv = TEColumnParallelLinear
        linear_proj = TERowParallelLinear
        linear_q = TEColumnParallelLinear
        normalization_class = TENorm
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
        linear_kv = IntelTEColumnParallelLinear
        linear_proj = IntelTERowParallelLinear
        linear_q = IntelTEColumnParallelLinear
        normalization_class = IntelTENorm
    spec.submodules.pre_cross_attn_layernorm = normalization_class
    spec.submodules.cross_attention = ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={"attn_mask_type": AttnMaskType.padding},
        submodules=CrossAttentionSubmodules(
            linear_q=linear_q,
            linear_kv=linear_kv,
            core_attention=core_attention_class,
            linear_proj=linear_proj,
        ),
    )
    spec.submodules.cross_attn_bda = ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm = ModuleSpec(
        module=RetroEncoderLayerNorm, submodules=normalization_class
    )
    spec.submodules.mlp = ModuleSpec(
        module=MLP, submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
    )
    return spec


def get_retro_encoder_layer_local_spec() -> ModuleSpec:
    """Retro encoder local spec (uses Megatron-Core components).

    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.

    Returns:
        A module spec if local modules.
    """
    spec = get_gpt_layer_local_spec()
    spec.submodules.pre_cross_attn_layernorm = LNImpl
    spec.submodules.cross_attention = ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={"attn_mask_type": AttnMaskType.padding},
        submodules=CrossAttentionSubmodules(
            linear_q=ColumnParallelLinear,
            linear_kv=ColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=RowParallelLinear,
        ),
    )
    spec.submodules.cross_attn_bda = ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm = ModuleSpec(module=RetroEncoderLayerNorm, submodules=LNImpl)
    spec.submodules.mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
    )
    spec.submodules.sharded_state_dict_keys_map = {
        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_'
    }  # pre_mlp_layernorm doesn't need remapping
    return spec


def get_retro_encoder_block_spec(
    config: RetroConfig, use_transformer_engine: bool
) -> TransformerBlockSubmodules:
    """Retro encoder block spec.

    The retro encoder block consists of one customized Retro encoder layer
    (layer 1), and all of the following layers are standard GPT layers.

    Args:
      config (RetroConfig): Retro config.
      use_transformer_engine (bool): If True, use Transformer Engine (instead of local modules).

    Returns:
        Transformer block submodules for the given spec.
    """

    # Num layers.
    num_layers = config.retro_encoder_num_layers
    retro_layer_numbers = [1]

    # Layer specs.
    gpt_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec()
        if use_transformer_engine
        else get_gpt_layer_local_spec()
    )
    get_retro_encoder_layer_spec = (
        get_retro_encoder_layer_te_spec
        if use_transformer_engine
        else get_retro_encoder_layer_local_spec
    )
    retro_layer_spec = get_retro_encoder_layer_spec()
    if use_transformer_engine:
        if HAVE_TE:
            module = TEDotProductAttention
        else:
            enable_fsdpa = False
            from intel_transformer_engine.utils import is_gaudi3

            if is_gaudi3() and enable_fsdpa:
                module = IntelTEDotProductAttention
            elif enable_fsdpa:
                module = FusedDotProductAttention
            else:
                module = DotProductAttention
    else:
        module = DotProductAttention
    for spec in (gpt_layer_spec, retro_layer_spec):
        spec.params["hidden_dropout"] = config.retro_encoder_hidden_dropout
        spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.padding
        spec.submodules.self_attention.submodules.core_attention = ModuleSpec(
            module=module, params={"attention_dropout": config.retro_encoder_attention_dropout}
        )

    layer_specs = []
    for layer_number in range(1, num_layers + 1):
        if layer_number in retro_layer_numbers:
            layer_specs.append(retro_layer_spec)
        else:
            layer_specs.append(gpt_layer_spec)

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs)

    return block_spec
