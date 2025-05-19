# © 2024-2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add, get_bias_dropout_norm_add
from megatron.core.fusions.fused_dot_product_attention import FusedDotProductAttention
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.rmsnorm import RMSNorm
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import is_real_cuda_device_available, is_te_min_version
from megatron.core.version_utils import is_habana_frameworks_min_version

try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEColumnParallelGroupedLinear,
        IntelTEColumnParallelLinear,
        IntelTEDotProductAttention,
        IntelTENorm,
        IntelTERowParallelGroupedLinear,
        IntelTERowParallelGroupedLinearFP8Disabled,
        IntelTERowParallelLinear,
        IntelTERowParallelLinearFp8Disabled,
    )
except:
    pass

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelGroupedLinear,
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

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,
    enable_fsdpa: bool = False,
    fp8_coverage: dict = {},
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Flag to decide the linear layer spec for MoE. Defaults to None.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    mlp = _get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        fp8_coverage=fp8_coverage,
    )

    use_intel_te = not is_real_cuda_device_available()
    if use_intel_te:

        if is_habana_frameworks_min_version("1.21.0") and enable_fsdpa:
            core_attention_class = IntelTEDotProductAttention
        else:
            core_attention_class = FusedDotProductAttention if enable_fsdpa else DotProductAttention
        linear_col_proj = IntelTEColumnParallelLinear
        linear_proj = IntelTERowParallelLinear
        linear_qkv = IntelTEColumnParallelLinear
        normalization_class = IntelTENorm
    else:
        core_attention_class = TEDotProductAttention
        linear_col_proj = TEColumnParallelLinear
        linear_proj = TERowParallelLinear
        linear_qkv = TELayerNormColumnParallelLinear
        normalization_class = TENorm

    if multi_latent_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=normalization_class,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=linear_col_proj,
                        linear_q_down_proj=linear_col_proj,
                        linear_q_up_proj=linear_col_proj,
                        linear_kv_down_proj=linear_col_proj,
                        linear_kv_up_proj=linear_col_proj,
                        core_attention=core_attention_class,
                        linear_proj=linear_proj,
                        q_layernorm=normalization_class if qk_layernorm else IdentityOp,
                        kv_layernorm=normalization_class if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=normalization_class if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:

        # TENorm significantly harms convergence when used
        # for QKLayerNorm if TE Version < 1.9;
        # we instead use the Apex implementation.
        qk_norm = TENorm if is_te_min_version("1.9.0") else LNImpl
        if use_intel_te:
            qk_norm = IntelTENorm

        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=normalization_class if use_intel_te else IdentityOp,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=linear_qkv,
                        core_attention=core_attention_class,
                        linear_proj=linear_proj,
                        q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                        k_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=(
                    normalization_class if use_intel_te or num_experts else IdentityOp
                ),
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    normalization_type: str = 'LayerNorm',
    enable_fsdpa: bool = False,
    use_pre_norm=True,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """

    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    if normalization_type not in ('LayerNorm', 'RMSNorm'):
        raise Exception(
            f'Only LayerNorm and RMSNorm are currently supported, configured {normalization_type}'
        )
    normalization_class = None
    if normalization_type == "LayerNorm":
        normalization_class = LNImpl
    elif normalization_type == "RMSNorm":
        normalization_class = RMSNorm
    core_attention_class = None
    if is_real_cuda_device_available() or not enable_fsdpa:
        core_attention_class = DotProductAttention
    else:
        core_attention_class = FusedDotProductAttention
    get_bda = get_bias_dropout_add if use_pre_norm else get_bias_dropout_norm_add

    if multi_latent_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=normalization_class,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=ColumnParallelLinear,
                        linear_q_down_proj=ColumnParallelLinear,
                        linear_q_up_proj=ColumnParallelLinear,
                        linear_kv_down_proj=ColumnParallelLinear,
                        linear_kv_up_proj=ColumnParallelLinear,
                        core_attention=core_attention_class,
                        linear_proj=RowParallelLinear,
                        q_layernorm=normalization_class if qk_layernorm else IdentityOp,
                        kv_layernorm=normalization_class if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bda,
                pre_mlp_layernorm=normalization_class,
                mlp=mlp,
                mlp_bda=get_bda,
            ),
        )
    else:
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
                self_attn_bda=get_bda,
                pre_mlp_layernorm=normalization_class,
                mlp=mlp,
                mlp_bda=get_bda,
                sharded_state_dict_keys_map={
                    'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                    'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                },
            ),
        )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,
    fp8_coverage: dict = {},
) -> ModuleSpec:
    """Helper function to get module spec for MLP"""
    if num_experts is not None:
        moe_spec = _get_moe_module_spec(
            use_te=True,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            fp8=fp8,
            fp8_coverage=fp8_coverage,
        )
        return moe_spec

    if use_te:
        if is_real_cuda_device_available():
            linear_fc1 = TELayerNormColumnParallelLinear
            linear_fc2 = TERowParallelLinear
        else:
            linear_fc1 = IntelTEColumnParallelLinear
            linear_fc2 = (
                IntelTERowParallelLinear
                if fp8_coverage.get('mlp_row_parallel', True)
                else IntelTERowParallelLinearFp8Disabled
            )
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=linear_fc1 if use_te else ColumnParallelLinear,
            linear_fc2=linear_fc2 if use_te else RowParallelLinear,
        ),
    )


def _get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,
    fp8_coverage: dict = {},
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    if num_experts is None:
        return None
    cuda_available = is_real_cuda_device_available()
    if cuda_available:
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        elif use_te and fp8:
            linear_fc1 = TEColumnParallelLinear
            linear_fc2 = TERowParallelLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear
    else:
        if use_te:
            if moe_grouped_gemm:
                linear_fc1 = IntelTEColumnParallelGroupedLinear
                linear_fc2 = (
                    IntelTERowParallelGroupedLinear
                    if fp8_coverage.get('mlp_row_parallel', True)
                    else IntelTERowParallelGroupedLinearFP8Disabled
                )
            else:
                linear_fc1 = ColumnParallelLinear
                linear_fc2 = RowParallelLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

    if use_te:
        if cuda_available:
            se_linear_fc1 = TEColumnParallelLinear
            se_linear_fc2 = TERowParallelLinear
        else:
            se_linear_fc1 = IntelTEColumnParallelLinear
            # TODO: should we follow fp8 coverage here also ?
            se_linear_fc2 = (
                IntelTERowParallelLinear
                if fp8_coverage.get('mlp_row_parallel', True)
                else IntelTERowParallelLinearFp8Disabled
            )

    if is_real_cuda_device_available():
        use_te_grouped_gemm = use_te and HAVE_TE and TEColumnParallelGroupedLinear is not None
    else:
        use_te_grouped_gemm = use_te and IntelTEColumnParallelGroupedLinear is not None

    return ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=(
                MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                if not moe_grouped_gemm or use_te_grouped_gemm
                else None
            ),
            shared_experts=ModuleSpec(
                module=SharedExpertMLP,
                params={"gate": False},
                submodules=MLPSubmodules(
                    linear_fc1=se_linear_fc1 if use_te else ColumnParallelLinear,
                    linear_fc2=se_linear_fc2 if use_te else RowParallelLinear,
                ),
            ),
        ),
    )


def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    enable_fsdpa: bool = False,
    fp8_coverage: dict = {},
    normalization_type: str = 'LayerNorm',
    use_pre_norm=True,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    if use_transformer_engine:
        layer_norm_impl = TENorm if is_real_cuda_device_available() else IntelTENorm
    else:
        layer_norm_impl = LNImpl
        if config.normalization == "RMSNorm":
            layer_norm_impl = RMSNorm

    # Layer specs.
    dense_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            fp8=config.fp8,
            enable_fsdpa=enable_fsdpa,
            fp8_coverage=fp8_coverage,
        )
        if use_transformer_engine
        else get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            normalization_type=normalization_type,
            enable_fsdpa=enable_fsdpa,
            use_pre_norm=use_pre_norm,
        )
    )
    moe_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            fp8=config.fp8,
            enable_fsdpa=enable_fsdpa,
            fp8_coverage=fp8_coverage,
        )
        if use_transformer_engine
        else get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            normalization_type=normalization_type,
            enable_fsdpa=enable_fsdpa,
            use_pre_norm=use_pre_norm,
        )
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    offset = TransformerLayer._get_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=layer_norm_impl)

    return block_spec
