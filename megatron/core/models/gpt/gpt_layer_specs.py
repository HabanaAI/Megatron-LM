# © 2024-2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional, Union

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add, get_bias_dropout_norm_add
from megatron.core.fusions.fused_dot_product_attention import FusedDotProductAttention
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_offset,
    get_mtp_layer_spec_for_backend,
    get_mtp_num_layers_to_build,
)
from megatron.core.transformer.rmsnorm import RMSNorm
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import is_real_cuda_device_available, is_te_min_version
from megatron.core.version_utils import is_habana_frameworks_min_version

try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEDotProductAttention,
        IntelTEDotProductAttentionFp8Disabled,
        IntelTENorm,
        IntelTERowParallelLinear,
        IntelTERowParallelLinearFp8Disabled,
        IntelTERowParallelLinearFP8SmoothSwiglu,
    )
    from megatron.core.extensions.intel_transformer_engine_spec_provider import IntelTESpecProvider
except:
    pass

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEFusedMLP, TENorm
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import nvidia_kitchen  # pylint: disable=unused-import

    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    enable_fsdpa: bool = False,
    fp8_coverage: dict = {},
    moe_dynamic_hpu: Optional[bool] = False,
    fp8_smooth_swiglu: bool = False,
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_kitchen: bool = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_te_op_fuser (bool, optional): Use Transformer Engine's operation-based API, which may
                                          enable certain operation fusions. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules

    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    use_intel_te = not is_real_cuda_device_available()
    if use_kitchen and not use_intel_te:
        assert HAVE_KITCHEN
        backend: BackendSpecProvider = KitchenSpecProvider(fallback=TESpecProvider())
        if use_te_op_fuser:
            raise AssertionError("use_te_op_fuser not compatible with using kitchen in mlp.")
    elif use_intel_te:
        from megatron.core.extensions.intel_transformer_engine_spec_provider import (
            IntelTESpecProvider,
        )

        backend: BackendSpecProvider = IntelTESpecProvider()
    else:
        backend = TESpecProvider()
    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        fp8=fp8,
        fp8_coverage=fp8_coverage,
        moe_dynamic_hpu=moe_dynamic_hpu,
        fp8_smooth_swiglu=fp8_smooth_swiglu,
        use_te_op_fuser=use_te_op_fuser,
    )

    core_attention_class = backend.core_attention()
    if use_intel_te:
        if is_habana_frameworks_min_version("1.21.0") and enable_fsdpa:
            core_attention_class = (
                IntelTEDotProductAttention
                if fp8_coverage.get('attention', True)
                else IntelTEDotProductAttentionFp8Disabled
            )
        else:
            core_attention_class = FusedDotProductAttention if enable_fsdpa else DotProductAttention
        normalization_class = IntelTENorm
    else:
        normalization_class = TENorm

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        linear_q_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm and HAVE_TE
            else backend.column_parallel_linear()
        )
        linear_kv_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm and HAVE_TE
            else backend.column_parallel_linear()
        )
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=backend.layer_norm(),
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_q_down_proj=backend.column_parallel_linear(),
                        linear_q_up_proj=linear_q_up_proj,
                        linear_kv_down_proj=backend.column_parallel_linear(),
                        linear_kv_up_proj=linear_kv_up_proj,
                        core_attention=core_attention_class,
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            IdentityOp
                            if HAVE_TE
                            else normalization_class if qk_layernorm else IdentityOp
                        ),
                        kv_layernorm=(
                            IdentityOp
                            if HAVE_TE
                            else normalization_class if qk_layernorm else IdentityOp
                        ),
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
    else:

        # TENorm significantly harms convergence when used
        # for QKLayerNorm if TE Version < 1.9;
        # we instead use the Apex implementation.
        qk_norm = backend.layer_norm(for_qk=True) if is_te_min_version("1.9.0") else LNImpl
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
                        linear_qkv=(
                            backend.column_parallel_layer_norm_linear()
                            if HAVE_TE
                            else backend.column_parallel_linear()
                        ),
                        core_attention=core_attention_class,
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=(
                    normalization_class if use_intel_te or num_experts else IdentityOp
                ),
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                    "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                    "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                    "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                    "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                    "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
                },
            ),
        )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    normalization_type: str = 'LayerNorm',
    enable_fsdpa: bool = False,
    use_pre_norm=True,
    moe_dynamic_hpu: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    use_kitchen: bool = False,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """

    if use_kitchen:
        assert HAVE_KITCHEN
        backend = KitchenSpecProvider(fallback=LocalSpecProvider())
    else:
        backend = LocalSpecProvider()
    # Adjust for RMS norm.
    if normalization_type == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_local_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        use_te=False,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        moe_dynamic_hpu=moe_dynamic_hpu,
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
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=layer_norm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_q_down_proj=backend.column_parallel_linear(),
                        linear_q_up_proj=backend.column_parallel_linear(),
                        linear_kv_down_proj=backend.column_parallel_linear(),
                        linear_kv_up_proj=backend.column_parallel_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                        kv_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bda,
                pre_mlp_layernorm=layer_norm,
                mlp=mlp,
                mlp_bda=get_bda,
            ),
        )
    else:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=layer_norm,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=backend.column_parallel_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bda,
                pre_mlp_layernorm=layer_norm,
                mlp=mlp,
                mlp_bda=get_bda,
                sharded_state_dict_keys_map={
                    "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                    "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
                },
            ),
        )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    fp8_coverage: dict = {},
    moe_dynamic_hpu: Optional[bool] = False,
    fp8_smooth_swiglu: bool = False,
) -> ModuleSpec:
    warnings.warn(
        """This private function is on a deprecation track. Please switch to `get_mlp_module_spec`
        since it will be removed in a future release."""
    )

    return get_mlp_module_spec(
        use_te=use_te,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        fp8_smooth_swiglu=fp8_smooth_swiglu,
        fp8_coverage=fp8_coverage,
        moe_dynamic_hpu=moe_dynamic_hpu,
    )


def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    fp8_coverage: dict = {},
    moe_dynamic_hpu: Optional[bool] = False,
    fp8_smooth_swiglu: bool = False,
    use_te_op_fuser: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    if use_te_op_fuser:
        if not is_te_min_version("1.13.0"):
            raise ValueError(
                "Transformer Engine operation-based API requires Transformer Engine 1.13+"
            )
        if num_experts is not None:
            raise ValueError(
                "Transformer Engine operation-based API does not support mixture-of-experts"
            )

    if use_te is not None and use_te:
        backend = TESpecProvider() if HAVE_TE else IntelTESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        fp8=fp8,
        use_te_op_fuser=use_te_op_fuser,
        fp8_coverage=fp8_coverage,
        moe_dynamic_hpu=moe_dynamic_hpu,
        fp8_smooth_swiglu=fp8_smooth_swiglu,
    )


def get_mlp_module_spec_for_backend(
    use_te: Optional[bool] = True,
    backend: BackendSpecProvider = None,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    use_te_op_fuser: Optional[bool] = False,
    fp8_coverage: dict = {},
    moe_dynamic_hpu: Optional[bool] = False,
    fp8_smooth_swiglu: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    linear_fc2 = backend.row_parallel_linear()
    use_intel_te = not is_real_cuda_device_available()
    if use_te and use_intel_te:
        linear_fc2 = (
            IntelTERowParallelLinearFP8SmoothSwiglu
            if fp8_smooth_swiglu
            else (
                IntelTERowParallelLinear
                if fp8_coverage.get('mlp_row_parallel', True)
                else IntelTERowParallelLinearFp8Disabled
            )
        )
        assert linear_fc2 is not None

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        if use_te_op_fuser:
            return ModuleSpec(module=TEFusedMLP)
        elif backend.fuse_layernorm_and_linear():
            linear_fc1 = backend.column_parallel_layer_norm_linear()
            if use_intel_te:
                linear_fc1 = backend.column_parallel_linear()
            assert linear_fc1 is not None
        else:
            linear_fc1 = backend.column_parallel_linear()
        return ModuleSpec(
            module=MLP, submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec_for_backend(
            backend=backend,
            use_te=use_te,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            fp8_coverage=fp8_coverage,
            moe_dynamic_hpu=moe_dynamic_hpu,
            # We need to use fp8 here, as IntelDynamicMLP for bf16 and fp8
            # have separate implementations
            fp8=fp8 is not None,
            fp8_smooth_swiglu=fp8_smooth_swiglu,
        )


def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    enable_fsdpa: bool = False,
    fp8_coverage: dict = {},
    normalization_type: str = 'LayerNorm',
    use_pre_norm=True,
    moe_dynamic_hpu: Optional[bool] = False,
    fp8_smooth_swiglu: bool = False,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
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
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            enable_fsdpa=enable_fsdpa,
            fp8_coverage=fp8_coverage,
            moe_dynamic_hpu=moe_dynamic_hpu,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
        if use_transformer_engine
        else get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            fp8=config.fp8,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            normalization_type=normalization_type,
            enable_fsdpa=enable_fsdpa,
            use_pre_norm=use_pre_norm,
            moe_dynamic_hpu=moe_dynamic_hpu,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    )
    moe_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            fp8=config.fp8,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            enable_fsdpa=enable_fsdpa,
            fp8_coverage=fp8_coverage,
            moe_dynamic_hpu=moe_dynamic_hpu,
            fp8_smooth_swiglu=fp8_smooth_swiglu,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
        if use_transformer_engine
        else get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            fp8=config.fp8,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            normalization_type=normalization_type,
            enable_fsdpa=enable_fsdpa,
            use_pre_norm=use_pre_norm,
            moe_dynamic_hpu=moe_dynamic_hpu,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
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
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )

    return block_spec


def get_gpt_mtp_block_spec(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    use_transformer_engine: bool,
    vp_stage: Optional[int] = None,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    if use_transformer_engine:
        backend: BackendSpecProvider = (
            KitchenSpecProvider(fallback=TESpecProvider())
            if config.use_kitchen
            else TESpecProvider()
        )
    else:
        backend = (
            KitchenSpecProvider(fallback=LocalSpecProvider())
            if config.use_kitchen
            else LocalSpecProvider()
        )
    return get_gpt_mtp_block_spec_for_backend(
        config=config, spec=spec, backend=backend, vp_stage=vp_stage
    )


def get_gpt_mtp_block_spec_for_backend(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    backend: BackendSpecProvider,
    vp_stage: Optional[int] = None,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = get_mtp_num_layers_to_build(config, vp_stage=vp_stage)
    if num_layers_to_build == 0:
        return None

    if isinstance(spec, TransformerBlockSubmodules):
        # get the spec for the last layer of decoder block
        transformer_layer_spec = spec.layer_specs[-1]
    elif isinstance(spec, ModuleSpec) and spec.module == TransformerLayer:
        transformer_layer_spec = spec
    else:
        raise ValueError(f"Invalid spec: {spec}")

    mtp_layer_spec = get_mtp_layer_spec_for_backend(
        transformer_layer_spec=transformer_layer_spec, backend=backend
    )
    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    offset = get_mtp_layer_offset(config)
    # split the mtp layer specs to only include the layers that are built in this pipeline stage.
    mtp_layer_specs = mtp_layer_specs[offset : offset + num_layers_to_build]
    if len(mtp_layer_specs) > 0:
        assert (
            len(mtp_layer_specs) == config.mtp_num_layers
        ), +f"currently all of the mtp layers must stage in the same pipeline stage."
        mtp_block_spec = MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs)
    else:
        mtp_block_spec = None

    return mtp_block_spec
