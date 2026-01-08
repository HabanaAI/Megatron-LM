# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import (
    GroupedMLP,
    IntelDynamicMLP,
    IntelFP8DynamicMLP,
    SequentialMLP,
    TEGroupedMLP,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.version_utils import is_habana_frameworks_min_version

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEColumnParallelGroupedLinear,
        IntelTEColumnParallelLinear,
        IntelTEMixtureOfExperts,
        IntelTEMixtureOfExpertsSmoothSwiglu,
        IntelTERowParallelGroupedLinear,
        IntelTERowParallelGroupedLinearFP8Disabled,
        IntelTERowParallelLinear,
        IntelTERowParallelLinearFp8Disabled,
    )
except:
    pass


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    fp8_coverage: dict = {},
    moe_dynamic_hpu: Optional[bool] = False,
    fp8: Optional[bool] = False,
    fp8_smooth_swiglu: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_moe_module_spec_for_backend(
        use_te=use_te,
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        fp8_coverage=fp8_coverage,
        moe_dynamic_hpu=moe_dynamic_hpu,
        fp8=fp8,
        fp8_smooth_swiglu=fp8_smooth_swiglu,
    )


def get_moe_module_spec_for_backend(
        use_te: Optional[bool] = True,
        backend: BackendSpecProvider = None,
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
        moe_use_legacy_grouped_gemm: Optional[bool] = False,
        fp8_coverage: dict = {},
        moe_dynamic_hpu: Optional[bool] = False,
        fp8: Optional[bool] = False,
        fp8_smooth_swiglu: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = (
        TERowParallelLinear if HAVE_TE else IntelTERowParallelLinear
        if fp8_coverage.get('mlp_row_parallel', True)
        else IntelTERowParallelLinearFp8Disabled
    )
    mlp = MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)

    # [REBASE Comment] Check for a Real Cuda Device somewhere here.
    expert_module, expert_submodule = backend.grouped_mlp_modules(
        moe_grouped_gemm is not None and moe_grouped_gemm,
        moe_use_legacy_grouped_gemm is not None and moe_use_legacy_grouped_gemm,
    )

    # [REBASE Comment] Below if-else clauses does moe module initializations.
    # This probably needs to be moved to moe module specs.
    # experts spec

    if moe_grouped_gemm:
        # use GroupedMLP
        if (
            use_te
            and (
                (HAVE_TE and TEColumnParallelGroupedLinear is not None)
                or (not HAVE_TE and IntelTEColumnParallelGroupedLinear is not None)
            )
            and not moe_use_legacy_grouped_gemm
        ):
            # use TEGroupedLinear
            expert_module = TEGroupedMLP
            iterpgl = None
            if not HAVE_TE:
                iterpgl = (
                    IntelTERowParallelGroupedLinear
                    if fp8_coverage.get('mlp_row_parallel', True)
                    else IntelTERowParallelGroupedLinearFP8Disabled
                )
            expert_submodule = MLPSubmodules(
                linear_fc1=(
                    TEColumnParallelGroupedLinear if HAVE_TE else IntelTEColumnParallelGroupedLinear
                ),
                linear_fc2=TERowParallelGroupedLinear if HAVE_TE else iterpgl,
            )
        else:
            # use legacy GroupedMLP
            expert_module = GroupedMLP
            expert_submodule = None
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
    elif moe_dynamic_hpu and fp8 and not HAVE_TE:
        expert_module = IntelFP8DynamicMLP
        submodule_cls = (
            IntelTEMixtureOfExpertsSmoothSwiglu if fp8_smooth_swiglu else IntelTEMixtureOfExperts
        )
        expert_submodule = MLPSubmodules(linear_fc12=submodule_cls)
    elif moe_dynamic_hpu and not HAVE_TE:
        expert_module = IntelDynamicMLP
        expert_submodule = None
    else:
        # use SequentialMLP
        expert_module = SequentialMLP
        if use_te and not (
            is_te_min_version("1.7.0.dev0") or is_habana_frameworks_min_version("1.21.0")
        ):
            warnings.warn(
                "Only transformer-engine>=1.7.0 supports MoE experts, "
                f"but your version is {get_te_version()}. Use local linear implementation instead."
            )
            expert_submodule = MLPSubmodules(
                linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
            )
        else:
            expert_submodule = mlp

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec
