# © 2024-2025 Intel Corporation
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Intel Corporation proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.

from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.rmsnorm import RMSNorm
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import divide
from megatron.core.version_utils import is_habana_frameworks_min_version

try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    HAVE_APEX = False
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm

try:
    import intel_transformer_engine as te
    from intel_transformer_engine.utils import is_gaudi3
except:
    print("Could not import Intel TE package")


def condition_init_method(config, init_method):
    return init_method if config.perform_initialization else (lambda w: None)


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {"params_dtype": config.params_dtype}

    extra_transformer_engine_kwargs["device"] = "hpu"
    return extra_transformer_engine_kwargs


class IntelTENorm:
    """
    A conditional wrapper to initialize an instance of local `LayerNorm` or `RMSNorm`
    based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        if config.normalization == "LayerNorm":
            instance = LNImpl(
                config=config,
                hidden_size=hidden_size,
                eps=eps,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
            )
        elif config.normalization == "RMSNorm":
            instance = RMSNorm(config=config, hidden_size=hidden_size, eps=eps)
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance


class IntelTELinear(te.Linear):
    """
    Wrapper for the Intel Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: str,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: str = None,
        force_disable_fp8: bool = False,
    ):
        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        if self.config.cache_fp8_weight_fwd:
            self.is_first_microbatch = True
        else:
            self.is_first_microbatch = None
        self.force_disable_fp8 = force_disable_fp8
        if skip_weight_param_allocation:
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        extra_kwargs = _get_extra_te_kwargs(config)

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=condition_init_method(config, init_method),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=parallel_mode,
            minimize_memory=not self.config.cache_fp8_weight,
            **extra_kwargs,
        )

    def forward(self, x):
        _is_first_microbatch = self.is_first_microbatch
        ctx = te.fp8_autocast(enabled=False) if self.force_disable_fp8 else nullcontext()
        with ctx:
            out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        if self.is_first_microbatch:
            self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None


class IntelTEColumnParallelLinear(IntelTELinear):
    """
    Wrapper for the Intel Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )


class IntelTERowParallelLinear(IntelTELinear):
    """
    Wrapper for the Intel Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str = None,
        **kwargs,
    ):
        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,  # We don't currently use this for row parallel layers
            **kwargs,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )


class IntelTERowParallelLinearFp8Disabled(IntelTERowParallelLinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer and force-disabled FP8.
    """

    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__(
            input_size=input_size, output_size=output_size, **kwargs, force_disable_fp8=True
        )


if is_habana_frameworks_min_version("1.21.0"):

    class IntelTEGroupedLinear(te.GroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer.

        Note that if Megatron's parallel_state has not been initialized
        yet, the tp_group passed to TE will be None and must be set later
        via set_tensor_parallel_group().
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            parallel_mode: str,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            force_disable_fp8=False,
        ):
            self.config = config
            # If `force_disable_fp8` is `True`, TE module runs using `params_dtype` from `config`.
            self.force_disable_fp8 = force_disable_fp8

            # TE returns a zero length Tensor when bias=False and
            # return_bias=True, but we prefer None. So in that case we
            # tell TE to not return the bias, and return None
            # ourselves. This way our forward always returns two values
            # and we don't have to deal with the zero length Tensor.
            self.te_return_bias = skip_bias_add and bias
            self.is_first_microbatch = True

            extra_kwargs = _get_extra_te_kwargs(config)

            self.expert_parallel = self.config.expert_model_parallel_size > 1
            if self.expert_parallel:
                extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

            # For MoE models, the comms between TP and EP group is explicitly handled by
            # MoE token dispatcher. So we disable comms by making TE agnostic of model parallel.
            self.explicit_expert_comm = is_expert and (
                config.tensor_model_parallel_size > 1 or self.expert_parallel
            )
            tp_group = get_tensor_model_parallel_group(check_initialized=False)
            if self.explicit_expert_comm and config.expert_tensor_parallel_size > 1:
                tp_size = parallel_state.get_expert_tensor_and_model_parallel_world_size()
            else:
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if self.explicit_expert_comm:
                # Nvidia sets `tp_group` to `None`, we need it for amax reduction,
                # hence we leave it unchanged.
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                parallel_mode = None

            super().__init__(
                num_gemms=num_gemms,
                in_features=input_size,
                out_features=output_size,
                sequence_parallel=self.config.sequence_parallel,
                tp_group=tp_group,
                tp_size=tp_size,
                get_rng_state_tracker=(
                    get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
                ),
                init_method=condition_init_method(config, init_method),
                bias=bias,
                return_bias=self.te_return_bias,
                parallel_mode=parallel_mode,
                **extra_kwargs,
            )

            for param in self.parameters():
                setattr(param, 'allreduce', not (is_expert and self.expert_parallel))

        def forward(self, x, m_splits):
            _is_first_microbatch = self.is_first_microbatch
            ctx = te.fp8_autocast(enabled=False) if self.force_disable_fp8 else nullcontext()
            with ctx:
                out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)
            self.is_first_microbatch = False

            # TE only returns a tuple when return_bias is True, otherwise
            # it returns a single Tensor, we always want to return two
            # values regardless of the arguments.
            if self.te_return_bias:
                return out
            return out, None

        def _sharded_state_dict_grouped(
            self, tp_axis_map, prefix='', sharded_offsets=(), metadata=None
        ):
            """
            prefix should be module_name to make keys identical to sequetial ones.
            """
            sharded_state_dict = {}
            full_state_dict = self.state_dict(prefix='', keep_vars=True)
            num_global_experts = (
                parallel_state.get_expert_model_parallel_world_size() * self.num_gemms
            )
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_gemms
            )
            ep_axis = len(sharded_offsets)
            for gemm_idx in range(self.num_gemms):
                state_dict = {
                    f'{gemm_idx}.weight': full_state_dict[f'weight{gemm_idx}'],
                    f'{gemm_idx}._extra_state': full_state_dict['_extra_state'],
                }
                if self.use_bias:
                    state_dict[f'{gemm_idx}.bias'] = full_state_dict[f'bias{gemm_idx}']
                sub_sd = make_sharded_tensors_for_checkpoint(
                    state_dict,
                    '',
                    tp_axis_map,
                    (
                        *sharded_offsets,
                        (ep_axis, local_expert_indices_offset + gemm_idx, num_global_experts),
                    ),
                )
                # Remove expert layers indexing from sharded keys
                replace_prefix_for_sharding(sub_sd, f'{gemm_idx}.', prefix)
                sharded_state_dict.update(
                    {
                        f'{prefix}weight{gemm_idx}': sub_sd[f'{gemm_idx}.weight'],
                        # TODO: TE's GroupedLinear only has one _extra_state for all experts.
                        # We need sharding or build/merge fn to handle _extra_state correctly.
                        f'{prefix}_extra_state{"" if gemm_idx == 0 else gemm_idx}': sub_sd[
                            f'{gemm_idx}._extra_state'
                        ],
                    }
                )
                if self.use_bias:
                    sharded_state_dict[f'{prefix}bias{gemm_idx}'] = sub_sd[f'{gemm_idx}.bias']
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in sharded_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                sh_ten.replica_id = (
                    *replica_id[:2],
                    parallel_state.get_data_modulo_expert_parallel_rank(),
                )
            return sharded_state_dict

    class IntelTEColumnParallelGroupedLinear(IntelTEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to column-parallel style.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: str = None,
            force_disable_fp8=False,
        ):

            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="column",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                force_disable_fp8=force_disable_fp8,
            )

        def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 0, bias sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {}
            for gemm_idx in range(self.num_gemms):
                tp_axis_map.update({f'{gemm_idx}.weight': 0, f'{gemm_idx}.bias': 0})
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

    class IntelTERowParallelGroupedLinear(IntelTEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to row-parallel style.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: str = None,
            force_disable_fp8=False,
        ):

            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="row",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                force_disable_fp8=force_disable_fp8,
            )

        def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 1, bias not sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {f'{gemm_idx}.weight': 1 for gemm_idx in range(self.num_gemms)}
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

    class IntelTERowParallelGroupedLinearFP8Disabled(IntelTERowParallelGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to row-parallel style and force-disabled FP8.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: str = None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                config=config,
                init_method=init_method,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                force_disable_fp8=True,
            )

else:
    IntelTEGroupedLinear = None
    IntelTEColumnParallelGroupedLinear = None
    IntelTERowParallelGroupedLinear = None
    IntelTERowParallelGroupedLinearFP8Disabled = None


class IntelTEDotProductAttention(te.FusedAttention):
    """
    Wrapper for the Intel Transformer-Engine's `FusedAttention` layer.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        force_disable_fp8=not is_gaudi3(),
    ):
        self.config = config
        self.force_disable_fp8 = force_disable_fp8
        self.use_fast_softmax = "fast" if config.use_fast_softmax else "None"

        assert config.window_size is None, "Window attention not supported yet!"

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)

        super().__init__(
            scale=None,
            attention_dropout=attention_dropout if attention_dropout is not None else 0.0,
            enable_recompute=self.config.use_fused_sdpa_with_recompute,
            cp_group=parallel_state.get_context_parallel_group(check_initialized=False),
            cp_global_ranks=parallel_state.get_context_parallel_global_ranks(
                check_initialized=False
            ),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        assert (
            attention_bias is None
        ), "Attention bias is not supported for IntelTEDotProductAttention."
        assert packed_seq_params is None, "packed_seq_params are not supported."

        # [sq, b, np, hn] -> [b, np, sq, hn]
        q, k, v = [x.transpose(0, 1).transpose(1, 2) for x in [query, key, value]]
        causal = attn_mask_type == AttnMaskType.causal
        attn_mask = None if causal else attention_mask

        ctx = te.fp8_autocast(enabled=False) if self.force_disable_fp8 else nullcontext()
        with ctx:
            context_layer = super().forward(q, k, v, attn_mask, causal, self.use_fast_softmax)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class IntelTEDelayedScaling(te.recipe.DelayedScaling):
    """
    Wrapper for the Intel Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        fp8_format: int,
        override_linear_precision: tuple = (False, False, False),
    ):
        super().__init__(
            margin=config.fp8_margin,
            interval=config.fp8_interval,
            fp8_format=fp8_format,
            amax_compute_algo=config.fp8_amax_compute_algo,
            amax_history_len=config.fp8_amax_history_len,
            override_linear_precision=override_linear_precision,
            reduce_amax=config.fp8_amax_reduce,
        )
