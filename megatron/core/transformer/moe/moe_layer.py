# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import (
    GroupedMLP,
    IntelDynamicMLP,
    SequentialMLP,
    TEGroupedMLP,
)
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.moe_extended_tp:
            self.num_local_experts = self.config.num_moe_experts
            local_expert_indices_offset = 0
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
            )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules, MLPSubmodules):
                self.experts = TEGroupedMLP(self.num_local_experts, self.config, self.submodules)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        elif self.config.moe_dynamic_hpu:
            self.experts = IntelDynamicMLP(
                self.num_local_experts, self.local_expert_indices, self.config
            )
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == None:
            self.token_dispatcher = None
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.ep_size = parallel_state.get_expert_model_parallel_world_size()

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, indices = self.router(hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, indices
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            return output, mlp_bias

        def custom_forward_hpu(hidden_states):
            probs, indices = self.router(hidden_states)

            if self.tp_size > 1 or self.ep_size > 1:
                with torch.no_grad():
                    global_indices = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                        indices
                    )
                global_probs = tensor_parallel.gather_from_sequence_parallel_region_to_moe(probs)
                global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                    hidden_states, use_global_buffer=True
                )
            else:
                global_hidden_states = hidden_states
                global_probs = probs
                global_indices = indices

            # process tokens with expert MLPs
            expert_output, mlp_bias = self.experts(
                global_hidden_states, global_probs, global_indices
            )

            if self.tp_size > 1 or self.ep_size > 1:
                output = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    expert_output
                )
            else:
                output = expert_output

            output = output.view(hidden_states.shape)

            return output, mlp_bias

        fn = custom_forward_hpu if self.config.moe_dynamic_hpu else custom_forward

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(fn, False, hidden_states)
        else:
            output, mlp_bias = fn(hidden_states)

        return output, mlp_bias
