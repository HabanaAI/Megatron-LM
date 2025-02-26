# © 2024-2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Any, OrderedDict, Union

import torch
import torch.distributed
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.aux_loss import aux_losses_tracker_track_metrics
from megatron.core.parallel_state import (
    get_expert_model_parallel_group,
    get_tensor_and_expert_parallel_group,
)
from megatron.core.transformer.moe.capacity_bins import CapacityBins, optimize_bins
from megatron.core.utils import is_lazy_mode, is_real_cuda_device_available


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
    sequence_partition_group=None,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.
                                          Shape in [num_experts]
        topk (int): The number of experts selected for each token.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
        sequence_partition_group (optional): The parallel group over which the sequence is
                                             partitioned. If None, no partitioning is applied.
                                             Defaults to None.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for
        # `tokens_per_expert`, saving one allreduce operation for `aggregated_probs_per_expert`.
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    num_tokens = probs.shape[0] * num_sub_sequence
    num_experts = probs.shape[1]

    # The formula of aux_loss: aux_loss = sum((probs_per_expert/num_tokens) *
    # (tokens_per_expert/(num_tokens*topk))) * num_experts * moe_aux_loss_coeff.
    # This can be simplified to fuse the division and multiplication operations.
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
    )
    return aux_loss


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss


def sinkhorn(cost: torch.Tensor, tol: float = 0.0001):
    """Sinkhorn based MoE routing function"""
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    """
    Calculate the capacity of each expert.

    Args:
        num_tokens (int): num of the input tokens.
        num_experts (int): num of the experts.
        capacity_factor (float): Capacity factor.
        min_capacity (int, optional): Minimum capacity. Defaults to None.

    Returns:
        Tensor: Capacity of each expert.
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each
       token separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of
                                [num_tokens] or [num_tokens, topk].
        num_out_tokens (int, optional): The effective output token count, when enabling the
                                        capacity factor, should equal the number of tokens not
                                        dropped. By default, set to None, meaning no tokens are
                                        dropped.
        padded_mode (bool, optional): If True, indicating the indices are padded to
                                      [num_expert, capacity] to denote selected tokens per expert.
                                      Defaults to False.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        indices = indices.unsqueeze(1)

    topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    moe_gather_indices = (sorted_indices // topk).unsqueeze(1).expand(-1, tokens.size(-1))
    permuted_tokens = moe_gather.apply(tokens, moe_gather_indices)

    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the
    tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): 2D tensor [num_tokens*topk, hidden]. The tensor of permuted
                                        tokens to be unpermuted.
        sorted_indices (torch.Tensor): 1D tensor [num_tokens*topk]. The tensor of sorted indices
                                       used to unpermute the tokens.
        probs (torch.Tensor, optional): 2D tensor [num_tokens, topk]. The tensor of probabilities
                                        corresponding to the permuted tokens. If provided,
                                        the unpermuted tokens will be merged with their respective
                                        probabilities.
        padded_mode (bool, optional): If True, indicating the indices are padded to
                                      [num_expert, capacity] to denote selected tokens per expert.
                                      Defaults to False.
        restore_shape (torch.Size, optional): The input shape before permutation, only used in
                                              padding mode. Defaults to None.

    Returns:
        torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
    """
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(
        0
    ), f"Got {sorted_indices.numel()} != {permuted_tokens.size(0)}."
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        assert probs.dim() == 2, f"Expected 2D tensor for probs, got {probs.dim()} dims."
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    output_size = [num_unpermuted_tokens, permuted_tokens.shape[-1]]
    moe_scatter_indices = sorted_indices.unsqueeze(1).expand(-1, permuted_tokens.size(-1))
    unpermuted_tokens = moe_scatter.apply(permuted_tokens, moe_scatter_indices, output_size)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def permute_with_padded_tokens(tokens, indices):
    """Permute the tokens based on the indices, only used in padding mode.
       The input indices shape is [num_expert, capacity], it indicates which tokens were selected
       by each expert separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected
                                tokens for each expert.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))

    return permuted_tokens, indices


def unpermute_with_padded_tokens(
    permuted_tokens: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermutes a padded permuted tokens based on sorted indices and merges the tokens with their
    corresponding probabilities.

    This function takes a tensor of permuted tokens and reorders them according to the provided
    indices. It also combines the tokens with their associated probabilities.

    Parameters:
        permuted_tokens (torch.Tensor): A 2D tensor containing permuted tokens.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected
                                tokens for each expert.
        probs (torch.Tensor): A tensor with the same shape as indices, containing probabilities
                              corresponding to each token.
        restore_shape (torch.Size): The target shape for the unpermuted tokens tensor.

    Returns:
        torch.Tensor: A tensor of unpermuted tokens, merged with their probabilities.

    """
    # Ensure permuted_tokens is 2D
    assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dim()}D."
    if probs is not None:
        # Reshape and expand probabilities and indices to match permuted_tokens
        probs = probs.view(-1).unsqueeze(-1)
        # Combine tokens with their probabilities
        permuted_tokens = probs * permuted_tokens

    # Scatter the combined tokens back to their original positions
    if is_lazy_mode():
        indices = indices.flatten()
        assert (
            permuted_tokens.shape[0] == indices.shape[0]
        ), f"Shape mismatch between permuted_tokens and indices."
        unpermuted_tokens = moe_scatter_1d.apply(permuted_tokens, indices, restore_shape)
    else:
        indices = indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
        assert (
            permuted_tokens.shape == indices.shape
        ), "Shape mismatch between permuted_tokens and indices."
        unpermuted_tokens = moe_scatter.apply(permuted_tokens, indices, restore_shape)
    return unpermuted_tokens


def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output


def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: float = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
    capacity_bins: Union[CapacityBins, None] = None,
):
    """Apply capacity and padding to the top-k selection.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        capacity_factor (int): The capacity factor of each expert. Will drop tokens if the number
                               of tokens exceeds the capacity.
        pad_to_capacity (bool): Whether to need padding in token drop mode.
        drop_policy (str): The policy to drop tokens. Can be either "prob" or "position".
                           If "prob", the tokens with the lowest probabilities will be dropped.
                           If "position", tokens at the end of each batch will be dropped.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Probs, indices and tokens_per_expert
                                                         tensor.

        (1) If there's no token padding, the shape of probs and indices is [tokens, top_k],
            indicating the selected experts for each token.
        (2) If there's token padding, the shape of probs and indices is [num_expert, capacity],
            indicating the tokens selected for each expert.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."

    assert (
        capacity_bins is None or capacity_factor is None
    ), f"Capacity bins can't be used with capacity factor, set one to None, got {capacity_bins} and {capacity_factor}"

    num_tokens, num_experts = logits.shape

    if use_pre_softmax:
        # Pre softmax
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs, top_indices = torch.topk(scores, k=topk, dim=1)
    else:
        # Post softmax
        if topk == 1:
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")
        scores, top_indices = torch.topk(logits, k=topk, dim=1)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

    # TopK selection, maskout unused experts
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_mask = torch.zeros_like(logits).scatter(1, top_indices, 1)
    tokens_per_expert_before_capacity = topk_mask.sum(dim=0)

    if capacity_factor is None and capacity_bins is None:  # keep all tokens
        return probs, top_indices, tokens_per_expert_before_capacity

    else:

        if capacity_bins is None:
            # TopK with capacity factor
            expert_capacity = get_capacity(
                num_tokens=num_tokens * topk,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            )
        else:
            # TopK with capacity bins
            expert_capacity = capacity_bins.get_binned_capacity(
                gate_output=logits, capacity=torch.max(tokens_per_expert_before_capacity)
            )

            # sync max in ep-tp group
            ep_tp_group = get_tensor_and_expert_parallel_group()
            ep_group = get_expert_model_parallel_group()
            group = ep_tp_group if ep_tp_group is not None else ep_group
            torch.distributed.all_reduce(
                expert_capacity, op=torch.distributed.ReduceOp.MAX, group=group
            )

        # Maskout exceeded tokens
        # If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.
        if drop_policy == "probs":
            capacity_probs, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
            capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_probs, final_indices = (
                capacity_probs.T.contiguous(),
                capacity_indices.T.contiguous(),
            )

        else:
            # Get exceed mask and maskout exceeded probs and indices
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            final_mask = torch.logical_and(topk_mask, capacity_mask)
            drop_mask = torch.logical_not(final_mask)
            exceed_mask = torch.gather(drop_mask, 1, top_indices)
            final_probs = probs * torch.logical_not(exceed_mask)
            mask_value = (
                num_experts if not is_real_cuda_device_available() else torch.iinfo(torch.long).max
            )
            final_indices = top_indices.clone().masked_fill_(exceed_mask, mask_value)

        return final_probs, final_indices, tokens_per_expert_before_capacity


def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    avg_group: torch.distributed.ProcessGroup = None,
):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
        mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    if name not in tracker:
        tracker[name] = {}
        tracker[name]["values"] = torch.zeros(num_layers, device=loss.device)
    tracker[name]["values"][layer_number - 1] += loss.detach()  # Aggregate the loss for the layer.
    tracker[name]["reduce_group"] = reduce_group
    tracker[name]["avg_group"] = avg_group


def clear_aux_losses_tracker():
    """Clear the auxiliary losses."""
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    for name in tracker:
        tracker[name]["values"].zero_()
        tracker[name]["reduce_group"] = None
        tracker[name]["avg_group"] = None


def reduce_aux_losses_tracker_across_ranks():
    """Collect and reduce the auxiliary losses across ranks."""
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    for name in tracker:
        values = tracker[name]["values"]
        # Collect aux losses across PP.
        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce aux losses across ranks.
        if tracker[name].get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker[name].get('reduce_group'))
        if tracker[name].get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker[name]['avg_group'], op=torch.distributed.ReduceOp.AVG
            )


def track_moe_metrics(
    loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None, per_layer_logging=False
):
    """Track the MoE metrics for logging."""
    # Aux loss logging
    reduce_aux_losses_tracker_across_ranks()
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    aux_losses_tracker_track_metrics(
        tracker,
        loss_scale,
        iteration,
        writer,
        wandb_writer,
        total_loss_dict,
        per_layer_logging,
        per_layer_prefix='moe/',
    )


def save_token_distribution_tracker(
    name: str,
    tokens_per_expert: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
):
    """Save the token distribution for logging.
    Args:
        name (str): The name of the token distribution.
        tokens_per_expert (torch.Tensor): Token distribution tensor.
        layer_number (int): Layer index of the token distribution.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
    """
    # Skip token distribution logging if layer_number is None.
    if layer_number is None:
        return

    tracker = parallel_state.get_moe_token_distribution_logging_tracker()
    if name not in tracker:
        tracker[name] = {}
        values_shape = (num_layers, tokens_per_expert.size(-1))
        tracker[name]['values'] = torch.zeros(
            values_shape, device=tokens_per_expert.device, dtype=tokens_per_expert.dtype
        )
    tracker[name]['values'][layer_number - 1] += tokens_per_expert.clone().detach().to(torch.long)
    tracker[name]['reduce_group'] = reduce_group


def generate_per_layer_token_distribution(history: list[Tensor]) -> list[Tensor]:
    # history: List[Tensor[num_layers, num_experts]]
    if not history:
        return []
    num_layers = history[0].size(0)
    per_layer_token_distribution = []
    for i in range(num_layers):
        tokens_per_expert = torch.stack([history[j][i] for j in range(len(history))], dim=0)
        per_layer_token_distribution.append(tokens_per_expert)

    return per_layer_token_distribution


def generate_token_distribution_log(name: str, per_layer_token_distribution: list[Tensor]) -> str:
    from tabulate import tabulate

    console_log = f'{name}:\n'
    for layer, tokens_per_expert in enumerate(per_layer_token_distribution):
        console_log += f'Layer {layer + 1}:\n'
        console_log += tabulate(
            tokens_per_expert.tolist(),
            headers=[f'Expert {i}' for i in range(tokens_per_expert.size(1))],
            tablefmt='github',
        )
        console_log += '\n'
    return console_log


def write_token_distribution_plot(
    name: str, per_layer_token_distribution: list[Tensor], writer: Any, iteration: int
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    for layer, tokens_per_expert in enumerate(per_layer_token_distribution):
        tokens_per_expert = (tokens_per_expert - tokens_per_expert.min()) / (
            tokens_per_expert.max() - tokens_per_expert.min()
        )
        figure_name = f'{name}_layer_{layer}'
        plt.figure(figsize=(10, 6))
        sns.heatmap(tokens_per_expert.numpy(), annot=False, cmap='Blues')
        plt.title(figure_name)
        plt.xlabel('Experts')
        figure = plt.gcf()
        writer.add_figure(figure_name, figure, iteration)
        plt.close(figure)


def track_moe_token_distribution_metrics(
    report_token_distribution_flag: bool, iteration: int, writer: Any
) -> str:
    tracker = parallel_state.get_moe_token_distribution_logging_tracker()

    # reduce values
    for name in tracker:
        if tracker[name].get('reduce_group') is not None:
            torch.distributed.all_reduce(
                tracker[name]['values'],
                torch.distributed.ReduceOp.SUM,
                group=tracker[name].get('reduce_group'),
            )

    # save history
    for name in tracker:
        if 'history' not in tracker[name]:
            tracker[name]['history'] = []
    for name in tracker:
        values = tracker[name]['values'].cpu()
        tracker[name]['history'].append(values)
    console_log = ''
    if report_token_distribution_flag:
        for name in tracker:
            per_layer_token_distribution = generate_per_layer_token_distribution(
                tracker[name]['history']
            )
            console_log += generate_token_distribution_log(name, per_layer_token_distribution)
            if writer is not None:
                write_token_distribution_plot(name, per_layer_token_distribution, writer, iteration)
            # clear history
            tracker[name]['history'] = []

    # clear values, keep history
    for name in tracker:
        tracker[name]["values"].zero_()
        tracker[name]["reduce_group"] = None

    return console_log


class moe_gather(torch.autograd.Function):
    """Gather the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_):
        """Gather the input tensor based on the map tensor."""
        ctx.input_size = input_.size()
        ctx.map = map_
        return torch.gather(input_, 0, map_)

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter the grad_output tensor based on the map tensor."""
        input_size = ctx.input_size
        map_ = ctx.map

        output = torch.zeros(
            input_size, dtype=grad_output.dtype, device=torch.cuda.current_device()
        )
        output.scatter_add_(0, map_, grad_output)
        return output, None, None


class moe_scatter_1d(torch.autograd.Function):
    """Scatter the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_, output_size_=None):
        # ctx, unpermuted_tokens, indices, output_size
        """Scatter the input tensor based on the map tensor."""
        ctx.save_for_backward(map_)
        # Prepare a tensor of zeros with the desired output shape
        if output_size_ is not None:
            output_ = torch.zeros(output_size_, dtype=input_.dtype, device=input_.device)
        else:
            output_ = torch.zeros_like(input_)
        return output_.scatter_add_(0, map_, input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Gather the grad_output tensor based on the map tensor."""
        map_ = ctx.saved_tensors[0]
        grad_input = torch.index_select(grad_output, 0, map_)
        return grad_input, None, None, None


class moe_scatter(torch.autograd.Function):
    """Scatter the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_, output_size=None):
        """Scatter the input tensor based on the map tensor."""
        ctx.map = map_

        if output_size is not None:
            output = torch.zeros(output_size, dtype=input_.dtype, device=input_.device)
        else:
            output = torch.zeros_like(input_)

        output.scatter_add_(0, map_, input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Gather the grad_output tensor based on the map tensor."""
        map_ = ctx.map
        grad_input = torch.gather(grad_output, 0, map_)
        return grad_input, None, None, None


def optimize_moe_capacity(
    global_rank,
    gate_modules,
    num_experts,
    step,
    max_grouped_experts: int = 4,
    max_overhead_factor: float = 0.0,
):
    """Optimize MoE gate capacity bins

    If MoE is using capacity bins, optimize the bins based on running stats.
    In order to reduce the number of compilation recipes, we optimize a set
    of grouped gates together.
    The grouped gates must have same number of experts.
    """
    # find all gates with capacity factor
    gate_with_capacity_bins_idx = [
        i for i, gate in enumerate(gate_modules) if gate.has_capacity_bins()
    ]
    if len(gate_with_capacity_bins_idx) == 0:
        return

    # handle only gates have capacity bins usage statistics
    gate_capacity_bin_stats = OrderedDict()
    for i in gate_with_capacity_bins_idx:
        gate = gate_modules[i]
        if hasattr(gate, 'get_stats'):
            stats = gate.get_stats(bins_state_name='optimize_moe')
            if stats is not None and 'capacity_bins' in stats:
                gate_capacity_bin_stats[i] = stats['capacity_bins']
    if len(gate_capacity_bin_stats) == 0:
        return

    del gate_with_capacity_bins_idx  # removing the list because it is out of date

    # divide gates into groups up to max_grouped_experts or until different num_experts encountered
    gate_groups = []
    first_gate_idx = list(gate_capacity_bin_stats.keys())[0]
    current_group = [first_gate_idx]
    current_group_n_experts = num_experts[first_gate_idx]
    for i in list(gate_capacity_bin_stats.keys())[1:]:
        if num_experts[i] == current_group_n_experts and len(current_group) < max_grouped_experts:
            current_group.append(i)
        else:
            gate_groups.append(current_group)
            current_group = [i]
            current_group_n_experts = num_experts[i]
    gate_groups.append(current_group)

    # print new optimized groups for each pipeline stage (no sharing across pp stages)
    dp_rank = torch.distributed.get_rank(group=parallel_state.get_data_parallel_group())
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    log_ranks = [global_rank] if dp_rank == 0 and tp_rank == 0 else []

    # for each group, (1) accumulate stats (2) calculate optimized capacity and (3) reconfigure bins
    for gate_group in gate_groups:
        skip_bins_optimization = True
        group_stats = []
        for i in gate_group:
            group_stats.append(gate_capacity_bin_stats[i])
            # Update a group only if at least one gates capacity overhead is above the threshold.
            skip_bins_optimization &= (
                'capacity_overhead_factor' in gate_capacity_bin_stats[i]
                and gate_capacity_bin_stats[i]['capacity_overhead_factor'] <= max_overhead_factor
            )

        if skip_bins_optimization:
            continue

        # sanity - verify all gates in groups have same bins edges
        bins_edges = [stats['edges'] for stats in group_stats]
        same_edges = all(torch.equal(bins_edges[0], tensor) for tensor in bins_edges[1:])
        assert (
            same_edges
        ), f'Got different capacity bin edges for group={gate_group} edges={bins_edges}'

        # accumulate usage
        stacked_usage = torch.stack([stats['usage'] for stats in group_stats], dim=0)
        total_group_usage = torch.sum(stacked_usage, dim=0)

        # find optimized bins for this group
        min_range = group_stats[0]['min_range']
        current_bins = group_stats[0]['edges']
        alignment = group_stats[0]['alignment']
        min_bin_size = group_stats[0]['min_bin_size']
        new_bins = optimize_bins(
            min_range=min_range,
            bins=current_bins,
            bins_usage=total_group_usage,
            alignment=alignment,
            min_bin_size=min_bin_size,
        )

        # configure gates in group with new bins
        for i in gate_group:
            gate = gate_modules[i]
            capacity_bins = gate.get_capacity_bins()
            capacity_bins.set_bins(new_bins)
