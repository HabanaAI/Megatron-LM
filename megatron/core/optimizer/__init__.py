# © 2024-2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD
    except ImportError:
        import warnings

        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        # See https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16.
        from torch.optim import AdamW as Adam
        from torch.optim import SGD

from megatron.core import mpu

from ..distributed.param_and_grad_buffer import _ParamAndGradBuffer
from ..transformer.module import MegatronModule
from ..utils import log_single_rank
from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


def _get_param_groups(
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable],
    scale_lr_cond: Optional[Callable],
    lr_mult: float,
    lr: float,
    min_lr: float,
    decoupled_lr: Optional[float],
    decoupled_min_lr: Optional[float],
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func, optional): function to determine whether a
            parameter should not perform weight decay.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        decoupled_lr (Optional[float]): optional decoupled learning rate.
        decoupled_min_lr (Optional[float]): optional decoupled minimum learning rate.

    Returns:
        List of parameter groups.
    """

    use_decoupled_learning_rate = decoupled_lr is not None

    # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, _lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, _lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, _lr_mult = 0.0, 1.0
            else:
                wd_mult, _lr_mult = 0.0, lr_mult

            is_decoupled_lr = False
            # For input/embedding and output layer: embedding.word_embeddings.weight /
            # output_layer.weight.
            if use_decoupled_learning_rate and getattr(
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

    param_groups = []
    for (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():
        assert len(params) > 0
        param_group = {
            'params': params,
            'wd_mult': wd_mult,
            'lr_mult': _lr_mult,
            'is_expert_parallel': is_expert_parallel,
            'is_decoupled_lr': is_decoupled_lr,
        }
        param_groups.append(param_group)

    param_groups = _update_min_and_max_lr_in_param_groups(
        param_groups,
        lr=lr,
        min_lr=min_lr,
        decoupled_lr=decoupled_lr,
        decoupled_min_lr=decoupled_min_lr,
    )

    return param_groups


def _update_min_and_max_lr_in_param_groups(
    param_groups: List[Dict],
    lr: float,
    min_lr: float,
    decoupled_lr: Optional[float],
    decoupled_min_lr: Optional[float],
) -> List[Dict]:
    """
    Updates `max_lr` and `min_lr` values in each parameter group, and returns new list.
    By default, each group will use `lr` / `min_lr` as `max_lr` / `min_lr`.
    If `decoupled_lr` is provided, then `decoupled_lr` / `decoupled_min_lr` will be used
    as `max_lr` / `min_lr` for the input and output layer.

    Args:
        param_groups (List): parameter groups whose 'max_lr' and `min_lr` fields need to
            be adjusted.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        decoupled_lr (Optional[float]): optional decoupled learning rate.
        decoupled_min_lr (Optional[float]): optional decoupled minimum learning rate.

    Returns:
        List of adjusted parameter groups.
    """

    if decoupled_min_lr is None:
        decoupled_min_lr = min_lr

    for param_group in param_groups:
        if param_group['is_decoupled_lr']:
            assert decoupled_lr is not None
            param_group['max_lr'] = decoupled_lr
            param_group['min_lr'] = decoupled_min_lr
        else:
            param_group['max_lr'] = lr
            param_group['min_lr'] = min_lr
    return param_groups


def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    no_weight_decay_cond: Optional[Callable],
    scale_lr_cond: Optional[Callable],
    lr_mult: float,
    filter_fn: Callable,
    buffer_name: str,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        model_chunk_offset (int): offset of model_chunks in global model_chunks list.
        config (OptimizerConfig): optimizer configuration object.
        no_weight_decay_cond (func, optional): function to determine whether a
            parameter should not perform weight decay.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        filter_fn (callable): filtering function for param_groups.
        buffer_name (str): name of buffer.

    Returns:
        List of parameter groups and dictionary of model chunk IDs to buffers.
    """
    param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )
    param_groups = list(filter(filter_fn, param_groups))
    buffers = {}
    for model_chunk_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name):
            buffers[model_chunk_idx + model_chunk_offset] = getattr(model_chunk, buffer_name)

    return param_groups, buffers


def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (list): list of model chunks.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.

    Returns:
        Instance of MegatronOptimizer.
    """
    if config.optimizer in ['adam', 'fusedadamw']:
        optimizer_class = None
        if config.optimizer == 'adam':
            optimizer_class = Adam
        elif config.optimizer == 'fusedadamw':
            from habana_frameworks.torch.hpex.optimizers import FusedAdamW

            optimizer_class = FusedAdamW
        optimizer = optimizer_class(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )

        def init_state_fn(opt):
            for group in opt.param_groups:
                for p in group['params']:
                    if len(opt.state[p]) == 0:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

    elif config.optimizer == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.sgd_momentum,
        )
        init_state_fn = None
    else:
        raise Exception('{} optimizer is not supported.'.format(config.optimizer))

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
        if config.use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                *optimizer_args,
                model_chunks=model_chunks,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
            )
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)
            setattr(optimizer, 'model_parallel_group', model_parallel_group)
    else:
        # FP32 optimizer.
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'model_parallel_group', model_parallel_group)

    return optimizer


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.

    Returns:
        Instance of MegatronOptimizer.
    """

    log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

    # Separate out first model chunk if overlapping param AG with optimizer step.
    if config.overlap_param_gather_with_optimizer_step:
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]]
        overlap_param_gather_with_optimizer_step_flags = [True, False]
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())

    optimizers = []
    model_chunk_offset = 0
    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ):
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset,
            config=config,
            no_weight_decay_cond=no_weight_decay_cond,
            scale_lr_cond=scale_lr_cond,
            lr_mult=lr_mult,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='dense_buffers',
        )
        for model_chunk in dense_model_chunks:
            model_chunk.overlap_param_gather_with_optimizer_step = (
                overlap_param_gather_with_optimizer_step
            )
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config,
                model_chunks=dense_model_chunks,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mpu.get_model_parallel_group(),
                data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(
                    with_context_parallel=True
                ),
                data_parallel_group_idx=model_parallel_rank,
            )
        )
        model_chunk_offset += 1

    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        no_weight_decay_cond=no_weight_decay_cond,
        scale_lr_cond=scale_lr_cond,
        lr_mult=lr_mult,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
    )
    if len(moe_param_groups) > 0:
        model_parallel_world_size = torch.distributed.get_world_size(mpu.get_model_parallel_group())
        expert_parallel_rank = mpu.get_expert_model_parallel_rank()
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=mpu.get_model_parallel_group(with_expert_parallel=True),
                data_parallel_group=mpu.get_data_modulo_expert_parallel_group(
                    with_context_parallel=True
                ),
                data_parallel_group_gloo=mpu.get_data_modulo_expert_parallel_group_gloo(
                    with_context_parallel=True
                ),
                data_parallel_group_idx=expert_parallel_rank * model_parallel_world_size
                + model_parallel_rank,
            )
        )

    if len(optimizers) == 1:
        return optimizers[0]

    return ChainedOptimizer(optimizers)
