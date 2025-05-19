# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Mcore model schemas."""

import typing as T

from schema_base import ModelSchema

from megatron.core.utils import is_real_cuda_device_available


def get_mcore_transformer_block_key(model_key):
    return {
        "GPT" : "decoder",
        "BERT" : "encoder",
    }[model_key]


class MCoreSchema(ModelSchema):

    def __init__(self, model_type, layer_schema):
        block_key = get_mcore_transformer_block_key(model_type)
        super().__init__({
            "embeddings" : {
                "pos" : "embedding.position_embeddings.weight",
                "word" : "embedding.word_embeddings.weight",
            },
            "layer_prefix" : f"{block_key}.layers",
            "layer" : layer_schema,
            "final_norm" : {
                "weight" : f"{block_key}.final_layernorm.weight",
                "bias" : f"{block_key}.final_layernorm.bias",
            },
            "output_layer" : {
                "weight" : "output_layer.weight",
            },
            "pooler" : {
                "weight" : "pooler.dense.weight",
                "bias" : "pooler.dense.bias",
            },
            "lm_head" : {
                "dense_weight" : "lm_head.dense.weight",
                "dense_bias" : "lm_head.dense.bias",
                "norm_weight" : "lm_head.layer_norm.weight",
                "norm_bias" : "lm_head.layer_norm.bias",
            },
            "binary_head" : {
                "weight" : "binary_head.weight",
                "bias" : "binary_head.bias",
            },
        })


class MCoreLocalSchema(MCoreSchema):

    def __init__(self, model_type):
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "input_layernorm.weight",
            "self_attn_norm_bias" : "input_layernorm.bias",
            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",
            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias",

        })


class MCoreTESchema(MCoreSchema):

    def __init__(self, model_type, use_legacy_models):
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "self_attention.linear_qkv.layer_norm_weight" if use_legacy_models else "input_layernorm.weight",
            "self_attn_norm_bias" : "self_attention.linear_qkv.layer_norm_bias" if use_legacy_models else "input_layernorm.bias",
            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",

            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "mlp.linear_fc1.layer_norm_weight" if use_legacy_models else "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "mlp.linear_fc1.layer_norm_bias" if use_legacy_models else "pre_mlp_layernorm.bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias",

        })


class MCoreMoETESchema(MCoreSchema):

    def __init__(self, model_type, num_experts, expert_model_parallel_size, moe_dynamic_hpu, use_capacity_bins, use_legacy_models):
        num_local_experts = num_experts // expert_model_parallel_size

        layer_schema = {
            # Self attention.
            "self_attn_norm_weight" : "self_attention.linear_qkv.layer_norm_weight" if use_legacy_models else "input_layernorm.weight",
            "self_attn_norm_bias" : "self_attention.linear_qkv.layer_norm_bias" if (use_legacy_models or not is_real_cuda_device_available()) else "input_layernorm.bias",

            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",

            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",

            "router_weight" : "mlp.router.weight",
        }

        if moe_dynamic_hpu:
            layer_schema.update(
                **{f"mlp_fc1_weight.{expert_idx}" : f"mlp.experts.local_expert_{expert_idx}_linear_fc1_weight" for expert_idx in range(num_local_experts) },
                **{f"mlp_fc2_weight.{expert_idx}" : f"mlp.experts.local_expert_{expert_idx}_linear_fc2_weight" for expert_idx in range(num_local_experts) },
            )
        else:
            layer_schema.update(
                **{f"mlp_fc1_weight.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.linear_fc1.weight" for expert_idx in range(num_local_experts) },
                **{f"mlp_fc2_weight.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.linear_fc2.weight" for expert_idx in range(num_local_experts) },
            )

        if use_capacity_bins:
            # Capacity bins
            layer_schema.update({
                "bins_usage": "mlp.router.capacity_bins.bins_usage",
                "total_requested_capacity": "mlp.router.capacity_bins.total_requested_capacity",
                "bins_usage_last": "mlp.router.capacity_bins.optimize_moe_bins_usage_last",
                "total_requested_capacity_last": "mlp.router.capacity_bins.optimize_moe_total_requested_capacity_last",
                "capacity_bins": "mlp.router.capacity_bins.capacity_bins",
            })

        super().__init__(model_type, layer_schema)


def get_model_schema(
    model_type: T.Literal["GPT", "BERT"],
    transformer_impl: T.Literal["transformer_engine", "local"],
    num_experts: T.Optional[int] = None,
    expert_model_parallel_size: T.Optional[int] = None,
    use_legacy_models: bool = False,
    moe_dynamic_hpu: bool = False,
    use_capacity_bins: bool = False,
) -> MCoreSchema:
    te_impl_kwargs={}
    if transformer_impl == "transformer_engine" or not is_real_cuda_device_available():
        te_impl_kwargs.update({'use_legacy_models': use_legacy_models})
    if num_experts is not None and num_experts > 0:
        # Only support TE setter for MOE
        if is_real_cuda_device_available():
            assert transformer_impl == "transformer_engine"
        assert isinstance(expert_model_parallel_size, int)
        return MCoreMoETESchema(model_type, num_experts, expert_model_parallel_size, moe_dynamic_hpu, use_capacity_bins, **te_impl_kwargs)
    return {
        "local" : MCoreLocalSchema,
        "transformer_engine" : MCoreTESchema,
    }[transformer_impl](model_type, **te_impl_kwargs)
