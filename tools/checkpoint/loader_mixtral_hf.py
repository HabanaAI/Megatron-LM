# © 2024-2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import transformers
from transformers import MixtralConfig
from tqdm import tqdm
import types

from megatron.core.utils import is_real_cuda_device_available

device = "cpu"

def add_arguments(parser):
    group = parser.add_argument_group(title='Mixtral HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')
    parser.add_argument('--bf16', action='store_true',
                        help='Whether to load weights in bf16.')
    parser.add_argument('--load-capacity-bins',  action='store_true',
                        help='Load capacity bins parameters.')


def load_args_from_checkpoint(margs, args, use_source_margs_file=False):
    # Read Mixtral 8x7B args from source margs file or use default values.

    # During MLM -> HF conversion, many training, model arguments are not saved
    # in the final checkpoint or in config.json, so we need to load them from
    # the source checkpoint.
    # `source_megatron_args.json` is created during MLM to HF conversion,
    # essential to make the final MLM, MLM (1) -> HF -> MLM (2) checkpoint
    # consistent.

    mixtral_config = MixtralConfig.from_pretrained(margs.load)

    if use_source_margs_file:
        print(f"Loading arguments from {args.source_margs_file} ")
        exclusions = {
            "sequence_parallel",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "use_distributed_optimizer",
            "verify_checkpoint",
            "load",
            "use_rotary_position_embeddings",
            "transformer_pipeline_model_parallel_size",
            "consumed_train_samples",
            "consumed_valid_samples",
            "tokenizer_model",
            "encoder_num_layers",
            "encoder_seq_length",
            "start_weight_decay",
            "end_weight_decay",

        }
        with open(args.source_margs_file, "r") as f:
            src_megatron_args = json.load(f)

        for key, val in src_megatron_args.items():
            if key == 'pipeline_model_parallel_size' and hasattr(margs, key):
                margs.previous_pipeline_model_parallel_size = val
            if key == 'tensor_model_parallel_size' and hasattr(margs, key):
                margs.previous_tensor_model_parallel_size = val
            if key == 'expert_tensor_parallel_size' and hasattr(margs, key):
                margs.previous_expert_tensor_parallel_size = val

            if key not in exclusions and hasattr(margs, key) and val != getattr(margs, key):
                print(f"key: {key} replacing margs {getattr(margs, key)} with {val}.")
                setattr(margs, key, val)
    else:
        print("Argument `--source-margs-file` is not set or path doesn't exit. Default megatron arguments would be set.")

        # Update Megatron args.
        margs.seq_length = 32768
        margs.global_batch_size = 128
        margs.tokenizer_type = "GPTSentencePieceTokenizer"
        margs.disable_bias_linear = True
        margs.untie_embeddings_and_output_weights = (not mixtral_config.tie_word_embeddings)
        # Max position embeddings must be larger or equal sequence length
        if mixtral_config.max_position_embeddings >= margs.seq_length:
            margs.max_position_embeddings = mixtral_config.max_position_embeddings
        else:
            margs.max_position_embeddings = margs.seq_length
        margs.hidden_size = mixtral_config.hidden_size
        margs.num_attention_heads = mixtral_config.num_attention_heads
        margs.num_layers = mixtral_config.num_hidden_layers
        margs.use_rotary_position_embeddings = True
        margs.swiglu = True
        margs.normalization = "RMSNorm"

        margs.norm_epsilon = mixtral_config.rms_norm_eps
        margs.vocab_size = mixtral_config.vocab_size
        margs.padded_vocab_size = mixtral_config.vocab_size
        margs.ffn_hidden_size = mixtral_config.intermediate_size
        margs.num_experts = mixtral_config.num_local_experts
        # margs.expert_tensor_parallel_size = 1
        margs.transformer_impl = 'transformer_engine'

        margs.previous_tensor_model_parallel_size = 1
        margs.previous_pipeline_model_parallel_size = 1
        margs.previous_expert_tensor_parallel_size = 1

        if mixtral_config.num_key_value_heads:
            margs.group_query_attention = True
            margs.num_query_groups = mixtral_config.num_key_value_heads


    margs.iteration = 1 # '0', 'release' don't work
    margs.padded_vocab_size = mixtral_config.vocab_size
    margs.verify_checkpoint_model_type = 'MIXTRAL'

    if '--use-cpu-initialization' in sys.argv:
        margs.use_cpu_initialization = True
    else:
        margs.use_cpu_initialization = False

    return margs

def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major >= 4 and minor >= 36

def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)

def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    model.output_layer.weight.data.copy_(hf_model.lm_head.weight)

def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    num_heads = args.num_attention_heads // tp
    num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads) // tp
    num_querys_per_group = num_heads // num_query_groups
    dim = args.kv_channels
    assert num_heads % num_querys_per_group == 0

    # Copy weights (re-order dimensions for Megatron).
    attn.linear_qkv.weight.data.copy_(torch.cat([
        hf_attn.q_proj.weight.reshape((num_query_groups, num_querys_per_group*dim, -1)),
        hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
        hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)

def set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''

    if args.moe_router_fp32:
        layer.mlp.router.to(torch.float32)
        hf_layer.block_sparse_moe.gate.weight.to(torch.float32)

    layer.mlp.router.weight.data.copy_(hf_layer.block_sparse_moe.gate.weight)

    hf_experts = hf_layer.block_sparse_moe.experts

    if not args.moe_dynamic_hpu:
        mcore_experts = layer.mlp.experts.local_experts

        for expert_idx in range(args.num_experts):
            mcore_experts[expert_idx].linear_fc1.weight.data.copy_(
                torch.cat([
                    hf_experts[expert_idx].w1.weight,
                    hf_experts[expert_idx].w3.weight
                ], dim=0)
            )
            mcore_experts[expert_idx].linear_fc2.weight.data.copy_(
                hf_experts[expert_idx].w2.weight
            )
    else:
        mcore_experts_fc1 = [f'local_expert_{expert_idx}_linear_fc1_weight' for expert_idx in range(args.num_experts)]
        mcore_experts_fc2 = [f'local_expert_{expert_idx}_linear_fc2_weight' for expert_idx in range(args.num_experts)]

        for expert_idx in range(args.num_experts):
            linear_fc1 = getattr(layer.mlp.experts, mcore_experts_fc1[expert_idx])
            linear_fc2 = getattr(layer.mlp.experts, mcore_experts_fc2[expert_idx])

            linear_fc1.data.copy_(
                torch.cat([
                    hf_experts[expert_idx].w1.weight,
                    hf_experts[expert_idx].w3.weight
                ], dim=0)
            )
            linear_fc2.data.copy_(
                hf_experts[expert_idx].w2.weight
            )


def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)

    if is_real_cuda_device_available():
        layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)
    else:
        layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)

def load_checkpoint_to_model(args):
    '''Set model params.'''

    from pretrain_gpt import model_provider
    from transformers import MixtralForCausalLM

    # Load Huggingface model.
    hf_model = MixtralForCausalLM.from_pretrained(args.load, device_map=device)

    # Init Megatron model.
    model = model_provider(True, True).to(args.params_dtype).to(device)

    # Set model state.
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)
    return model


def _load_checkpoint(queue, args):

    # Llama-2 requires HF transformers >=4.31.0.
    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--use-mcore-models',
                '--disable-bias-linear',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--mock-data', # To pass the "blend data checks" in arguments.py
                '--transformer-impl', 'transformer_engine',
                '--load', args.load_dir,
                '--no-one-logger',
                ]

    margs = parse_args()

    use_source_margs_file = args.source_margs_file is not None and os.path.exists(args.source_margs_file)

    margs = load_args_from_checkpoint(margs, args, use_source_margs_file)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.tokenizer_model = args.tokenizer_model
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    margs.bf16 = args.bf16
    margs.params_dtype = torch.bfloat16 if args.bf16 else torch.float32

    if margs.moe_capacity_bins_num > 0:
        sys.argv.extend(['--moe-capacity-bins-num', str(margs.moe_capacity_bins_num)])

    capacity_bins_path = os.path.join(args.load_dir, 'capacity_bins.pt')
    # Optionally load capacty bins
    if args.load_capacity_bins:
        assert os.path.exists(capacity_bins_path), 'The file for loading capacity bins parameters is missing.'
        capacity_bins = torch.load(capacity_bins_path, weights_only=False)

    validate_args(margs)

    def check_for_arg(arg_name, dest_arg_name=None, default=None):
        if getattr(margs, arg_name, None) is None and getattr(margs, dest_arg_name, None):
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('expert_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('disable_bias_linear', 'add_bias_linear')
    check_for_arg('params_dtype')
    check_for_arg('swiglu')

    # Determine how to make our models.
    assert args.model_type == 'GPT', 'Mixtral is a GPT model.'
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    if is_real_cuda_device_available():
        fused_kernels.load(margs)

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.previous_tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.previous_pipeline_model_parallel_size
    md.previous_expert_tensor_parallel_size = margs.previous_expert_tensor_parallel_size
    md.true_vocab_size = margs.vocab_size # skips padding in saver
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.num_experts = margs.num_experts
    md.moe_capacity_bins_num = margs.moe_capacity_bins_num
    md.qkv_bias = margs.add_qkv_bias
    md.moe_router_fp32 = margs.moe_router_fp32

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.embedding, 'position_embeddings')

    queue_put("embeddings", message)

    for layer_idx in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.decoder.layers[layer_idx]
        if is_real_cuda_device_available():
            message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
        else:
            message["input norm weight"] = layer.input_layernorm.weight.data
        message["post norm weight"] = layer.pre_mlp_layernorm.weight.data

        # Simple concat of the rest.
        message["qkv weight"] = layer.self_attention.linear_qkv.weight.data
        message["dense weight"] = layer.self_attention.linear_proj.weight.data

        # Grab all parallel tensors for this layer.
        layer = model.decoder.layers[layer_idx]
        if margs.moe_dynamic_hpu:
            experts = layer.mlp.experts
        else:
            experts = layer.mlp.experts.local_experts

        message["router weight"] = layer.mlp.router.weight.data

        if args.load_capacity_bins:
            message["bins usage"] = capacity_bins["bins_usage"][layer_idx]
            message["total requested capacity"] = capacity_bins["total_requested_capacity"][layer_idx]
            message["bins usage last"] = capacity_bins["optimize_moe_bins_usage_last"][layer_idx]
            message["total requested capacity last"] = capacity_bins["optimize_moe_total_requested_capacity_last"][layer_idx]
            message["capacity bins"] = capacity_bins['capacity_bins'][layer_idx]
        
        if not margs.moe_dynamic_hpu:
            if is_real_cuda_device_available() and md.swiglu:
                chunked_mlp_l0_weight =  [torch.chunk(local_expert.linear_fc1.weight.data, 2, dim=0) for local_expert in experts]
                message["mlp l0 weight W"] = torch.stack([local_weight[0] for local_weight in chunked_mlp_l0_weight], dim=0)
                message["mlp l0 weight V"] = torch.stack([local_weight[1] for local_weight in chunked_mlp_l0_weight], dim=0)
            else:
                message["mlp l0 weight"] = torch.stack([local_expert.linear_fc1.weight.data for local_expert in experts])
            message["mlp l1 weight"] = torch.stack([local_expert.linear_fc2.weight.data for local_expert in experts], dim=0)
        else:
            mcore_experts_fc1 = [f'local_expert_{expert_idx}_linear_fc1_weight' for expert_idx in range(margs.num_experts)]
            mcore_experts_fc2 = [f'local_expert_{expert_idx}_linear_fc2_weight' for expert_idx in range(margs.num_experts)]

            message["mlp l0 weight"] = torch.stack([getattr(experts, linear_fc1).data for linear_fc1 in mcore_experts_fc1])
            message["mlp l1 weight"] = torch.stack([getattr(experts, linear_fc2).data for linear_fc2 in mcore_experts_fc2])

        queue_put(f"transformer layer {layer_idx}", message)

    queue_put("final norm", {
        "weight": model.decoder.final_layernorm.weight.data,
    })

    if md.output_layer:
        queue_put("output layer", {
            "weight": model.output_layer.weight.data
        })

    queue.put("done")

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
