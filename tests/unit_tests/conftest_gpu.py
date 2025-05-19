# © 2024-2025 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="function", autouse=True)
def set_env():
    if is_te_min_version("1.3"):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir

    else:
        yield tmp_dir


# Failures on Titan XP GPU
list_of_skip = (
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[True-8-1-1]",
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[True-4-2-1]",
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[True-1-1-8]",
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[True-2-1-4]",
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[True-2-2-2]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_finetune_doesnt_load_optimizer[src_tp_pp0-dest_tp_pp0-False]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_finetune_doesnt_load_optimizer[src_tp_pp1-dest_tp_pp1-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_finetune_doesnt_load_optimizer[src_tp_pp2-dest_tp_pp2-False]",
    "tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestFullyParallelSaveAndLoad::test_memory_usage[cpu]",
    "tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestFullyParallelSaveAndLoad::test_memory_usage[cuda]",
    "tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-te-te]",
    "tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-te-local]",
    "tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-local-te]",
    "tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-local-local]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp0-dest_tp_pp_exp0-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-True-src_tp_pp_exp1-dest_tp_pp_exp1-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp2-dest_tp_pp_exp2-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-True-src_tp_pp_exp3-dest_tp_pp_exp3-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp4-dest_tp_pp_exp4-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp5-dest_tp_pp_exp5-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-True-src_tp_pp_exp6-dest_tp_pp_exp6-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp7-dest_tp_pp_exp7-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp8-dest_tp_pp_exp8-False]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp9-dest_tp_pp_exp9-True]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp10-dest_tp_pp_exp10-True]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-True-src_tp_pp_exp11-dest_tp_pp_exp11-True]",
    "tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[False-False-src_tp_pp_exp12-dest_tp_pp_exp12-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp0-dest_tp_pp_exp0-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp1-dest_tp_pp_exp1-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp2-dest_tp_pp_exp2-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp3-dest_tp_pp_exp3-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp4-dest_tp_pp_exp4-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp5-dest_tp_pp_exp5-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp6-dest_tp_pp_exp6-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp7-dest_tp_pp_exp7-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp8-dest_tp_pp_exp8-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp9-dest_tp_pp_exp9-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp10-dest_tp_pp_exp10-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp11-dest_tp_pp_exp11-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp12-dest_tp_pp_exp12-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp13-dest_tp_pp_exp13-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp14-dest_tp_pp_exp14-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp0-dest_tp_pp_exp0-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp1-dest_tp_pp_exp1-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp2-dest_tp_pp_exp2-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp3-dest_tp_pp_exp3-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp4-dest_tp_pp_exp4-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp5-dest_tp_pp_exp5-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp6-dest_tp_pp_exp6-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp7-dest_tp_pp_exp7-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp8-dest_tp_pp_exp8-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-src_tp_pp_exp9-dest_tp_pp_exp9-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp10-dest_tp_pp_exp10-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp11-dest_tp_pp_exp11-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp12-dest_tp_pp_exp12-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp13-dest_tp_pp_exp13-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp14-dest_tp_pp_exp14-False]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp15-dest_tp_pp_exp15-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp16-dest_tp_pp_exp16-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp17-dest_tp_pp_exp17-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp18-dest_tp_pp_exp18-True]",
    "tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py::TestGroupedMLPReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-src_tp_pp_exp19-dest_tp_pp_exp19-True]",
    "tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py::TestParallelMLPWithGLU::test_parallel_reconfiguration_e2e[src_tp_pp0-dest_tp_pp0]",
    "tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py::TestParallelMLPWithGLU::test_parallel_reconfiguration_e2e[src_tp_pp1-dest_tp_pp1]",
    "tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py::TestParallelMLPWithGLU::test_parallel_reconfiguration_e2e[src_tp_pp2-dest_tp_pp2]",
    "tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py::TestParallelMLPWithGLU::test_parallel_reconfiguration_e2e[src_tp_pp3-dest_tp_pp3]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_partition_change_save_load[src_tp_pp0-dest_tp_pp0]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_partition_change_save_load[src_tp_pp1-dest_tp_pp1]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_partition_change_save_load[src_tp_pp2-dest_tp_pp2]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_partition_change_save_load[src_tp_pp3-dest_tp_pp3]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_reformulate_nd_flattened_tensors[src_tp_pp0-dest_tp_pp0-expected_ckpt_offsets_by_rank0]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_reformulate_nd_flattened_tensors[src_tp_pp1-dest_tp_pp1-expected_ckpt_offsets_by_rank1]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_load_tensor_metadata[src_tp_pp0]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_load_tensor_metadata[src_tp_pp1]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_load_tensor_metadata[src_tp_pp2]",
    "tests/unit_tests/dist_checkpointing/test_flattened_resharding.py::TestFlattenedResharding::test_load_tensor_metadata[src_tp_pp3]",
    "tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_generate_all_output_tokens_static_batch",
    "tests/unit_tests/inference/engines/test_mcore_engine.py::TestMCoreEngine::test_generate",
    "tests/unit_tests/models/test_clip_vit_model.py::TestCLIPViTModel::test_forward",
    "tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_post_process_forward",
    "tests/unit_tests/models/test_t5_model.py::TestT5Model::test_post_process_forward",
    "tests/unit_tests/models/test_multimodal_projector.py::TestMultimodalProjector::test_forward",
    "tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_forward",
    "tests/unit_tests/models/test_bert_model.py::TestBertModel::test_post_process_forward",
    "tests/unit_tests/pipeline_parallel/test_schedules.py::test_forward_backward_func_without_pipeline_parallel",
    "tests/unit_tests/pipeline_parallel/test_schedules.py::test_forward_backward_func_with_pipeline_parallel",
    "tests/unit_tests/pipeline_parallel/test_schedules.py::test_forward_backward_func_with_interleaving",
    "tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_gpu_forward",
    "tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_gpu_forward_full_checkpoint",
    "tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_gpu_forward_selective_checkpoint",
    "tests/unit_tests/transformer/test_retro_attention.py::TestRetroAttention::test_constructor",
    "tests/unit_tests/transformer/test_retro_attention.py::TestRetroAttention::test_gpu_forward",
    "tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_gpu_forward",
    "tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_fused_rope_gpu_forward",
    "tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_checkpointed_gpu_forward",
    "tests/unit_tests/transformer/test_transformer_layer.py::TestParallelTransformerLayer::test_gpu_forward",
    "tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_transformer_block_custom",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_forward_backward[1-8]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_forward_backward[8-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_forward_backward[4-2]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_forward_backward[1-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_a2aseq_forward_backward[1-8]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_a2aseq_forward_backward[8-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_a2aseq_forward_backward[4-2]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_a2aseq_forward_backward[1-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_padding_forward_backward[1-8]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_padding_forward_backward[8-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_padding_forward_backward[4-2]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_padding_forward_backward[1-1]",
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[4-2-1]",
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[1-1-8]",
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[2-1-4]",
    "tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[2-2-2]",
    "tests/unit_tests/transformer/moe/test_moe_layer.py::TestMoELayerInit::test_moe_with_late_initialize[2-2-False-alltoall]",
    "tests/unit_tests/transformer/moe/test_moe_layer.py::TestInterleaveTransformerBlock::test_interleave_transformer_block[moe_layer_freq1]",
    "tests/unit_tests/transformer/moe/test_moe_layer.py::TestInterleaveTransformerBlock::test_interleave_transformer_block[moe_layer_freq2]",
    "tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_weight_init_value_the_same",
    "tests/unit_tests/transformer/moe/test_upcycling.py::TestGPTModel::test_upcycling[tp_pp_ep0-False-False]",
    "tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[8-1]",
    "tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[1-8]",
    "tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[2-4]",
    "tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[1-1]",
    "tests/unit_tests/data/test_preprocess_data.py::test_preprocess_data_bert",
    "tests/unit_tests/inference/model_inference_wrappers/gpt/test_gpt_inference_wrapper.py::TestGPTInferenceWrapper::test_inference_pipeline_parallel_small_size",
    "tests/unit_tests/inference/model_inference_wrappers/gpt/test_gpt_inference_wrapper.py::TestGPTInferenceWrapper::test_inference_pipeline_parallel_large__size",
    "tests/unit_tests/inference/model_inference_wrappers/gpt/test_gpt_inference_wrapper.py::TestGPTInferenceWrapper::test_inference_only_tensor_parallel",
    "tests/unit_tests/dist_checkpointing/test_fp8.py::TestFP8::test_fp8_save_load[True-src_tp_pp0-dest_tp_pp0-broadcast]",
    "tests/unit_tests/dist_checkpointing/test_fp8.py::TestFP8::test_fp8_save_load[True-src_tp_pp1-dest_tp_pp1-gather_rounds]",
    "tests/unit_tests/dist_checkpointing/test_fp8.py::TestFP8::test_fp8_save_load[False-src_tp_pp2-dest_tp_pp2-None]",
    "tests/unit_tests/dist_checkpointing/test_fp8.py::TestFP8::test_simple_broadcast[0-fp8]",
    "tests/unit_tests/dist_checkpointing/test_fp8.py::TestFP8::test_simple_broadcast[6-fp8]",
    "tests/unit_tests/models/test_mamba_model.py::TestMambaModel::test_forward",
    "tests/unit_tests/models/test_mamba_model.py::TestMambaModel::test_inference",
    "tests/unit_tests/models/test_t5_model.py::TestT5Model::test_forward_output_encoder_hidden_only",
    "tests/unit_tests/models/test_t5_model.py::TestT5Model::test_forward_with_encoder_hidden_states",
    "tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_gpu_forward_full_checkpoint_fp8",
    "tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_gpu_forward_selective_checkpoint_fp8",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[2-1-8]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[2-8-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[2-4-2]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[2-1-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[4-1-8]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[4-8-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[4-4-2]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[4-1-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[6-1-8]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[6-8-1]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[6-4-2]",
    "tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_bins[6-1-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[True-True-1-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[True-True-8-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[True-True-4-2]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[True-False-1-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[True-False-8-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[True-False-4-2]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[False-True-1-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[False-True-8-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[False-True-4-2]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[False-False-1-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[False-False-8-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp[False-False-4-2]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[silu-1-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[silu-8-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[silu-4-2]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[gelu-1-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[gelu-8-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[gelu-4-2]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[relu-1-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[relu-8-1]",
    "tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[relu-4-2]",
    # R10 skips
    "tests/unit_tests/ssm/test_mamba_mixer.py::TestMambaMixer::test_gpu_forward[True]",
    "tests/unit_tests/ssm/test_mamba_mixer.py::TestMambaMixer::test_gpu_forward[False]",
    "tests/unit_tests/ssm/test_mamba_layer.py::TestMambaLayer::test_gpu_forward",
    "tests/unit_tests/ssm/test_mamba_block.py::TestMambaBlock::test_gpu_forward",
    "tests/unit_tests/inference/model_inference_wrappers/t5/test_t5_inference_wrapper.py::TestT5InferenceWrapper::test_inference_only_tensor_parallel",
    "tests/unit_tests/inference/engines/test_mcore_engine.py::TestMCoreEngine::test_generate_empty_prompt",
    "tests/unit_tests/inference/text_generation_controllers/test_encoder_decoder_text_generation_controller.py::TestEncoderDecoderTextGenerationController::test_generate_all_output_tokens_static_batch",
    "tests/unit_tests/export/trtllm/test_distributed_fp8.py::TestTRTLLMSingleDeviceConverterFP8::test_get_model_weights_converter",
    "tests/unit_tests/export/trtllm/test_single_device_fp8.py::TestTRTLLMSingleDeviceConverterFP8::test_get_model_weights_converter",
    "tests/unit_tests/inference/engines/test_mcore_engine.py::TestMCoreEngine::test_generate_empty_prompt",
    "tests/unit_tests/inference/model_inference_wrappers/t5/test_t5_inference_wrapper.py::TestT5InferenceWrapper::test_inference_only_tensor_parallel",
    "tests/unit_tests/ssm/test_mamba_block.py::TestMambaBlock::test_gpu_forward",
    "tests/unit_tests/ssm/test_mamba_layer.py::TestMambaLayer::test_gpu_forward",
    "tests/unit_tests/ssm/test_mamba_mixer.py::TestMambaMixer::test_gpu_forward[True]",
    "tests/unit_tests/ssm/test_mamba_mixer.py::TestMambaMixer::test_gpu_forward[False]",
    "tests/unit_tests/dist_checkpointing/test_local.py::TestLocalCheckpointing::test_sharded_tensors[True-2-4]",
    "tests/unit_tests/dist_checkpointing/test_local.py::TestLocalCheckpointing::test_sharded_tensors[False-2-4]",
    "tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-True-src_tp_pp_ep_etp14-dest_tp_pp_ep_etp14-True]",
    "tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp15-dest_tp_pp_ep_etp15-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp1-dest_tp_pp_exp1-True-False-True-True-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp3-dest_tp_pp_exp3-False-False-False-True-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp3-dest_tp_pp_exp3-False-False-True-True-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp3-dest_tp_pp_exp3-True-False-False-True-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp3-dest_tp_pp_exp3-True-False-True-True-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp2-dest_tp_pp_exp2-True-False-True-True-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp2-dest_tp_pp_exp2-True-False-False-True-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp2-dest_tp_pp_exp2-False-False-True-True-True]",
    "tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp2-dest_tp_pp_exp2-False-False-False-True-True]",
    "tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestSimpleTextGenerationController::test_generate_all_output_tokens_static_batch",
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in list_of_skip:
            item.add_marker(pytest.mark.xfail(run=False))
