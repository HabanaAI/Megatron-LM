# © 2024-2025 Intel Corporation

import os

import torch

major, minor, patch = [eval(ss) for ss in torch.__version__.split('+')[0].split('.')]

try:
    import habana_frameworks.torch.gpu_migration
except:
    pass

import pytest


def is_torch_min_version(check_major, check_minor, check_patch):
    if major >= check_major and minor >= check_minor and patch >= check_patch:
        return True
    return False


# Key in the expected_fail_tests can be an exact node_id or module or directory
def test_in_xfail_dict(test_dict, nodeid):
    for key in test_dict:
        if key.endswith('::') or key.endswith('.py') or key.endswith('/'):
            if nodeid.startswith(key):
                return True
        elif key == nodeid:
            return True

    return False


def get_reason_for_xfail(test_dict, nodeid):
    for key in test_dict:
        if key.endswith('::') or key.endswith('.py') or key.endswith('/'):
            if nodeid.startswith(key):
                return test_dict[key]
        elif key == nodeid:
            return test_dict[key]

    return ""


unit_tests_to_deselect = {
    'TODO_0.13.1': [
        'tests/unit_tests/test_optimizer_cpu_offloading.py',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_a2aseq_forward_backward[True-1-8]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_logprobs_and_topn_consistency[dtype0-True-0]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_logprobs_and_topn_consistency[dtype0-True-4]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_logprobs_and_topn_consistency[dtype0-False-0]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_logprobs_and_topn_consistency[dtype0-False-4]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_logprobs_and_topn_consistency[dtype1-True-0]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_logprobs_and_topn_consistency[dtype1-True-4]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_logprobs_and_topn_consistency[dtype1-False-0]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_logprobs_and_topn_consistency[dtype1-False-4]',
        'tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestCrossRanksReads::test_cross_dp_access_does_not_disturb_the_distribution',
        'tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestCrossRanksReads::test_out_of_order_load',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_constructor[heterogeneous_gpt_model0-1486901248]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_constructor[heterogeneous_gpt_model2-1444954112]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_constructor[heterogeneous_gpt_model4-1310736384]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_constructor[heterogeneous_gpt_model6-1461735424]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_constructor[heterogeneous_gpt_model8-1327517696]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_post_process_forward[heterogeneous_gpt_model0-1486901248]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_post_process_forward[heterogeneous_gpt_model2-1444954112]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_post_process_forward[heterogeneous_gpt_model4-1310736384]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_post_process_forward[heterogeneous_gpt_model6-1461735424]',
        'tests/unit_tests/models/test_heterogeneous_gpt_model.py::TestHeterogeneousGPTModel::test_post_process_forward[heterogeneous_gpt_model8-1327517696]',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_constructor',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_forward',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_forward_fsdp',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_freeze',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_preprocess_data',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_save_load',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_set_input_tensor',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelTokenParallel::test_process_embedding_token_parallel[1-8-True-True]',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelTokenParallel::test_process_embedding_token_parallel[2-4-False-True]',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelVisionEncoders::test_constructor[radio-g]',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelVisionEncoders::test_constructor[siglip]',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelVisionEncoders::test_set_input_tensor[radio-g]',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelVisionEncoders::test_set_input_tensor[siglip]',
        'tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptGPTModel::test_inference',
        'tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptGPTModel::test_sharded_state_dict_restore',
        'tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptLlama4MoE::test_inference',
        'tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptLlama4MoE::test_sharded_state_dict_restore',
        'tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptMLAMoE::test_inference',
        'tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptMLAMoE::test_sharded_state_dict_restore',
        'tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptMambaModel::test_inference',
        'tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptMambaModel::test_sharded_state_dict_restore',
        'tests/unit_tests/tensor_parallel/test_random.py::test_checkpoint_without_output',
        'tests/unit_tests/test_checkpointing.py::test_dist_checkpoint_versioning',
        'tests/unit_tests/test_checkpointing.py::test_load_checkpoint[torch]',
        'tests/unit_tests/test_utils.py::test_nvtx_range[None-None]',
        'tests/unit_tests/test_utils.py::test_nvtx_range[None-test_suffix]',
        'tests/unit_tests/test_utils.py::test_nvtx_range[test_message-None]',
        'tests/unit_tests/test_utils.py::test_nvtx_range[test_message-test_suffix]',
        'tests/unit_tests/transformer/test_attention.py::TestSelfAttention::test_self_attention_independent_pg_smoke',
        'tests/unit_tests/transformer/test_attention.py::TestSelfAttention::test_self_attention_mpu',
        'tests/unit_tests/transformer/test_attention_packed_seq.py::TestParallelAttentionWithPackedPaddedSequence::test_checkpointed_gpu_forward',
        'tests/unit_tests/transformer/test_attention_packed_seq.py::TestParallelAttentionWithPackedPaddedSequence::test_gpu_forward',
        'tests/unit_tests/transformer/test_attention_packed_seq.py::TestParallelAttentionWithPackedSequence::test_checkpointed_gpu_forward',
        'tests/unit_tests/transformer/test_attention_packed_seq.py::TestParallelAttentionWithPackedSequence::test_cpu_forward',
        'tests/unit_tests/transformer/test_attention_packed_seq.py::TestParallelAttentionWithPackedSequence::test_gpu_forward',
        'tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPrediction::test_sharded_state_dict[1]',
        'tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPrediction::test_sharded_state_dict[2]',
        'tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPrediction::test_sharded_state_dict[4]',
        'tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPredictionLayer::test_constructor_local[2]',
        'tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPredictionLayer::test_constructor_local[4]',
        'tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPredictionLayer::test_constructor_ues_te[1]',
        'tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPredictionLayer::test_constructor_ues_te[2]',
        'tests/unit_tests/transformer/test_multi_token_prediction.py::TestMultiTokenPredictionLayer::test_constructor_ues_te[4]',
        'tests/unit_tests/transformer/test_transformer_block.py::TestMixedProcessGroups::test_mixed_pg_transformer_block[2-4]',
        'tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_constructor',
        'tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_cpu_forward',
        'tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_gpu_forward_full_checkpoint_fp8',
        'tests/unit_tests/transformer/test_transformer_block.py::TestParallelTransformerBlock::test_gpu_forward_selective_checkpoint_fp8',
        'tests/unit_tests/transformer/test_transformer_layer.py::TestParallelTransformerLayer::test_chunked_mlp',
        'tests/unit_tests/transformer/test_transformer_layer.py::TestParallelTransformerLayer::test_constructor',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[8-1-1]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[4-2-1]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[1-1-8]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[2-1-4]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[2-2-2]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[True-8-1-1]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[True-4-2-1]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[True-1-1-8]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[True-2-1-4]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[True-2-2-2]',
        'tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[silu-8-1]',
        'tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[silu-4-2]',
        'tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[gelu-8-1]',
        'tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[gelu-4-2]',
        'tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[relu-8-1]',
        'tests/unit_tests/transformer/moe/test_dynamic_mlp.py::test_dynamic_mlp_fwd_bwd[relu-4-2]',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_constructor',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_gpu_forward_backward',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_gpu_forward_backward_with_no_tokens_allocated',
        'tests/unit_tests/transformer/moe/test_moe_layer.py::TestMoELayerInit::test_te_moe_layer[True-1-allgather]',
        'tests/unit_tests/transformer/moe/test_moe_layer.py::TestMoELayerInit::test_te_moe_layer[True-1-alltoall]',
        'tests/unit_tests/transformer/moe/test_moe_layer.py::TestMoELayerInit::test_te_moe_layer[True-2-allgather]',
        'tests/unit_tests/transformer/moe/test_moe_layer.py::TestMoELayerInit::test_te_moe_layer[True-2-alltoall]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_dispatcher_discrepancy[1-2-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_dispatcher_discrepancy[2-1-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_dispatcher_discrepancy[8-1-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_dispatcher_discrepancy[2-2-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_dispatcher_discrepancy[2-4-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_dispatcher_discrepancy[4-2-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[1-1-True-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[1-2-True-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[1-8-True-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[2-1-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[2-1-True-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[8-1-True-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[2-2-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[2-2-True-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[2-4-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[2-4-True-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[4-2-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[4-2-True-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_a2a_dispatcher_discrepancy[2-1-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_a2a_dispatcher_discrepancy[4-1-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_a2a_dispatcher_discrepancy[8-1-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_a2a_dispatcher_discrepancy[2-4-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_a2a_dispatcher_discrepancy[4-2-False-8]',
        'tests/unit_tests/transformer/moe/test_routers.py::TestTopRouter::test_constructor',
        'tests/unit_tests/transformer/moe/test_routers.py::TestTopRouter::test_router_forward[2-True]',
        'tests/unit_tests/transformer/moe/test_routers.py::TestTopRouter::test_router_forward[2-False]',
        'tests/unit_tests/transformer/moe/test_routers.py::TestTopRouter::test_router_forward[3-True]',
        'tests/unit_tests/transformer/moe/test_routers.py::TestTopRouter::test_router_forward[3-False]',
        'tests/unit_tests/transformer/moe/test_routers.py::TestTopRouter::test_aux_loss[2]',
        'tests/unit_tests/transformer/moe/test_routers.py::TestTopRouter::test_aux_loss[3]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[8-1]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[1-8]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[2-4]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[1-1]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_moe_tp_forward_backward[1-1-8]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_moe_tp_forward_backward[1-2-4]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_moe_tp_forward_backward[1-4-2]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_moe_tp_forward_backward[2-2-4]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_dispatcher_discrepancy[1-8-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_ag_dispatcher_discrepancy[8-1-False-8]',
        'tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py::TestMoELayerDispatcherDiscrepancy::test_moe_layer_dispatcher_discrepancy[1-1-False-8]',
        # Flaky
        'tests/unit_tests/dist_checkpointing/test_serialization.py',
        'tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestCrossRanksReads::test_full_dp_reads',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-1-False-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-1-True-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-1-True-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-1-False-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-1-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-1-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-1-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-1-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-1-False-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-1-True-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-1-True-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-1-False-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-1-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-1-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-1-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-1-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-2-False-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-2-True-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-2-True-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-2-False-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-2-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-2-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-2-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-2-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-2-False-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-2-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-2-False-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-2-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-2-True-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-2-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-2-True-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-2-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-1-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-1-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-1-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-1-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-2-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-2-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-2-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-2-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-1-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-1-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-1-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-1-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-2-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-2-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-2-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-2-True-True-True]',
        # #34
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_dp_sharding[tp_pp0-2-2-False-initialize_small_model]',
    ],
    'https://jira.habana-labs.com/browse/SW-238787': [
        'tests/unit_tests/dist_checkpointing/test_async_save.py::TestAsyncSave::test_async_is_equivalent_to_sync[True]',
        'tests/unit_tests/dist_checkpointing/test_fp8.py::TestFP8::test_fp8_save_load[True-src_tp_pp0-dest_tp_pp0-broadcast]',
        'tests/unit_tests/dist_checkpointing/test_fp8.py::TestFP8::test_fp8_save_load[False-src_tp_pp2-dest_tp_pp2-None]',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelVisionEncoders::test_constructor[siglip]',
        'tests/unit_tests/test_optimizer.py::test_optim_sharded_state_dict[bf16-False]',
        'tests/unit_tests/test_optimizer.py::test_optim_sharded_state_dict[bf16-True]',
        'tests/unit_tests/test_optimizer.py::test_optim_sharded_state_dict[fp32-False]',
        'tests/unit_tests/test_optimizer.py::test_optim_sharded_state_dict[fp32-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_dp_sharding[tp_pp0-2-2-True-initialize_small_model]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_dp_sharding[tp_pp0-2-2-True-initialize_gpt_model]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_dp_sharding[tp_pp0-2-2-True-initialize_1d_flatten_tensor_model]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp0-dest_tp_pp0-False-True-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp0-dest_tp_pp0-True-True-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp0-dest_tp_pp0-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp1-dest_tp_pp1-False-True-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp1-dest_tp_pp1-True-True-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp1-dest_tp_pp1-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp2-dest_tp_pp2-False-True-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp2-dest_tp_pp2-True-True-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp2-dest_tp_pp2-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp3-dest_tp_pp3-False-True-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp3-dest_tp_pp3-True-True-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp3-dest_tp_pp3-True-True-True]',
        # fails after missing PRs
        'tests/unit_tests/dist_checkpointing/test_global_metadata_reuse.py::TestGlobalMetadataReuse::test_global_metadata_reuse[2-4]',
        'tests/unit_tests/dist_checkpointing/test_global_metadata_reuse.py::TestGlobalMetadataReuse::test_no_global_metadata_reuse_on_different_parallelism[2-4]',
        'tests/unit_tests/dist_checkpointing/test_nonpersistent.py::TestLegacySaveAndLoad::test_basic_save_load_scenario[2-4]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_dp_sharding[tp_pp0-2-2-False-initialize_gpt_model]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_dp_sharding[tp_pp0-2-2-True-initialize_gpt_model]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_finetune_doesnt_load_optimizer[src_tp_pp0-dest_tp_pp0-False]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_finetune_doesnt_load_optimizer[src_tp_pp1-dest_tp_pp1-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_finetune_doesnt_load_optimizer[src_tp_pp2-dest_tp_pp2-False]',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_single_process_save_load',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_multi_process_save',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_multi_process_save_log_difference',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_partition_change_save_load',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_load_tensors_metadata',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_can_mix_sharded_tensors_and_factories',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_load_error_msg',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_sharded_object_serialization',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_tensor_shape_mismatch',
        'tests/unit_tests/models/test_bert_model.py::TestBertModel::test_constructor',
        'tests/unit_tests/models/test_bert_model.py::TestBertModel::test_set_input_tensor',
        'tests/unit_tests/models/test_bert_model.py::TestBertModel::test_post_process_forward',
        'tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_local_spec',
        'tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_local_spec_exception',
        'tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_transformer_engine_version_1_10',
        'tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_transformer_engine_version_1_7_to_1_10_flash_attn',
        'tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_transformer_engine_version_1_7_to_1_10_rng_error',
        'tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_transformer_engine_version_1_7_to_1_10_unfused_attention',
        'tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_transformer_engine_version_less_than_1_7',
    ],
    'https://jira.habana-labs.com/browse/SW-Flaky': [
        'tests/unit_tests/data/test_builder.py::test_builder',
        'tests/unit_tests/test_parallel_state.py::test_tensor_model_parellel_world_size[tp-cp-pp-ep-dp]',
        'tests/unit_tests/data/test_preprocess_data.py::test_preprocess_data_gpt',
        'tests/unit_tests/data/test_gpt_dataset.py::test_mock_gpt_dataset',
        'tests/unit_tests/data/test_gpt_dataset.py::test_mock_gpt_dataset',
        'tests/unit_tests/data/test_multimodal_dataset.py::test_mock_multimodal_dataset',
        'tests/unit_tests/data/test_preprocess_mmdata.py::test_preprocess_mmdata',
        'tests/unit_tests/test_parallel_state.py::test_initialize_and_destroy_model_parallel[tp-cp-pp-ep-dp]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestSimpleTextGenerationController::test_generate_all_output_tokens_static_batch',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_forward_backward[True-8-1]',
    ],
    'https://jira.habana-labs.com/browse/SW-237069': [
        'tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestFullyParallelSaveAndLoad::test_load_distribution[True]'
    ],
    'https://jira.habana-labs.com/browse/SW-222660': [
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py',
        'tests/unit_tests/inference/test_modelopt_module_spec.py::TestModelOptMambaModel::test_sharded_state_dict_restore',
        'tests/unit_tests/inference/test_modelopt_module_spec.py::TestModelOptMambaModel::test_inference',
        'tests/unit_tests/ssm/test_mamba_block.py',
        'tests/unit_tests/ssm/test_mamba_hybrid_layer_allocation.py',
        'tests/unit_tests/ssm/test_mamba_layer.py',
        'tests/unit_tests/ssm/test_mamba_mixer.py',
        'tests/unit_tests/models/test_mamba_model.py::TestMambaModel::test_layer_numbers',
    ],
    'https://jira.habana-labs.com/browse/SW-222670': [
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp0-dest_tp_pp_exp0-False-False-False-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp0-dest_tp_pp_exp0-False-False-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp0-dest_tp_pp_exp0-True-False-False-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp0-dest_tp_pp_exp0-True-False-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp1-dest_tp_pp_exp1-False-False-False-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp1-dest_tp_pp_exp1-False-False-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp1-dest_tp_pp_exp1-True-False-False-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp1-dest_tp_pp_exp1-True-False-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp3-dest_tp_pp_exp3-False-False-False-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp3-dest_tp_pp_exp3-False-False-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp3-dest_tp_pp_exp3-True-False-False-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp3-dest_tp_pp_exp3-True-False-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp2-dest_tp_pp_exp2-True-False-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp2-dest_tp_pp_exp2-True-False-False-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp2-dest_tp_pp_exp2-False-False-True-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_chained_optimizer_resharding[src_tp_pp_exp2-dest_tp_pp_exp2-False-False-False-True-True]',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelTokenParallel::test_process_embedding_token_parallel[1-8-True-True]',
    ],
    'https://jira.habana-labs.com/browse/SW-222893': [
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[1-False-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-False-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-1-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-False-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-False-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-False-False-False]',
    ],
    'https://jira.habana-labs.com/browse/SW-222894': [
        'tests/unit_tests/export/trtllm/test_distributed_fp8.py::TestTRTLLMSingleDeviceConverterFP8::test_get_model_weights_converter',
        'tests/unit_tests/export/trtllm/test_single_device_fp8.py::TestTRTLLMSingleDeviceConverterFP8::test_get_model_weights_converter',
    ],
    'https://jira.habana-labs.com/browse/SW-222895': [
        'tests/unit_tests/inference/model_inference_wrappers/t5/test_t5_inference_wrapper.py::TestT5InferenceWrapper::test_inference_only_tensor_parallel',
        'tests/unit_tests/inference/text_generation_controllers/test_encoder_decoder_text_generation_controller.py::TestEncoderDecoderTextGenerationController::test_generate_all_output_tokens_static_batch',
    ],
    'https://jira.habana-labs.com/browse/SW-222896': [
        'tests/unit_tests/tensor_parallel/test_initialization.py::Test::test_te_col_init',
        'tests/unit_tests/tensor_parallel/test_initialization.py::Test::test_te_row_init',
    ],
    'https://jira.habana-labs.com/browse/SW-222899': [
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModelSigLIP::test_constructor'
    ],
    'https://jira.habana-labs.com/browse/SW-201768': [
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-False-True-None]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-False-True-9000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-False-True-9025]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-False-True-9050]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-False-True-18000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-False-True-18050]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-False-True-20000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-True-None]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-True-9000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-True-9025]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-True-9050]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-True-18000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-True-18050]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-True-20000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-False-True-None]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-False-True-9000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-False-True-9025]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-False-True-9050]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-False-True-18000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-False-True-18050]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-False-True-20000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[1-False-False-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[1-False-True-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[1-True-False-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[1-True-True-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[2-True-False-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[2-True-True-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[2-False-False-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[2-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-True-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-True-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[1-2-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-False-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-True-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-True-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-1-True-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-False-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-False-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-False-True-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-True-False-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-True-False-True]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-True-True-False]',
        'tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py::test_grad_sync[2-2-True-True-True]',
        'tests/unit_tests/distributed/test_torch_fully_sharded_parallel.py::test_fsdp2_constructor',
        'tests/unit_tests/distributed/test_torch_fully_sharded_parallel.py::test_fsdp2_constructor_with_process_group',
    ],
    'https://jira.habana-labs.com/browse/SW-201767': [
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_constructor',
        'tests/unit_tests/inference/test_modelopt_gpt_model.py::TestModelOptGPTModel::test_load_te_state_dict_pre_hook',
        'tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_build_module',
    ],
    'https://jira.habana-labs.com/browse/SW-202752': [
        'tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_sliding_window_attention'
    ],
    'https://jira.habana-labs.com/browse/SW-202755': [
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp0-dest_tp_pp_ep_etp0-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-True-src_tp_pp_ep_etp1-dest_tp_pp_ep_etp1-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp2-dest_tp_pp_ep_etp2-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-True-src_tp_pp_ep_etp3-dest_tp_pp_ep_etp3-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp4-dest_tp_pp_ep_etp4-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-True-src_tp_pp_ep_etp5-dest_tp_pp_ep_etp5-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp6-dest_tp_pp_ep_etp6-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp7-dest_tp_pp_ep_etp7-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-True-src_tp_pp_ep_etp8-dest_tp_pp_ep_etp8-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp9-dest_tp_pp_ep_etp9-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp10-dest_tp_pp_ep_etp10-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp11-dest_tp_pp_ep_etp11-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp12-dest_tp_pp_ep_etp12-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp13-dest_tp_pp_ep_etp13-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-True-src_tp_pp_ep_etp14-dest_tp_pp_ep_etp14-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-sequential-False-src_tp_pp_ep_etp15-dest_tp_pp_ep_etp15-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp0-dest_tp_pp_ep_etp0-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-True-src_tp_pp_ep_etp1-dest_tp_pp_ep_etp1-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp2-dest_tp_pp_ep_etp2-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-True-src_tp_pp_ep_etp3-dest_tp_pp_ep_etp3-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp4-dest_tp_pp_ep_etp4-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-True-src_tp_pp_ep_etp5-dest_tp_pp_ep_etp5-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp6-dest_tp_pp_ep_etp6-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp7-dest_tp_pp_ep_etp7-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-True-src_tp_pp_ep_etp8-dest_tp_pp_ep_etp8-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp9-dest_tp_pp_ep_etp9-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp10-dest_tp_pp_ep_etp10-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp11-dest_tp_pp_ep_etp11-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp12-dest_tp_pp_ep_etp12-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp13-dest_tp_pp_ep_etp13-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-True-src_tp_pp_ep_etp14-dest_tp_pp_ep_etp14-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_parallel_reconfiguration_e2e[tp-ep-dp-pp-tp-ep-dp-pp-grouped-False-src_tp_pp_ep_etp15-dest_tp_pp_ep_etp15-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp0-dest_tp_pp_exp0-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp1-dest_tp_pp_exp1-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp2-dest_tp_pp_exp2-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp3-dest_tp_pp_exp3-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp4-dest_tp_pp_exp4-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp5-dest_tp_pp_exp5-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp6-dest_tp_pp_exp6-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp7-dest_tp_pp_exp7-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp8-dest_tp_pp_exp8-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[sequential-grouped-src_tp_pp_exp9-dest_tp_pp_exp9-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp0-dest_tp_pp_exp0-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp1-dest_tp_pp_exp1-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp2-dest_tp_pp_exp2-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp3-dest_tp_pp_exp3-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp4-dest_tp_pp_exp4-False]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp5-dest_tp_pp_exp5-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp6-dest_tp_pp_exp6-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp7-dest_tp_pp_exp7-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp8-dest_tp_pp_exp8-True]',
        'tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_interchangeable[grouped-sequential-src_tp_pp_exp9-dest_tp_pp_exp9-True]',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_constructor',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_weight_init_value_the_same',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gpu_forward',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gpu_forward_with_no_tokens_allocated',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gradient_with_no_tokens_allocated',
    ],
    'https://jira.habana-labs.com/browse/SW-206537': [
        'tests/unit_tests/dist_checkpointing/test_flattened_resharding.py'
    ],
    'https://jira.habana-labs.com/browse/SW-206543': [
        'tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestFullyParallelSaveAndLoad::test_memory_usage[cuda]',
        'tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestFullyParallelSaveAndLoad::test_memory_usage[cpu]',
    ],
    'https://jira.habana-labs.com/browse/SW-206546': [
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestDistributedOptimizer::test_can_load_deprecated_bucket_space_format',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestFP32Optimizer::test_fp32_optimizer_resharding[src_tp_pp0-dest_tp_pp0]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestFP32Optimizer::test_fp32_optimizer_resharding[src_tp_pp1-dest_tp_pp1]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestFP32Optimizer::test_fp32_optimizer_resharding[src_tp_pp2-dest_tp_pp2]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp0-dest_tp_pp0-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp0-dest_tp_pp0-False-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp1-dest_tp_pp1-False-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp1-dest_tp_pp1-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp2-dest_tp_pp2-False-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp2-dest_tp_pp2-True-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp3-dest_tp_pp3-False-True]',
        'tests/unit_tests/dist_checkpointing/test_optimizer.py::TestOptimizerResharding::test_optimizer_resharding[src_tp_pp3-dest_tp_pp3-True-True]',
    ],
    'https://jira.habana-labs.com/browse/SW-206557': [
        'tests/unit_tests/transformer/moe/test_sequential_mlp.py::TestParallelSequentialMLP::test_gpu_forward',
        'tests/unit_tests/transformer/test_mlp.py::TestParallelMLP::test_gpu_forward',
    ],
    'https://jira.habana-labs.com/browse/SW-206558': [
        'tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_constructor',
        'tests/unit_tests/models/test_multimodal_projector.py::TestMultimodalProjector::test_constructor',
    ],
    'https://jira.habana-labs.com/browse/SW-206559': [
        'tests/unit_tests/data/test_preprocess_data.py::test_preprocess_data_bert'
    ],
    'https://jira.habana-labs.com/browse/SW-206560': [
        'tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_transformer_block_custom'
    ],
    'https://jira.habana-labs.com/browse/SW-206561': [
        'tests/unit_tests/test_utils.py::test_straggler_detector'
    ],
    'https://jira.habana-labs.com/browse/SW-214505': [
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp0-dest_tp_pp_exp0-False]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp1-dest_tp_pp_exp1-False]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp2-dest_tp_pp_exp2-False]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp3-dest_tp_pp_exp3-False]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp4-dest_tp_pp_exp4-False]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp5-dest_tp_pp_exp5-False]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp6-dest_tp_pp_exp6-False]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp7-dest_tp_pp_exp7-False]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp8-dest_tp_pp_exp8-True]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp9-dest_tp_pp_exp9-True]',
        'tests/unit_tests/dist_checkpointing/models/test_mamba.py::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp10-dest_tp_pp_exp10-True]',
        'tests/unit_tests/models/test_mamba_model.py::TestMambaModel::test_constructor',
        'tests/unit_tests/models/test_mamba_model.py::TestMambaModel::test_set_input_tensor',
        'tests/unit_tests/models/test_mamba_model.py::TestMambaModel::test_forward',
        'tests/unit_tests/models/test_mamba_model.py::TestMambaModel::test_inference',
        'tests/unit_tests/models/test_mamba_model.py::TestMambaModel::test_save_load',
    ],
    'https://jira.habana-labs.com/browse/SW-214903': [
        'tests/unit_tests/dist_checkpointing/test_fp8.py::TestFP8::test_fp8_save_load[True-src_tp_pp1-dest_tp_pp1-gather_rounds]'
    ],
    'https://jira.habana-labs.com/browse/SW-214904': [
        'tests/unit_tests/dist_checkpointing/test_nonpersistent.py::TestNonPersistentSaveAndLoad::test_basic_save_load_scenarios[2-4]'
    ],
    'https://jira.habana-labs.com/browse/SW-214827': [
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_preprocess_data'
    ],
    'https://jira.habana-labs.com/browse/SW-214828': [
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_forward'
    ],
    'https://jira.habana-labs.com/browse/SW-225513': [
        'tests/unit_tests/inference/test_modelopt_module_spec.py::TestModelOptGPTModel::test_sharded_state_dict_restore'
    ],
    'https://jira.habana-labs.com/browse/SW-225515': [
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_generate_all_output_tokens_static_batch[dtype0]',
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_generate_all_output_tokens_static_batch[dtype1]',
    ],
    'https://jira.habana-labs.com/browse/SW-225516': [
        'tests/unit_tests/inference/text_generation_controllers/test_vlm_text_generation_controller.py::TestVLMTextGenerationController::test_generate_all_output_tokens_static_batch'
    ],
    'https://jira.habana-labs.com/browse/SW-225546': [
        'tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp3-2]',
        'tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp4-2]',
        'tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp5-2]',
    ],
    'https://jira.habana-labs.com/browse/SW-241257': [
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py'
    ],
    'DONT-FIX': [
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file47---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file147---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file152---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file153---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file155---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file156---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file186---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file187---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file189---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file190---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file191---log-memory-to-tensorboard]",
        "tests/unit_tests/test_model_configs.py::test_model_config_tracks_memory[yaml_file192---log-memory-to-tensorboard]",
    ],
}

unit_tests_to_deselect_eager_only = {
    'https://jira.habana-labs.com/browse/SW-TODO': [
        'tests/unit_tests/inference/',  # Fails to exit gracefully 9/11 passed.
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_generate_all_output_tokens_static_batch',  # Fails to exit gracefully
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-te-te]',
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-local-local]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5ModelReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_encpp1-dest_tp_pp_encpp1-t5-local-local]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5ModelReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_encpp2-dest_tp_pp_encpp2-t5-local-local]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5ModelReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_encpp3-dest_tp_pp_encpp3-t5-local-local]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5ModelReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_encpp4-dest_tp_pp_encpp4-t5-local-local]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5ModelReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_encpp5-dest_tp_pp_encpp5-t5-local-local]',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_remove_sharded_tensors',
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_empty_load',
    ],
    'https://jira.habana-labs.com/browse/SW-216976': [
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBERTModelReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp1-dest_tp_pp1-src_layer_spec1-dst_layer_spec1]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBERTModelReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp0-dest_tp_pp0-src_layer_spec0-dst_layer_spec0]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[True-tp-dp-pp-tp-pp-dp-src_tp_pp2-dest_tp_pp2-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-dp-pp-tp-dp-pp-src_tp_pp0-dest_tp_pp0-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-dp-pp-tp-dp-pp-src_tp_pp0-dest_tp_pp0-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-pp-dp-tp-pp-dp-src_tp_pp1-dest_tp_pp1-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[True-tp-dp-pp-tp-pp-dp-src_tp_pp2-dest_tp_pp2-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-dp-pp-tp-dp-pp-src_tp_pp3-dest_tp_pp3-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[True-tp-pp-dp-tp-pp-dp-src_tp_pp4-dest_tp_pp4-get_gpt_layer_local_spec-get_gpt_layer_local_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-dp-pp-tp-pp-dp-src_tp_pp8-dest_tp_pp8-get_gpt_layer_local_spec-get_gpt_layer_local_spec]',
    ],
    'https://jira.habana-labs.com/browse/SW-206331': [
        'tests/unit_tests/transformer/test_module.py::TestFloat16Module::test_fp16_module'
    ],
    'https://jira.habana-labs.com/browse/SW-206335': [
        'tests/unit_tests/transformer/test_module.py::TestFloat16Module::test_bf16_module'
    ],
    'https://jira.habana-labs.com/browse/SW-206337': [
        'tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_fused_rope_gpu_forward'
    ],
    'https://jira.habana-labs.com/browse/SW-206551': [
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModel::test_sharded_state_dict_save_load[get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_local_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModel::test_sharded_state_dict_save_load[get_gpt_layer_local_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-dp-pp-tp-pp-dp-src_tp_pp5-dest_tp_pp5-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_local_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[True-tp-dp-pp-tp-dp-pp-src_tp_pp6-dest_tp_pp6-get_gpt_layer_local_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-pp-dp-tp-pp-dp-src_tp_pp7-dest_tp_pp7-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_local_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-te-local]',
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-local-te]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5Model::test_sharded_state_dict_save_load[t5-te-local]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5Model::test_sharded_state_dict_save_load[t5-local-te]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBertModel::test_sharded_state_dict_save_load[dst_layer_spec0-src_layer_spec1]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBertModel::test_sharded_state_dict_save_load[dst_layer_spec1-src_layer_spec0]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBERTModelReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp5-dest_tp_pp5-src_layer_spec5-dst_layer_spec5]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBERTModelReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp6-dest_tp_pp6-src_layer_spec6-dst_layer_spec6]',
    ],
    'https://jira.habana-labs.com/browse/SW-206537': [
        'tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py',
        'tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py',
        'tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py',
    ],
    'https://jira.habana-labs.com/browse/SW-217295': [
        'tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_tensor_shape_mismatch'
    ],
    'https://jira.habana-labs.com/browse/SW-225547': [
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_forward_fsdp'
    ],
}

unit_tests_to_deselect_lazy_only = {
    '0.13.1-rebase': [
        'tests/unit_tests/distributed/test_finalize_model_grads.py::TestAllReduceLNGrads::test_allreduce_word_embedding_grads[True-2-True]',
        'tests/unit_tests/distributed/test_finalize_model_grads.py::TestAllReduceLNGrads::test_allreduce_word_embedding_grads[False-2-True]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-sgd-0.5-False-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-sgd-0.5-False-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-sgd-0.5-True-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-sgd-0.5-True-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-sgd-1.0-False-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-sgd-1.0-False-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-sgd-1.0-True-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-sgd-1.0-True-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-adam-0.5-False-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-adam-0.5-False-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-adam-0.5-True-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-adam-0.5-True-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-adam-1.0-False-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-adam-1.0-False-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-adam-1.0-True-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[False-adam-1.0-True-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-sgd-0.5-False-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-sgd-0.5-False-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-sgd-0.5-True-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-sgd-0.5-True-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-sgd-1.0-False-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-sgd-1.0-False-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-sgd-1.0-True-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-sgd-1.0-True-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-adam-0.5-False-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-adam-0.5-False-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-adam-0.5-True-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-adam-0.5-True-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-adam-1.0-False-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-adam-1.0-False-10]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-adam-1.0-True-1]',
        'tests/unit_tests/test_optimizer_cpu_offloading.py::test_multi_device_hybrid_optimizer[True-adam-1.0-True-10]',
        'tests/unit_tests/test_inference.py::test_completions',
    ],
    'https://jira.habana-labs.com/browse/SW-206540': [
        'tests/unit_tests/dist_checkpointing/test_serialization.py',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py',
        'tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py',
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py',
    ],
}

unit_tests_to_deselect_sim_only = {
    'SW-TODO-Flaky': ['tests/unit_tests/dist_checkpointing/', 'tests/unit_tests/distributed/'],
    'SW-SLOW-TESTS': ['tests/unit_tests/models/'],
}

all_xfails_dict = {
    node_id: jira for jira in unit_tests_to_deselect for node_id in unit_tests_to_deselect[jira]
}

eager_only_xfail_dict = {
    node_id: jira
    for jira in unit_tests_to_deselect_eager_only
    for node_id in unit_tests_to_deselect_eager_only[jira]
}

lazy_only_xfail_dict = {
    node_id: jira
    for jira in unit_tests_to_deselect_lazy_only
    for node_id in unit_tests_to_deselect_lazy_only[jira]
}

sim_only_xfail_dict = {
    node_id: jira
    for jira in unit_tests_to_deselect_sim_only
    for node_id in unit_tests_to_deselect_sim_only[jira]
}

if os.getenv("PT_HPU_LAZY_MODE", "0") == "0":
    all_xfails_dict.update(eager_only_xfail_dict)
else:
    all_xfails_dict.update(lazy_only_xfail_dict)

if os.getenv("MLM_DUT", "0") == "SIMULATOR":
    all_xfails_dict.update(sim_only_xfail_dict)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if test_in_xfail_dict(all_xfails_dict, item.nodeid):
            reason_str = get_reason_for_xfail(all_xfails_dict, item.nodeid)
            xfail_marker = pytest.mark.xfail(run=False, reason=reason_str)
            item.user_properties.append(("xfail", "true"))
            item.add_marker(xfail_marker)
