# © 2024-2025 Intel Corporation

import os

try:
    import habana_frameworks.torch.gpu_migration
except:
    pass

import pytest


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
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[False-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[True-True]',
    ],
    'https://jira.habana-labs.com/browse/SW-201767': [
        'tests/unit_tests/models/test_clip_vit_model.py::TestCLIPViTModel::test_constructor',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_constructor',
        'tests/unit_tests/inference/test_modelopt_gpt_model.py::TestModelOptGPTModel::test_load_te_state_dict_pre_hook',
        'tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_build_module',
    ],
    'https://jira.habana-labs.com/browse/SW-202752': [
        'tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_sliding_window_attention'
    ],
    'https://jira.habana-labs.com/browse/SW-202755': [
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_constructor',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_weight_init_value_the_same',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gpu_forward',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gpu_forward_with_no_tokens_allocated',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gradient_with_no_tokens_allocated',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_constructor',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_gpu_forward_backward',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_gpu_forward_backward_with_no_tokens_allocated',
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
    'https://jira.habana-labs.com/browse/SW-206636': [
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[True-8-1]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[True-1-8]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward[True-2-4]',
        'tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_extend_tp_forward_backward[True-2-4]',
    ],
}

unit_tests_to_deselect_eager_only = {
    'https://jira.habana-labs.com/browse/SW-TODO': [
        'tests/unit_tests/inference/',  # Fails to exit gracefully 9/11 passed.
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_generate_all_output_tokens_static_batch',  # Fails to exit gracefully
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
}


unit_tests_to_deselect_lazy_only = {
    'https://jira.habana-labs.com/browse/SW-206540': [
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_generate_all_output_tokens_static_batch',
        'tests/unit_tests/dist_checkpointing/test_serialization.py',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py',
        'tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py',
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py',
    ],
    'https://jira.habana-labs.com/browse/SW-214827': [
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_preprocess_data'
    ],
    'https://jira.habana-labs.com/browse/SW-214828': [
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_forward'
    ],
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

if os.getenv("PT_HPU_LAZY_MODE") == "0":
    all_xfails_dict.update(eager_only_xfail_dict)
else:
    all_xfails_dict.update(lazy_only_xfail_dict)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if test_in_xfail_dict(all_xfails_dict, item.nodeid):
            reason_str = get_reason_for_xfail(all_xfails_dict, item.nodeid)
            xfail_marker = pytest.mark.xfail(run=False, reason=reason_str)
            item.user_properties.append(("xfail", "true"))
            item.add_marker(xfail_marker)
