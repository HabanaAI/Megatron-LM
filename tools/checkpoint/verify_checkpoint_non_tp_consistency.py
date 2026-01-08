# © 2024-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from collections import namedtuple
from dataclasses import dataclass
import os
import re
import glob
import tqdm
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))


ParallelConfig = namedtuple('ParallelConfig', 'dp_degree tp_degree pp_degree ep_degree')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", default=None, type=str, help="Checkpoint folder"
    )
    parser.add_argument(
        "--model_type",
        default="LLAMA",
        type=str,
        help="Type of the model",
        choices=["LLAMA", "MIXTRAL", "DEEPSEEK"],
    )
    args = parser.parse_args()
    print(f"args = {args}")
    return args


class MLMCheckpoint:
    def __init__(self, folder, args, model_type) -> None:
        if hasattr(args, "use_dist_ckpt"):
            self.use_dist_ckpt = args.use_dist_ckpt
        if hasattr(args, "dist_ckpt_format"):
            self.dist_ckpt_format = args.dist_ckpt_format
        if hasattr(args, "use_distributed_optimizer"):
            self.use_distributed_optimizer = args.use_distributed_optimizer
        if not os.path.basename(folder).startswith("iter_"):
            filename = os.path.join(folder, "latest_checkpointed_iteration.txt")
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    latest_checkpointed_iteration = int(f.readline().rstrip())
                folder = os.path.join(folder, f"iter_{latest_checkpointed_iteration:07d}")
        self.ckpt_folder = folder

        tp, pp, ep = (1, 1, 1)
        if hasattr(args, "data_parallel_size") and hasattr(args, "tensor_model_parallel_size") and hasattr(args, "pipeline_model_parallel_size") and (not model_type in ['MIXTRAL', 'DEEPSEEK'] or hasattr(args, "expert_model_parallel_size")):
            dp = args.data_parallel_size
            tp = args.tensor_model_parallel_size
            pp = args.pipeline_model_parallel_size
            if model_type in ['MIXTRAL', 'DEEPSEEK']:
                ep = args.expert_model_parallel_size
        else:
            files = glob.glob(os.path.join(folder, 'mp_rank_*'))
            if hasattr(args, "use_dist_ckpt"):
                self.use_dist_ckpt = len(files) == 0
            if not self.use_dist_ckpt:
                files = [os.path.basename(file).split("mp_rank_")[1].split("_") for file in files]
                # Megatron-LM checkpoints can take the following formats:
                # For LLama:
                # TT, TT_PPP
                # Additionally, for Mixtral:
                # TT_EEE, TT_PPP_EEE
                # In scenarios where two parallelization methods are applied
                # with Mixtral, the file name alone does not allow us to
                # determine whether the second method is expert parallelism or
                # pipeline parallelism. Therefore, for validation purposes
                # only, we assume it to be pipeline parallelism.

                tp_files = [file[0] for file in files]
                tp = len(set(tp_files))

                if len(files[0]) > 1:
                    pp_files = [file[1] for file in files]
                    pp = len(set(pp_files))

                if len(files[0]) > 2:
                    ep_files = [file[2] for file in files]
                    ep = len(set(ep_files))
                dp = 'x'
        self.distrib_optim_filename = "distrib_optim.pt"
        self.model_optim_rng_filename = "model_optim_rng.pt"
        self.parallel_config = ParallelConfig(dp_degree=dp, tp_degree=tp, pp_degree=pp, ep_degree=ep)


    def get_folder(self, pp, ep, pp_rank, ep_rank, tp_rank):
        folder_template = ""
        if pp == 1 and ep == 1:
            folder_template = "mp_rank_{:02d}"
        elif pp == 1 and ep > 1:
            folder_template = f"mp_rank_{{:02d}}_{ep_rank:03d}"
        elif pp > 1 and ep == 1:
            folder_template = f"mp_rank_{{:02d}}_{pp_rank:03d}"
        elif pp > 1 and ep > 1:
            folder_template = f"mp_rank_{{:02d}}_{pp_rank:03d}_{ep_rank:03d}"
        else:
            raise ValueError("Incorrect pipeline parallel and expert parallel sizes.")
        
        folder_name = folder_template.format(tp_rank)
        folder = os.path.join(self.ckpt_folder, folder_name)

        return folder


    def validate_files(self, model_type):
        if not self.use_dist_ckpt:
            tp = self.parallel_config.tp_degree
            pp = self.parallel_config.pp_degree
            ep = self.parallel_config.ep_degree

            for tp_rank in range(tp):
                for pp_rank in range(pp):
                    # Expert parallelism is used only in Mixtral; for other
                    # models, just for the purpose of validation it is
                    # simplified by setting it to 1.
                    for ep_rank in range(ep):
                        folder = self.get_folder(pp, ep, pp_rank, ep_rank, tp_rank)

                        if model_type in ["MIXTRAL", "DEEPSEEK"]:
                            error_message = f"{folder=} does not exist, {pp=}, {tp=}, {ep=}"
                        else:
                            error_message = f"{folder=} does not exist, {pp=}, {tp=}"

                        assert os.path.exists(folder), f"{error_message}"
                        files = os.listdir(folder)
                        # Filtering only the files.
                        files = [f for f in files if os.path.isfile(os.path.join(folder, f))]
                        num_files = 1
                        if self.use_distributed_optimizer:
                            num_files += 1
                        assert len(files) >= num_files
                        if self.use_distributed_optimizer:
                            assert self.distrib_optim_filename in files
                        assert self.model_optim_rng_filename in files


def show_3d(mlm_checkpoint, model_type):
    parallel_config = mlm_checkpoint.parallel_config
    dp, tp, pp, ep = parallel_config.dp_degree, parallel_config.tp_degree, parallel_config.pp_degree, parallel_config.ep_degree
    if model_type in ['MIXTRAL', 'DEEPSEEK']:
        print(f"4D configuration: DP={dp} TP={tp} PP={pp}, EP={ep}")
    else:
        print(f"3D configuration: DP={dp} TP={tp} PP={pp}")


def get_model_optim_rng_patterns_for_non_sharded(model_type):
    if model_type == "LLAMA":
        return [
            r"embedding.word_embeddings.bias",
            r"embedding.position_embeddings.weight",
            r"decoder.layers.+\d+.input_layernorm.weight",
            r"decoder.layers.+\d+.input_layernorm.bias",
            r"decoder.layers.+\d+.self_attention.linear_qkv.norm.weight",
            r"decoder.layers.+\d+.self_attention.linear_qkv.norm.bias",
            r"decoder.layers.+\d+.self_attention.linear_qkv.bias",
            r"decoder.layers.+\d+.self_attention.linear_proj.bias",
            r"decoder.layers.+\d+.pre_mlp_layernorm.weight",
            r"decoder.layers.+\d+.pre_mlp_layernorm.bias",
            r"decoder.layers.+\d+.mlp.linear_fc1.norm.weight",
            r"decoder.layers.+\d+.mlp.linear_fc1.norm.bias",
            r"decoder.layers.+\d+.mlp.linear_fc1.bias",
            r"decoder.layers.+\d+.mlp.linear_fc2.bias",
            r"decoder.final_layernorm.weight",
            r"decoder.final_layernorm.bias",
        ]
    elif model_type in ["MIXTRAL", "DEEPSEEK"]:
        return [
            r"decoder.layers.+\d+.input_layernorm.weight",
            r"decoder.layers.+\d+.self_attention.linear_qkv.norm.weight",
            r"decoder.layers.+\d+.pre_mlp_layernorm.weight",
            r"decoder.layers.+\d+.mlp.linear_fc1.norm.weight",
            r"decoder.final_layernorm.weight",
        ]


@dataclass
class ParamInfo:
    pp: int
    tp: int
    dp: int
    ep: int
    data: torch.Tensor
    numel: int


def verify_equal_params(params, tp):
    failed = 0
    report = {}
    for name, info in params.items():
        n = len(info)
        if n != tp:
            ok = False
            print(f"{name}: FAILED expected n={n} == tp={tp}")
        elif n == 1:
            ok = True
        else:
            ok = all([(x.numel == info[0].numel) for x in info[1:]])
            if not ok:
                print(f"{name}: FAILED numel comparison [n={n}]")
            else:
                ok = all([x.data.eq(info[0].data).all().item() for x in info[1:]])
                if not ok:
                    print(f"{name}: FAILED data comparison [n={n}]")
        failed += ok == False
        report[name] = (ok, n)
        if ok:
            print(f"{name}: OK [n={n}]")
    return failed, report


def update_model_optim_rng_non_sharded_params(params, model_type, filename, pp_index, tp_index, ep_index):
    sd = torch.load(filename, map_location=torch.device("cpu"), weights_only=False)['model']
    model_optim_rng_patterns = get_model_optim_rng_patterns_for_non_sharded(model_type)
    for key in sd.keys():
        if not any(re.match(model_optim_rng_pattern, key) for model_optim_rng_pattern in model_optim_rng_patterns):
            continue
        if key not in params:
            params[key] = []
        info = ParamInfo(
            pp=pp_index, tp=tp_index, dp=-1, ep=(ep_index if model_type in ['MIXTRAL', 'DEEPSEEK'] else None), data=sd[key], numel=sd[key].numel()
        )
        params[key].append(info)
    return params


def verify_model_optim_rng_files(mlm_checkpoint, model_type):
    parallel_config = mlm_checkpoint.parallel_config
    tp, pp, ep = parallel_config.tp_degree, parallel_config.pp_degree, parallel_config.ep_degree

    total_failed = 0
    if not mlm_checkpoint.use_dist_ckpt:
        for pp_index in range(pp):
            for ep_index in range(ep):
                if model_type in ['MIXTRAL', 'DEEPSEEK']:
                    print(f"\nChecking pp_stage={pp_index}, ep_stage={ep_index}")
                else:
                    print(f"\nChecking pp_stage={pp_index}")
                params = {}
                for tp_index in range(tp):
                    folder = mlm_checkpoint.get_folder(pp, ep, pp_index, ep_index, tp_index)
                    filename = os.path.join(folder, mlm_checkpoint.model_optim_rng_filename)
                    update_model_optim_rng_non_sharded_params(
                        params, model_type, filename, pp_index, tp_index, ep_index
                    )
                failed, report = verify_equal_params(params, tp)
                total_failed += failed
    return total_failed


def verify_checkpoint(folder, model_type, args=None):
    mlm_checkpoint = MLMCheckpoint(folder, args=args, model_type=model_type)
    mlm_checkpoint.validate_files(model_type)
    show_3d(mlm_checkpoint, model_type)

    if not mlm_checkpoint.use_dist_ckpt:
        print("\nVerify ** model_optim_rng ** files")
    total_failed_model_optim_rng = verify_model_optim_rng_files(mlm_checkpoint, model_type)
    if total_failed_model_optim_rng == 0:
        if not mlm_checkpoint.use_dist_ckpt:
            print("\nCheckpoint model_optim_rng files OK")
    else:
        if not mlm_checkpoint.use_dist_ckpt:
            print(f"\nCheckpoint model_optim_rng files BAD with total_failed={total_failed_model_optim_rng}")

    # TODO: if possible find/explore a way to verify distributed optimizer ckpt also

    return total_failed_model_optim_rng == 0


def main():
    print(f"Verify Checkpoint consistency for non-TP-sharded parameters")
    args = parse_arguments()
    assert (
        verify_checkpoint(args.folder, args.model_type, args) is True
    ), "Checkpoint verification failed"


if __name__ == "__main__":
    main()
