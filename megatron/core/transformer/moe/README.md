# Megatron Core MoE Key Features

Megatron-Core offers rich parallelism mappings, combining Expert Parallelism with tensor, data, sequence, and pipeline parallelism. This boosts Mixtral 8X7B bf16 training to achieve **468 TFLOPS** as of MCore v0.9.


### Parallelism
- **Expert Parallelism**
    - A specific method of parallelism for MoE models, where experts are partitioned onto different workers and each worker processes a different batch of training samples, each worker process one or more experts for each MoE layer.
- **3D Parallelism**: Data Parallelism, Tensor Parallelism, Pipeline Parallelism
    - Note: When using MoE with expert parallelism and tensor parallelism, sequence parallelism must be enabled.
- **Context Parallelism**:
    - Split the sequence dimension to support long context training.
- **Richer parallel mappings**: EP can be combined with DP/TP/PP/CP for handling larger MoE variants.
- **Full distributed optimizer support.**

### Router and Load Balancing
- Router type:
    - Top-K MLP router
- Load Balancing algorithms:
    - Sinkhorn (S-BASE)
    - Aux loss / Load balancing loss

### Performance Optimizations
- Fused MoE Kernel
    - Supported dtypes: bf16, fp32
    - Supported only on HPU
    - Performance improvement and faster load balancing
    - Enable `--moe-dynamic-hpu`
- GroupedGEMM when num local experts > 1
    - Supported dtype: bf16
    - Performance improvements for larger MoE models
- Enable `--tp-comm-overlap` for MoE
- FP8 training support

### Token Dispatch Mechanism

- Dropless / No token drop
- Token drop and padding
- Capacity bins (no drop with optimized discrete padding)

### Ease of use
- Checkpoint converter for Mixtral models, see the [example](https://github.com/HabanaAI/Megatron-LM/tree/main/examples/mixtral) for details.
- Distributed checkpoining
- Per-layer logging
- Upcycling Support
- Granular upcycling

## Upcoming features
- New Parallelism for Large-scale MoE training
- FP8 support for GroupedGEMM
- Token permutation / Unpermutation fusion
- TopK Router Fusion
- MoE Layer Frequency
- Fused Sinkhorn Kernel
- Context Parallel with MoE
- FP8 training support
- TEGroupedMLP support for HPU

# User Guide

### MoE Related Arguments

| Item | Description |
| --- | --- |
| --num-experts | Number of Experts in MoE (None means no MoE) |
| --expert-model-parallel-size | Degree of expert model parallelism. Default is 1. |
| --moe-ffn-hidden-size | MoE Feed-Forward Network hidden size. Default is None. |
| --expert-tensor-parallel-size | Degree of tensor model parallelism of expert layer. Default is same to --tensor-model-parallel-size. |
| --moe-layer-freq | Frequency between MoE layers and Dense layers. Accepts either: 1) An integer N for 1:N ratio (one expert layer for every N-1 dense layers), 2) A string "N" for the same ratio, or 3) A string with Python list expression for custom patterns like `([1]*3+[0]*1)*3` which gives [1,1,1,0,1,1,1,0,1,1,1,0] where 1=expert layer and 0=dense layer. Examples: `([0]+[1]*23)` for 1 dense layer followed by 23 experts layers, `([1]*3+[0]*2)*2` for three expert layers followed by two dense layers, repeated twice. Default is 1. |
| --moe-grouped-gemm | When there are multiple experts per rank, launch multiple local GEMM kernels in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine. |
| --moe-router-load-balancing-type | Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss". |
| --moe-router-topk | Number of experts to route to for each token. The default is 2. |  
| --moe-aux-loss-coeff | Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended. Default is 0.0. |
| --moe-z-loss-coeff | Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended. Default is None. |
| --moe-input-jitter-eps | Add noise to the input tensor by applying jitter with a specified epsilon value. Default is None. |
| --moe-token-dispatcher-type | Determines the token dispatcher type. Choices are "allgather", "alltoall" and "alltoall_seq". Default is "allgather". We recommend using 'alltoall' if expert parallelism is applied. We have upgraded the "alltoall" dispatcher in place during MCore v0.9, while retaining the original implementation, renamed as "alltoall_seq".|
| --moe-per-layer-logging | Enable per-layer logging for MoE, currently supports auxiliary loss and z loss. |
| --moe-expert-capacity-factor | The capacity factor for each expert, None means no token will be dropped. Default is None. |
| --moe-pad-expert-input-to-capacity | Pads the input for each expert to match the expert capacity length, effective only after the --moe-expert-capacity-factor is set. |
| --moe-token-drop-policy | The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped. |
| --moe-layer-recompute | Enable activation checkpointing for moe_layer, should be used when memory is not sufficient. |
| --moe-router-fp32 | Explicit casting of the router input to fp32 may be used to improve model accuracy and softmax numerical stablility. |
| --moe-shared-expert-intermediate-size | Set shared expert total ffn hidden size. It should be equal to `num_shared_experts * ffn_size_of_each_shared_expert` if there are multiple shared experts. None means no shared expert. |
| --moe-shared-expert-overlap | (Experimental, may changed) If this is set, the communications/computations in the shared experts and the dispatcher will overlap (The `alltoall` dispatcher is needed.) Otherwise, the shared expert runs after the routed experts. |
| --moe-use-upcycling | Load the dense model checkpoint, convert it into an MoE model at runtime and start training. The converted model will be saved to the path specified by `--save` before training begins. Upcycling is implemented on the top of distributed checkpointing, so it supports parallel modes different from the dense model.|

### Fused HPU MoE Kernel

Fused Dynamic MoE Kernel encapsulates expert MLP computation and token scatter-gather operations in and between EP/TP groups into a single operation executed fully on HPU. It ensures precision without compromising performance due to excessive input token padding while maintaining the compiled HPU graph integrity - even with dynamic input shapes. This feature is enabled with a new MoE Module `IntelDynamicMLP` and is integrated to be used with a standard `MoELayer` formulation. The kernel supports BF16 and FP32 precision.

| Item | Description |
| --- | --- |
| --moe-dynamic-hpu | Enable fused dynamic kernel. Default is True. |
| --moe-permuted-weights | FFN weight format for inference frameworks compliance. Default is True. |
| --moe-fused-weights | SwiGLU packed weight format. Default is True. |

### Capacity Bins - performance optimization in dropless scenario

In addition, Capacity Bins functionality is enabled for Mixtral.
Capacity bins provide a performance-efficient solution for managing dynamic workloads in Mixture of Expert (MoE) layers.
Expert capacity values are limited to a predefined set of values defined by bins. The bins are automatically optimized
at specified intervals based on the usage frequencies of previous bins. Using capacity bins ensures no tokens are dropped
and reduces padding compared to using `--moe-expert-capacity-factor num_experts/topk`. To enable capacity bins,
set the `--moe-capacity-bins-num` parameter. The recommended configuration is to set `--moe-capacity-bins-num` to 10
while keeping other parameters at their default values. Currently, capacity bins functionality is supported only
with the `AllToAll` token dispatcher.

| Item | Description |
| --- | --- |
| --moe-capacity-bins-num | Size of fixed set of expert capacity values. Default is 0. |
| --moe-capacity-bins-base | Exponential base for initialization of capacity bins. Bins are generated with exponential growing bins width. Bins that are closer to the start are smaller and thus have less extra non-required capacity. Default is 1.5. |
| --moe-capacity-bins-alignment | Every capacity bin value (initialized or optimized) will be a multiple of this alignment. Default is 64. |
| --moe-capacity-bins-oprimize-interval | Steps interval for auto-optimization of MoE capacity bins. Default is 300. |
| --moe-capacity-bins-oprimize-max-group | Maximum group size of adjacent MoE gates that their capacity bins are optimized jointly. Default is 4. |
| --moe-capacity-bins-max-overhead-factor | Value of capacity bins overhead that will trigger bins optimization. Overhead is defined as relative additional capacity used with bins, compared to requested capacity value. Default is 0.0. |


## Usage

### Quick Start
To train a top-2 MoE model with 8 experts and auxiliary loss, include the following arguments:

```bash
--num-experts 8
--expert-model-parallel-size 8
--moe-grouped-gemm
--moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
--moe-router-topk 2
--moe-aux-loss-coeff 1e-2
--use-distributed-optimizer
--moe-token-dispatcher-type alltoall
```

To enable the token drop mechanism, such as GShard and SwitchTransformer, include the following arguments:

```bash
--moe-expert-capacity-factor 1.0
--moe-pad-expert-input-to-capacity # Optional
```

Instead of a fixed uniform capacity, use a discrete set of capacity values that will be asinged automatically to each expert per layer, minimizing redundant padding:

```bash
--moe-capacity-bins-num 10
--moe-pad-expert-input-to-capacity # Required
```

To run the model on Gaudi, enable fused kernel for the best performance:

```bash
--num-experts 8
--moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
--moe-router-topk 2
--moe-dynamic-hpu
```

The following figure illustrates different dropping strategies in MCore:
<!-- This image is uncommented for now as Sphinx cannot resolve this path. Sphinx imports this markdown file, and from the imported location this relative path does not exist anymore. Ideally, this markdown should not live here but rather in the `docs/` directory that Sphinx uses. -->
<!-- ![Token Droppling Strategies](../../../../docs/source/images/moe/token_drop.png) -->

1. The default dropless strategy will not drop or pad any token.
2. By setting `--moe-expert-capacity-factor`, the tokens exceed the capacity of expert will be dropped based on their selected probabilities. 
   The dropping is performed before the token exchange operation between EP ranks when EP > 1. 
   The formula of capacity is `capacity = num_tokens_per_rank * topk * capacity_factor / num_experts`.
3. By setting `--moe-pad-expert-input-to-capacity`, the experts with tokens less than capacity will be padded to the capacity.

### Fine-tuning Mixtral Models
Megatron-Core has full support for Mixtral MoE models, and we provide the checkpoint converter for Mixtral models from huggingface format to MCore format. 
<!-- See more details in the [mixtral example](../../../../examples/mixtral/README.md). -->


### Distributed Checkpointing
MCore v0.7 introduced fully parallel and asynchronous saving capabilities to distributed checkpointing, 
which addresses the issues of low efficiency in the traditional checkpoint saving methods. 
It also solved the problem of incompatibility between checkpoints of different parallel mappings in the traditional format.
With the new distributed checkpointing solution, MCore can achieve flexible parallelism configurations by saving and loading the unified format checkpoints.
Compared to native PyTorch solution, MCore achieves up to 50x reduction in checkpointing overhead.

From MCore v0.8, MoE supports Distributed Checkpointing, which means users can save and load with any combination of parallelism and it is currently available, including expert parallel.
1. Loading weight and distributed optimizer states with TPxCPxEPxPP resharding with SequentialMLP is supported in version 0.8.
2. GroupedMLP weight resharding is supported in version 0.8.0 and optimizer state resharding is supported in version 0.10.0. Switching between GroupedMLP/SequentialMLP when loading and saving is partially supported.
3. TEGroupedMLP has fully support on distributed checkpointing and is fully exchangable with SequentialMLP in version 0.9.0.
4. Optimizer state resharding cannot do across EP=1 with EP>1 due to the different optimizer type.

Usage
- `--ckpt-format torch_dist` The main argument, it will attempt to save and load using distributed checkpointing.
- `--auto-detect-ckpt-format` With this, it can load both distributed checkpointing and legacy checkpointing.

Checkpoint compatibility across SequentialMLP, GroupedMLP, and TEGroupedMLP:
```text
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐     
    │   GroupedMLP  │          │ SequentialMLP │          │ TEGroupedMLP  │     
    │               │          │               │          │               │     
    │               │          │               │          │               │     
    │ ┌───────────┐ │          │ ┌───────────┐ │          │ ┌───────────┐ │     
    │ │legacy ckpt│ │          │ │legacy ckpt│ │          │ │legacy ckpt│ │     
    │ └─────┬─────┘ │          │ └─────┬─────┘ │          │ └─────┬─────┘ │     
    │       ▼       │          │       ▼       │          │       ▼       │     
    │  ┌─────────┐  │          │  ┌─────────┐  │          │  ┌─────────┐  │     
    │  │dist ckpt│  │          │  │dist ckpt│  │          │  │dist ckpt│  │     
┌──►│  │ weight  │  │◄────────►│  │ weight  │  │◄────────►│  │ weight  │  │◄──┐ 
│   │  └─────────┘  │          │  └─────────┘  │          │  └─────────┘  │   │ 
└───┼───────────────┼──────────┼───────────────┼──────────┼───────────────┼───┘ 
    │┌─────────────┐│          │┌─────────────┐│          │┌─────────────┐│     
    ││  dist ckpt  ││          ││  dist ckpt  ││          ││  dist ckpt  ││     
    ││optim states ││          ││optim states ││◄────────►││optim states ││     
    │└─────────────┘│          │└─────────────┘│          │└─────────────┘│     
    └───────────────┘          └───────────────┘          └───────────────┘     
```

Best practices for distributed checkpointing:
1. Convert a legacy checkpoint to a distributed checkpoint. To achieve this, we can add both `--ckpt-format torch_dist --auto-detect-ckpt-format`, then it will load the legacy one and save as the distributed checkpoint format later when the training progress tries to save checkpoints.
2. Convert checkpoint of the legacy GroupedMLP to TEGroupedMLP. This is only supported for the weight parts. To achieve this, we can use the above method to convert the legacy checkpoint to a distributed checkpoint of the legacy GroupedMLP. After updating the libraries and using TEGroupedMLP, we can directly load the previously saved checkpoint by adding argument `--no-load-optim`.

### Shared Experts
MCore v0.9 introduced the shared expert feature. We can enable this feature by setting suitable `--moe-shared-expert-intermediate-size`.

The parallelism patterns of the shared experts follow the settings of the dense part, i.e., the attention module. The shared experts are not distributed but replicated in EP ranks.

We also have an experimental feature that tries to overlap the communications and computations in the shared experts and the dispatcher.
We can set `--moe-shared-expert-overlap` and use `alltoall` dispatcher to enable it.
The overlapping relies on the envirionment setting `CUDA_DEVICE_MAX_CONNECTIONS=1`.
The `AllGather` and `ReduceScatter` communications in the shared experts are overlapped with `permute`/`unpermute` in the dispatcher.
The `MLP` computation part in the shared experts are overlapped with the `AlltoAll` communications in the dispatcher.
Both the forward and the backward pass can overlap. But to get the overlapping in the backward pass, the PyTorch version should `>= 2.2.0`.

### Upcycling
Use `--moe-use-upcycling` to enable upcycling, which loads the dense model from the `--load` directory, converts it to an MoE model at runtime, and starts training. The converted model is saved to the `--save` path before training begins. Upcycling is built on distributed checkpointing, supporting parallel modes different from existing dense checkpoints, such as arbitrary expert parallelism during upcycling.

We currently only support the default upcycling strategy, which duplicates the existing MLP to multiple experts, with each expert starting from a copy of the MLP. In the future, we will support more state-of-the-art upcycling strategies, such as Granular upcycling from [our recent research work](https://arxiv.org/abs/2410.07524).

Note: The MoE model structure is defined through script arguments. All MoE-related arguments (such as `--num-experts`) can be customized; however, other model structure arguments must be consistent with those of the dense model.

## MoE training example:
* For the HPU training example, follow the instructions in [mixtral example](../../../../examples/mixtral/README.md).
* For the GPU training example, see below:
<details>
<summary>Click here. </summary>

```bash
#!/bin/bash

# Runs Mixtral 8x7B model on 32 H100/A100 GPUs
# The Dropless MoE suffers from an imbalanced token distribution at the early stage of training (the first few hundred iterations), which may lead to poor performance and out-of-memory (OOM) issues.
# To check the performance of a Dropless MoE model, we should run the model for at least 500 iterations or resume from trained checkpoints.

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1
TOKENIZER_MODEL=$2
DATA_PATH=$3

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 8
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --num-layers-per-virtual-pipeline-stage 8
    --sequence-parallel
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```
</details>

# Performance Best Practice

### Tuning Guide of Parallel Mappings

To find a good parallel mapping that help you achieve a high throughput of a new model, there are some general rule that could help. Here is an overview of properties in different aspects for each parallel strategy.

| Parallel Strategy | Peak Activation Memory          | Weight Memory  | Optimizer states                  | Communication (Per-Layer) |
|:-----------------:|:-------------------------------:|:--------------:|:---------------------------------:|:-------------------------:|
| TP                | 1/N (with SP on)                | 1/N            | 1/N                               |        High               |
| EP                | 1                               | 1/N in MoELayer| 1/N                               |       Medium              |
| PP                | 1 (>1 with virtual pipeline)    | 1/N            | 1/N                               |       Medium              |
| CP                | 1/N                             | 1              | 1/N (with distributed optimizer)  |       Medium              |
| DP                | 1                               | 1              | 1/N (with distributed optimizer)  |        Low                |

For a specific model, the best parallel mapping varies based on the model architecture, trained sequence length and the hardware platform.
Here we provide some general rules to get better performance:
1. Keep the model parallism size as small as possible. 
    - For the large language models, model parallism is often required to prevent OOM, but it will bring communication overhead and hurt performance. 
    - With distributed optimizer, master weights and optimizer states will be sharded across all DP ranks with slight communication overhead.
    So try to reduce the model parallism size and increase data parallism size when there are lots of free GPU memory during training.
2. Ensure the EPxTP communication winthin the NVLink domain.
    - Communications of EP and TP should remain within the NVLink domain as much as possible, as both are communication-intensive.
    - If the model is too large and requires scaling across multiple nodes, consider PP before TP and EP. See item 3 for details.
3. Use Pipeline Parallelism to scale the model further.
    - Enable Virtual Pipeline Parallelism(VPP) to reduce pp bubbles when PP_size >= 2 by setting `num_layers_per_virtual_pipeline_stage`.
    - VPP_size tuning: the legal values of vpp_size are all common divisors of num_layers/pp_size, E.g., num_layers=24, pp_size=4, then we can pick vpp_size from {1, 2, 3, 6}. The larger the vpp_size, the lower the pipeline bubbles, while the larger number of P2P communications between each PP stages. Empirically a value in the middle often gives the best trade-off. `VPP_size=num_layers / PP_size / num_layers_per_virtual_pipeline_stage`
4. Prefer EP over TP for the expert layer when possible:
    - TP saves more memory than EP, but EP can achieve better GEMM efficiency and less communication overhead than TP.
    - If EP size increased to the number of expert, the local token permutation/un-permutation for experts computation are omitted.
    - Simplify the computation graph of MoE layers, more convenient for performing potential comm-computation overlapping.
    - In practice, EP8TP1 is better than EP4TP2 for 8x7B.
5. Enable Context Parallelism for long context training.
    - The efficiency of CP largely depends on whether its communication can be overlapped with computation. 
    - Emperically, use CP when sequence length >= 8K.
6. To measure roofline model performance use `--moe-expert-capacity-factor 1.0 --moe-pad-expert-input-to-capacity` which simulates perfect uniform token distribution. Other training methods, such as capacity bins, may converge to this performance within a few epochs by optimizing aux/z-loss. Please keep in mind, that the uniform distribution is not guaranteed and depends on used dataset characteristics.

### MoE Parallel Folding

MoE Parallel Folding separates the MoE related parallel groups from Dense groups.
1. Traditional MoE parallel groups are entangled with dense by using a 5-dimension parallel group generator with default order `tp-cp-ep-dp-pp`. The EP group in MoE is a sub-group of DP in Attention.
2. With MoE Parallel Fodling, we use a parallel group generator with `tp-cp-dp-pp` for Attention, and another with `tp-ep-dp-pp` for MoE. The EPxTP group in MoE is a sub-group of DPxCPxTP in Attention.

By setting `--expert-tensor-parallel-size`, we can set MoE-specific TP size.

#### Advantages of MoE Parallel Folding
1. The CP and EP group are folded together by defualt, such that:
    1. It reduces the minimal required GPUs to turn on both CP and EP. For example, the traditional way with (CP=8, EP=8) needs at least 64 GPUs, for now it only requires 8 GPUs.
    2. The CP and EP communication can be both put in the NVLink domain.
2. We can set different TP sizes for Attention and MoE part.
    1. For MoE, EP is often more efficient than TP. But in the traditional way, only using EP can get OOM for most models.
    2. With MoE parallel folding, we can turn on TP for Attention part and setting TP=1 for MoE models, which often gets better MFU.

### End-to-End Training Practice
**Use the latest NVIDIA PyTorch or NeMo Docker Image**
- [NGC PyTorch Image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [NGC NeMo Image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)

**Token Dispatcher Choices**
- Token Dispatcher sends tokens to the designated expert, involves tensor rearangement and communications.
- Dispatcher `allgather` is the default option. It achieves better performance and efficiency when only tensor parallelism is used or when the Top-k value is very large.
- Dispatcher `alltoall` is recommended if expert parallelism is applied.
- Dispatcher `alltoall_seq` is the original implementation of `alltoall` and is retained for potential compatibility risk.

**Enable Communication Overlap**
- Enable `--overlap-param-gather` and `--overlap-grad-reduce` with distributed optimizer.
- Enable `--tp-comm-overlap` when TP>1.
- Enable p2p comm overlap when PP > 1 by setting `num_layers_per_virtual_pipeline_stage`.

**Enable GroupedGEMM when num_local_experts>1 with `--moe-grouped-gemm`**
- GroupedGEMM has higher efficiency than vanilla sequential GEMMs for each expert.
- Recommend to use the TE version of Grouped GEMM (by upgrading to MCore v0.8 and TE v1.9), which support Gradient Accumulation Fusion and FP8 Training.

**OOM Caused by Token Distribution Imbalance when Training From Scratch**  
MoE suffers from a severe load imbalance issue when the router is under-trained, leading to the model easily running out of memory (OOM), which typically occurs in the first 100~300 steps when training from scratch. 
Therefore, there are two recommended ways during the first 200 steps to avoid the OOM problem, which can be removed after the token distribution is more stable:
1. Increase the `expert-tensor-parallel-size` and decrease `expert-model-parallel-size` to replace EP with TP in MoELayer, this can prevent the load imbalancing between EP ranks. Since current ETP implementation has some memeory overhead, you can further enable activation recomputation only for MoE Layer by adding `--moe-layer-recompute`.
2. Setting capacity factor to a relatively small number like 1.0 by adding `--moe-token-capacity-factor 1.0`.

### Reference Best Parallel Mapping

Here are the reference parallel mappings of MCore v0.8 for Mixtral 8x7B and 8x22B models:
|        Model            | Vocab Size| Dispatcher | Precision | #GPUs | SEQ LEN | TP | EP | PP | VP | MBS | GBS |
|:-----------------------:|:---------:|:----------:|:---------:|:-----:|:-------:|:--:|:--:|:--:|:--:|:---:|:---:|
| Mixtral 8x7B(Dropless)  |   32K     | All-to-All | BF16      | 64    | 4096    | 1  | 8  | 4  | 8  | 1   | 256 |
| Mixtral 8x22B(Dropless) |   32K     | All-to-All | BF16      | 128   | 4096    | 4  | 2  | 8  | 7  | 1   | 256 |

Detailed Benchmark Information:  
Server:
- 8xH100 80GB HBM3 
- NVLink 4th Generation
- InfiniBand 8x400 Gbit/s

Docker Image:
- PyTorch 24.09 with TransformerEngine v1.11
