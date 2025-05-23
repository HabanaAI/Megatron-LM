#!/bin/bash

# © 2024-2025 Intel Corporation

# -------------------------------------------
# Mixtral model
# Paper: https://arxiv.org/pdf/2401.04088.pdf
# -------------------------------------------

set -ex

# ----------------------------
# User configurable parameters

LAUNCHER_TYPE=${HL_LAUNCHER_TYPE:-mpirun}

DATA_DIR=${HL_DATA_DIR_ROOT:-/data/datasets/red_pajama}
CACHE_PATH=${HL_CACHE_PATH:-}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-medium}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-GPTSentencePieceTokenizer}
TOKENIZER_MODEL=${HL_TOKENIZER_MODEL:-}

NUM_NODES=${HL_NUM_NODES:-1}
DEVICES_PER_NODE=${HL_DEVICES_PER_NODE:-8}
DP=${HL_DP:-1}
TP=${HL_TP:-8}
PP=${HL_PP:-1}
CP=${HL_CP:-1}
MOE_EP=${HL_MOE_EP:-}
MOE_TP=${HL_MOE_TP:-}
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-1}
HOSTSFILE=${HL_HOSTSFILE:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH:-}
DIST_OPTIMIZER=${HL_DIST_OPTIMIZER:-0}

MIXTRAL_MODEL=${HL_MIXTRAL_MODEL:-8x7b}
MICRO_BATCH=${HL_MICRO_BATCH:-1}
OPTIMIZER=${HL_OPTIMIZER:-fusedadamw}
DROPOUT=${HL_DROPOUT:-0.0}
LR_WARMUP_ITERS=${HL_LR_WARMUP_ITERS:-2000}
USE_CPU_INIT=${HL_USE_CPU_INIT:-0}

# General MoE Arguments
MOE_EXPERT_CAPACITY_FACTOR=${HL_MOE_EXPERT_CAPACITY_FACTOR:-}
MOE_TOKEN_DROP_POLICY=${HL_MOE_TOKEN_DROP_POLICY:-probs}  # either probs or position
MOE_PAD_EXPERT_INPUT_TO_CAPACITY=${HL_MOE_PAD_EXPERT_INPUT_TO_CAPACITY:-0}
MOE_TOPK=${HL_MOE_TOPK:-2}
MOE_ZLOSS_COEFF=${HL_MOE_ZLOSS_COEFF:-}
MOE_AUX_LOSS_COEFF=${HL_MOE_AUX_LOSS_COEFF:-2e-2}
MOE_TOKEN_DISTRIBUTION_LOGGING=${HL_MOE_TOKEN_DISTRIBUTION_LOGGING:-0}
MOE_TOKEN_DISTRIBUTION_LOGGING_INTERVAL=${HL_MOE_TOKEN_DISTRIBUTION_LOGGING_INTERVAL:-50}
MOE_ROUTER_PRE_SOFTMAX=${HL_MOE_ROUTER_PRE_SOFTMAX:-1}
MOE_ROUTER_FP32=${HL_MOE_ROUTER_FP32:-1}
MOE_TOKEN_DISPATCHER_TYPE=${HL_TOKEN_DISPATCHER_TYPE:-alltoall}
MOE_LOAD_BALANCING_TYPE=${HL_MOE_LOAD_BALANCING_TYPE:-aux_loss}

# Dynamic MOE with IntelDynamicMLP
MOE_DYNAMIC=${HL_MOE_DYNAMIC:-0} # Default for HPU
MOE_PERMUTED_WEIGHTS=${HL_MOE_PERMUTED_WEIGHTS:-1}
MOE_FUSED_WEIGHTS=${HL_MOE_FUSED_WEIGHTS:-1}

# Capacity Bins with SequentialMLP
MOE_NUM_CAPACITY_BINS=${HL_MOE_NUM_CAPACITY_BINS:-10} # 10 is default for 32k Sequence Length
MOE_CAPACITY_BINS_EXP_BASE=${HL_MOE_CAPACITY_BINS_EXP_BASE:-1.5}
MOE_CAPACITY_BINS_ALIGNMENT=${HL_MOE_CAPACITY_BINS_ALIGNMENT:-64}
MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL=${HL_MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL:-300}
MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP=${HL_MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP:-4}
MOE_CAPACITY_BINS_MAX_OVERHEAD_FACTOR=${HL_MOE_CAPACITY_BINS_MAX_OVERHEAD_FACTOR:-0.0}

USE_LAZY_MODE=${HL_USE_LAZY_MODE:-1}
USE_TORCH_COMPILE=${HL_USE_TORCH_COMPILE:-0}
USE_TORCH_COMPILED_AUTOGRAD=${HL_USE_TORCH_COMPILED_AUTOGRAD:-0}
TORCH_COMPILE_DISABLE=${HL_TORCH_COMPILE_DISABLE:-0}

USE_FUSED_SDPA=${HL_USE_FUSED_SDPA:-1}
USE_FUSED_SDPA_WITH_RECOMPUTE=${HL_USE_FUSED_SDPA_WITH_RECOMPUTE:-0}
USE_FUSED_RMSNORM=${HL_USE_FUSED_RMSNORM:-1}
USE_FAST_SOFTMAX=${HL_USE_FAST_SOFTMAX:-1}
USE_FLASH_ATTN=${HL_USE_FLASH_ATTN:-0}

CKP_ACT=${HL_CKP_ACT:-0}
RECOMPUTE_NUM_LAYERS=${HL_RECOMPUTE_NUM_LAYERS:-1}

CHECKPOINT_SAVE=${HL_SAVE:-0}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
CKPT_FORMAT=${HL_CKPT_FORMAT:-torch} # torch, torch_dist and zarr
OVERRIDE_OPT_PARAM_SCHEDULER=${HL_OVERRIDE_OPT_PARAM_SCHEDULER:-0}
USE_CKPT_OPT_PARAM_SCHEDULER=${HL_USE_CKPT_OPT_PARAM_SCHEDULER:-0}
NO_LOAD_STRICT=${HL_NO_LOAD_STRICT:-0}
NO_LOAD_OPTIM=${HL_NO_LOAD_OPTIM:-0}
NO_LOAD_RNG=${HL_NO_LOAD_RNG:-0}
LOAD_DIR=${HL_LOAD_DIR:-}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
TRAIN_ITERS=${HL_TRAIN_ITERS:-250000}
EVAL_ITERS=${HL_EVAL_ITERS:-100}
EVAL_INTERVAL=${HL_EVAL_INTERVAL:-1000}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
SAVE_DISTRIB_OPTIMIZER_METHOD=${HL_SAVE_DISTRIB_OPTIMIZER_METHOD:-parallel_multi_node}
LOAD_DISTRIB_OPTIMIZER_METHOD=${HL_LOAD_DISTRIB_OPTIMIZER_METHOD:-serial_per_node}

PROFILE_TYPE=${HL_PROFILE_TYPE:-}  # provide either of pt, pt-full
PROFILE_STEP_START=${HL_PROFILE_STEP_START:-3}
PROFILE_STEP_END=${HL_PROFILE_STEP_END:-4}
PROFILE_RANKS=${HL_PROFILE_RANKS:-"0"} # "0 1 4 7"

FP8=${HL_FP8:-0}
BF16=${HL_BF16:-1}
TRANSFORMER_IMPL=${HL_TRANSFORMER_IMPL:-transformer_engine}
FP8_FORMAT=${HL_FP8_FORMAT:-hybrid} # hybrid or e5m2
FP8_MARGIN=${HL_FP8_MARGIN:-0}
FP8_AMAX_COMPUTE_ALGO=${HL_FP8_AMAX_COMPUTE_ALGO:-max} # max or most_recent
FP8_COVERAGE=${HL_FP8_COVERAGE:-"mlp_row_parallel=False"}
MOE_GROUPED_GEMM=${HL_MOE_GROUPED_GEMM:-0}

NUM_WORKERS=${HL_NUM_WORKERS:-0}
ENV_FLAGS=${HL_ENV_FLAGS:-} #"a=1,b=2,c=3"

# Following configuration are dependant on specific model definitions, but can
# be overridden for debug purposes
# - HL_MOE_NUM_EXPERTS
# - HL_NUM_LAYERS
# - HL_SEQ_LEN
# - HL_GBS
# - HL_TRAIN_ITERS

# --------------------
# Mixtral architecture

if [ $MIXTRAL_MODEL == "8x7b" ]; then
    # Mixtral-8x7B model architecture
    MOE_NUM_EXPERTS=${HL_MOE_NUM_EXPERTS:-8}
    NUM_LAYERS=${HL_NUM_LAYERS:-32}
    SEQ_LEN=${HL_SEQ_LEN:-32768}
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=14336
    NUM_HEADS=32
    NUM_QUERY_GROUPS=8
    LR=3e-4
    MIN_LR=3e-6  # using 0.01 of max-lr (DeepSpeed-MoE https://arxiv.org/pdf/2201.05596.pdf section 3.2)
else
    echo "Unsupported HL_MIXTRAL_MODEL=$MIXTRAL_MODEL"
    exit 1
fi

# ------------------------------
# Verify supported configuration

NUM_DEVICES=$(($DP * $TP * $PP * $CP))
NUM_DEVICES_GOT=$(($DEVICES_PER_NODE * $NUM_NODES))
if [ $NUM_DEVICES -ne $NUM_DEVICES_GOT ]; then
    echo "Bad devices configuration. DPxTPxPPxCP=${NUM_DEVICES} != N_NODES*N_DEVICES_PER_NODE=${NUM_DEVICES_GOT}"
    exit 1
fi

if [ $(( NUM_LAYERS % PP )) -ne 0 ]; then
    echo 'HL_NUM_LAYERS must be divisible by PP'
    exit 1
fi

if [[ $TP -gt 1 && $SEQ_PARALLEL -eq 0 ]]; then
    SEQ_PARALLEL=1
    echo "SEQ_PARALLEL set to 1 because TP > 1"
fi

if [ -z "${MOE_EP}" ]; then
  if [[ $MOE_NUM_EXPERTS -gt $NUM_DEVICES ]]; then
      MOE_EP=${NUM_DEVICES}
  else
      MOE_EP=${MOE_NUM_EXPERTS}
  fi
fi
# Check valid expert capacity configuration
if [[ $MOE_DYNAMIC -eq 1 && $MOE_NUM_CAPACITY_BINS -gt 0 ]]; then
    echo "MOE_DYNAMIC=1 and MOE_NUM_CAPACITY_BINS > 0 are mutually exclusive."
    exit 1
fi

if [[ $MOE_DYNAMIC -eq 1 && -n "$MOE_EXPERT_CAPACITY_FACTOR" ]]; then
    echo "MOE_DYNAMIC=1 and MOE_EXPERT_CAPACITY_FACTOR are mutually exclusive."
    exit 1
fi

if [[ $MOE_NUM_CAPACITY_BINS -gt 0 && -n "$MOE_EXPERT_CAPACITY_FACTOR" ]]; then
    echo "MOE_NUM_CAPACITY_BINS > 0 and MOE_EXPERT_CAPACITY_FACTOR are mutually exclusive."
    exit 1
fi

if [[ $MOE_DYNAMIC -eq 1 && $FP8 -eq 1 ]]; then
        echo "Fused HPU kernel for MoE 'IntelDynamicMLP' supports FP32 and BF16 precision, got HL_FP8=1"
        exit 1
fi
if [[ $MOE_DYNAMIC -eq 1 ]]; then
    MOE_PAD_EXPERT_INPUT_TO_CAPACITY=""
    MOE_TOKEN_DISPATCHER_TYPE=""
fi

if [[ $MOE_NUM_CAPACITY_BINS -gt 0 ]]; then
    MOE_PAD_EXPERT_INPUT_TO_CAPACITY=1
    MOE_TOKEN_DISPATCHER_TYPE="alltoall"
fi

if [[ $MOE_EXPERT_CAPACITY_FACTOR -gt 0 ]]; then
    MOE_TOKEN_DISPATCHER_TYPE="alltoall"
fi

if [[ "${USE_FUSED_SDPA}" = "1" || "${USE_FUSED_SDPA_WITH_RECOMPUTE}" = "1" ]]; then
    CMD="${CMD} --no-create-attention-mask-in-dataloader"
fi

echo "Using Num Experts=${MOE_NUM_EXPERTS} with MoE EP=${MOE_EP}"

# ------------------------------------------------------------------------
# Training configuration: Mixtral paper has no details on training regime.
# Therefore using LLAMA1 regime.
# So, assuming LLAMA1 regime with few exceptions:
# - seq_len = 32768
# - smaller min_lr

TOKENS_IN_BATCH=$((2 ** 22))  # 4M tokens
CALCULATED_GBS=$(($TOKENS_IN_BATCH / $SEQ_LEN))
GLOBAL_BATCH=${HL_GBS:-$CALCULATED_GBS}
TOTAL_TOKENS=$((250000 * $TOKENS_IN_BATCH))  # ~1T tokens

# -----
# PATHs

if [[ -z "$MEGATRON_LM_ROOT" ]]; then
    MEGATRON_LM_ROOT=$(realpath $(dirname $0)/../../)
fi

if [[ -z "$TOKENIZER_MODEL" ]]; then
    TOKENIZER_MODEL="${DATA_DIR}/tokenizer.model"
fi

DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}

RUNTIME=`date +"%Y%m%d_%H%M"`
# output paths
if [ -z "$OUTPUT_DIR" ]; then
    # Experiment name
    if [ -z "$EXP_NAME" ]; then
        EXP_NAME="default"
    fi
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/mixtral_${MIXTRAL_MODEL}/${EXP_NAME}_ckpact${CKP_ACT}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_ffn${FFN_HIDDEN_SIZE}_moe_exp${MOE_NUM_EXPERTS}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_E${MOE_EP}_devices${NUM_DEVICES}_${RUNTIME}
fi

if [ -z "$CHECKPOINTS_DIR" ]; then
    CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
fi

if [ -z "$LOAD_DIR" ]; then
    LOAD_DIR=$CHECKPOINTS_DIR
fi

if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINTS_DIR}
mkdir -p ${TENSORBOARD_DIR}

# --------------
# Create command

# configure multi-node/multi-hpu command
if [ "$NUM_NODES" -ne "1" -a -z "$HOSTSFILE" ]; then
    HOSTSFILE=${MEGATRON_LM_ROOT}/examples/hostsfile
    if [ -f $HOSTSFILE ]; then
        cat /dev/null > ${HOSTSFILE}
    fi
    etc_mpi_hostfile="/etc/mpi/hostfile"
    if [ ! -f $etc_mpi_hostfile ]; then
        echo "$etc_mpi_hostfile not available, set HL_HOSTSFILE"
        exit 1
    fi
    cat $etc_mpi_hostfile | xargs -I{} echo {} slots=8 >> ${HOSTSFILE}
fi

PT_HPU_GPU_MIGRATION=1
CMD=""

if [ "$LAUNCHER_TYPE" = "mpirun" ]; then
    CMD="$CMD mpirun"
    CMD="$CMD --allow-run-as-root"
    CMD="$CMD -n ${NUM_DEVICES}"
    [[ -n "$HL_PE" ]] && __MAP_BY="socket:PE=${HL_PE}"
    [[ -n "$HL_PE" ]] && [[ -n "${HL_PPR}" ]] && __MAP_BY="ppr:${HL_PPR}:socket:PE=${HL_PE}"
    if [[ -n "${__MAP_BY}" ]]; then
        CMD="${CMD} --bind-to core --rank-by core --report-bindings --map-by ${__MAP_BY}"
    else
        CMD="${CMD} --bind-to none"
    fi
    CMD="$CMD -x PT_HPU_GPU_MIGRATION=$PT_HPU_GPU_MIGRATION"
    CMD="${CMD} -x PT_HPU_LAZY_MODE=${USE_LAZY_MODE}"
    if [ "${TORCH_COMPILE_DISABLE}" = "1" ]; then
        CMD="${CMD} -x TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE}"
    fi
    IFS=',' read -ra ENV_FLAGS_ARR <<< "$ENV_FLAGS"
    for ENV_FLAG in "${ENV_FLAGS_ARR[@]}"; do
        CMD="${CMD} -x ${ENV_FLAG}"
    done
    if [ "$NUM_NODES" -ne "1" ]; then
        CMD="$CMD -hostfile $HOSTSFILE"
        CMD="$CMD -x MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p)"
    else
        CMD="$CMD -x MASTER_ADDR=localhost"
    fi
    CMD="$CMD -x MASTER_PORT=12345"
elif [ "$LAUNCHER_TYPE" = "torchrun" ]; then
    if [ "$NUM_NODES" -ne "1" ]; then
        echo "NUM_NODES greater than 1 not supported by torchrun"
        exit 1
    fi
    export PT_HPU_GPU_MIGRATION=$PT_HPU_GPU_MIGRATION
    export PT_HPU_LAZY_MODE=${USE_LAZY_MODE}
    if [ "${TORCH_COMPILE_DISABLE}" = "1" ]; then
        export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE}
    fi
    IFS=',' read -ra ENV_FLAGS_ARR <<< "$ENV_FLAGS"
    for ENV_FLAG in "${ENV_FLAGS_ARR[@]}"; do
        export "${ENV_FLAG?}"
    done
    CMD="$CMD torchrun"
    CMD="$CMD --nnodes $NUM_NODES"
    CMD="$CMD --nproc-per-node $DEVICES_PER_NODE"
    CMD="$CMD --no-python"
    CMD="$CMD --node-rank 0"
    CMD="$CMD --master-addr localhost"
    CMD="$CMD --master-port 12345"
else
    echo "Unsupported launcher type = $LAUNCHER_TYPE"
    exit 2
fi

# training script command
MLM_SCRIPT="${MEGATRON_LM_ROOT}/pretrain_gpt.py"

CMD="${CMD} \
    python ${MLM_SCRIPT} \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --use-torch-compile ${USE_TORCH_COMPILE} \
    --use-torch-compiled-autograd ${USE_TORCH_COMPILED_AUTOGRAD} \
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS} \
    --attention-dropout ${DROPOUT} \
    --hidden-dropout ${DROPOUT} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --use-fused-sdpa ${USE_FUSED_SDPA} \
    --use-fused-sdpa-with-recompute ${USE_FUSED_SDPA_WITH_RECOMPUTE} \
    --use-fused-rmsnorm ${USE_FUSED_RMSNORM} \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --rotary-base 1000000 \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --weight-decay 0.1 \
    --train-iters ${TRAIN_ITERS} \
    --clip-grad 1.0 \
    --optimizer ${OPTIMIZER} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --expert-model-parallel-size ${MOE_EP} \
    --log-interval ${LOG_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --exit-interval ${EXIT_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --load ${LOAD_DIR} \
    --data-path ${DATA_PATH} \
    --log-throughput \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --num-workers ${NUM_WORKERS} \
    --use-fast-softmax ${USE_FAST_SOFTMAX} \
    --distributed-timeout-minutes 60 \
    "

# -------------
# Precision fp32->bf16
if [ $BF16 -eq 1 ]; then
    CMD="${CMD} --bf16"
fi

# -------------
# GroupedMLP
if [ "$MOE_GROUPED_GEMM" -eq 1 ] || [ "$FP8" -eq 1 ]; then
    CMD="${CMD} --moe-grouped-gemm"
fi

# -------------
# Flash Attention

if [ $USE_FLASH_ATTN -eq 1 ]; then
    CMD="${CMD} --use-flash-attn"
fi

# -------------
# MoE arguments
# -------------

# MoE Dynamic Kernel arguments

if [ $MOE_DYNAMIC -eq 1 ]; then
    CMD="${CMD} --moe-dynamic-hpu"
    if [ $MOE_PERMUTED_WEIGHTS -eq 1 ]; then
        CMD="${CMD} --moe-permuted-weights"
    fi
    if [ $MOE_FUSED_WEIGHTS -eq 1 ]; then
        CMD="${CMD} --moe-fused-weights"
    fi
fi

# MoE Capacity Bins arguments

if [[ $MOE_NUM_CAPACITY_BINS -gt 0 ]]; then
    CMD="${CMD} --moe-capacity-bins-num ${MOE_NUM_CAPACITY_BINS}"
    CMD="${CMD} --moe-capacity-bins-exp-base ${MOE_CAPACITY_BINS_EXP_BASE}"
    CMD="${CMD} --moe-capacity-bins-alignment ${MOE_CAPACITY_BINS_ALIGNMENT}"
    CMD="${CMD} --moe-capacity-bins-optimize-interval ${MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL}"
    CMD="${CMD} --moe-capacity-bins-optimize-max-group ${MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP}"
    CMD="${CMD} --moe-capacity-bins-max-overhead-factor ${MOE_CAPACITY_BINS_MAX_OVERHEAD_FACTOR}"

fi

CMD="${CMD} --num-experts ${MOE_NUM_EXPERTS}"
CMD="${CMD} --moe-router-topk ${MOE_TOPK}"
CMD="${CMD} --moe-router-load-balancing-type ${MOE_LOAD_BALANCING_TYPE}"
CMD="${CMD} --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF}"
if [ -n "$MOE_TOKEN_DISPATCHER_TYPE" ]; then
    CMD="${CMD} --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE}"
fi
if [ -n "$MOE_ZLOSS_COEFF" ]; then
    CMD="${CMD} --moe-z-loss-coeff ${MOE_ZLOSS_COEFF}"
fi
if [ $MOE_TOKEN_DISTRIBUTION_LOGGING -eq 1 ]; then
    CMD="${CMD} --moe-token-distribution-logging"
    CMD="${CMD} --moe-token-distribution-logging-interval ${MOE_TOKEN_DISTRIBUTION_LOGGING_INTERVAL}"
fi
if [ $MOE_ROUTER_PRE_SOFTMAX -eq 1 ]; then
    CMD="${CMD} --moe-router-pre-softmax"
fi
if [ $MOE_ROUTER_FP32 -eq 1 ]; then
    CMD="${CMD} --moe-router-fp32"
fi
if [ ! -z "$MOE_EXPERT_CAPACITY_FACTOR" ]; then
    CMD="${CMD} --moe-expert-capacity-factor ${MOE_EXPERT_CAPACITY_FACTOR}"
    CMD="${CMD} --moe-token-drop-policy ${MOE_TOKEN_DROP_POLICY}"
fi
if [ $MOE_PAD_EXPERT_INPUT_TO_CAPACITY -eq 1 ]; then
    CMD="${CMD} --moe-pad-expert-input-to-capacity"
fi

# Additonal arguments
if [[ "${USE_FUSED_SDPA}" = "1" || "${USE_FUSED_SDPA_WITH_RECOMPUTE}" = "1" ]]; then
    CMD="${CMD} --no-create-attention-mask-in-dataloader"
fi

if [ $SEQ_PARALLEL -eq 1 ]; then
    CMD="${CMD} --sequence-parallel"
fi

if [ $MOE_TP -eq 1 ]; then
    CMD="${CMD} --expert-tensor-parallel-size ${TP}"
fi

if [ $CKP_ACT -eq 1 ]; then
    CMD="${CMD} --recompute-granularity full"
    CMD="${CMD} --recompute-method uniform"
    CMD="${CMD} --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [ $CKP_ACT -eq 2 ]; then
    CMD="${CMD} --recompute-granularity selective"
elif [ $CKP_ACT -eq 3 ]; then
    CMD="${CMD} --moe-layer-recompute"
fi

if [ $DIST_OPTIMIZER -eq 1 ]; then
    CMD="${CMD} --use-distributed-optimizer"

    if [ -n "$SAVE_DISTRIB_OPTIMIZER_METHOD" ]; then
        CMD="${CMD} --save-distrib-optimizer-method ${SAVE_DISTRIB_OPTIMIZER_METHOD}"
    fi
    if [ -n "$LOAD_DISTRIB_OPTIMIZER_METHOD" ]; then
        CMD="${CMD} --load-distrib-optimizer-method ${LOAD_DISTRIB_OPTIMIZER_METHOD}"
    fi
fi

if [ ! -z "$KILL_SWITCH_FILE" ]; then
    CMD="${CMD} --kill-switch-file ${KILL_SWITCH_FILE}"
fi

if [ ! -z "$PROFILE_TYPE" ]; then
    CMD="${CMD} --profile"
    CMD="${CMD} --use-pytorch-profiler"
    CMD="${CMD} --profile-type ${PROFILE_TYPE}"
    CMD="${CMD} --profile-step-start ${PROFILE_STEP_START}"
    CMD="${CMD} --profile-step-end ${PROFILE_STEP_END}"
    CMD="${CMD} --profile-ranks ${PROFILE_RANKS}"
fi

if [ $CHECKPOINT_SAVE -eq 1 ]; then
    CMD="${CMD} --save ${CHECKPOINTS_DIR}"
    CMD="${CMD} --save-interval ${SAVE_INTERVAL}"
    CMD="${CMD} --ckpt-format ${CKPT_FORMAT}"
fi

if [[ $OVERRIDE_OPT_PARAM_SCHEDULER -eq 1 && $USE_CKPT_OPT_PARAM_SCHEDULER -eq 1 ]]; then
    echo "Both OVERRIDE_OPT_PARAM_SCHEDULER and USE_CKPT_OPT_PARAM_SCHEDULER are set"
    exit 1
fi
if [ $OVERRIDE_OPT_PARAM_SCHEDULER -eq 1 ]; then
    CMD="${CMD} --override-opt_param-scheduler"
fi
if [ $USE_CKPT_OPT_PARAM_SCHEDULER -eq 1 ]; then
    CMD="${CMD} --use-checkpoint-opt_param-scheduler"
fi

if [ $NO_LOAD_STRICT -eq 1 ]; then
    CMD="${CMD} --no-load-strict"
fi
if [ $NO_LOAD_OPTIM -eq 1 ]; then
    CMD="${CMD} --no-load-optim"
fi
if [ $NO_LOAD_RNG -eq 1 ]; then
    CMD="${CMD} --no-load-rng"
fi

if [ "$TOKENIZER_TYPE" = "GPTSentencePieceTokenizer" ]; then
    CMD="${CMD} --tokenizer-type GPTSentencePieceTokenizer"
    CMD="${CMD} --tokenizer-model ${TOKENIZER_MODEL}"
elif [ "$TOKENIZER_TYPE" = "GPT2BPETokenizer" ]; then
    CMD="${CMD} --tokenizer-type GPT2BPETokenizer"
    CMD="${CMD} --vocab-file ${DATA_DIR}/gpt2-vocab.json"
    CMD="${CMD} --merge-file ${DATA_DIR}/gpt2-merges.txt"
else
    echo "incorrect HL_TOKENIZER_TYPE=${TOKENIZER_TYPE} is set"
    exit 1
fi

if [ ! -z "$CACHE_PATH" ]; then
    CMD="${CMD} --data-cache-path ${CACHE_PATH}"
fi

if [ $USE_CPU_INIT -eq 1 ]; then
    CMD="${CMD} --use-cpu-initialization"
fi

# fp8 args
if [[ "${TRANSFORMER_IMPL}" == "transformer_engine" && $FP8 -eq 1 ]]; then
    FP8_MEASURE_INTERVAL=${HL_FP8_MEASURE_INTERVAL:-$(( GLOBAL_BATCH / MICRO_BATCH / DP ))}
    FP8_AMAX_HISTORY_LEN=${HL_FP8_AMAX_HISTORY_LEN:-$(( GLOBAL_BATCH / MICRO_BATCH / DP ))}
    FP8_AMAX_REDUCE=${HL_FP8_AMAX_REDUCE:-1}

    CMD="${CMD} --fp8-interval ${FP8_MEASURE_INTERVAL}"
    CMD="${CMD} --fp8-margin ${FP8_MARGIN}"
    CMD="${CMD} --fp8-amax-compute-algo ${FP8_AMAX_COMPUTE_ALGO}"
    CMD="${CMD} --fp8-amax-history-len ${FP8_AMAX_HISTORY_LEN}"
    CMD="${CMD} --fp8-format ${FP8_FORMAT}"
    CMD="${CMD} --fp8-coverage ${FP8_COVERAGE}"

    if [[ "${FP8_AMAX_REDUCE}" -eq 1 ]]; then
        CMD="${CMD} --fp8-amax-reduce"
    fi
fi

# run script
${CMD}
