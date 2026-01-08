#!/bin/bash

# © 2024-2025 Intel Corporation

set -ex

MEGATRON_LM_ROOT=$(realpath "$(dirname "$0")"/../..)


# Distributed training variables
LAUNCHER_TYPE=${HL_LAUNCHER_TYPE:-mpirun}
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/dataset_idx/red_pajama_deepseek}
DATA_CACHE_DIR=${HL_DATA_CACHE_DIR:-}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-rp_wiki_24B_deepseek_tok_text_document}
TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-HuggingFaceTokenizer}
TOKENIZER_MODEL=${HL_TOKENIZER_MODEL:-}
TRANSFORMER_IMPL=${HL_TRANSFORMER_IMPL:-transformer_engine}

# Parallelism variables
NUM_NODES=${HL_NUM_NODES:-1}
SSH_PORT=${HL_SSH_PORT:-22}
HNIC=${HL_HNIC:-0}

# asserting that NUM_NODES is greater than 1 when HNIC is set to 1
if [ "${HNIC}" = "1" ] && [ "${NUM_NODES}" -le 1 ]; then
  echo "Exiting: Host NIC is enabled and NUM_NODES is not greater than 1"
  exit 1
fi
#Host NIC variables
if [[ "${HNIC}" -eq "1"  ]]; then
    HCCL_OVER_OFI=${HL_HCCL_OVER_OFI:-1}
    HCCL_GAUDI_DIRECT=${HL_HCCL_GAUDI_DIRECT:-1}
    FI_PROVIDER=${HL_FI_PROVIDER:-verbs}
fi

TP=${HL_TP:-2}
PP=${HL_PP:-2}
DP=${HL_DP:-2}
EP=${HL_EP:-2}
MICRO_BATCH_SIZE=${HL_MICRO_BATCH:-1}
# TODO implement gradually increasing batch size mode
GLOBAL_BATCH_SIZE=${HL_GBS:-3072} # 15360 after 469B tokens
SEQ_LEN=${HL_SEQ_LEN:-4096}
TRAIN_ITERS=${HL_TRAIN_ITERS:-500000} 
LR_DECAY_ITER=${HL_LR_DECAY_ITER:-320000}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
CKPT_FORMAT=${HL_CKPT_FORMAT:-torch}
USE_DISTRIBUTED_OPTIMIZER=${HL_USE_DISTRIBUTED_OPTIMIZER:-1}
LOAD_DIR=${HL_LOAD_DIR:-}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
VERIFY_CKPT=${HL_VERIFY_CKPT:-0}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH_FILE:-}
HOSTSFILE=${HL_HOSTSFILE:-}
CKP_ACT=${HL_CKP_ACT:-0}
RECOMPUTE_NUM_LAYERS=${HL_RECOMPUTE_NUM_LAYERS:-1}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
DEVICES_PER_NODE=${HL_DEVICES_PER_NODE:-8}
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-1}
OPTIMIZER=${HL_OPTIMIZER:-fusedadamw}
DROPOUT=${HL_DROPOUT:-0.0}
EVAL_ITERS=${HL_EVAL_ITERS:-10}
EVAL_INTERVAL=${HL_EVAL_INTERVAL:-1000}
USE_FUSED_SDPA=${HL_USE_FUSED_SDPA:-1}
USE_FUSED_SDPA_WITH_RECOMPUTE=${HL_USE_FUSED_SDPA_WITH_RECOMPUTE:-1}
USE_FAST_SOFTMAX=${HL_USE_FAST_SOFTMAX:-1} 
USE_LAZY_MODE=${HL_USE_LAZY_MODE:-0}
USE_TORCH_COMPILE=${HL_USE_TORCH_COMPILE:-0}
USE_TORCH_COMPILED_AUTOGRAD=${HL_USE_TORCH_COMPILED_AUTOGRAD:-0}

USE_FUSED_RMSNORM=${HL_USE_FUSED_RMSNORM:-1}

PROFILE_TYPE=${HL_PROFILE_TYPE:-} # provide either of pt, pt-full, hltv
PROFILE_STEP_START=${HL_PROFILE_STEP_START:-3}
PROFILE_STEP_END=${HL_PROFILE_STEP_END:-4}
PROFILE_RANKS=${HL_PROFILE_RANKS:-}

REDIRECT_LOGS=${HL_REDIRECT_LOGS:-0}
DETERMINISTIC_MODE=${HL_DETERMINISTIC_MODE:-0}
TORCHRUN_MULTINODE=${HL_TORCHRUN_MULTINODE:-0}
TORCHRUN_NODE_RANK=${HL_TORCHRUN_NODE_RANK:-0}
TORCHRUN_MASTER_ADDR=${HL_TORCHRUN_MASTER_ADDR:-localhost}

NUM_WORKERS=${HL_NUM_WORKERS:-2}
ENABLE_PARAM_GATHER_OVERLAP=${HL_ENABLE_PARAM_GATHER_OVERLAP:-0}
ENABLE_GRAD_REDUCE_OVERLAP=${HL_ENABLE_GRAD_REDUCE_OVERLAP:-0}
# MoE
TOKEN_DROP=${HL_TOKEN_DROP:-0} # 0: original dropless, 1: drop, 2: capacity dropless
TOKEN_DISPATCHER_TYPE=${HL_TOKEN_DISPATCHER_TYPE:-allgather}
MOE_LAYER_RECOMPUTE=${HL_MOE_LAYER_RECOMPUTE:-0}
MOE_GROUPED_GEMM=${HL_MOE_GROUPED_GEMM:-1}

ENABLE_SHARED_EXPERT_OVERLAP=${HL_ENABLE_SHARED_EXPERT_OVERLAP:-0}
# first 3 layers dense
MOE_SKIP_FIRST_LAYERS=${HL_MOE_SKIP_FIRST_LAYERS:-3}
# PP
FISRT_PP_STAGE_LAYERS=${HL_FISRT_PP_STAGE_LAYERS:-0}
LAST_PP_STAGE_LAYERS=${HL_LAST_PP_STAGE_LAYERS:-0}

if [[ -z "${MEGATRON_LM_ROOT}" ]]; then
    MEGATRON_LM_ROOT=$(realpath "$(dirname "$0")"/../../)
fi

if [[ $((NUM_NODES*DEVICES_PER_NODE)) -ne $((DP*TP*PP)) ]]; then
    echo "NUM_NODES*DEVICES_PER_NODE != DP*TP*PP"
    exit 1
fi

# Model size variables
MAX_SEQ_LEN=${SEQ_LEN}
EXTRA_VOCAB_SIZE=467 # 466 for unsloth because of 128816 unsloth tokenizer.jso

HIDDEN_SIZE=7168
FFN_HIDDEN_SIZE=18432
EXPERT_FFN_HIDDEN_SIZE=2048
SHARED_EXPERT_FFN_HIDDEN_SIZE=2048
NUM_HEADS=128
NUM_KV_HEADS=128
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_HEAD_DIM=128
QK_POS_EMB_HEAD_DIM=64
V_HEAD_DIM=128
NUM_EXPERTS=256
TOPK=8
NUM_LAYERS=${HL_NUM_LAYERS:-61}
KV_CHANNELS=128

LR=2.2e-4
MIN_LR=2.2e-5
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR_WARMUP_ITERS=2000
ROTARY_BASE=10000
INIT_STD=6e-3

AUX_LOSS_COEFF=1e-4

if [[ ${FISRT_PP_STAGE_LAYERS} -eq 0 && ${LAST_PP_STAGE_LAYERS} -eq 0 && $(( NUM_LAYERS % PP )) -ne 0 ]]; then
    echo 'HL_NUM_LAYERS must be divisible by PP'
    exit 1
fi

# Paths
SRC_PATH="${MEGATRON_LM_ROOT}/pretrain_gpt.py"
DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}

if [[ -z "${TOKENIZER_MODEL}" ]]; then
    TOKENIZER_MODEL="${DATA_DIR}"
fi

NUM_DEVICES=$((DEVICES_PER_NODE*NUM_NODES))

RUNTIME=$(date +"%Y%m%d_%H%M.%S.%N")
# Experiment name
if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="default"
fi
# output paths
if [[ -z "${OUTPUT_DIR}" ]]; then
    data_type="bf16"
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/deepseek_v3/${data_type}_${TRANSFORMER_IMPL}_${EXP_NAME}_nl${NUM_LAYERS}_gb${GLOBAL_BATCH_SIZE}_mb${MICRO_BATCH_SIZE}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_E${EP}_devices${NUM_DEVICES}_${RUNTIME}
fi
if [[ -z "${CHECKPOINTS_DIR}" ]]; then
    CHECKPOINTS_DIR=${OUTPUT_DIR}/checkpoints
fi
if [[ -z "${LOAD_DIR}" ]]; then
    LOAD_DIR=${CHECKPOINTS_DIR}
fi

if [[ -z "${TENSORBOARD_DIR}" ]]; then
    TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
fi
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINTS_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

if [[ "${NUM_NODES}" -ne "1" ]] && [[ -z "${HOSTSFILE}" ]] && [[ "${LAUNCHER_TYPE}" = "mpirun" ]]; then
    HOSTSFILE=${MEGATRON_LM_ROOT}/examples/hostsfile
    if [[ -f "${HOSTSFILE}" ]]; then
        cat /dev/null > "${HOSTSFILE}"
    fi
    etc_mpi_hostfile="/etc/mpi/hostfile"
    if [[ ! -f ${etc_mpi_hostfile} ]]; then
        echo "${etc_mpi_hostfile} not available, set HL_HOSTSFILE"
        exit 1
    fi
    cat ${etc_mpi_hostfile} | xargs -I{} echo {} slots=8 >> "${HOSTSFILE}"
fi

# Setting the environment variables
PT_HPU_GPU_MIGRATION=1

# Set training command
CMD=""
if [[ "${LAUNCHER_TYPE}" = "mpirun" ]]; then
    CMD="${CMD} mpirun"
    CMD="${CMD} --allow-run-as-root"
    CMD="${CMD} --mca plm_rsh_args -p${SSH_PORT}"
    [[ -n "${MPI_ROOT}" ]] && CMD="$CMD --prefix ${MPI_ROOT}"
    CMD="${CMD} -n ${NUM_DEVICES}"
    [[ -n "${HL_PE}" ]] && __MAP_BY="socket:PE=${HL_PE}"
    [[ -n "${HL_PE}" ]] && [[ -n "${HL_PPR}" ]] && __MAP_BY="ppr:${HL_PPR}:socket:PE=${HL_PE}"
    if [[ -n "${__MAP_BY}" ]]; then
        CMD="${CMD} --bind-to core --rank-by slot --report-bindings --map-by ${__MAP_BY}"
    else
        CMD="${CMD} --bind-to none"
    fi
    if [[ "${HNIC}" -eq "1" ]]; then
        CMD="${CMD} -x HCCL_OVER_OFI -x HCCL_GAUDI_DIRECT -x FI_PROVIDER -x LD_LIBRARY_PATH"
    fi
    CMD="${CMD} -x PT_HPU_GPU_MIGRATION=${PT_HPU_GPU_MIGRATION}"
    CMD="${CMD} -x PT_HPU_LAZY_MODE=${USE_LAZY_MODE}"
    IFS=',' read -ra ENV_FLAGS_ARR <<< "$ENV_FLAGS"
    for ENV_FLAG in "${ENV_FLAGS_ARR[@]}"; do
        CMD="${CMD} -x ${ENV_FLAG}"
    done
    if [[ "${NUM_NODES}" -ne "1" ]]; then
        CMD="${CMD} -hostfile ${HOSTSFILE}"
        CMD="${CMD} -x MASTER_ADDR=$(head -n 1 "${HOSTSFILE}" | sed -n s/[[:space:]]slots.*//p)"
    else
        CMD="${CMD} -x MASTER_ADDR=localhost"
    fi
    CMD="${CMD} -x MASTER_PORT=12345"
elif [[ "${LAUNCHER_TYPE}" = "torchrun" ]]; then
    if [[ "${TORCHRUN_MULTINODE}" -ne "1" ]] && [[ "${NUM_NODES}" -ne "1" ]]; then
        echo "NUM_NODES greater than 1 not supported by torchrun"
        exit 1
    fi
    export PT_HPU_GPU_MIGRATION=${PT_HPU_GPU_MIGRATION}
    export PT_HPU_LAZY_MODE=${USE_LAZY_MODE}
    if [[ "${HNIC}" -eq "1" ]]; then
        export HCCL_OVER_OFI=${HCCL_OVER_OFI}
        export HCCL_GAUDI_DIRECT=${HCCL_GAUDI_DIRECT}
        export FI_PROVIDER=${FI_PROVIDER}
    fi

    IFS=',' read -ra ENV_FLAGS_ARR <<< "$ENV_FLAGS"
    for ENV_FLAG in "${ENV_FLAGS_ARR[@]}"; do
        export "${ENV_FLAG?}"
    done
    CMD="${CMD} torchrun"
    CMD="${CMD} --nnodes ${NUM_NODES}"
    CMD="${CMD} --nproc-per-node ${DEVICES_PER_NODE}"
    CMD="${CMD} --no-python"
    CMD="${CMD} --node-rank ${TORCHRUN_NODE_RANK}"
    CMD="${CMD} --master-addr ${TORCHRUN_MASTER_ADDR}"
    CMD="${CMD} --master-port 12345"
else
    echo "Unsupported launcher type = ${LAUNCHER_TYPE}"
    exit 2
fi
# prepare MoE args
MOE_ARGS="--num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${TOPK} \
    --expert-model-parallel-size ${EP} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-bias-update-rate 0.001 \
    --moe-router-score-function sigmoid \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-enable-expert-bias \
    --moe-aux-loss-coeff ${AUX_LOSS_COEFF} \
    --moe-token-dispatcher-type ${TOKEN_DISPATCHER_TYPE} \
    --moe-ffn-hidden-size ${EXPERT_FFN_HIDDEN_SIZE} \
    --moe-shared-expert-intermediate-size ${SHARED_EXPERT_FFN_HIDDEN_SIZE}"
    

if [[ "${TOKEN_DISPATCHER_TYPE}" = "alltoall" && ${ENABLE_SHARED_EXPERT_OVERLAP} -eq 1 ]]; then
    MOE_ARGS="${MOE_ARGS} \
        --moe-shared-expert-overlap"
fi

if [[ ${TOKEN_DROP} -eq 1 ]]; then
    MOE_ARGS="${MOE_ARGS} \
        --moe-expert-capacity-factor 1.0 \
        --moe-pad-expert-input-to-capacity \
        --moe-token-drop-policy probs"
fi

if [[ ${MOE_LAYER_RECOMPUTE} -eq 1 ]]; then
    MOE_ARGS="${MOE_ARGS} \
        --recompute-granularity selective --recompute-modules moe"
fi

if [[ ${MOE_SKIP_FIRST_LAYERS} -gt 0 ]]; then
    NUM_MOE_LAYERS=$((NUM_LAYERS - MOE_SKIP_FIRST_LAYERS))
    MOE_ARGS="${MOE_ARGS} \
       --moe-layer-freq ([0]*${MOE_SKIP_FIRST_LAYERS}+[1]*${NUM_MOE_LAYERS})"
fi

FISRT_PP_STAGE_LAYERS_ARG=""
if [[ ${FISRT_PP_STAGE_LAYERS} -gt 0 ]]; then
    FISRT_PP_STAGE_LAYERS_ARG="--decoder-first-pipeline-num-layers ${FISRT_PP_STAGE_LAYERS}"
fi

LAST_PP_STAGE_LAYERS_ARG=""
if [[ ${LAST_PP_STAGE_LAYERS} -gt 0 ]]; then
    LAST_PP_STAGE_LAYERS_ARG="--decoder-last-pipeline-num-layers ${LAST_PP_STAGE_LAYERS}"
fi

MLA_ARGS="--multi-latent-attention \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_HEAD_DIM} \
    --qk-layernorm \
    --qk-pos-emb-head-dim ${QK_POS_EMB_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM}"


CMD="${CMD} \
    python ${SRC_PATH} \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    ${FISRT_PP_STAGE_LAYERS_ARG} \
    ${LAST_PP_STAGE_LAYERS_ARG} \
    --distributed-backend nccl \
    --seq-length ${SEQ_LEN} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --num-query-groups ${NUM_KV_HEADS} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --position-embedding-type rope \
    --rotary-base ${ROTARY_BASE} \
    --max-position-embeddings ${MAX_SEQ_LEN} \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --attention-dropout ${DROPOUT} \
    --hidden-dropout ${DROPOUT} \
    ${MOE_ARGS} \
    ${MLA_ARGS} \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 ${ADAM_BETA1}\
    --adam-beta2 ${ADAM_BETA2} \
    --adam-eps ${ADAM_EPS} \
    --lr ${LR} \
    --lr-decay-style cosine \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --min-lr ${MIN_LR} \
    --use-fused-sdpa-with-recompute ${USE_FUSED_SDPA_WITH_RECOMPUTE} \
    --use-fused-sdpa ${USE_FUSED_SDPA} \
    --use-fused-rmsnorm ${USE_FUSED_RMSNORM} \
    --use-fast-softmax ${USE_FAST_SOFTMAX} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-iters ${LR_DECAY_ITER} \
    --log-interval ${LOG_INTERVAL} \
    --log-throughput \
    --disable-bias-linear \
    --optimizer ${OPTIMIZER} \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --use-mcore-models \
    --bf16 \
    --exit-interval ${EXIT_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --load ${LOAD_DIR} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --data-path ${DATA_PATH} \
    --init-method-std ${INIT_STD} \
    --num-workers ${NUM_WORKERS} \
    --no-rope-fusion \
    --kv-channels ${KV_CHANNELS} \
    --use-torch-compile=${USE_TORCH_COMPILE} \
    --use-torch-compiled-autograd=${USE_TORCH_COMPILED_AUTOGRAD} \
    "

# --log-memory-to-tensorboard
# -------------
# GroupedMLP
if [[ "$MOE_GROUPED_GEMM" -eq 1 ]] ; then
    CMD="${CMD} --moe-grouped-gemm"
fi

if [[ ${ENABLE_PARAM_GATHER_OVERLAP} -eq 1 ]]; then
    CMD="${CMD} --overlap-param-gather"
fi

if [[ ${ENABLE_GRAD_REDUCE_OVERLAP} -eq 1 ]]; then
    CMD="${CMD} --overlap-grad-reduce"
fi

if [[ "${SEQ_PARALLEL}" -eq 1 ]]; then
    CMD="${CMD} --sequence-parallel"
fi

if [[ "${CKP_ACT}" -eq 1 ]]; then
    CMD="${CMD} --recompute-granularity=full"
    CMD="${CMD} --recompute-method uniform"
    CMD="${CMD} --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [[ "${CKP_ACT}" -eq 2 ]]; then
    CMD="${CMD} --recompute-granularity selective"
fi

if [[ "${USE_DISTRIBUTED_OPTIMIZER}" -eq 1 ]]; then
    CMD="${CMD} --use-distributed-optimizer"
fi

if [[ "${DETERMINISTIC_MODE}" -eq 1 ]]; then
    CMD="${CMD} --deterministic-mode"
fi

# handle kill switch file argument
if [[ -n "${KILL_SWITCH_FILE}" ]]; then
    CMD="${CMD} --kill-switch-file ${KILL_SWITCH_FILE}"
fi

if [[ -n "${PROFILE_TYPE}" ]]; then
    CMD="${CMD} --profile-type ${PROFILE_TYPE}"
    CMD="${CMD} --profile-step-start ${PROFILE_STEP_START}"
    CMD="${CMD} --profile-step-end ${PROFILE_STEP_END}"
    if [ -n "${PROFILE_RANKS}" ]; then
        CMD="${CMD} --profile-ranks ${PROFILE_RANKS}"
    fi
fi

if [[ "${CHECKPOINT_SAVE}" -eq 1 ]]; then
    CMD="${CMD} --save ${CHECKPOINTS_DIR}"
    CMD="${CMD} --save-interval ${SAVE_INTERVAL}"
    CMD="${CMD} --ckpt-format ${CKPT_FORMAT}"
    if [[ "${VERIFY_CKPT}" -eq 1 ]]; then
       CMD="${CMD} --verify-checkpoint"
       CMD="${CMD} --verify-checkpoint-model-type DEEPSEEK"
    fi
fi

if [[ "${TOKENIZER_TYPE}" = "HuggingFaceTokenizer" || "${TOKENIZER_TYPE}" = "GPTSentencePieceTokenizer" || "${TOKENIZER_TYPE}" = "Llama2Tokenizer" || "${TOKENIZER_TYPE}" = "Llama3Tokenizer" ]]; then
    CMD="${CMD} --tokenizer-type ${TOKENIZER_TYPE}"
    CMD="${CMD} --tokenizer-model ${TOKENIZER_MODEL}"
elif [[ "${TOKENIZER_TYPE}" = "DeepSeekV2Tokenizer" ]]; then
    CMD="${CMD} --tokenizer-type ${TOKENIZER_TYPE}"
    CMD="${CMD} --tokenizer-model ${TOKENIZER_MODEL}"
    CMD="${CMD} --extra-vocab-size ${EXTRA_VOCAB_SIZE}"
elif [[ "${TOKENIZER_TYPE}" = "GPT2BPETokenizer" ]]; then
    CMD="${CMD} --tokenizer-type GPT2BPETokenizer"
    CMD="${CMD} --vocab-file ${DATA_DIR}/gpt2-vocab.json"
    CMD="${CMD} --merge-file ${DATA_DIR}/gpt2-merges.txt"
else
    echo "incorrect HL_TOKENIZER_TYPE=${TOKENIZER_TYPE} is set"
    exit 1
fi

if [[ -n "${DATA_CACHE_DIR}" ]]; then
    CMD="${CMD} --data-cache-path ${DATA_CACHE_DIR}"
fi

if [[ "${REDIRECT_LOGS}" -eq 1 ]]; then
    ${CMD} 2>&1 | tee "${OUTPUT_DIR}"/log_"${EXP_NAME}"_"${RUNTIME}".txt
else
    ${CMD}
fi
