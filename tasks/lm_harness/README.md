# Using lm evaluation harness with Megatron LM

Using lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness) with a Megatron-LM saved checkpoint.

Verified for lm-evaluation-harness version 0.4.8 evaluating Llama 3.1 8B.

Evaluation tasks supported in the python script and include:
- wikitext (ppl)
- lambada_openai (ppl + acc)
- hellaswag (acc)
- piqa (acc)
- winogrande (acc)

# Setup Instructions
Set environment variables

```bash
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH="$MEGATRON_LM_ROOT:$PYTHONPATH"
```

```bash
pip install -r $MEGATRON_LM_ROOT/tasks/lm_harness/requirements.txt
```

# Usage
Load the Megatron-LM checkpoint with all of the relevant arguments. Then run tasks/lm_evaluation_harness.py.
For example, run the following bash script, which loads arguments and runs evaluation:

```bash
HL_TOKENIZER_MODEL= HL_TP= HL_NUM_NODES=1 HL_DP=1 HL_CP=1 HL_PP=1 HL_SEQ_PARALLEL=0 HL_NUM_WORKERS=8 HL_CHECKPOINTS_DIR= HL_LM_EVAL=1 HL_LLAMA_VER= HL_LLAMA_MODEL_SIZE= HL_MICRO_BATCH= $MEGATRON_LM_ROOT/examples/llama/pretrain_llama.sh
```