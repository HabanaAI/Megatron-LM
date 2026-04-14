# DeepSeek V3 for PyTorch

This directory provides examples of the DeepSeek V3 Mixture-of-Experts (MoE) model training in the Megatron-LM repository on Intel® Gaudi® AI accelerators.
Before you get started, make sure to review the [Supported Configuration](../../README.md#supported-configuration).

## Table of Contents
* [Setup](#setup)
* [Dataset Preparation](#dataset-preparation)
* [Mpirun Settings](#mpirun-settings)
* [Training Script Settings](#training-script-settings)
* [DeepSeek V3 Training and Examples](#deepseek-v3-training-and-examples)

# Setup
Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi 2.

## How to Use
Users bear sole liability and responsibility to follow and comply with any third party licenses, and Intel Corporation disclaims and will bear no liability with respect to users' use or compliance with third party licenses.
* Third-Party Models
  * In the course of using Megatron-LM, users may choose to download models created and distributed by third parties after reviewing background information about the models and agreeing to the license governing those models.
  * Notice: Intel does not create the content and does not warrant its accuracy or quality. By accessing the third-party content, or using materials trained on or with such content, you are indicating your acceptance of the terms associated with that content and warranting that your use complies with the applicable license.
  * Intel expressly disclaims the accuracy, adequacy, or completeness of any such third-party content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. You agree Intel is not liable for any liability or damages relating to your use of third-party content.
  * Intel's identification of these resources does not expand or otherwise alter Intel's applicable published warranties or warranty disclaimers for Intel products or solutions, and you agree that no additional obligations, indemnifications, or liabilities arise from Intel identifying such resources. Intel reserves the right, without notice, to make corrections, enhancements, improvements, and other changes to its materials.
  * The table below contains links to the licenses for certain third-party models and detailed information about the capabilities, limitations, and best practices for those models.

    | Model/Component        | Framework         | Mode                | Detailed Information | License |
    | ---------------------- | ----------------- | ------------------- | -------------------- | ------- |
    | DeepSeek V3            | PyTorch           | Pretraining         | https://github.com/deepseek-ai/DeepSeek-V3 | [License](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE) |

## Prerequisites
* When creating Docker container, set the shared memory size as 10 GB through the Docker run command:
  ```bash
  --shm-size=10g
  ```

## Clone Intel Gaudi Megatron-LM
In the Docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.
```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Megatron-LM
```
Set the required environment variables as shown below:
```
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_ROOT:$PYTHONPATH
```

## Install DeepSeek Requirements
* In the Docker container, go to the Megatron-LM directory:
  ```bash
  cd $MEGATRON_LM_ROOT
  ```

* Install the required packages using pip:
  ```bash
  pip install -r megatron/core/requirements.txt
  ```
* Review and accept the [DeepSeek V3 license conditions](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE) before using it
  ```bash
  pip install -r examples/deepseek/requirements.txt
  ```

* To run training on more than 128 cards, apply the below configuration changes:
  ```bash
  echo '*    soft nofile  unlimited' >> /etc/security/limits.conf
  echo '*    hard nofile  unlimited' >> /etc/security/limits.conf
  echo 'root soft nofile  unlimited' >> /etc/security/limits.conf
  echo 'root hard nofile  unlimited' >> /etc/security/limits.conf
  ```

## Dataset Preparation
Follow the instructions in https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T to download the Red Pajama 1T dataset from HuggingFace. This dataset is used for training DeepSeek V3 models.

### Dataset Preparation Example
The below provides the steps required to prepare your dataset using the Red Pajama 1T dataset. This example shows how to download and preprocess the data.

### Step 0: Install Preprocessing Requirements
```bash
pip install -r $MEGATRON_LM_ROOT/examples/deepseek/requirements_preprocess.txt
```

### Step 1: Download Dataset
```bash
# Create dataset directory
mkdir -p /data/deepseek_dataset
cd /data/deepseek_dataset

# Download the full Wikipedia JSONL file (24B tokens) from below url
wget https://data.together.xyz/redpajama-data-1T/v1.0.0/wikipedia/wiki.jsonl
```

### Step 2: Download DeepSeek Tokenizer
Review and accept the [license conditions](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE) before using it
```bash
# Create tokenizer directory
mkdir -p /data/deepseek_tokenizer

# Download tokenizer files using HuggingFace (recommended)
hf download deepseek-ai/DeepSeek-V3 tokenizer.json tokenizer_config.json --local-dir /data/deepseek_tokenizer

# Or download individual files manually
# Download tokenizer files from HuggingFace
cd /data/deepseek_tokenizer
wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json
wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer_config.json
```

### Step 3: Tokenize the Dataset
Use the methods below to tokenize the dataset. You can use any number of workers based on the CPU cores.

#### Full Wikipedia Dataset (24B tokens)
```bash
mkdir -p /data/deepseek_preprocessed

$PYTHON $MEGATRON_LM_ROOT/tools/preprocess_data.py \
    --input /data/deepseek_dataset/wiki.jsonl \
    --output-prefix /data/deepseek_preprocessed/rp_wiki_24B_deepseek_tok \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /data/deepseek_tokenizer/ \
    --workers 32 \
    --append-eod \
    --partitions 4
```

### Step 4: Verify Preprocessed Data
```bash
# Check that binary files were created successfully
ls -la /data/deepseek_preprocessed/
# Expected files:
# rp_wiki_24B_deepseek_tok_text_document.bin
# rp_wiki_24B_deepseek_tok_text_document.idx
```

# Mpirun Settings
These are system specific settings. Use these parameters for efficient allocation of resources and optimized performance. Please refer [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration) for more details.
* parallel environments (PEs) value is used to define how many processing elements(CPU cores) to be used for a given job. It is used as --map-by socket:PE=n. i.e. bind 'n' CPU cores to each MPI process.
  ```
  HL_PE=13
  ```
* processes per resource (PPR) specifies how many MPI processes should be launched per specific resource (socket). It is mostly used in multi-node training, used as --map-by ppr:n:socket:PE=m. i.e. 'n' MPI processes on each processor socket & bind 'm' CPU cores to each MPI process.
  ```
  HL_PPR=4
  ```

# Training Script Settings
* Default launcher to run training is mpirun. torchrun launcher is also supported but needs manual steps to launch training job on all workers involved in the multinode configuration. Set following flags on all workers correctly `HL_TORCHRUN_MULTINODE`, `HL_TORCHRUN_NODE_RANK`, `HL_TORCHRUN_MASTER_ADDR`.
  ```
  HL_LAUNCHER_TYPE=mpirun
  ```
* Based on the tokenization method, update the tokenizer type:
  ```
  HL_TOKENIZER_TYPE=HuggingFaceTokenizer
  ```
* Update data root dir with the path of your choice:
  ```
  HL_DATA_DIR_ROOT=/data/deepseek_preprocessed
  ```
* Update data file prefix(*.bin and *.idx) based on file name in data root dir:
  ```
  HL_DATA_FILE_PREFIX=rp_wiki_24B_deepseek_tok_text_document
  ```
* Update tokenizer path:
  ```
  HL_TOKENIZER_MODEL=/data/deepseek_tokenizer
  ```
* To run in lazy mode
  ```
  HL_USE_LAZY_MODE=1
  ```

Note: For the training commands, make sure to change the IP addresses in hostsfile according to your setup.
`HL_RESULTS_DIR` and `HL_DATA_DIR_ROOT` must be shared writable across all nodes and launchers when running training on more than 8 cards.
The same applies to `HL_CHECKPOINTS_DIR`, `HL_TENSORBOARD_DIR` and `HL_KILL_SWITCH` if specified.
If `HL_DATA_DIR_ROOT` is not writable, then `HL_DATA_CACHE_DIR` must be set to a writable location and
must be shared and accessible across all nodes and launchers when running training on more than 8 cards.

# DeepSeek V3 Training and Examples
* Training of DeepSeek V3 is based on the architecture described in the [DeepSeek V3 paper](https://huggingface.co/deepseek-ai/DeepSeek-V3)
* DeepSeek V3 is a Mixture-of-Experts (MoE) model with 671B total parameters and 37B activated parameters

## Model Architecture Details

DeepSeek V3 features:
- **Standard Transformer Layers**: First 3 layers (0-2) with regular MLP
- **MoE Layers**: Remaining layers use Mixture-of-Experts with 256 experts per layer
- **Expert Parallelism**: Various EP configuration supported for distributing experts across devices
- **Multi-Head Latent Attention (MLA)**: Advanced attention mechanism for efficiency
- **DeepSeekMoE Architecture**: Enhanced MoE with load balancing and routing

Key parameters:
- Total Parameters: 671B
- Activated Parameters: 37B per token
- Hidden Size: 7168
- Number of Experts: 256 per MoE layer
- Top-K Experts: 8 active experts per token
- Context Length: Up to 128K tokens

## Multi-Card Training Examples

### DeepSeek V3 Basic Training
* Run DeepSeek V3 with Reduced configuration sequence length 4k and number of layers 4 on 8 HPUs with BF16 precision:

  ```bash
  # Retain default settings for optimal performance.

  HL_USE_LAZY_MODE=1 \
  HL_TOKENIZER_TYPE=HuggingFaceTokenizer \
  HL_DATA_DIR_ROOT=/data/deepseek_preprocessed \
  HL_DATA_FILE_PREFIX=rp_wiki_24B_deepseek_tok_text_document \
  HL_TOKENIZER_MODEL=/data/deepseek_tokenizer \
  HL_NUM_NODES=1 \
  HL_DEVICES_PER_NODE=8 \
  HL_DP=8 \
  HL_TP=1 \
  HL_PP=1 \
  HL_EP=8 \
  HL_GBS=8 \
  HL_NUM_LAYERS=4 \
  HL_TOKEN_DROP=1 \
  HL_TOKEN_DISPATCHER_TYPE=alltoall \
  HL_MOE_SKIP_FIRST_LAYERS=3 \
  HL_SEQ_LEN=4096 \
  $MEGATRON_LM_ROOT/examples/deepseek/pretrain_deepseek.sh
  ```

### Multi-Node Training Examples

* Run DeepSeek V3 on 32 HPUs (4 nodes) with BF16 precision:
  ```bash
  HL_USE_LAZY_MODE=1 \
  HL_TOKENIZER_TYPE=HuggingFaceTokenizer \
  HL_DATA_DIR_ROOT=/data/deepseek_preprocessed \
  HL_DATA_FILE_PREFIX=rp_wiki_24B_deepseek_tok_text_document \
  HL_TOKENIZER_MODEL=/data/deepseek_tokenizer \
  HL_CKP_ACT=1 \
  HL_NUM_NODES=4 \
  HL_DEVICES_PER_NODE=8 \
  HL_DP=16 \
  HL_TP=1 \
  HL_PP=2 \
  HL_EP=16 \
  HL_GBS=1024 \
  HL_NUM_LAYERS=16 \
  HL_TOKEN_DROP=1 \
  HL_TOKEN_DISPATCHER_TYPE=alltoall \
  HL_MOE_SKIP_FIRST_LAYERS=3 \
  HL_SEQ_LEN=4096 \
  $MEGATRON_LM_ROOT/examples/deepseek/pretrain_deepseek.sh
  ```


### MoE-Specific Configuration

Key environment variables for MoE configuration:
```bash
# Expert Parallelism settings
export HL_EP=8                    # Expert parallelism degree
export HL_TOKEN_DROP=1            # Token dropping policy (0: dropless, 1: drop)
export HL_MOE_GROUPED_GEMM=1      # Use grouped GEMM for experts
export HL_MOE_SKIP_FIRST_LAYERS=3 # Number of dense layers before MoE layers

# Load balancing settings
export HL_AUX_LOSS_COEFF=1e-4     # Auxiliary loss coefficient for load balancing

# Token dispatcher settings
export HL_TOKEN_DISPATCHER_TYPE=alltoall  # allgather or alltoall
```

# Known Issues
* Deepseek model was tested with 4k sequnce length.
* Only scripts and configurations mentioned in this README are supported and verified.
