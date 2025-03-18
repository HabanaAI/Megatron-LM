# LLM for PyTorch

This directory provides scripts to train the GPT-based models in the Megatron-LM repository on Intel® Gaudi® 2 & Gaudi® 3 AI accelerators.
Before you get started, make sure to review the [Supported Configurations](#supported-configurations).

## Table of Contents
* [Megatron Overview](#megatron-overview)
* [Setup](#setup)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

# Megatron Overview
This implementation is based on https://github.com/NVIDIA/Megatron-LM at core_r0.9.0.

This repository comprises two essential components: Megatron-LM and Megatron-Core. Megatron-LM serves as a research-oriented framework leveraging Megatron-Core for large language model (LLM) training. Megatron-Core, on the other hand, is a library of optimized training techniques including versioned APIs and regular releases. Alternatively, you can integrate Megatron-Core's building blocks into your preferred training framework.

## Megatron-LM
First introduced in 2019, Megatron ([1](https://arxiv.org/pdf/1909.08053), [2](https://arxiv.org/pdf/2104.04473), and [3](https://arxiv.org/pdf/2205.05198)) sparked a wave of innovation in the AI community, enabling researchers and developers to utilize the underpinnings of this library to further LLM advancements.

## Megatron-Core
Megatron-Core is an open-source PyTorch-based library that contains optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on accelerated computing infrastructure.

Megatron-Core offers core building blocks such as attention mechanisms, transformer blocks and layers, normalization layers, and embedding techniques. Additional functionality like activation recomputation, distributed checkpointing is also natively built-in to the library. The building blocks and functionality are all optimized, and can be built with advanced parallelization strategies for optimal training speed and stability on Accelerated Computing Infrastructure. Another key component of the Megatron-Core library includes advanced model parallelism techniques (tensor, sequence, pipeline, context, and MoE expert parallelism).


## How to Use
Users bear sole liability and responsibility to follow and comply with any third party licenses, and Habana Labs disclaims and will bear no liability with respect to users’ use or compliance with third party licenses.
* Third-Party Models
  * In the course of using Megatron-LM, users may choose to download models created and distributed by third parties after reviewing background information about the models and agreeing to the license governing those models.
  * Notice: Intel does not create the content and does not warrant its accuracy or quality. By accessing the third-party content, or using materials trained on or with such content, you are indicating your acceptance of the terms associated with that content and warranting that your use complies with the applicable license.
  * Intel expressly disclaims the accuracy, adequacy, or completeness of any such third-party content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. You agree Intel is not liable for any liability or damages relating to your use of third-party content.
  * Intel’s identification of these resources does not expand or otherwise alter Intel’s applicable published warranties or warranty disclaimers for Intel products or solutions, and you agree that no additional obligations, indemnifications, or liabilities arise from Intel identifying such resources. Intel reserves the right, without notice, to make corrections, enhancements, improvements, and other changes to its materials.


# Setup
Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi 2 and Gaudi 3.

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
## Install Megatron-LM Requirements
* In the Docker container, go to the Megatron-LM directory:
  ```bash
  cd $MEGATRON_LM_ROOT
  ```

* Install the required packages using pip:
  ```bash
  pip install -r megatron/core/requirements.txt
  ```

* To run training on more than 128 cards, apply the below configuration changes:
  ```bash
  echo '*    soft nofile  unlimited' >> /etc/security/limits.conf
  echo '*    hard nofile  unlimited' >> /etc/security/limits.conf
  echo 'root soft nofile  unlimited' >> /etc/security/limits.conf
  echo 'root hard nofile  unlimited' >> /etc/security/limits.conf
  ```

## Dataset Preparation
Follow the instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar to download oscar-en full dataset. Note that the dataset takes around 550G of disk space. This dataset is used for training LLaMA & LLaMA 2.
### Dataset Preparation Example
The below provides the steps required to prepare your dataset. It is based on instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar.  The dataset in the example is intended to be `zh`
### Step 0 :
```bash
git clone https://github.com/bigscience-workshop/bigscience.git
cd bigscience/data/oscar
# Edit the `oscar-to-jsonl.py` in the list language_subsets and remove the comment on unshuffled_deduplicated_zh and comment out unshuffled_deduplicated_en
vi oscar-to-jsonl.py
```
### Step 1 :
```bash
# -s can be added for subset of data
$PYTHON oscar-to-jsonl.py
```
### Step 2 :
  ```bash
mkdir -p zh
mv oscar*.jsonl zh
cd zh
  ```
### Step 3 :
Use one of the three methods below to tokenize the dataset. You can use any number of workers based on the CPU cores.
*  Tokenize the dataset using GPT2BPETokenizer:
    ```bash
    # download gpt2 vocab and merge files
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    # tokenize individual jsonl files
    # loop count will change based on number of files for a given dataset
    mkdir zh_tokenized
    for i in $(seq 0 4);
    do
      $PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/preprocess_data.py --input oscar-${i}.jsonl --output-prefix zh_tokenized/tokenized${i} --tokenizer-type GPT2BPETokenizer --vocab-file gpt2-vocab.json --merge-file gpt2-merges.txt --append-eod --workers 80
    done
    ```
  * Tokenize the dataset using GPTSentencePieceTokenizer:
    ```bash
    # download tokenizer.model based on model trying to train
    # tokenize individual jsonl files
    # loop count will change based on number of files for a given dataset
    mkdir zh_tokenized
    for i in $(seq 0 4);
    do
      $PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/preprocess_data.py --input oscar-${i}.jsonl --output-prefix zh_tokenized/tokenized${i} --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /path/to/tokenizer.model --append-eod --workers 80
    done
    ```
  * Tokenize the dataset using HFTokenizer:
    ```bash
    # path to tokenizer can be local directory path and to run custom code from it, trust remote code option(--trust-remote-code) should be passed
    #  or
    # path to tokenizer can be link to huggingface repo model card
    # if huggingface repo model card is a gated repo, Log in using a token from huggingface.co/settings/tokens with below command
    # huggingface-cli login
    # --seq-length value need to be passed explicitly from huggingface repo model card or local directory path which has model_max_length in tokenizer_config.json file
    # tokenize individual jsonl files
    # loop count will change based on number of files for a given dataset
    mkdir zh_tokenized
    for i in $(seq 0 4);
    do
      $PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/preprocess_data.py --input oscar-${i}.jsonl --output-prefix zh_tokenized/tokenized${i} --tokenizer-type HuggingFaceTokenizer --tokenizer-model /path/to/tokenizer --append-eod --workers 4 --seq-length 1000000000000000019884624838656
    done
    ```
### Step 4 :
 * Multiple tokenized dataset files are merged into a single file using the below method:
    ```bash
    # merge tokenized files
    mkdir zh_tokenized_merged
    $PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/merge_datasets.py --input zh_tokenized --output-prefix zh_tokenized_merged/tokenized_text_document
    # use the tokenized files generated from above command to train
    ```

# Supported Configurations
| Model                                       | Mode        | Intel Gaudi software Version | PyTorch Version | Validated on Gaudi 2 | Validated on Gaudi 3 |
| --------------------------------------------| ----------- | ---------------------------- | --------------- | -------------------- | -------------------- |
| [LLaMA 3.1](examples/llama/README.md)       | Pretraining | 1.20.0                       | 2.6.0           | :heavy_check_mark:   | :heavy_check_mark:*  |
| [Mixtral 8x7B](examples/mixtral/README.md)  | Pretraining | 1.20.0                       | 2.6.0           | :heavy_check_mark:** |                      |

*Sporadic numerical instability can occur when training with fp8 precision.

**Only BF16 configurations are currently enabled.

# Changelog
## 1.20.0
- Changed the default behavior of "accumulate_allreduce_grads_in_fp32" for the bfloat16 data type. Instead of performing gradient accumulation and all-reduce in fp32, it is sufficient to do so in bfloat16 for some configurations. However, for configurations where Gradient Accumulutation Steps * data parallel size > 256, switching to fp32 is necessary.
- Rebased code to upstream [core_r0.9.0](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.9.0) release.

## 1.19.0
 - Added support for Gaudi 3.
 - Added LLaMA 3.1 support and set as default.
 - Added Megatron-LM to Hugging Face LLaMA and Mixtral checkpoint conversion support. Usage example is available [here](./tools/checkpoint/README.md#llama-convert-megatron-lm-to-hugging-face-checkpoint).
 - Added Hugging Face to Megatron-LM LLaMA and Mixtral checkpoint conversion support. Usage example is available [here](./tools/checkpoint/README.md#llama-convert-hugging-face-checkpoint-to-megatron-lm).
 - Added Mixtral 8x7b BF16 support (preview version) [here](./examples/mixtral/README.md).

## 1.18.0
 - Initial release.

### Script Modifications
Major changes done to the original code from [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.9.0) repository:
* Changed README file content.
* Added HPU support.
* Added local RMSNorm support.
* Added support for HPU fused ops.
* Added checkpoint verification.
* Added kill-switch mechanism to gracefully stop training.


# Known Issues
* Only recipes mentioned in this README are supported and verified.
* Gaudi 3 scale out configurations can exhibit performance fluctuations.
