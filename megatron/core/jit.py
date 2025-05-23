# Copyright (C) 2024 Intel Corporation
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.utils import is_lazy_mode, is_real_cuda_device_available, is_torch_min_version

jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2
if is_torch_min_version("2.2.0a0"):
    jit_fuser = torch.compile

if not is_real_cuda_device_available() and is_lazy_mode():

    def dummy(func):
        return func

    jit_fuser = dummy
