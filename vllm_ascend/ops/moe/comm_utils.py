# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
import torch_npu

from vllm.forward_context import get_forward_context

import vllm_ascend.envs as envs_ascend
from vllm_ascend.utils import get_ascend_soc_version, AscendSocVersion

COMM_STREAM = None


def async_all_to_all(input_,
                     output_split_sizes,
                     input_split_sizes,
                     group,
                     event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(
                device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True)
    else:
        handle = dist.all_to_all_single(a2a_out,
                                        input_.contiguous(),
                                        output_split_sizes=output_split_sizes,
                                        input_split_sizes=input_split_sizes,
                                        group=group,
                                        async_op=True)
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size,
                             dtype=input_.dtype,
                             device=torch.npu.current_device())
        torch.distributed.all_gather_into_tensor(output,
                                                 input_.contiguous(),
                                                 group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size,
                             dtype=input_.dtype,
                             device=torch.npu.current_device())
        output_tensor_list = list(
            torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


def is_enable_fusion_gmm_all2allv2() -> bool:
    """check if gmmall2allv2 operation is enabled"""
    model_type = get_forward_context().model_type
    soc_info = get_ascend_soc_version()
    hccl_op_expansion_mode = os.getenv("HCCL_OP_EXPANSION_MODE")
    return (envs_ascend.VLLM_ASCEND_ENABLE_GMM_All2AllV2
            and "qwen3_moe" == model_type
            and soc_info in [AscendSocVersion.A2, AscendSocVersion.A3]
            and hccl_op_expansion_mode == 'AIV')


def trans_scale_from_float_to_int64(scale):
    """trans scale dtype from fp16/bf16 to int64, used for fusion op such as gmmall2allv"""
    original_shape = scale.shape
    scale = torch.from_numpy(np.frombuffer(scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32)
                             .reshape(original_shape)
                             .astype(np.int64)).to(scale.device)
    return scale
