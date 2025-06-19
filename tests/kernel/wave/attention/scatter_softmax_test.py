# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import math
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_ones,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
import os
from torch.nn import functional as F
from ..common.utils import (
    require_e2e,
    require_cdna3,
    dump_generated_mlir,
    perf_test,
    param_bool,
    enable_scheduling_barriers,
)

from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions

# Symbols
M = tkl.sym.M
N = tkl.sym.N
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


@require_e2e
def test_read_actual_data():
    # Constraintss
    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 8, N: LOAD_ELEMS_PER_THREAD},
        ),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N),
    ]

    # Mapping (identity)
    i = tkw.IndexMapping.iterator(0)
    # j = tkw.IndexMapping.iterator(1)

    d0 = tkw.IndexMapping.dynamic_val(0)
    mapping1 = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: i},
        outputs={M: i},
    )

    mapping_gather = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: d0},
        outputs={M: i},
        dynamic_val_mappings={M: i},
    )

    # Define kernel
    @tkw.wave(constraints)
    def read_kernel(
        a: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.i32],
        index: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.i32],
        lds_exp: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
        lds_max: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        """
        Compute softmax(x) using the formula:
            softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
        """

        index_reg = tkw.read(
            index, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=mapping1
        )
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=mapping1)

        casted_reg = tkw.cast(
            a_reg, tkl.f32
        )  # Ensure a_reg is in float32 for exp calculation

        ##Nominator calculation for softmax
        tkw.scatter_max(
            a_reg,
            index_reg,
            dim=0,
            memory=lds_max,
            mapping=mapping1,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )  # lds_max contains max values for each node

        max_value = tkw.read(
            lds_max,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
            mapping=mapping_gather,
            mapping_dynamic_vals=(index_reg,),
        )
        casted_max_val = tkw.cast(max_value, tkl.f32)
        substract = a_reg - max_value
        casted_substract = tkw.cast(substract, tkl.f32)

        nominator = tkw.exp(casted_substract)

        # Denominator calculation for softmax
        casted_reg -= casted_max_val
        exp = tkw.exp(casted_reg)
        tkw.scatter_add(
            exp,
            index_reg,
            dim=0,
            memory=lds_exp,
            mapping=mapping1,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )  # lds_exp contains denomainator values for softmax

        ## Division
        denominator = tkw.read(
            lds_exp,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
            mapping=mapping_gather,
            mapping_dynamic_vals=(index_reg,),
        )

        softmax = nominator / denominator

        tkw.write(softmax, b, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # Compile kernel
    options = WaveCompileOptions(
        subs={
            M: 8,
            N: 16,
            BLOCK_M: 8,
            BLOCK_N: 16,
            LOAD_ELEMS_PER_THREAD: 1,
            STORE_ELEMS_PER_THREAD: 1,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        kernel_usages=[
            tkl.kernel_buffer.KernelBufferUsage.INPUT,
            tkl.kernel_buffer.KernelBufferUsage.INPUT,
            tkl.kernel_buffer.KernelBufferUsage.INPUT,
            tkl.kernel_buffer.KernelBufferUsage.INPUT,
            tkl.kernel_buffer.KernelBufferUsage.OUTPUT,
        ],
        compile_to_mlir=False,
        canonicalize=True,
        run_bench=False,
        print_signature=True,
        # print_ir_before=["expand_graph"],
        # print_ir_after=["expand_graph"]
    )
    options = set_default_run_config(options)

    read_fn = wave_compile(options, read_kernel)
    print(read_fn.asm)
    # Input tensors

    input = device_arange(8, dtype=torch.int32).view(-1).contiguous()
    index = device_ones(8, dtype=torch.int32).view(-1).contiguous()
    lds1 = device_zeros(8, dtype=torch.float32).view(-1).contiguous()
    lds2 = device_zeros(8, dtype=torch.int32).view(-1).contiguous()
    output = device_zeros(8, dtype=torch.float32).view(-1).contiguous()

    # Run kernel
    read_fn(input, index, lds1, lds2, output)

    print("Input a:")
    print(input.cpu())
    print("Input index:")
    print(index.cpu())
    print("Output:")
    print(output.cpu())

    # ###TORCH baseline
    def scatter_softmax_baseline(input, index, output_size=8):
        """
        Simulates scatter_softmax using PyTorch ops.
        For each group of values in `input` that share the same index,
        computes softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
        Returns a tensor of the same shape as input.
        """

        index = index.to(dtype=torch.int64).contiguous()
        input = input.to(dtype=torch.float32).contiguous()

        # Step 1: compute max per index group
        max_per_index = torch.full(
            (output_size,), float("-inf"), dtype=input.dtype, device=input.device
        )
        max_per_index.scatter_reduce_(
            dim=0, index=index, src=input, reduce="amax", include_self=False
        )

        # Step 2: gather max values for each element from index
        max_gathered = max_per_index[index]

        # Step 3: subtract and exponentiate
        shifted = input - max_gathered
        exp_shifted = torch.exp(shifted)

        # Step 4: compute denominator (sum of exponentials per group)
        sum_exp_per_index = torch.zeros_like(max_per_index)
        sum_exp_per_index.scatter_add_(dim=0, index=index, src=exp_shifted)

        # Step 5: gather denominator values
        sum_exp_gathered = sum_exp_per_index[index]

        # Step 6: compute softmax
        softmax = exp_shifted / sum_exp_gathered

        return softmax

    torch_output = scatter_softmax_baseline(input, index)

    print("torch_output:")
    print(torch_output)

    torch.testing.assert_close(output, torch_output)

    print("Test passed! values scattered correctly.")