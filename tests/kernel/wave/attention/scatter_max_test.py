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

    # d0 = [tkw.IndexMapping.dynamic_val(i) for i in range(1)]
    d0 = tkw.IndexMapping.dynamic_val(0)

    mapping1 = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: i},
        outputs={M: i},
    )
    mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: d0},
        outputs={M: i},
        dynamic_val_mappings={M: i},
    )

    mapping_scatter = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: i},
        outputs={M: i},
        dynamic_val_mappings={M: i},
    )

    # Define kernel
    @tkw.wave(constraints)
    def read_kernel(
        a: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.i32],
        index: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.i32],
        lds: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):

        index_reg = tkw.read(
            index, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=mapping1
        )

        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=mapping1)

        tkw.scatter_max(
            a_reg,
            index_reg,
            dim=0,
            memory=lds,
            mapping=mapping1,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )

        lds_reg = tkw.read(
            lds, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=mapping1
        )
        casted_lds_reg = tkw.cast(lds_reg, tkl.f32)

        tkw.write(casted_lds_reg, b, elements_per_thread=STORE_ELEMS_PER_THREAD)

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
    outputsize = 8  # Number of output "rows" for scatter max
    lds = device_zeros(8, dtype=torch.int32).view(-1).contiguous()
    output = device_zeros(8, dtype=torch.float32).view(-1).contiguous()

    # Run kernel
    read_fn(input, index, lds, output)

    print("Input a:")
    print(input.cpu())
    print("Input index:")
    print(index.cpu())
    print("Output:")
    print(output.cpu())

    ###TORCH baseline
    def scatter_max_baseline(input, index):
        """
        Simulates scatter_max by reducing values in `input` by maximum per `index` value.
        Returns a 1D tensor of shape [output_size].
        """

        index = index.to(dtype=torch.int64).contiguous()
        input = input.contiguous()
        input = input.to(dtype=torch.float32)

        if input.dtype.is_floating_point:
            init_val = torch.finfo(input.dtype).min
        else:
            init_val = torch.iinfo(input.dtype).min

        output = device_zeros(8, dtype=torch.float32).reshape(8).contiguous()

        output.scatter_reduce_(
            dim=0, index=index, src=input, reduce="amax", include_self=False
        )

        return output

    torch_output = scatter_max_baseline(input, index)

    print("torch_output:")
    print(torch_output)

    torch.testing.assert_close(output, torch_output)

    print("Test passed! values scattered correctly.")
