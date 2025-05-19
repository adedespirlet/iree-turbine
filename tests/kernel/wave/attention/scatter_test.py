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
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
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
from ..common.shapes import get_test_shapes
from torch.testing import assert_close

from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
    get_bshd_attention_kernel,
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

        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1,1,1), vector_shapes={M: 16, N: LOAD_ELEMS_PER_THREAD}),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M/2),
        tkw.WaveConstraint(N, BLOCK_N),
    ]

    # Mapping (identity)
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i,N:j},
        outputs={M: i,N:j},
    )

    # Define kernel
    @tkw.wave(constraints)
    def read_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
        index: tkl.Memory[M,N,GLOBAL_ADDRESS_SPACE, tkl.i32],
        lds: tkl.Memory[M,N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
       
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD,mapping=mapping)
       
        index_reg=tkw.read(index,elements_per_thread=LOAD_ELEMS_PER_THREAD)
        #index_reg = tkw.broadcast(index_reg, target_shape=[M, N])
        tkw.scatter_add(a_reg,index_reg,dim=0,memory=lds,mapping=mapping,elements_per_thread=LOAD_ELEMS_PER_THREAD)
        
        lds_reg=tkw.read(lds, elements_per_thread=LOAD_ELEMS_PER_THREAD,mapping=mapping)
        tkw.write(lds_reg,b,elements_per_thread=STORE_ELEMS_PER_THREAD,mapping=mapping)

        #tkw.write(a_reg,b,elements_per_thread=STORE_ELEMS_PER_THREAD,mapping=mapping)

 
    # Compile kernel
    options = WaveCompileOptions(
        subs={
            M: 16,
            N: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
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

   
    input = device_arange(16*16, dtype=torch.int32).reshape(16, 16).contiguous()
    #index = device_ones(16*16, dtype=torch.int16).contiguous()
    index = device_ones(16*16, dtype=torch.int32).reshape(16, 16).contiguous()
    lds= device_zeros((16, 16), dtype=torch.int32).reshape(16,16).contiguous()
    output = device_zeros((16, 16), dtype=torch.int32).reshape(16, 16).contiguous()
    

    # Run kernel
    read_fn(input,index,lds,output)

    print("Input a:")
    print(input.cpu())
    print("Input index:")
    print(index.cpu())
    print("Output:")  
    print(output.cpu())

    ####TORCH baseline
    def scatter_baseline(input, index):
        index = index.to(dtype=torch.int64)  
        if index.shape != input.shape:
            while index.dim() < input.dim():
                index = index.unsqueeze(-1)
            index = index.expand_as(input)
        baseline_output = device_zeros(input.shape, dtype=torch.int32) 
        baseline_output = baseline_output.scatter_add(dim=0, index=index, src=input)
        return baseline_output
        
    torch_output=scatter_baseline(input,index)
    print("torch_output:")
    print(torch_output)

    torch.testing.assert_close(output, torch_output)

    print("Test passed! values scattered correctly.")



# # Symbols
# M = tkl.sym.M
# N = tkl.sym.N
# BLOCK_M = tkl.sym.BLOCK_M
# BLOCK_N = tkl.sym.BLOCK_N
# LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
# STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
# ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


# @require_e2e
# def test_read_actual_data():
#     # Constraints
#     constraints = [
#         tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 64}),
#         tkw.WorkgroupConstraint(M, BLOCK_M, 0),
#         tkw.WaveConstraint(M, BLOCK_M),
    
#     ]

#     # Mapping (identity)
#     i = tkw.IndexMapping.iterator(0)
#     mapping = tkw.IndexMapping(
#         num_iterators=1,
#         inputs={M: i},
#         outputs={M: i},
#     )

#     # Define kernel
#     @tkw.wave(constraints)
#     def read_kernel(
#         a: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.i32],
#         index: tkl.Memory[M,GLOBAL_ADDRESS_SPACE, tkl.i32],
#         lds: tkl.Memory[M,ADDRESS_SPACE, tkl.i32],
#         b: tkl.Memory[M,GLOBAL_ADDRESS_SPACE, tkl.i32],
#     ):
#         a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD,mapping=mapping)
        
#         index_reg=tkw.read(index,elements_per_thread=LOAD_ELEMS_PER_THREAD)
#         tkw.scatter_add(a_reg,index_reg,dim=0,memory=lds,mapping=mapping,elements_per_thread=LOAD_ELEMS_PER_THREAD)
        
#         lds_reg=tkw.read(lds, elements_per_thread=LOAD_ELEMS_PER_THREAD,mapping=mapping)
#         tkw.write(lds_reg,b,elements_per_thread=STORE_ELEMS_PER_THREAD,mapping=mapping)

#         #index_reg = tkw.broadcast(index_reg, target_shape=[M, N])
#         #tkw.write(a_reg,b,elements_per_thread=STORE_ELEMS_PER_THREAD,mapping=mapping)
#         return 
    
#     options = WaveCompileOptions(
#         subs={
#             M: 64,
#             N: 64,
#             BLOCK_M: 64,
#             BLOCK_N: 64,
#             LOAD_ELEMS_PER_THREAD: 2,
#             STORE_ELEMS_PER_THREAD: 2,
#             ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            
#         },
#         kernel_usages=[
#         tkl.kernel_buffer.KernelBufferUsage.INPUT,   
#         tkl.kernel_buffer.KernelBufferUsage.INPUT,
#         tkl.kernel_buffer.KernelBufferUsage.INPUT,     
#         tkl.kernel_buffer.KernelBufferUsage.OUTPUT,  
#         ],
#         compile_to_mlir=False,
#         canonicalize=True,
#         run_bench=False,
#     )
#     options = set_default_run_config(options)
#     read_fn = wave_compile(options, read_kernel)
#     print(read_fn.asm)
 
#     input = device_ones(64, dtype=torch.int32).reshape(64).contiguous()
#     index = device_ones(64, dtype=torch.int32).reshape(64).contiguous()
#     lds= device_zeros(64, dtype=torch.int32).reshape(64).contiguous()
#     output = device_zeros(64, dtype=torch.int32).reshape(64).contiguous()
   
#     read_fn(input,index,lds,output)

#     print("Input a:")
#     print(input.cpu())
#     print("Input index:")
#     print(index.cpu())
#     print("Output:")  
#     print(output.cpu()) 

#     ####TORCH baseline
#     def scatter_baseline(input, index):
#         index = index.to(dtype=torch.int64)  
#         baseline_output = device_zeros(input.shape, dtype=torch.int32) 
#         baseline_output = baseline_output.scatter_add(dim=0, index=index, src=input)
#         return baseline_output
        
#     torch_output=scatter_baseline(input,index)
#     print("torch_output:")
#     print(torch_output)

#     torch.testing.assert_close(output, torch_output)

#     print("Test passed! values scattered correctly.")
