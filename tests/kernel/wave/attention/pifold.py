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
from iree.turbine.kernel.wave.constraints import MMAType, MMAOperand, GenericDot

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


K1=tkl.sym.K1
M=tkl.sym.M
N = tkl.sym.N
E = tkl.sym.E
FT = tkl.sym.FT
FT2 =  tkl.sym.FT2
F_IN = tkl.sym.F_IN
F_OUT = tkl.sym.F_OUT
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K1 = tkl.sym.BLOCK_K1
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

@require_e2e
@pytest.mark.parametrize(
    "mfma_variant",
    [
        GenericDot(k_mult=8,k_vec_size=1, out_vec_size=1, along_dim=MMAOperand.M),
    ],
)

def test_neighbor_attention( mfma_variant: MMAType):


    # mfma_variant=(MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16)
    
    # if mfma_variant[1] == MMAType.F32_16x16x16_F16:
    #     Mvec = 16
    #     Nvec = 16
    # if mfma_variant[1] == MMAType.F32_32x32x8_F16:
    #     Mvec = 32
    #     Nvec = 32

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),  
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.TilingConstraint(K1, BLOCK_K1),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={M: 8, N: 1, K1:8},
        ), 
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i, N: j},
    )
    
    @tkw.wave(constraints)
    def neighbor_attention(
        h_V_dst: tkl.Memory[M, K1, ADDRESS_SPACE, tkl.f16],
        mlp_weight: tkl.Memory[N,K1,ADDRESS_SPACE, tkl.f16 ],
        # edge_src: tkl.Memory[E, ADDRESS_SPACE, tkl.index],
        # edge_dst: tkl.Memory[E, ADDRESS_SPACE, tkl.index],
        out_V: tkl.Memory[M,N, ADDRESS_SPACE, tkl.f32],
    ):
        imm_reg = tkl.Register[M, N, tkl.f32](0.0)
        @tkw.iterate(K1, init_args=[imm_reg])
        def repeat(inner_acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            V_dst_reg = tkw.read(h_V_dst, elements_per_thread=LOAD_ELEMS_PER_THREAD)   
            mlp_reg = tkw.read(mlp_weight, elements_per_thread=LOAD_ELEMS_PER_THREAD) 
            #tkw.broadcast
            inner_acc=tkw.mma(V_dst_reg,mlp_reg,inner_acc)
            return inner_acc
        
        # Gather destination node features: [E, F]
        #h_dst = tkw.write(h_V_reg, dyn_values=dst_idx)


        # Compute raw attention score: simple dot product (per edge) , more involved is to pass both concatened through MLP
        # src = E_dst_reg*E_reg
        # attention_logits = tkw.sum(src, acc,dim=FT)  # [E, 1]

        tkw.write(repeat ,out_V,elements_per_thread=STORE_ELEMS_PER_THREAD,mapping=mapping)


    # Hyperparams
    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        M:16,
        N:1,
        K1:16,
        BLOCK_M: 16,
        BLOCK_N:1,
        BLOCK_K1:16,
        LOAD_ELEMS_PER_THREAD:1,
        STORE_ELEMS_PER_THREAD:1,
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=False,
        kernel_usages=[
        tkl.kernel_buffer.KernelBufferUsage.INPUT,   
        tkl.kernel_buffer.KernelBufferUsage.INPUT, 
        tkl.kernel_buffer.KernelBufferUsage.OUTPUT,  
        ],
        print_signature=True,
        print_ir_before=["decompose_dot_mma"],
        print_ir_after=["decompose_dot_mma"]
    )
    options = set_default_run_config(options)
    neighbor_attention = wave_compile(options, neighbor_attention)
    print(neighbor_attention.asm)

    h_V_dst = torch.arange(16*16, dtype=torch.float16).reshape(16,16).contiguous().cuda()
    
    #h_E = torch.zeros((16, 8), dtype=torch.int32).contiguous().cuda()
    mlp_weight = torch.arange(16*1, dtype=torch.float16).reshape(1,16).contiguous().cuda()

    output = torch.zeros((16, 1), dtype=torch.float32).contiguous().cuda()

    neighbor_attention(h_V_dst, mlp_weight, output)

    def matmul_baseline(h_V_dst, mlp_weight):
    # Treat `index` as matrix B, even though it's just ones
        return torch.matmul(h_V_dst.to(torch.float16), mlp_weight.T.to(torch.float16))

    print("Input a:")
    print(h_V_dst.cpu())
    print("Output:")  
    print(output.cpu())

    torch_output=matmul_baseline(h_V_dst,mlp_weight)
    print("torch_output:")
    print(torch_output)

# V = tkl.sym.V
# E = tkl.sym.E
# F_IN = tkl.sym.F_IN
# F_OUT = tkl.sym.F_OUT
# BLOCK_V = tkl.sym.BLOCK_V
# ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

# @run_test
# def test_neighbor_attention():
#     constraints = [
#         tkw.WorkgroupConstraint(V, BLOCK_V, 0), 
#         tkw.WorkgroupConstraint(E, BLOCK_V, 1), 
#         tkw.WorkgroupConstraint(F_IN, BLOCK_M, 2), #or F_OUT
#         tkw.WaveConstraint(M, BLOCK_M/2),
#         tkw.WaveConstraint(N, BLOCK_N),
#         tkw.HardwareConstraint(
#             threads_per_wave=64,
#             waves_per_block=(1, 1, 1),
#             vector_shapes={V: 1}
#         ),
#     ]

#     @tkw.wave(constraints)
#     def neighbor_attention(
#         h_V: tkl.Memory[V, F_IN, ADDRESS_SPACE, tkl.i32],
#         h_E: tkl.Memory[E, F_IN, ADDRESS_SPACE, tkl.i32],
#         edge_src: tkl.Memory[E, ADDRESS_SPACE, tkl.index],
#         edge_dst: tkl.Memory[E, ADDRESS_SPACE, tkl.index],
#         out_V: tkl.Memory[V, F_OUT, ADDRESS_SPACE, tkl.i32],
#     ):
#         h_V_reg = tkw.read(h_V, elements_per_thread=1)         # [V, F]
#         h_E_reg = tkw.read(h_E, elements_per_thread=1)         # [E, F]
#         src_idx = tkw.read(edge_src, elements_per_thread=1)    # [E]
#         dst_idx = tkw.read(edge_dst, elements_per_thread=1)    # [E]

#         acc = tkl.Register[E, 1, tkl.i32](0.0)

#         # Gather destination node features: [E, F]

#         # Compute raw attention score: simple dot product (per edge) , more involved is to pass both concatened through MLP
#         src = h_V_reg * h_E_reg
#         attention_logits = tkw.sum(src, acc,dim=1)  # [E, 1]

#         tkw.write(attention_logits,out_V)

#         # tkw.write(projected, out_V, elements_per_thread=1)

#     # Hyperparams
#     hyperparams = {
#         ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
#         V: 128,
#         E: 256,
#         F_OUT: 1,
#         F_IN: 32,
#         BLOCK_V: 64,
#     }

#     options = WaveCompileOptions(
#         subs=hyperparams,
#         canonicalize=True,
#         run_bench=False,
#         schedule=SchedulingType.NONE,
#         use_scheduling_barriers=False,
#         compile_to_mlir=True,
#     )

#     compiled = wave_compile(options, neighbor_attention)
#     print(compiled.asm)