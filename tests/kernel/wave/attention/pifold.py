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
    
    scale= 1.0 / math.sqrt(16.0)
    @tkw.wave(constraints)
    def neighbor_attention(
        concat_dst_edge_features: tkl.Memory[M, K1, ADDRESS_SPACE, tkl.f16],
        mlp_weights:tkl.Memory[N,K1,ADDRESS_SPACE, tkl.f16 ],
        out_V: tkl.Memory[M,N, ADDRESS_SPACE, tkl.f32],
    ):
        edge_scaling = tkl.Register[N, M, tkl.f32](scale)
        zero_accumulator = tkl.Register[M, N, tkl.f32](0.0)
        @tkw.iterate(K1, init_args=[zero_accumulator])
        def accumulate_dot_product(partial_sum: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # Gather destination node features, pass through MLP and obtain attention score for each edge : perform h_V_dst * mlp_weight
            concat_feat_reg = tkw.read(concat_dst_edge_features, elements_per_thread=LOAD_ELEMS_PER_THREAD)   
            mlp_reg = tkw.read(mlp_weights, elements_per_thread=LOAD_ELEMS_PER_THREAD) 
            partial_sum=tkw.mma(concat_feat_reg,mlp_reg,partial_sum)
            return partial_sum
        
        # Normalize attention scores
        # attention_score =  accumulate_dot_product 
        # result = attention_score * edge_scaling

        #SCATTER_SOFTMAX
        ##SCATTER_ADD

        tkw.write(accumulate_dot_product ,out_V,elements_per_thread=STORE_ELEMS_PER_THREAD,mapping=mapping)


    # Hyperparams
    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        M:8,
        N:1,
        K1:8,
        BLOCK_M: 8,
        BLOCK_N:1,
        BLOCK_K1:8,
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

    h_V_dst = torch.arange(8*8, dtype=torch.float16).reshape(8,8).contiguous().cuda()
    
    #h_E = torch.zeros((16, 8), dtype=torch.int32).contiguous().cuda()
    mlp_weight = torch.ones(8*1, dtype=torch.float16).reshape(1,8).contiguous().cuda()

    output = torch.zeros((8, 1), dtype=torch.float32).contiguous().cuda()

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

# GNN Message Passing – PiFold Code Implementation
# This section documents the message passing logic implemented for the GNN in PiFold.
# Inputs:
# Concatenated Features: For each edge, concatenate the destination node's features with the edge’s own features.
# MLP Weights: A learned weight matrix used to compute attention scores.
# Transformed Edge Features: Feature vectors associated with each edge (e.g., after projection or embedding).
# Computation Steps:
# Attention Score Computation
# Compute the attention score for each edge by taking the dot product between:
# The concatenated destination + edge features, and The MLP weight vector.
# This results in a scalar attention score for each edge ([num_edges × 1]).
# Normalization
# Normalize the attention scores by dividing each by the square root of the feature dimension to improve training stability.
# Attention Softmax (scatter_softmax)
# Apply a scatter-based softmax so that, for each destination node, the attention scores of all incoming edges sum to 1.
# Weighted Edge Features
# Multiply each edge’s feature vector by its corresponding normalized attention score.
# Message Aggregation (scatter_add)
# Aggregate the messages from all incoming edges to their destination nodes using scatter_add.