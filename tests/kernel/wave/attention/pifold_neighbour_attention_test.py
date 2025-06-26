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


K1 = tkl.sym.K1
M = tkl.sym.M
N = tkl.sym.N
E = tkl.sym.E
FT = tkl.sym.FT
FT2 = tkl.sym.FT2
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
        GenericDot(k_mult=32, k_vec_size=1, out_vec_size=1, along_dim=MMAOperand.M),
    ],
)
def test_neighbor_attention(mfma_variant: MMAType):

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M / 1),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.TilingConstraint(K1, BLOCK_K1),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={M: 2, N: 1, K1: 32},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    d0 = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i, N: j},
    )
    mapping_gather = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i},
    )
    mapping_scatter = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: d0, N: j},
        dynamic_val_mappings={M: i},
    )

    # mapping_2 = tkw.IndexMapping(
    #     num_iterators=2,
    #     inputs={M: i, K1: j},
    #     outputs={M: i, K1: j},
    # )

    # scale= 1.0 / math.sqrt(32.0)
    scale = 1.0

    @tkw.wave(constraints)
    def neighbor_attention(
        concat_dst_edge_features: tkl.Memory[M, K1, GLOBAL_ADDRESS_SPACE, tkl.f32],
        mlp_weights: tkl.Memory[N, K1, GLOBAL_ADDRESS_SPACE, tkl.f32],
        edge_dest: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
        edge_feat: tkl.Memory[M, K1, GLOBAL_ADDRESS_SPACE, tkl.f32],
        lds_max: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        lds_exp: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        lds_message: tkl.Memory[M, K1, ADDRESS_SPACE, tkl.f32],
        out_V: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        edge_scaling = tkl.Register[N, M, tkl.f32](scale)
        zero_accumulator = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K1, init_args=[zero_accumulator])
        def accumulate_dot_product(
            partial_sum: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            # Gather destination node features, pass through MLP and obtain attention score for each edge : perform h_V_dst * mlp_weight
            concat_feat_reg = tkw.read(
                concat_dst_edge_features, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            mlp_reg = tkw.read(mlp_weights, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            partial_sum = tkw.mma(concat_feat_reg, mlp_reg, partial_sum)
            return partial_sum

        # Normalize attention scores
        scaled_scores = accumulate_dot_product * edge_scaling

        edge_dest_reg = tkw.read(edge_dest)

        # SCATTER_SOFTMAX
        tkw.scatter_max(
            scaled_scores,
            edge_dest_reg,
            dim=0,
            memory=lds_max,
            mapping=mapping,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )

        max_value = tkw.read(
            lds_max,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
            mapping=mapping_gather,
            mapping_dynamic_vals=(edge_dest_reg,),
        )

        substract = scaled_scores - max_value
        nominator = tkw.exp(substract)

        ## Write operation for debugging purpose
        tkw.write(
            nominator,
            out_V,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
            mapping=mapping_scatter,
            mapping_dynamic_vals=(edge_dest_reg,),
        )

        # ## Denominator calculation
        # tkw.scatter_add(
        #     nominator,
        #     edge_dest_reg,
        #     dim=0,
        #     memory=lds_exp,
        #     mapping=mapping,
        #     elements_per_thread=LOAD_ELEMS_PER_THREAD,
        # )  # lds_exp contains denominator values for softmax

        # ## Division
        # denominator = tkw.read(
        #     lds_exp,
        #     elements_per_thread=LOAD_ELEMS_PER_THREAD,
        #     mapping=mapping_gather,
        #     mapping_dynamic_vals=(edge_dest_reg,),
        # )

        # softmax = nominator / denominator

        # ## Scale edge's features
        # edge_feat_reg = tkw.read(edge_feat)

        # edge_feat_reg*=softmax

        # ## Message Aggregation
        # tkw.scatter_add(
        #     edge_feat_reg,
        #     edge_dest_reg,
        #     dim=0,
        #     memory=lds_message,
        #     mapping=mapping_2,
        #     elements_per_thread=LOAD_ELEMS_PER_THREAD,
        # )

        # lds_reg = tkw.read(
        #     lds_message, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=mapping_2
        # )
        # tkw.write(
        #     lds_reg, out_V, elements_per_thread=STORE_ELEMS_PER_THREAD, mapping=mapping_2
        # )

    # Hyperparams
    hyperparams = {
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        M: 32,
        N: 1,
        K1: 32,
        BLOCK_M: 32,
        BLOCK_N: 1,
        BLOCK_K1: 32,
        LOAD_ELEMS_PER_THREAD: 1,
        STORE_ELEMS_PER_THREAD: 1,
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=False,
        print_signature=True,
    )
    options = set_default_run_config(options)
    neighbor_attention = wave_compile(options, neighbor_attention)
    print(neighbor_attention.asm)

    concat_dst_edge_features = (
        device_arange(32 * 32, dtype=torch.float32).reshape(32, 32).contiguous()
    )
    mlp_weight = device_ones(32, dtype=torch.float32).reshape(1, 32).contiguous()
    # edge_dest = device_ones(32, dtype=torch.int32).reshape(32, 1).contiguous()
    edge_dest = device_randint(0, 10, (32, 1), dtype=torch.int32).contiguous()
    edge_feat = device_arange(32 * 32, dtype=torch.float32).reshape(32, 32).contiguous()

    lds_max = device_zeros(32, dtype=torch.float32).reshape(32, 1).contiguous()
    lds_exp = device_zeros(32, dtype=torch.float32).reshape(32, 1).contiguous()
    lds_message = (
        device_zeros(32 * 32, dtype=torch.float32).reshape(32, 32).contiguous()
    )

    output = device_zeros(32, dtype=torch.float32).reshape(32, 1).contiguous()
    neighbor_attention(
        concat_dst_edge_features,
        mlp_weight,
        edge_dest,
        edge_feat,
        lds_max,
        lds_exp,
        lds_message,
        output,
    )

    def scatter_softmax_baseline(
        scores: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes scatter-based softmax:
            softmax[i] = exp(score[i] - max_group[index[i]]) / sum_group[index[i]]
        """
        scores = scores.to(torch.float32).contiguous()
        index = index.to(torch.int64).contiguous()

        # Flatten to [M] shape if needed
        if scores.dim() > 1 and scores.shape[1] == 1:
            scores = scores.view(-1)
        if index.dim() > 1 and index.shape[1] == 1:
            index = index.view(-1)

        # Step 1: max per group
        max_per_group = torch.zeros(
            index.max().item() + 1, dtype=torch.float32, device=scores.device
        )
        max_per_group.scatter_reduce_(
            dim=0, index=index, src=scores, reduce="amax", include_self=False
        )

        # Step 2: exp(score - max)
        gathered_max = max_per_group[index]
        exps = torch.exp(scores - gathered_max)

        # Step 3: sum per group
        sum_per_group = torch.zeros_like(max_per_group)
        sum_per_group.scatter_add_(0, index, exps)

        # Step 4: gather denominator and compute softmax
        gathered_sum = sum_per_group[index]
        softmax = exps / gathered_sum

        return softmax.view(-1, 1)  # reshape to match original shape if needed

    def matmul_baseline_with_scatter_softmax(h_V_dst, mlp_weight, edge_dest):
        """
        Performs matmul + scatter-based softmax using same logic as wave kernel.
        """
        # Matrix multiplication
        scaled_scores = h_V_dst @ mlp_weight.T  # [M, 1]
        return scatter_softmax_baseline(scaled_scores, edge_dest)

    print("Input a:")
    print(concat_dst_edge_features.cpu())
    print("Output:")
    print(output.cpu())

    torch_output = matmul_baseline_with_scatter_softmax(
        concat_dst_edge_features, mlp_weight, edge_dest
    )
    print("torch_output:")
    print(torch_output)

    # torch.testing.assert_close(output, torch_output)

    # print("Test passed! values scattered correctly.")


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

# """
# Test case simulating PiFold-style neighbor attention:

# - Nodes: N = 32 residues
# - Each node connects to K = 30 neighbors → 33240 directed edges
# - Edge features: 200-dimensional
# - Node features: 200-dimensional (used from destination node)
# - Concatenated input to MLP: [33240, 400]
# - MLP projects [400] → [1] to compute unnormalized attention scores

# To apply softmax over neighbors, we must group edge scores by their destination nodes.
# This requires mapping from 33240 edges to their corresponding 32 destination node indices,
# and then applying a scatter_max across incoming edges per node.
# """
