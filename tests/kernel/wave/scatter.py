from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import run_test
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
import torch

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_mma():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]  
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]  

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
        )
    ]

    @tkw.wave(constraints)
    def scatter(
        src: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        index: tkl.Memory[M, 1, ADDRESS_SPACE, tkl.i32],
        dest: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        src_reg = tkw.read(src, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        #broadcast index to match size [M,N] and such that each thread gets a matching index
        index_broad_casted=tkl.broadcast(index,M,N)
        index_reg = tkw.read(index_broad_casted, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        tkl.scatter_add(src_reg, index_reg, dim=0, dest=dest, NUM_ROWS=M, NUM_COLS=N)

    compile_options = WaveCompileOptions(
        subs={
            M: 64,
            BLOCK_M: 64, 
            N: 64,
            BLOCK_N: 64,
            LOAD_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    scatter = wave_compile(compile_options, scatter)
    print(scatter.asm)