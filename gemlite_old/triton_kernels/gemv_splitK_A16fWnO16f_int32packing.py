# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl

from .config import AUTOTUNE_ENABLE
from .utils import *

KEYS          = ['M', 'N', 'K', 'group_size', 'elements_per_sample']
MATMUL_TYPE   = "GEMV_SPLITK"
NATIVE_ATOMIC = gpu_supports_bfloat16_atomicadd()

def kernel_config_pruner(configs, nargs, **kwargs):
    global KEYS
    from ..core import GEMLITE_TRITON_CONFIG_CACHE
    
    m = nargs['M'] 
    n = nargs['N'] 
    k = nargs['K'] 
    g = nargs['group_size']
    e = nargs['elements_per_sample']

    #Check cache
    if(MATMUL_TYPE in GEMLITE_TRITON_CONFIG_CACHE):
        _signature = str(tuple([nargs[i] for i in KEYS]))
        if(_signature in GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE]):
            _config     = copy.deepcopy(GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE][_signature])
            _num_stages = _config.pop('num_stages')
            _num_warps  = _config.pop('num_warps')
            _num_ctas   = _config.pop('num_ctas')

            _config.pop('num_buffers_warp_spec', None)
            _config.pop('num_consumer_groups', None)
            _config.pop('reg_dec_producer', None)
            _config.pop('reg_inc_consumer', None)

            yield triton.Config(_config,
                num_stages=_num_stages,
                num_warps=_num_warps,
                pre_hook=init_to_zero("c_ptr") if _config['SPLIT_K'] > 1 else None,
            )

            return
    
    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M'] 
        block_size_n = min((2 ** int(math.ceil(math.log2(n)))), config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min((2 ** int(math.ceil(math.log2(k)))), config.kwargs['BLOCK_SIZE_K'])
        split_k      = config.kwargs['SPLIT_K']

        #Skip larger blocks
        if(block_size_k > k or block_size_n > n):
            continue

        #Only 1 is supported
        block_size_m = 1 
        
        #Skip blocks that are either too large or too small
        block_area = (block_size_k // split_k) * block_size_n
        if(block_area < 1024 or block_area > 4096 * 8): #128 * 8 * num_warps 
            continue

        #Constraints
        #BLOCK_SIZE_K >= group_size
        block_size_k = min(block_size_k, g)

        #Block size should be compatible with minimum-packing
        if(block_size_k < e):
            continue

        #K needs to be devisible by BLOCK_SIZE_K * SPLIT_K 
        if(not is_divisible(k, block_size_k * split_k)):
            continue

        #No need for a hook for split_k == 1
        if(split_k == 1):
            config.pre_hook = None

        A_load_order      = config.kwargs['A_load_order']
        meta_evict_policy = config.kwargs['meta_evict_policy']
        atomic_mode       = config.kwargs['atomic_mode']
        dot_prod_mode     = config.kwargs['dot_prod_mode']

        _key = (block_size_m, block_size_n, block_size_k, group_size_m, split_k, 
                A_load_order, meta_evict_policy, atomic_mode, dot_prod_mode,
                config.num_stages, config.num_warps,
                )
        
        if _key in used:
            continue

        used.add(_key)
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
                'BLOCK_SIZE_K': block_size_k,
                'GROUP_SIZE_M': group_size_m,
                'SPLIT_K'     : split_k,

                'A_load_order'      : A_load_order,
                'meta_evict_policy' : meta_evict_policy,
                'atomic_mode'       : atomic_mode,
                'dot_prod_mode'     : dot_prod_mode,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            pre_hook=init_to_zero("c_ptr") if split_k > 1 else None,
        )

def get_autotune_config():
    _configs = []
    for _M in [1]: 
        for _N in [1, 2, 4, 8, 16, 32, 64]:
            for _K in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
                for _w in [4, 8]:
                    for _s in [1, 2]:
                        for _sK in [1]:
                            for _A_load_order in [0]: #[0, 1, 2, 3]
                                for _dot_prod_mode in [0]: #[0, 1]
                                    for _meta_evict_policy in ['']: #[', 'evict_last']
                                        for _atomic_mode in ['relaxed']: #['release', 'relaxed']:
                                            _configs.append(
                                                    triton.Config(
                                                        {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 
                                                        'GROUP_SIZE_M': 8, 'SPLIT_K': _sK,
                                                        'A_load_order': _A_load_order, 'meta_evict_policy': _meta_evict_policy, 
                                                        'atomic_mode': _atomic_mode, 'dot_prod_mode': _dot_prod_mode,
                                                        }, 
                                                        num_stages=_s, num_warps=_w,
                                                        pre_hook=init_to_zero("c_ptr") if(_sK > 1) else None,
                                                        )
                                                    )
    return _configs


compute_capability = torch.cuda.get_device_capability(0)

#Optimized for low-batch size decoding: K needs to be divisible by BLOCK_SIZE_K * SPLIT_K = 256 !!!
def get_default_config():
    # Default
    config = [triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':8, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K': 1,
                            'A_load_order':2, 'meta_evict_policy':'', 'atomic_mode':'relaxed', 'dot_prod_mode':0}, 
                            num_warps=4, num_stages=2,#pre_hook=init_to_zero("c_ptr")
                            ),]

    if(compute_capability == (8, 0)): #A100
        config = [
                triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':8, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K': 1,
                                'A_load_order':0, 'meta_evict_policy':'', 'atomic_mode':'relaxed', 'dot_prod_mode':0}, 
                                num_warps=8, num_stages=4,
                                ),
                ]

    if(compute_capability == (9, 0)): #H100
        config = [
                triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':8, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K': 1,
                                'A_load_order':0, 'meta_evict_policy':'', 'atomic_mode':'relaxed', 'dot_prod_mode':0}, 
                                num_warps=8, num_stages=4,
                                ),
                ]

    return config


ENABLE_AUTOTUNE = AUTOTUNE_ENABLE.GEMV_SPLITK

@triton.autotune(
    configs=get_autotune_config() if ENABLE_AUTOTUNE else get_default_config(),
    key = KEYS,
    prune_configs_by = {'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
    warmup = 50, 
    rep = 50,
    use_cuda_graph = AUTOTUNE_ENABLE.USE_CUDA_GRAPH,
)

@triton.jit
def gemv_splitK_A16fWnO16f_int32packing_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N, K, 
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    ######### Strides #########
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_g, stride_meta_n,
    ######### Dtypes #########
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    ######### Meta-data mode #########
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    A_load_order: tl.constexpr, meta_evict_policy: tl.constexpr, atomic_mode: tl.constexpr, dot_prod_mode: tl.constexpr,
    data_contiguous: tl.constexpr,
    dump_b_val: tl.constexpr = 0, #Improve accuracy mainly for A16W8 with post looop scaling
):
    """
    Based on https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K * SPLIT_K <= group_size for imp1
    BLOCK_SIZE_K == SPLIT_K for imp2 (similar to original)
    """

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    #Swizzle?
    pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    #pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    #Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) 

    #Vectorized coalesced load
    ##############################
    if data_contiguous:
        offs_bn = offs_n  
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k
    offs_bk = offs_k
    ###############################

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)  
    a_mask  = offs_am[:, None] < M
    b_ptrs  = b_ptr + ((offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn)

    #Meta data stuff
    q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]

    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n
    stride_mul  = BLOCK_SIZE_K / group_size

    BLOCK_SIZE_K_U: tl.constexpr = (BLOCK_SIZE_K) * SPLIT_K
    BLOCK_SIZE_K_P: tl.constexpr = (BLOCK_SIZE_K // elements_per_sample) * SPLIT_K

    if(zero_is_scalar):
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')
    ##################################################################

    if(dot_prod_mode == 0):
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=acc_dtype)
    if(dot_prod_mode == 1):
        acc = tl.zeros((1, BLOCK_SIZE_N), dtype=acc_dtype)

    #for k in tl.range(0, num_pid_k, 1, num_stages=1):
    for k in range(num_pid_k):

        if(A_load_order == 0): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') 

        b = tl.load(b_ptrs, eviction_policy='evict_first')

        if(A_load_order == 1): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') 

        if(W_group_mode > 0):
            k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32) 

        if(W_group_mode >= 2): #[2, 3, 4]
            scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            scales = None

        if(W_group_mode == 1 or W_group_mode >= 3): #[1, 3, 4]
            if(zero_is_scalar):
                zeros = zero_scalar
            else:
                zeros = tl.load(zeros_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            zeros = None
        
        if(A_load_order == 2): #Mid load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')

        # Unpack and dequantize
        b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

        if(A_load_order == 3): #Late load 
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')

        if(dump_b_val > 0): b = b.to(tl.float32) * dump_b_val

        if(dot_prod_mode == 0):
            acc += a.reshape((BLOCK_SIZE_K, 1), can_reorder=False).to(acc_dtype) * b.to(acc_dtype)
        if(dot_prod_mode == 1):
            acc += tl.sum(a.reshape((BLOCK_SIZE_K, 1), can_reorder=False) * b.to(input_dtype), axis=0, keep_dims=True).to(acc_dtype) 

        #Advance
        a_ptrs += BLOCK_SIZE_K_U * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * stride_bk

    if(dot_prod_mode == 0):
        acc = tl.sum(acc, axis=0, keep_dims=True) 

    if(dump_b_val > 0): acc /= dump_b_val

    ##################################################################
    #Channel-wise scaling
    if(channel_scale_mode == 1): #weight-only
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * scales_b[None, :]

    if(channel_scale_mode == 2): #activation-only
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if(channel_scale_mode == 3): #weight + activation
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.load(scales_ptr   + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    ##################################################################

    #Output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask    = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if(SPLIT_K == 1):
        tl.store(c_ptrs, acc, mask=mask) 
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode) 


def gemv_splitK_A16fWnO16f_int32packing_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                                W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int,
                                                input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                                                channel_scale_mode: int, W_group_mode: int, data_contiguous: bool,
                                                ) -> Tensor: 
    
    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"

    native_atomic = True 
    #native_atomic = (output_dtype in [DType.FP16.value, DType.FP32.value]) or NATIVE_ATOMIC
    #WARNING: change this to the second check if SPLIT_K > 1 
    
    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype] if native_atomic else torch.float32) 
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META['SPLIT_K'])

    gemv_splitK_A16fWnO16f_int32packing_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K,
        W_nbits, group_size, unpack_mask, elements_per_sample,  
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        ################################################
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = DTYPE_TO_TRITON[output_dtype],
        acc_dtype    = DTYPE_TO_TRITON[acc_dtype],
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        ################################################
        channel_scale_mode = channel_scale_mode,
        W_group_mode       = W_group_mode,
        zero_is_scalar     = zeros.numel() == 1,
        data_contiguous    = data_contiguous,
        dump_b_val         = 0.001 if(W_group_mode in [0, 1] and acc_dtype == DType.FP16.value and W_nbits == 8) else 0, #Warning: Only use with INT8
    )

    if(not native_atomic):
        output = output.to(DTYPE_TO_TORCH[output_dtype])

    return output

class gemv_splitK_A16fWnO16f:
    kernel = gemv_splitK_A16fWnO16f_int32packing_kernel
    forward = gemv_splitK_A16fWnO16f_int32packing_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemv_splitK_A16fWnO16f"]

