# HRWKV-7 RNN-HYBRID Transformer "hxa078"
# 2025 OpenMOSE

import os
import os, torch as th
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "True"
os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "True"

import time
import torch
from collections import defaultdict
#from flash_attn import flash_attn_varlen_func, flash_attn_func

from torch.nn.attention import SDPBackend, sdpa_kernel


import torch._dynamo
torch._dynamo.config.cache_size_limit = 256  # 例えば32に拡張


# try:
#     import bitsandbytes as bnb
#     HAS_BITSANDBYTES = True
# except ImportError:
print('Bitsandbytes not found')
HAS_BITSANDBYTES = False
bnb = None


import torch
import torch.nn as nn
from typing import Optional,List
import types, gc, os, time, re
from torch.nn import functional as F
import numpy as np
import os, sys
import time
import functools
from einops import rearrange
from torch.nn import functional as F
from typing import Callable, Optional, Tuple, Union

from transformers.cache_utils import Cache, DynamicCache

#from rwkvengine.misc import PIPELINE
from rwkvengine.misc import PIPELINE, TimeMixState, ChannelMixState,BlockState,BlockStateList
from rwkvengine.matmularena import hybrid_matmul
from rwkvengine.quantization import fpx_matmul
# from rwkvengine.fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
# from rwkvengine.fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
from rwkvengine.fla.ops.rwkv7 import fused_recurrent_rwkv7
from rwkvengine.cuda.wkv7triton import rwkv7_attn_triton


from rwkvengine.flashattention_triton import triton_attention


MyStatic = torch.jit.script


# DTYPE = torch.bfloat16

# from torch.utils.cpp_extension import load
# HEAD_SIZE = 64
# current_path = os.path.dirname(os.path.abspath(__file__))

# device_props = th.cuda.get_device_properties(th.cuda.current_device())
# if 'AMD' in device_props.name:
#     load(name="rwkv7_state_fwd_fp16", sources=[f"{current_path}/cuda/rwkv7_state_fwd_fp16.cpp", f"{current_path}/cuda/rwkv7_state_fwd_fp16.cu"], is_python_module=False,
#                     verbose=True, extra_cuda_cflags=['-O3', '-ffast-math', '-DAMD', f"-D_N_={HEAD_SIZE}"])
# else:
#     load(name="rwkv7_state_fwd_fp16", sources=[f"{current_path}/cuda/rwkv7_state_fwd_fp16.cpp", f"{current_path}/cuda/rwkv7_state_fwd_fp16.cu"], is_python_module=False,
#                     verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
# class WKV_7(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, state, r, w, k, v, a, b):
#         with torch.no_grad():
#             T, C = r.size()
#             H = C // HEAD_SIZE
#             N = HEAD_SIZE
#             assert HEAD_SIZE == C // H
#             assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
#             assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
#             y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
#             torch.ops.rwkv7_state_fwd_fp16.forward(1, T, C, H, state, r, w, k, v, a, b, y)
#             return y
# def RWKV7_OP(state, r, w, k, v, a, b):
#     return WKV_7.apply(state, r, w, k, v, a, b)

# class WKV_7_batch(torch.autograd.Function):
#     @torch.compiler.disable
#     def forward(ctx, state, r, w, k, v, a, b):
#         with torch.no_grad():
#             B, T, C = r.size()
#             H = C // HEAD_SIZE
#             N = HEAD_SIZE
#             assert HEAD_SIZE == C // H
#             assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
#             assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
#             y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
#             torch.ops.rwkv7_state_fwd_fp16.forward(B, T, C, H, state, r, w, k, v, a, b, y)
#             return y
# def RWKV7_BATCH_OP(state, r, w, k, v, a, b):
#     return WKV_7_batch.apply(state, r, w, k, v, a, b)



@torch.compile
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads along the head dimension (GQA).
    Input:  (B, T, H_kv, D)
    Output: (B, T, H_kv * n_rep, D)
    """
    B, T, H_kv, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Expand head dim
    hidden_states = hidden_states[:, :, :, None, :]  # (B, T, H_kv, 1, D)
    hidden_states = hidden_states.expand(B, T, H_kv, n_rep, D)  # (B, T, H_kv, n_rep, D)
    return hidden_states.reshape(B, T, H_kv * n_rep, D).contiguous()
@torch.compile
def repeat_kv_original(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
@torch.compile
def T5RMSNorm(hidden_states,weight,variance_epsilon:float=1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)
 

def get_batch_rope_cache(cos_cache, sin_cache, batch_pos, seq_len):
    """
    事前計算されたcos/sin cacheから、バッチごとの開始位置に基づいて
    必要な長さ分のcos/sinを取得する
    
    Args:
        cos_cache: [1, max_seq_len, rotary_dim] - 事前計算されたcos値
        sin_cache: [1, max_seq_len, rotary_dim] - 事前計算されたsin値
        batch_pos: [B, 1] - 各バッチの開始位置
        seq_len: int - 処理する系列長 T
    
    Returns:
        cos: [B, T, rotary_dim] - バッチごとのcos値
        sin: [B, T, rotary_dim] - バッチごとのsin値
    """
    B = batch_pos.shape[0]
    rotary_dim = cos_cache.shape[-1]
    device = cos_cache.device
    dtype = cos_cache.dtype
    
    # 各バッチの位置インデックスを作成 [B, T]
    positions = batch_pos + torch.arange(seq_len, device=device, dtype=torch.long)
    
    # cos_cache と sin_cache から必要な部分を取得
    # cos_cache/sin_cache の shape は [1, max_seq_len, rotary_dim]
    # positions を使ってインデックシング
    cos_cache_squeezed = cos_cache.squeeze(0)  # [max_seq_len, rotary_dim]
    sin_cache_squeezed = sin_cache.squeeze(0)  # [max_seq_len, rotary_dim]
    
    # バッチごとに異なる位置から取得
    cos = cos_cache_squeezed[positions]  # [B, T, rotary_dim]
    sin = sin_cache_squeezed[positions]  # [B, T, rotary_dim]
    
    return cos, sin

def compute_qwen3_rope_cache(seq_len, rotary_dim, device, dtype, rope_theta):
            half_dim = rotary_dim // 2
            freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
            inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))
            positions = torch.arange(seq_len, dtype=dtype, device=device)
            freqs = torch.einsum("i,j->ij", positions, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos.unsqueeze(0), sin.unsqueeze(0), inv_freq

def compute_rope_cache_range(cache_pos, seq_len, rotary_dim, device, dtype, rope_theta):
    """
    RoPEのcos/sinをバッチごとに cache_pos から始まる Tトークン分生成

    Args:
        cache_pos: [B, 1]  - 各バッチの開始位置
        seq_len: int       - Tステップ分のRoPE計算を行う
        rotary_dim: int    - 回転に使う次元数（通常は Head_dim）
        rope_theta: float  - RoPEのthetaパラメータ（例: 10000.0）

    Returns:
        cos: [B, T, rotary_dim]
        sin: [B, T, rotary_dim]
        inv_freq: [half_dim]  # optional
    """
    B = cache_pos.shape[0]
    half_dim = rotary_dim // 2

    # [half_dim]
    freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))  # [half_dim]

    # [T]
    rel_pos = torch.arange(seq_len, dtype=dtype, device=device)  # 相対位置 0〜T-1

    # [B, T] = cache_pos + rel_pos
    pos = cache_pos.to(dtype) + rel_pos.view(1, -1)  # ブロードキャスト加算

    # [B, T, half_dim] = pos[b, t] × inv_freq[j]
    freqs = torch.einsum("bt,d->btd", pos, inv_freq)

    # rotary次元を2倍化：[B, T, rotary_dim]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin, inv_freq



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


 


def sdpa_attention_forward(
    #module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
    #     logger.warning_once(
    #         "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
    #         " Please set your attention to `eager` if you want any of these features."
    #     )

    # if hasattr(module, "num_key_value_groups"):
    #     key = repeat_kv_original(key, module.num_key_value_groups)
    #     value = repeat_kv_original(value, module.num_key_value_groups)

    # if attention_mask is not None and attention_mask.ndim == 4:
    #     attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    # query = query.contiguous()
    # key = key.contiguous()
    # value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    # if is_causal is None:
    #     # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
    #     # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
    #     is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    # if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
    #     is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


class HRWKV_7(nn.Module):
    @torch.compile()
    def hxa079_TimeMix(layer_id: int, H: int, N: int,
                        x_in, x_prev, v_first,k_first,state,cache_position,
                        calc_cos,calc_sin,
                        # w0, w1, w2, a0, a1, a2,
                        # v0, v1, v2, g1, g2,
                        # k0, k1, k2,
                        wavgk1,wavgk_split_list,
                        w0, w2, a0, a2,
                        v0, v2, g2,
                        k0, k2,
                        r_k, 
                        #R_, K_, V_,
                        RKV_,rkv_split_list,
                        O_,
                        #R_state,K_state,V_state,
                        RKV_state,
                        O_state,
                        R_bias, K_bias, V_bias, O_bias,
                        ln_r,ln_k,rmsnorm_epsilon:float,
                        ln1,ln2,rope_theta,
                        ebits:int, mbits:int
                        ):
        
        #idx = idx.clone()
        #state = state.clone() if state is not None else None
        
        x = T5RMSNorm(x_in,ln1,variance_epsilon=rmsnorm_epsilon)
        xw,xa,xv,xg,xk = (x @ wavgk1).split(wavgk_split_list, dim=-1)

        B, T, HN = x.shape

        rkv = fpx_matmul(x, RKV_,RKV_state, ebits, mbits)
        r,k,v = rkv.split(rkv_split_list,dim=-1)

        kv_dim = k.shape[2] # B,T,kv_dim
        kv_repeat = HN // kv_dim
        kv_per_head = kv_dim // N

        r = r.add_(R_bias)
        k = k.add_(K_bias).view(B, T, kv_per_head, N)
        v = v.add_(V_bias).view(B, T, kv_per_head, N)

        if ln_r == None:
            r = r.view(B,T,H,N)
            k = k
        else:
            r = T5RMSNorm(r.view(B,T,H,N), ln_r, variance_epsilon=rmsnorm_epsilon)
            k = T5RMSNorm(k, ln_k, variance_epsilon=rmsnorm_epsilon)

        # Rotary embedding
        cos, sin = calc_cos.to(k.dtype), calc_sin.to(k.dtype)
        r, k = apply_rotary_pos_emb(r, k, cos, sin,unsqueeze_dim=2)

        if layer_id == 0:
            v_first = v
            k_first = k 
        else:

            v = v + (v_first - v) * torch.sigmoid(xv@v2 + v0).view(B, T, kv_per_head, N)
            k = k + (k_first - k) * torch.sigmoid(xk@k2 + k0).view(B, T, kv_per_head, N)

 

        k = repeat_kv(k, kv_repeat).view(B, T, -1)
        v = repeat_kv(v, kv_repeat).view(B, T, -1)

  
        a = torch.sigmoid(xa @ a2 + a0)
        
        kk = F.normalize(k.view(B, T, H, N), p=2.0, dim=-1).view(B, T, H*N)
        w = torch.tanh(xw) @ w2 + w0
        w = -F.softplus(-w) - 0.5
        k = k * (1.0 - w + a)

        aa = -kk
        bb = kk * a

        w = -w.exp()#.to(dtype=torch.float32).exp()
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,N) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)

        
        xx = xx.view(B,T,-1) * (float(N) ** -0.5)

        xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,N)*k.view(B,T,H,N)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,HN)

        output = fpx_matmul((xx * (torch.sigmoid(xg) @ g2)), O_, O_state,ebits,mbits)

        x_in = x_in + output
        output = T5RMSNorm(x_in,ln2,variance_epsilon=rmsnorm_epsilon)

        return  output, x[:,-1], state, v_first, k_first, x_in
    

    #@torch.compile()
    @torch.compile(mode="max-autotune-no-cudagraphs")
    def hxa07A_TimeMix(layer_id: int, H: int, N: int,
                        x_in, x_prev,state,cache_position,
                     
                        wag1,wag_split_list,
                        w0, w2, a0, a2,
                        g2,
                        r_k, 
                        RKV_,rkv_split_list,
                        O_,
                        RKV_state,
                        O_state,
                        rmsnorm_epsilon:float,
                        ln1,ln2,
                        ebits:int, mbits:int
                        ):
        

        x = T5RMSNorm(x_in,ln1,variance_epsilon=rmsnorm_epsilon)
        xw,xa,xg = (x @ wag1).split(wag_split_list, dim=-1)

        B, T, C = x.shape

        rkv = fpx_matmul(x, RKV_,RKV_state, ebits, mbits)
        r,k,v = rkv.split(rkv_split_list,dim=-1)

        kv_dim = k.shape[2] # B,T,kv_dim
        kv_repeat = (H*N) // kv_dim
        kv_per_head = kv_dim // N
     

        k = repeat_kv(k.view(B, T, kv_per_head, N), kv_repeat).view(B, T, -1)#.contiguous()
        v = repeat_kv(v.view(B, T, kv_per_head, N), kv_repeat).view(B, T, -1)#.contiguous()

        a = torch.sigmoid(xa @ a2 + a0)
        
        kk = F.normalize(k.view(B, T, H, N), p=2.0, dim=-1).view(B, T, H*N)
        w = torch.tanh(xw) @ w2 + w0
        w = -F.softplus(-w) - 0.5
        k = k * (1.0 - w + a)

        aa = -kk
        bb = kk * a
        w = -w.exp()
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,N).to(dtype=torch.bfloat16) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        xx = xx.to(dtype=x.dtype)
        
        xx = xx.view(B,T,-1) * (float(N) ** -0.5)
        xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,N)*k.view(B,T,H,N)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,(H*N))

        #print(f'Layer:{layer_id} r={r.sum()} k={k.sum()} v={v.sum()} w={w.sum()} gatedxx = {gatedxx.sum()} x_in ={x_in.sum()} x = {x.sum()} x = {x}')

        output = fpx_matmul((xx * (torch.sigmoid(xg) @ g2)), O_, O_state,ebits,mbits)

        x_in = x_in + output
        output = T5RMSNorm(x_in,ln2,variance_epsilon=rmsnorm_epsilon)

        return  output, x[:,-1], state, x_in
    
  


    # @torch.compile
    # def hxa079_TimeMix_x(layer_id: int, H: int, N: int,
    #                     x_in, x_prev, v_first,k_first,state,cache_position,
    #                     calc_cos,calc_sin,
    #                     # w0, w1, w2, a0, a1, a2,
    #                     # v0, v1, v2, g1, g2,
    #                     # k0, k1, k2,
    #                     wavgk1,wavgk_split_list,
    #                     w0, w2, a0, a2,
    #                     v0, v2, g2,
    #                     k0, k2,
    #                     r_k, 
    #                     #R_, K_, V_,
    #                     RKV_,rkv_split_list,
    #                     O_,
    #                     #R_state,K_state,V_state,
    #                     RKV_state,
    #                     O_state,
    #                     R_bias, K_bias, V_bias, O_bias,
    #                     ln_r,ln_k,rmsnorm_epsilon:float,
    #                     ln1,ln2,rope_theta,
    #                     ebits:int, mbits:int
    #                     ):
        
    #     x = T5RMSNorm(x_in,ln1,variance_epsilon=rmsnorm_epsilon)
    #     xw,xa,xv,xg,xk = (x @ wavgk1).split(wavgk_split_list, dim=-1)

    #     B, T, C = x.shape
    #     HN = H*N

    #     rkv = fpx_matmul(x, RKV_,RKV_state, ebits, mbits)
    #     r,k,v = rkv.split(rkv_split_list,dim=-1)

    #     kv_dim = k.shape[2] # B,T,kv_dim
    #     kv_repeat = HN // kv_dim
    #     kv_per_head = kv_dim // N

    #     r = r.add_(R_bias)
    #     k = k.add_(K_bias).view(B, T, kv_per_head, N)
    #     v = v.add_(V_bias).view(B, T, kv_per_head, N)

    #     if ln_r == None:
    #         r = r.view(B,T,H,N)
    #         k = k
    #     else:
    #         r = T5RMSNorm(r.view(B,T,H,N), ln_r, variance_epsilon=rmsnorm_epsilon)
    #         k = T5RMSNorm(k, ln_k, variance_epsilon=rmsnorm_epsilon)

    #     # Rotary embedding
    #     cos, sin = calc_cos.to(k.dtype), calc_sin.to(k.dtype)
    #     r, k = apply_rotary_pos_emb(r, k, cos, sin,unsqueeze_dim=2)

    #     if layer_id == 0:
    #         v_first = v
    #         k_first = k 
    #     else:
    #         # gate = torch.sigmoid(xv@v2 + v0).view(B, T, kv_per_head, N)
    #         # v = v + (v_first - v) * gate
    #         # gate2 = torch.sigmoid(xk@k2 + k0).view(B, T, kv_per_head, N)
    #         # k = k + (k_first - k) * gate2
    #         v = v + (v_first - v) * torch.sigmoid(xv@v2 + v0).view(B, T, kv_per_head, N)
    #         k = k + (k_first - k) * torch.sigmoid(xk@k2 + k0).view(B, T, kv_per_head, N)

 

    #     k = repeat_kv(k, kv_repeat).view(B, T, -1)
    #     v = repeat_kv(v, kv_repeat).view(B, T, -1)

  
    #     a = torch.sigmoid(xa @ a2 + a0)
        
    #     kk = F.normalize(k.view(B, T, H, N), p=2.0, dim=-1).view(B, T, H*N)
    #     w = torch.tanh(xw) @ w2 + w0
    #     w = -F.softplus(-w) - 0.5
    #     k = k * (1.0 - w + a)

    #     aa = -kk
    #     bb = kk * a

    #     w = -w.to(dtype=torch.float32).exp()
    #     r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,N) for i in [r,w,k,v,aa,bb]]
    #     B,T,_,_ = r_.shape
    #     xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)

        
    #     xx = xx.view(B,T,-1) * (float(N) ** -0.5)

    #     xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,N)*k.view(B,T,H,N)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,HN)

    #     output = fpx_matmul((xx * (torch.sigmoid(xg) @ g2)), O_, O_state,ebits,mbits)

    #     x_in = x_in + output
    #     output = T5RMSNorm(x_in,ln2,variance_epsilon=rmsnorm_epsilon)

    #     return  output, x[:,-1], state, v_first, k_first, x_in


    @torch.compile
    def GQA_Attention_Nested_(layer_id: int, gqa_layer_id: int, H: int, N: int,
                        x_in, past_key_value, cache_position,
                        calc_cos, calc_sin,
                        #Q_, K_, V_,
                        QKV_,qkv_split_list,
                        O_,
                        #Q_state, K_state, V_state, 
                        QKV_state,
                        O_state,
                        Q_bias, K_bias, V_bias, O_bias,
                        ln_r, ln_k, rmsnorm_epsilon: float,
                        ln1, ln2, rope_theta,
                        ebits: int, mbits: int):

        B, T, C = x_in.size()
        x = T5RMSNorm(x_in, ln1, rmsnorm_epsilon)

        HN = H * N


        qkv = fpx_matmul(x, QKV_,QKV_state, ebits, mbits)
        q,k,v = qkv.split(qkv_split_list,dim=-1)

        q = q.add_(Q_bias).view(B, T, H, N)
        k = k.add_(K_bias)
        v = v.add_(V_bias)

        kv_dim = k.shape[2]  # B,T,kv_dim
        kv_per_head = kv_dim // N
        kv_repeat = HN // kv_dim

        k = k.view(B, T, kv_per_head, N)
        v = v.view(B, T, kv_per_head, N)

        if ln_r is not None:
            q = T5RMSNorm(q, ln_r, rmsnorm_epsilon)
            k = T5RMSNorm(k, ln_k, rmsnorm_epsilon)


        cache_shape = past_key_value.shape
        
        k_write = k.view(B, T, -1)
        v_write = v.view(B, T, -1)
        starts = cache_position[:, 0]
        
        #if is_new_layout:
        if B == 1 or (starts == starts[0]).all():
            start = starts[0].int()
            end = start + T
            past_key_value[gqa_layer_id, :, 0, start:end] = k_write
            past_key_value[gqa_layer_id, :, 1, start:end] = v_write
        else:
            batch_idx = torch.arange(B, device=k.device)[:, None]
            seq_idx = starts[:, None] + torch.arange(T, device=k.device)
            past_key_value[gqa_layer_id, batch_idx, 0, seq_idx] = k_write
            past_key_value[gqa_layer_id, batch_idx, 1, seq_idx] = v_write
  
        kv_heads = kv_dim // N
        
        # Get valid lengths for each batch item
        valid_lengths = cache_position[:, 0] + T  # Current position + new tokens
        
        # Create lists for K and V tensors per batch item
        k_list = []
        v_list = []
        q_list = []
        
        for b in range(B):
            valid_len = valid_lengths[b].int()
            
            #if is_new_layout:
            k_valid = past_key_value[gqa_layer_id, b, 0, :valid_len]
            v_valid = past_key_value[gqa_layer_id, b, 1, :valid_len]
            # else:
            #     k_valid = past_key_value[gqa_layer_id, b, :valid_len, 0]
            #     v_valid = past_key_value[gqa_layer_id, b, :valid_len, 1]
            
            # Reshape to (seq_len, kv_heads, N)
            k_valid = k_valid.view(valid_len, kv_heads, N)
            v_valid = v_valid.view(valid_len, kv_heads, N)
            
            # Repeat KV if needed (GQA)
            if kv_repeat > 1:
                k_valid = k_valid.repeat_interleave(kv_repeat, dim=1)  # (seq_len, H, N)
                v_valid = v_valid.repeat_interleave(kv_repeat, dim=1)
            
            # Transpose to (kv_heads, seq_len, N) for SDPA
            k_valid = k_valid.transpose(0, 1)  # (H, seq_len, N)
            v_valid = v_valid.transpose(0, 1)
            
            k_list.append(k_valid)
            v_list.append(v_valid)
            
            # Also prepare query for this batch item
            q_b = q[b].transpose(0, 1)  # (H, T, N)
            q_list.append(q_b)

        # Create Nested Tensors
        try:
            # Try using nested tensors if available
            q_nested = torch.nested.nested_tensor(q_list, dtype=q.dtype, device=q.device)
            k_nested = torch.nested.nested_tensor(k_list, dtype=k.dtype, device=k.device) 
            v_nested = torch.nested.nested_tensor(v_list, dtype=v.dtype, device=v.device)
            
            # Use SDPA with nested tensors (no mask needed, causal=True)
            #with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output_nested = F.scaled_dot_product_attention(
                    q_nested, k_nested, v_nested, 
                    attn_mask=None, 
                    is_causal=True
                )
            
            # Convert back to regular tensor
            attn_output_list = attn_output_nested.unbind()
            attn_output = torch.stack([
                out.transpose(0, 1) for out in attn_output_list  # (T, H, N)
            ], dim=0)  # (B, T, H, N)
            
        except (AttributeError, RuntimeError):
            #print('fallback :(')
            # Fallback: Use regular tensors with manual padding
            # This handles cases where nested tensors aren't fully supported
            max_len = max(valid_lengths).int()
            
            # Pad all tensors to max length
            k_padded = torch.zeros(B, kv_heads * kv_repeat, max_len, N, 
                                dtype=k.dtype, device=k.device)
            v_padded = torch.zeros(B, kv_heads * kv_repeat, max_len, N, 
                                dtype=v.dtype, device=v.device)
            
            for b in range(B):
                valid_len = valid_lengths[b].int()
                k_padded[b, :, :valid_len] = k_list[b]
                v_padded[b, :, :valid_len] = v_list[b]
            
            # Create causal mask for padded version
            causal_mask = torch.triu(
                torch.full((T, max_len), float('-inf'), dtype=q.dtype, device=q.device),
                diagonal=1
            )
            
            # Add padding mask
            for b in range(B):
                valid_len = valid_lengths[b].int()
                if valid_len < max_len:
                    causal_mask[:, valid_len:] = float('-inf')
            
            attn_output = F.scaled_dot_product_attention(
                q.transpose(1, 2), k_padded, v_padded,
                attn_mask=causal_mask,
                is_causal=False  # We're handling causality with explicit mask
            )
            attn_output = attn_output.transpose(1, 2)

        # Reshape and project output
        attn_output = attn_output.contiguous().view(B, T, HN)
        out = fpx_matmul(attn_output, O_, O_state, ebits, mbits)
        x_out = x_in + out

        return T5RMSNorm(x_out, ln2, rmsnorm_epsilon), x_out, past_key_value

    

    @torch.compile()
    def GQA_Attention_Nested(layer_id: int, gqa_layer_id: int, H: int, N: int,
                        x_in, past_key_value, cache_position,
                        calc_cos, calc_sin,
                        QKV_, qkv_split_list,
                        O_,
                        QKV_state,
                        O_state,
                        Q_bias, K_bias, V_bias, O_bias,
                        ln_r, ln_k, rmsnorm_epsilon: float,
                        ln1, ln2, rope_theta,
                        ebits: int, mbits: int):
    
        B, T, C = x_in.size()
        x = T5RMSNorm(x_in, ln1, rmsnorm_epsilon)
        
        HN = H * N
        
        # QKV projection
        qkv = fpx_matmul(x, QKV_, QKV_state, ebits, mbits)
        q, k, v = qkv.split(qkv_split_list, dim=-1)
        
        # Add bias and reshape
        q = q.add_(Q_bias).view(B, T, H, N)
        k = k.add_(K_bias)
        v = v.add_(V_bias)
        
        # KV dimensions for GQA
        kv_dim = k.shape[2]
        kv_heads = kv_dim // N
        kv_repeat = HN // kv_dim
        
        k = k.view(B, T, kv_heads, N)
        v = v.view(B, T, kv_heads, N)
        
        # Optional RMSNorm
        if ln_r is not None:
            q = T5RMSNorm(q, ln_r, rmsnorm_epsilon)
            k = T5RMSNorm(k, ln_k, rmsnorm_epsilon)
        
        # Write to KV cache
        k_write = k.view(B, T, -1)
        v_write = v.view(B, T, -1)
        starts = cache_position[:, 0]
        
        # Simple batched write to cache
        batch_idx = torch.arange(B, device=k.device)[:, None]
        seq_idx = starts[:, None] + torch.arange(T, device=k.device)
        past_key_value[gqa_layer_id, batch_idx, 0, seq_idx] = k_write
        past_key_value[gqa_layer_id, batch_idx, 1, seq_idx] = v_write
        
        # Get valid lengths for each batch
        valid_lengths = cache_position[:, 0] + T
        
        # Create lists for building nested tensors
        q_list = []
        k_list = []
        v_list = []
        
        # Process each batch item
        for b in range(B):
            valid_len = valid_lengths[b].int()
            
            # Extract valid K and V from cache
            k_valid = past_key_value[gqa_layer_id, b, 0, :valid_len]
            v_valid = past_key_value[gqa_layer_id, b, 1, :valid_len]
            
            # Reshape to (seq_len, kv_heads, N)
            k_valid = k_valid.view(valid_len, kv_heads, N)
            v_valid = v_valid.view(valid_len, kv_heads, N)
            
            # Repeat KV if needed (GQA)
            if kv_repeat > 1:
                k_valid = k_valid.repeat_interleave(kv_repeat, dim=1)  # (seq_len, H, N)
                v_valid = v_valid.repeat_interleave(kv_repeat, dim=1)
            
            # Transpose to (H, seq_len, N) for SDPA
            k_valid = k_valid.transpose(0, 1)  # (H, seq_len, N)
            v_valid = v_valid.transpose(0, 1)
            
            k_list.append(k_valid)
            v_list.append(v_valid)
            
            # Prepare query for this batch item
            q_b = q[b].transpose(0, 1)  # (H, T, N)
            q_list.append(q_b)
        
        # Create Nested Tensors with jagged layout
        q_nested = torch.nested.nested_tensor(q_list, dtype=q.dtype, device=q.device, layout=torch.jagged)
        k_nested = torch.nested.nested_tensor(k_list, dtype=k.dtype, device=k.device, layout=torch.jagged)
        v_nested = torch.nested.nested_tensor(v_list, dtype=v.dtype, device=v.device, layout=torch.jagged)
        
        # Apply scaled dot product attention
        attn_output_nested = F.scaled_dot_product_attention(
            q_nested, k_nested, v_nested,
            attn_mask=None,
            is_causal=True
        )
        
        # Convert back to regular tensor
        attn_output_list = attn_output_nested.unbind()
        attn_output = torch.stack([
            out.transpose(0, 1) for out in attn_output_list  # (T, H, N)
        ], dim=0)  # (B, T, H, N)
        
        # Reshape and output projection
        attn_output = attn_output.contiguous().view(B, T, HN)
        out = fpx_matmul(attn_output, O_, O_state, ebits, mbits)
        x_out = x_in + out
        
        return T5RMSNorm(x_out, ln2, rmsnorm_epsilon), x_out, past_key_value
    #@torch.compile()
    #@torch.compile
    #@torch.compiler.disable
    def triton_attention_wrapper(
            q_flash,
            k_flash,
            v_flash,
            o_flash,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,  # causal
            sm_scale,
            bias,  # bias

        ):
        return triton_attention(
            q_flash,
            k_flash,
            v_flash,
            o_flash,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,  # causal
            sm_scale,
            bias,  # bias
        )


    #@torch.compile#r.disable
    #@torch.compile(mode="max-autotune-no-cudagraphs")
    @torch.compiler.disable
    def GQA_Attention_Flash_(layer_id: int, gqa_layer_id: int, H: int, N: int,
                       x_in, past_key_value, cache_position,
                       calc_cos, calc_sin,
                       QKV_, qkv_split_list,
                       O_,
                       QKV_state,
                       O_state,
                       Q_bias, K_bias, V_bias, O_bias,
                       ln_r, ln_k, rmsnorm_epsilon: float,
                       ln1, ln2, rope_theta,
                       ebits: int, mbits: int):
        """
        GQA Attention using Flash Attention implementation via Triton with KV cache size management.
        
        This function replaces the PyTorch SDPA with the custom Flash Attention kernel
        and manages KV cache size by evicting old entries when approaching the limit.
        The max cache size is automatically determined from the KV cache tensor dimensions.
        """
        
        B, T, C = x_in.size()
        
        # Get max cache size from KV cache tensor dimensions
        # past_key_value shape: [num_layers, batch_size, 2, max_seq_len, hidden_dim]
        max_cache_size = past_key_value.shape[3]
        
        x = T5RMSNorm(x_in, ln1, rmsnorm_epsilon)
        
        HN = H * N
        
        # QKV projection
        qkv = fpx_matmul(x, QKV_, QKV_state, ebits, mbits)
        q, k, v = qkv.split(qkv_split_list, dim=-1)
        
        # Add bias and reshape
        q = q.add_(Q_bias).view(B, T, H, N)
        k = k.add_(K_bias)
        v = v.add_(V_bias)
        
        # KV dimensions for GQA
        kv_dim = k.shape[2]
        kv_heads = kv_dim // N
        kv_repeat = HN // kv_dim
        
        k = k.view(B, T, kv_heads, N)
        v = v.view(B, T, kv_heads, N)
        
        # Optional RMSNorm
        if ln_r is not None:
            q = T5RMSNorm(q, ln_r, rmsnorm_epsilon)
            k = T5RMSNorm(k, ln_k, rmsnorm_epsilon)
        
        # Write to KV cache with size management
        k_write = k.view(B, T, -1)
        v_write = v.view(B, T, -1)
        starts = cache_position[:, 0]
        
        # Check which batches need cache eviction
        new_lengths = starts + T
        batches_to_evict = new_lengths > max_cache_size
        
        # For batches that would exceed max_cache_size, perform eviction
        if batches_to_evict.any():
            # Calculate how many tokens to evict for each overflowing batch
            excess_tokens = torch.clamp(new_lengths - max_cache_size, min=0)
            
            for b in range(B):
                if batches_to_evict[b]:
                    evict_count = excess_tokens[b].item()
                    if evict_count > 0:
                        # Shift the cache to remove old tokens (FIFO eviction)
                        current_len = starts[b].item()
                        remaining_len = current_len - evict_count
                        
                        if remaining_len > 0:
                            # Clone the data to avoid memory overlap issues
                            k_data = past_key_value[gqa_layer_id, b, 0, evict_count:current_len].clone()
                            v_data = past_key_value[gqa_layer_id, b, 1, evict_count:current_len].clone()
                            
                            # Clear the cache for this batch
                            past_key_value[gqa_layer_id, b, 0, :current_len].zero_()
                            past_key_value[gqa_layer_id, b, 1, :current_len].zero_()
                            
                            # Write the shifted data back
                            past_key_value[gqa_layer_id, b, 0, :remaining_len] = k_data
                            past_key_value[gqa_layer_id, b, 1, :remaining_len] = v_data
                        else:
                            # If no tokens remain, clear the entire cache for this batch
                            past_key_value[gqa_layer_id, b, 0, :current_len].zero_()
                            past_key_value[gqa_layer_id, b, 1, :current_len].zero_()
                        
                        # Update the start position for this batch
                        starts[b] = remaining_len
        
        # Write new K, V to cache (after potential eviction)
        batch_idx = torch.arange(B, device=k.device)[:, None]
        seq_idx = starts[:, None] + torch.arange(T, device=k.device)
        
        # Ensure we don't exceed cache bounds
        valid_seq_idx = torch.clamp(seq_idx, 0, past_key_value.shape[3] - 1)
        
        past_key_value[gqa_layer_id, batch_idx, 0, valid_seq_idx] = k_write
        past_key_value[gqa_layer_id, batch_idx, 1, valid_seq_idx] = v_write
        
        # Get valid lengths for each batch (capped at max_cache_size)
        valid_lengths = torch.clamp(starts + T, max=max_cache_size)
        
        # Prepare for Flash Attention
        # For variable length sequences, we need cu_seqlens format
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        
        # Build cumulative sequence lengths
        for i in range(B):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + T
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + valid_lengths[i]
        
        # Prepare tensors for Flash Attention
        # Query: reshape to (total_q_tokens, H, N)
        q_flash = q.view(B * T, H, N)
        
        # Key and Value: gather all valid tokens from cache
        k_all_list = []
        v_all_list = []
        
        for b in range(B):
            valid_len = valid_lengths[b].int()
            
            # Extract valid K and V from cache
            k_valid = past_key_value[gqa_layer_id, b, 0, :valid_len]
            v_valid = past_key_value[gqa_layer_id, b, 1, :valid_len]
            
            # Reshape to (seq_len, kv_heads, N)
            k_valid = k_valid.view(valid_len, kv_heads, N)
            v_valid = v_valid.view(valid_len, kv_heads, N)
            
            k_all_list.append(k_valid)
            v_all_list.append(v_valid)
        
        # Concatenate all K and V
        k_flash = torch.cat(k_all_list, dim=0)  # (total_k_tokens, kv_heads, N)
        v_flash = torch.cat(v_all_list, dim=0)  # (total_v_tokens, kv_heads, N)
        
        # Calculate max sequence lengths
        max_seqlen_q = T
        max_seqlen_k = int(valid_lengths.max().item())
        
        # Prepare output tensor
        o_flash = torch.empty_like(q_flash)
        
        # Set up scaling
        sm_scale = 1.0 / (N ** 0.5)
        
        # Call Flash Attention
        # Note: The triton_attention function expects FP8 scale parameters
        # For non-FP8 case, we pass None
        fp8_scales = None
        fp8_out_scale = None
        
        # Handle GQA by ensuring K/V are properly shaped
        # If we have fewer KV heads than Q heads, Flash Attention handles this internally
        
        # Import the Flash Attention implementation
        #from flash_attn_triton import triton_attention
        
        # Call the Flash Attention kernel
        attn_output, _ = triton_attention(
            q_flash,
            k_flash,
            v_flash,
            o_flash,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            True,  # causal
            sm_scale,
            None,  # bias
            # fp8_scales,
            # fp8_out_scale,
        )
        
        # Reshape output back to (B, T, H, N)
        #heattn_output = attn_output.view(B, T, H, N)
        
        # For GQA, if we had repeated KV heads, the output already accounts for this
        # Reshape and output projection
        attn_output = attn_output.contiguous().view(B, T, HN)
        out = fpx_matmul(attn_output, O_, O_state, ebits, mbits)
        x_out = x_in + out
        
        # Update cache_position to reflect any eviction that occurred
        cache_position[:, 0] = starts
        
        return T5RMSNorm(x_out, ln2, rmsnorm_epsilon), x_out, past_key_value



    # def GQA_Attention_Flash__(layer_id: int, gqa_layer_id: int, H: int, N: int,
    #                    x_in, past_key_value, cache_position,
    #                    calc_cos, calc_sin,
    #                    QKV_, qkv_split_list,
    #                    O_,
    #                    QKV_state,
    #                    O_state,
    #                    Q_bias, K_bias, V_bias, O_bias,
    #                    ln_r, ln_k, rmsnorm_epsilon: float,
    #                    ln1, ln2, rope_theta,
    #                    ebits: int, mbits: int):
    #     """
    #     GQA Attention using Flash Attention implementation via Triton.
        
    #     This function replaces the PyTorch SDPA with the custom Flash Attention kernel.
    #     """
        
    #     B, T, C = x_in.size()
    #     x = T5RMSNorm(x_in, ln1, rmsnorm_epsilon)
        
    #     HN = H * N
        
    #     # QKV projection
    #     qkv = fpx_matmul(x, QKV_, QKV_state, ebits, mbits)
    #     q, k, v = qkv.split(qkv_split_list, dim=-1)
        
    #     # Add bias and reshape
    #     q = q.add_(Q_bias).view(B, T, H, N)
    #     k = k.add_(K_bias)
    #     v = v.add_(V_bias)
        
    #     # KV dimensions for GQA
    #     kv_dim = k.shape[2]
    #     kv_heads = kv_dim // N
    #     kv_repeat = HN // kv_dim
        
    #     k = k.view(B, T, kv_heads, N)
    #     v = v.view(B, T, kv_heads, N)
        
    #     # Optional RMSNorm
    #     if ln_r is not None:
    #         q = T5RMSNorm(q, ln_r, rmsnorm_epsilon)
    #         k = T5RMSNorm(k, ln_k, rmsnorm_epsilon)
        
    #     # Write to KV cache
    #     k_write = k.view(B, T, -1)
    #     v_write = v.view(B, T, -1)
    #     starts = cache_position[:, 0]
        
    #     # Simple batched write to cache
    #     batch_idx = torch.arange(B, device=k.device)[:, None]
    #     seq_idx = starts[:, None] + torch.arange(T, device=k.device)
    #     past_key_value[gqa_layer_id, batch_idx, 0, seq_idx] = k_write
    #     past_key_value[gqa_layer_id, batch_idx, 1, seq_idx] = v_write
        
    #     # Get valid lengths for each batch
    #     valid_lengths = cache_position[:, 0] + T
        
    #     # Prepare for Flash Attention
    #     # For variable length sequences, we need cu_seqlens format
    #     cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    #     cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        
    #     # Build cumulative sequence lengths
    #     for i in range(B):
    #         cu_seqlens_q[i + 1] = cu_seqlens_q[i] + T
    #         cu_seqlens_k[i + 1] = cu_seqlens_k[i] + valid_lengths[i]
        
    #     # Prepare tensors for Flash Attention
    #     # Query: reshape to (total_q_tokens, H, N)
    #     q_flash = q.reshape(B * T, H, N)
        
    #     # Key and Value: gather all valid tokens from cache
    #     k_all_list = []
    #     v_all_list = []
        
    #     for b in range(B):
    #         valid_len = valid_lengths[b].int()
            
    #         # Extract valid K and V from cache
    #         k_valid = past_key_value[gqa_layer_id, b, 0, :valid_len]
    #         v_valid = past_key_value[gqa_layer_id, b, 1, :valid_len]
            
    #         # Reshape to (seq_len, kv_heads, N)
    #         k_valid = k_valid.view(valid_len, kv_heads, N)
    #         v_valid = v_valid.view(valid_len, kv_heads, N)
            
    #         k_all_list.append(k_valid)
    #         v_all_list.append(v_valid)
        
    #     # Concatenate all K and V
    #     k_flash = torch.cat(k_all_list, dim=0)  # (total_k_tokens, kv_heads, N)
    #     v_flash = torch.cat(v_all_list, dim=0)  # (total_v_tokens, kv_heads, N)
        
    #     # Calculate max sequence lengths
    #     max_seqlen_q = T
    #     max_seqlen_k = int(valid_lengths.max().item())
        
    #     # Prepare output tensor
    #     o_flash = torch.empty_like(q_flash)
        
    #     # Set up scaling
    #     sm_scale = 1.0 / (N ** 0.5)
        
    #     # Call Flash Attention
    #     # Note: The triton_attention function expects FP8 scale parameters
    #     # For non-FP8 case, we pass None
    #     fp8_scales = None
    #     fp8_out_scale = None
        
    #     # Handle GQA by ensuring K/V are properly shaped
    #     # If we have fewer KV heads than Q heads, Flash Attention handles this internally
        
    #     # Import the Flash Attention implementation
    #     #from flash_attn_triton import triton_attention
        
    #     # Call the Flash Attention kernel
    #     attn_output, _ = triton_attention(
    #         q_flash,
    #     k_flash,
    #     v_flash,
    #     o_flash,
    #     cu_seqlens_q,
    #     cu_seqlens_k,
    #     max_seqlen_q,
    #     max_seqlen_k,
    #     True,  # causal
    #     sm_scale,
    #     None,  # bias
    #     # fp8_scales,
    #     # fp8_out_scale,
    #     )
        
    #     # Reshape output back to (B, T, H, N)
    #     attn_output = attn_output.view(B, T, H, N)
        
    #     # For GQA, if we had repeated KV heads, the output already accounts for this
    #     # Reshape and output projection
    #     attn_output = attn_output.contiguous().view(B, T, HN)
    #     out = fpx_matmul(attn_output, O_, O_state, ebits, mbits)
    #     x_out = x_in + out
        
    #     return T5RMSNorm(x_out, ln2, rmsnorm_epsilon), x_out, past_key_value

    




    #from flash_attn import flash_attn_varlen_func, flash_attn_func
    import torch
    import torch.nn.functional as F





    
    # def SwiGLU_MLP_forward_fpx_w_add(x,gate_,down_,up_,gate_state,down_state,up_state,ebits,mbits,x_in):
    #     step1 = F.silu(fpx_matmul(x,gate_,gate_state,ebits,mbits)) * fpx_matmul(x,up_,up_state,ebits,mbits)
    #     xx = fpx_matmul(step1,down_,down_state,ebits,mbits)
    #     x_in = x_in + xx
    #     return xx, x_in
    @torch.compile
    def SwiGLU_MLP_forward_fpx_w_add(x,
                                    gateup_,gateup_split_list,
                                    down_,
                                    gateup_state,down_state,
                                    ebits,mbits,x_in):
        #step1 = F.silu(fpx_matmul(x,gate_,gate_state,ebits,mbits)) * fpx_matmul(x,up_,up_state,ebits,mbits)
        gate,up = fpx_matmul(x.to(device=gateup_.device),gateup_,gateup_state,ebits,mbits).split(gateup_split_list,dim=-1)
        xx = fpx_matmul((F.silu(gate) * up),down_,down_state,ebits,mbits).to(device=x.device)
        x_in = x_in + xx
        return xx, x_in


    def SwiGLU_MLP_forward_fpx(x,
                                    gateup_,gateup_split_list,
                                    down_,
                                    gateup_state,down_state,
                                    ebits,mbits):
        gate,up = fpx_matmul(x.to(device=gateup_.device),gateup_,gateup_state,ebits,mbits).split(gateup_split_list,dim=-1)
        xx = fpx_matmul((F.silu(gate) * up),down_,down_state,ebits,mbits).to(device=x.device)
        return xx

    def SwiGLU_MoEMLP_forward_fpx_w_add(hidden_states,
                                    z,
                                    HRWKV_Misc,
                                    num_experts,
                                    top_k,
                                    norm_topk_prob, #True or False
                                    layer_id,
                                    ebits,mbits,x_in):
        #step1 = F.silu(fpx_matmul(x,gate_,gate_state,ebits,mbits)) * fpx_matmul(x,up_,up_state,ebits,mbits)
        # gate,up = fpx_matmul(x.to(device=gateup_.device),gateup_,gateup_state,ebits,mbits).split(gateup_split_list,dim=-1)
        # xx = fpx_matmul((F.silu(gate) * up),down_,down_state,ebits,mbits).to(device=x.device)
        # x_in = x_in + xx
        # return xx, x_in
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        #router_logits = F.linear(hidden_states,z[f'model.layers.{layer_id}.mlp.gate.weight'])#self.gate(hidden_states)
        router_logits = hidden_states @ z[f'model.layers.{layer_id}.mlp.gate.weight']#self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            #expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            expert_idx_int = int(expert_idx)

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)


            current_hidden_states = HRWKV_7.SwiGLU_MLP_forward_fpx(current_state,
                                                   z[f'model.layers.{layer_id}.mlp.experts.{expert_idx_int}.gateup.weight'],
                                                   HRWKV_Misc[f'model.layers.{layer_id}.mlp.experts.{expert_idx_int}.gateup_split_list'],
                                                   z[f'model.layers.{layer_id}.mlp.experts.{expert_idx_int}.down_proj.weight'],

                                                   z[f'model.layers.{layer_id}.mlp.experts.{expert_idx_int}.gateup.weight.qstate'],
                                                   z[f'model.layers.{layer_id}.mlp.experts.{expert_idx_int}.down_proj.weight.qstate'],
                                                   ebits,
                                                   mbits,
                                                    ) * routing_weights[top_x, idx, None]


            
            #current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        x_in = x_in + final_hidden_states
        return final_hidden_states, x_in

    @torch.compile
    def SwiGLU_MoEMLP_forward_fpx_w_add2(
        hidden_states, z, HRWKV_Misc, num_experts, top_k, 
        norm_topk_prob, layer_id, ebits, mbits, x_in
    ):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # ルーティング（変更なし）
        router_logits = hidden_states @ z[f'model.layers.{layer_id}.mlp.gate.weight']
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        
        if norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        
        # 重要な変更: expert_hittedを使わない
        # 代わりに全エキスパートをチェック（GPU上で並列）
        for expert_idx in range(num_experts):  # Python intのループ
            # GPU上でマスクを計算（CPU同期なし）
            mask = (selected_experts == expert_idx)
            if not mask.any():
                continue
                
            # バッチでインデックスを取得
            token_indices, k_indices = mask.nonzero(as_tuple=True)
            
            if len(token_indices) == 0:
                continue
                
            # バッチ処理
            current_state = hidden_states[token_indices]
            current_weights = routing_weights[token_indices, k_indices]
            
            # expert_idxは既にPython int
            current_hidden_states = HRWKV_7.SwiGLU_MLP_forward_fpx(
                current_state,
                z[f'model.layers.{layer_id}.mlp.experts.{expert_idx}.gateup.weight'],
                HRWKV_Misc[f'model.layers.{layer_id}.mlp.experts.{expert_idx}.gateup_split_list'],
                z[f'model.layers.{layer_id}.mlp.experts.{expert_idx}.down_proj.weight'],
                z[f'model.layers.{layer_id}.mlp.experts.{expert_idx}.gateup.weight.qstate'],
                z[f'model.layers.{layer_id}.mlp.experts.{expert_idx}.down_proj.weight.qstate'],
                ebits,
                mbits,
            )
            
            # 重み付けして加算
            weighted_output = current_hidden_states * current_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_output)
        
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        x_in = x_in + final_hidden_states
        return final_hidden_states, x_in
    


    #@torch.compile
   # @torch.compile()
    def hxa079r_forward(self, idx, 
                last_wkv_states: List[torch.Tensor], kv_cache: List[torch.Tensor],pos_cache,  full_output:bool=False):
        
        with torch.no_grad(): 
            z = self.z

            if z['model.embed_tokens.weight'].device.type == 'cpu':
                x = z['model.embed_tokens.weight'][idx.cpu()].to(device=self.device,dtype=self.base_precision)
            else:
                x = z['model.embed_tokens.weight'][idx].to(dtype=self.base_precision)

            

            v_first = torch.empty_like(x)
            k_first = torch.empty_like(x)

            cache_pos = pos_cache

            B, T, C = x.shape

            calc_cos, calc_sin = get_batch_rope_cache(self.cos, self.sin, cache_pos, T)

            dummytensor = self.dummytensor

            ln_r =  z.get(f'model.layers.{0}.self_attn.r_norm.weight',None)
            rk_normmode = False
            if ln_r is not None:
                rk_normmode = True


            RWKV = 0.0
            GQA = 0.0
            MLP = 0.0

            RWKVBlockCount = 0    
            GQABlockCount = 0     
            MLPBlockCount = 0  

            for i in range(self.n_layer):
                bbb = f'model.layers.{i}.'
                att = f'model.layers.{i}.self_attn.'
                ffn = f'model.layers.{i}.mlp.'
                ffn1 = f'model.layers.{i+1}.mlp.'

                time_mix_shift = dummytensor
                channel_mix_state = dummytensor


                if self.HRWKV_Block_Mode[i][0] == 0:
                    time_mix_state = last_wkv_states[self.HRWKV_Block_Mode[i][2]]

                    xx, time_mix_shift, time_mix_state, v_first,k_first, x = HRWKV_7.hxa079_TimeMix(self.HRWKV_Block_Mode[i][2], self.n_head, self.head_size, x, time_mix_shift, v_first,k_first, time_mix_state,cache_pos,
                                                                        calc_cos,calc_sin,
                                                                        # z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                        # z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'], z[att+'g1'], z[att+'g2'],
                                                                        # z[att+'k0'], z[att+'k1'], z[att+'k2'], #key residual
                                                                        z[att+'wavgk_fused'],self.HRWKV_Misc[att+'wavgk_split_list'],
                                                                        z[att+'w0'], z[att+'w2'], z[att+'a0'], z[att+'a2'], z[att+'v0'], z[att+'v2'], z[att+'g2'],
                                                                        z[att+'k0'], z[att+'k2'], #key residual

                                                                        z[att+'r_k'],
                                                                        z[att+'rkv_fused.weight'],self.HRWKV_Misc[att+'rkv_split_list'],
                                                                        #z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], 
                                                                        z[att+'output.weight'],
                                                                        #z[att+'receptance.weight.qstate'], z[att+'key.weight.qstate'], z[att+'value.weight.qstate'], 
                                                                        z[att+'rkv_fused.weight.qstate'],
                                                                        z[att+'output.weight.qstate'],
                                                                        z[att+'receptance.bias'], z[att+'key.bias'], z[att+'value.bias'], dummytensor,
                                                                        z.get(att+'r_norm.weight',None),z.get(att+'k_norm.weight',None),self.rms_norm_eps,
                                                                        z[bbb+'input_layernorm.weight'],z[bbb+'post_attention_layernorm.weight'],self.rope_theta,
                                                                        self.attn_ebits,self.attn_mbits
                                                                        )
                    last_wkv_states[self.HRWKV_Block_Mode[i][2]] = time_mix_state

                else:

                    xx, x, kv_cache= HRWKV_7.GQA_Attention_Flash_(i,self.HRWKV_Block_Mode[i][2],self.n_head,self.head_size,x,kv_cache,cache_pos,
                                                                        calc_cos,calc_sin,
                                                                        #z[att+'q_proj.weight'], z[att+'k_proj.weight'], z[att+'v_proj.weight'],
                                                                        z[att+'qkv_fused.weight'],self.HRWKV_Misc[att+'qkv_split_list'],
                                                                        z[att+'o_proj.weight'],
                                                                        #z[att+'q_proj.weight.qstate'], z[att+'k_proj.weight.qstate'], z[att+'v_proj.weight.qstate'],
                                                                        z[att+'qkv_fused.weight.qstate'],
                                                                        z[att+'o_proj.weight.qstate'],
                                                                        z[att+'q_proj.bias'], z[att+'k_proj.bias'], z[att+'v_proj.bias'], dummytensor,
                                                                        z.get(att+'q_norm.weight',None),z.get(att+'k_norm.weight',None),self.rms_norm_eps,
                                                                        z[bbb+'input_layernorm.weight'],z[bbb+'post_attention_layernorm.weight'],self.rope_theta,
                                                                        self.attn_ebits,self.attn_mbits )
                #print(f'layer = {i}')
                # if i==0:
                #     z[ffn+'gateup.weight'] = z[ffn+'gateup.weight'].to(device=self.device)
                #     z[ffn+'down_proj.weight'] = z[ffn+'down_proj.weight'].to(device=self.device)

                # if i<self.n_layer-1 and i>=0:
                #     z[ffn1+'gateup.weight'] = z[ffn1+'gateup.weight'].to(device=self.device,non_blocking=True)
                #     z[ffn1+'down_proj.weight'] = z[ffn1+'down_proj.weight'].to(device=self.device,non_blocking=True)


                xx,x = HRWKV_7.SwiGLU_MLP_forward_fpx_w_add(xx,
                                                            z[ffn+'gateup.weight'],self.HRWKV_Misc[ffn+'gateup_split_list'],
                                                            z[ffn+'down_proj.weight'],

                                                            z[ffn+'gateup.weight.qstate'],
                                                            z[ffn+'down_proj.weight.qstate'],
                                                            self.mlp_ebits,self.mlp_mbits,
                                                            x
                                                            )
                # xx=xx.to(device='cuda')
                # x=x.to(device='cuda')
                # if i> 0:
                #     z[ffn+'gateup.weight'] = z[ffn+'gateup.weight'].to(device='cpu',non_blocking=True)
                #     z[ffn+'down_proj.weight'] = z[ffn+'down_proj.weight'].to(device='cpu',non_blocking=True)


            x = T5RMSNorm(x,z['model.norm.weight'],variance_epsilon=self.rms_norm_eps)
            x = fpx_matmul(x , z['lm_head.weight'],z.get('lm_head.weight.qstate',None),self.head_ebits,self.head_mbits)
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持
            pos_cache += T

            return x, None, last_wkv_states, kv_cache, pos_cache
    @torch.compile(mode="max-autotune-no-cudagraphs")
    def hxa07A_forward(self, idx, 
                last_wkv_states: List[torch.Tensor], kv_cache: List[torch.Tensor],pos_cache,  full_output:bool=False):
        
        with torch.no_grad(): 
            z = self.z

            if z['model.embed_tokens.weight'].device.type == 'cpu':
                x = z['model.embed_tokens.weight'][idx.cpu()].to(device=self.device,dtype=self.base_precision)
            else:
                x = z['model.embed_tokens.weight'][idx].to(dtype=self.base_precision)

            
            cache_pos = pos_cache

            B, T, C = x.shape

            dummytensor = self.dummytensor

       

            for i in range(self.n_layer):
                bbb = f'model.layers.{i}.'
                att = f'model.layers.{i}.self_attn.'
                ffn = f'model.layers.{i}.mlp.'

                time_mix_shift = dummytensor


                if self.HRWKV_Block_Mode[i][0] == 0:
                    time_mix_state = last_wkv_states[self.HRWKV_Block_Mode[i][2]]

                    xx, time_mix_shift, time_mix_state, x = HRWKV_7.hxa07A_TimeMix(self.HRWKV_Block_Mode[i][2], self.rwkv_n_head, self.rwkv_head_size, x, time_mix_shift, time_mix_state,cache_pos,
                                                 
                                                                        z[att+'wag_fused'],self.HRWKV_Misc[att+'wag_split_list'],
                                                                        z[att+'w0'], z[att+'w2'], z[att+'a0'], z[att+'a2'], 
                                                                        z[att+'g2'],
                                                                     
                                                                        z[att+'r_k'],
                                                                        z[att+'rkv_fused.weight'],self.HRWKV_Misc[att+'rkv_split_list'],
                                                                        z[att+'output.weight'],
                                                                        z[att+'rkv_fused.weight.qstate'],
                                                                        z[att+'output.weight.qstate'],
                                                                        self.rms_norm_eps,
                                                                        z[bbb+'input_layernorm.weight'],z[bbb+'post_attention_layernorm.weight'],
                                                                        self.attn_ebits,self.attn_mbits
                                                                        )
                 
                    last_wkv_states[self.HRWKV_Block_Mode[i][2]] = time_mix_state

                else:

                    xx, x, kv_cache= HRWKV_7.GQA_Attention_Flash_(i,self.HRWKV_Block_Mode[i][2],self.att_n_head,self.att_head_size,x,kv_cache,cache_pos,
                                                                        None,None,
                                                                        z[att+'qkv_fused.weight'],self.HRWKV_Misc[att+'qkv_split_list'],
                                                                        z[att+'o_proj.weight'],
                                                                        z[att+'qkv_fused.weight.qstate'],
                                                                        z[att+'o_proj.weight.qstate'],
                                                                        z[att+'q_proj.bias'], z[att+'k_proj.bias'], z[att+'v_proj.bias'], dummytensor,
                                                                        z.get(att+'q_norm.weight',None),z.get(att+'k_norm.weight',None),self.rms_norm_eps,
                                                                        z[bbb+'input_layernorm.weight'],z[bbb+'post_attention_layernorm.weight'],self.rope_theta,
                                                                        self.attn_ebits,self.attn_mbits )
         

                xx,x = HRWKV_7.SwiGLU_MLP_forward_fpx_w_add(xx,
                                                            z[ffn+'gateup.weight'],self.HRWKV_Misc[ffn+'gateup_split_list'],
                                                            z[ffn+'down_proj.weight'],

                                                            z[ffn+'gateup.weight.qstate'],
                                                            z[ffn+'down_proj.weight.qstate'],
                                                            self.mlp_ebits,self.mlp_mbits,
                                                            x
                                                            )
         


            x = T5RMSNorm(x,z['model.norm.weight'],variance_epsilon=self.rms_norm_eps)
            x = fpx_matmul(x , z['lm_head.weight'],z.get('lm_head.weight.qstate',None),self.head_ebits,self.head_mbits)
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持
            pos_cache += T

            return x, None, last_wkv_states, kv_cache, pos_cache

    @torch.compile()
    def hxa079r_moe_forward(self, idx, 
                last_wkv_states: List[torch.Tensor], kv_cache,pos_cache,  full_output:bool=False):
        
        with torch.no_grad(): 
            z = self.z

            if z['model.embed_tokens.weight'].device.type == 'cpu':
                x = z['model.embed_tokens.weight'][idx.cpu()].to(device=self.device,dtype=self.base_precision)
            else:
                x = z['model.embed_tokens.weight'][idx]

            

            v_first = torch.empty_like(x)
            k_first = torch.empty_like(x)

            cache_pos = pos_cache

            B, T, C = x.shape

            calc_cos, calc_sin = get_batch_rope_cache(self.cos, self.sin, cache_pos, T)

            dummytensor = self.dummytensor

            ln_r =  z.get(f'model.layers.{0}.self_attn.r_norm.weight',None)
            rk_normmode = False
            if ln_r is not None:
                rk_normmode = True


            RWKV = 0.0
            GQA = 0.0
            MLP = 0.0

            RWKVBlockCount = 0    
            GQABlockCount = 0     
            MLPBlockCount = 0  

            LayerTime = []

            for i in range(self.n_layer):
                
                bbb = f'model.layers.{i}.'
                att = f'model.layers.{i}.self_attn.'
                ffn = f'model.layers.{i}.mlp.'
                ffn1 = f'model.layers.{i+1}.mlp.'

                time_mix_shift = dummytensor
                channel_mix_state = dummytensor


                if self.HRWKV_Block_Mode[i][0] == 0:
                    time_mix_state = last_wkv_states[self.HRWKV_Block_Mode[i][2]]

                    xx, time_mix_shift, time_mix_state, v_first,k_first, x = HRWKV_7.hxa079_TimeMix_x(self.HRWKV_Block_Mode[i][2], self.n_head, self.head_size, x, time_mix_shift, v_first,k_first, time_mix_state,cache_pos,
                                                                        calc_cos,calc_sin,
                                                                        # z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                        # z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'], z[att+'g1'], z[att+'g2'],
                                                                        # z[att+'k0'], z[att+'k1'], z[att+'k2'], #key residual
                                                                        z[att+'wavgk_fused'],self.HRWKV_Misc[att+'wavgk_split_list'],
                                                                        z[att+'w0'], z[att+'w2'], z[att+'a0'], z[att+'a2'], z[att+'v0'], z[att+'v2'], z[att+'g2'],
                                                                        z[att+'k0'], z[att+'k2'], #key residual

                                                                        z[att+'r_k'],
                                                                        z[att+'rkv_fused.weight'],self.HRWKV_Misc[att+'rkv_split_list'],
                                                                        #z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], 
                                                                        z[att+'output.weight'],
                                                                        #z[att+'receptance.weight.qstate'], z[att+'key.weight.qstate'], z[att+'value.weight.qstate'], 
                                                                        z[att+'rkv_fused.weight.qstate'],
                                                                        z[att+'output.weight.qstate'],
                                                                        z[att+'receptance.bias'], z[att+'key.bias'], z[att+'value.bias'], dummytensor,
                                                                        z[att+'r_norm.weight'],z[att+'k_norm.weight'],self.rms_norm_eps,
                                                                        z[bbb+'input_layernorm.weight'],z[bbb+'post_attention_layernorm.weight'],self.rope_theta,
                                                                        self.attn_ebits,self.attn_mbits
                                                                        )
                    last_wkv_states[self.HRWKV_Block_Mode[i][2]] = time_mix_state

                else:

                    xx, x, kv_cache= HRWKV_7.GQA_Attention_Flash_ZeroCopy(i,self.HRWKV_Block_Mode[i][2],self.n_head,self.head_size,x,kv_cache,cache_pos,
                                                                        calc_cos,calc_sin,
                                                                        #z[att+'q_proj.weight'], z[att+'k_proj.weight'], z[att+'v_proj.weight'],
                                                                        z[att+'qkv_fused.weight'],self.HRWKV_Misc[att+'qkv_split_list'],
                                                                        z[att+'o_proj.weight'],
                                                                        #z[att+'q_proj.weight.qstate'], z[att+'k_proj.weight.qstate'], z[att+'v_proj.weight.qstate'],
                                                                        z[att+'qkv_fused.weight.qstate'],
                                                                        z[att+'o_proj.weight.qstate'],
                                                                        z[att+'q_proj.bias'], z[att+'k_proj.bias'], z[att+'v_proj.bias'], dummytensor,
                                                                        z[att+'q_norm.weight'],z[att+'k_norm.weight'],self.rms_norm_eps,
                                                                        z[bbb+'input_layernorm.weight'],z[bbb+'post_attention_layernorm.weight'],self.rope_theta,
                                                                        self.attn_ebits,self.attn_mbits )


                # xx,x = HRWKV_7.SwiGLU_MLP_forward_fpx_w_add(xx,
                #                                             z[ffn+'gateup.weight'],self.HRWKV_Misc[ffn+'gateup_split_list'],
                #                                             z[ffn+'down.weight'],

                #                                             z[ffn+'gateup.weight.qstate'],
                #                                             z[ffn+'down.weight.qstate'],
                #                                             self.ebits,self.mbits,
                #                                             x
                #                                             )
                t0 = time.perf_counter()
                xx,x = HRWKV_7.SwiGLU_MoEMLP_forward_fpx_w_add2(xx,
                                                        z,
                                                        self.HRWKV_Misc,
                                                        self.num_experts,
                                                        self.num_experts_per_tok,
                                                        self.norm_topk_prob, #True or False
                                                        i,
                                                        self.mlp_ebits,self.mlp_mbits,
                                                        x
                                                        )

                t1 = time.perf_counter()
                LayerTime.append((t1-t0)*1000)


            for i in range(self.n_layer):
                print(f'layer = {i} {LayerTime[i]} ms')


            x = T5RMSNorm(x,z['model.norm.weight'],variance_epsilon=self.rms_norm_eps)
            x = fpx_matmul(x , z['lm_head.weight'],z.get('lm_head.weight.qstate',None),self.head_ebits,self.head_mbits)
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持
            pos_cache += T

            return x, None, last_wkv_states, kv_cache, pos_cache


    # def hxa079r_forward_time(self, idx, 
    #             last_wkv_states: List[torch.Tensor], kv_cache, pos_cache, full_output: bool = False):
    
    #     # Timing storage
    #     timemix_times = []
    #     attention_times = []
        
    #     with torch.no_grad(): 
    #         z = self.z

    #         if self.emboncpu:
    #             x = z['emb.weight'][idx.cpu()].to(device=self.device, dtype=self.dtype)
    #         else:
    #             x = z['emb.weight'][idx]

    #         v_first = torch.empty_like(x)
    #         k_first = torch.empty_like(x)

    #         cache_pos = pos_cache

    #         B, T, C = x.shape

    #         calc_cos, calc_sin = get_batch_rope_cache(self.cos, self.sin, cache_pos, T)

    #         dummytensor = self.dummytensor

    #         ln_r = z.get(f'blocks.{0}.att.ln_r.weight', None)
    #         rk_normmode = False
    #         if ln_r is not None:
    #             rk_normmode = True

    #         RWKV = 0.0
    #         GQA = 0.0
    #         MLP = 0.0

    #         RWKVBlockCount = 0    
    #         GQABlockCount = 0     
    #         MLPBlockCount = 0  

    #         for i in range(self.n_layer):
    #             bbb = f'blocks.{i}.'
    #             att = f'blocks.{i}.att.'
    #             ffn = f'blocks.{i}.ffn.'

    #             time_mix_shift = dummytensor
    #             channel_mix_state = dummytensor

    #             if self.HRWKV_Block_Mode[i][0] == 0:
    #                 # TimeMix block
    #                 time_mix_state = last_wkv_states[self.HRWKV_Block_Mode[i][2]]
                    
    #                 # Synchronize before timing
    #                 if torch.cuda.is_available():
    #                     torch.cuda.synchronize()
    #                 start_time = time.perf_counter()
                    
    #                 xx, time_mix_shift, time_mix_state, v_first, k_first, x = HRWKV_7.hxa079_TimeMix(
    #                     self.HRWKV_Block_Mode[i][2], self.n_head, self.head_size, x, time_mix_shift, v_first, k_first, time_mix_state, cache_pos,
    #                     calc_cos, calc_sin,
    #                     z[att+'wavgk_fused'], self.HRWKV_Misc[att+'wavgk_split_list'],
    #                     z[att+'w0'], z[att+'w2'], z[att+'a0'], z[att+'a2'], z[att+'v0'], z[att+'v2'], z[att+'g2'],
    #                     z[att+'k0'], z[att+'k2'],
    #                     z[att+'r_k'],
    #                     z[att+'rkv_fused.weight'], self.HRWKV_Misc[att+'rkv_split_list'],
    #                     z[att+'output.weight'],
    #                     z[att+'rkv_fused.weight.qstate'],
    #                     z[att+'output.weight.qstate'],
    #                     z[att+'receptance.bias'], z[att+'key.bias'], z[att+'value.bias'], dummytensor,
    #                     z[att+'ln_r.weight'], z[att+'ln_k.weight'], self.rms_norm_eps,
    #                     z[bbb+'ln1.weight'], z[bbb+'ln2.weight'], self.rope_theta,
    #                     self.ebits, self.mbits
    #                 )
                    
    #                 # Synchronize after operation
    #                 if torch.cuda.is_available():
    #                     torch.cuda.synchronize()
    #                 end_time = time.perf_counter()
                    
    #                 timemix_times.append(end_time - start_time)
    #                 last_wkv_states[self.HRWKV_Block_Mode[i][2]] = time_mix_state
    #                 RWKVBlockCount += 1

    #             else:
    #                 # GQA Attention block
    #                 if torch.cuda.is_available():
    #                     torch.cuda.synchronize()
    #                 start_time = time.perf_counter()
                    
    #                 xx, x, kv_cache = HRWKV_7.GQA_Attention_Flash(
    #                     i, self.HRWKV_Block_Mode[i][2], self.n_head, self.head_size, x, kv_cache, cache_pos,
    #                     calc_cos, calc_sin,
    #                     z[att+'qkv_fused.weight'], self.HRWKV_Misc[att+'qkv_split_list'],
    #                     z[att+'o_proj.weight'],
    #                     z[att+'qkv_fused.weight.qstate'],
    #                     z[att+'o_proj.weight.qstate'],
    #                     z[att+'q_proj.bias'], z[att+'k_proj.bias'], z[att+'v_proj.bias'], dummytensor,
    #                     z[att+'ln_r.weight'], z[att+'ln_k.weight'], self.rms_norm_eps,
    #                     z[bbb+'ln1.weight'], z[bbb+'ln2.weight'], self.rope_theta,
    #                     self.ebits, self.mbits
    #                 )
                    
    #                 if torch.cuda.is_available():
    #                     torch.cuda.synchronize()
    #                 end_time = time.perf_counter()
                    
    #                 attention_times.append(end_time - start_time)
    #                 GQABlockCount += 1

    #             # MLP forward (not timed separately)
    #             xx, x = HRWKV_7.SwiGLU_MLP_forward_fpx_w_add(
    #                 xx,
    #                 z[ffn+'gateup.weight'], self.HRWKV_Misc[ffn+'gateup_split_list'],
    #                 z[ffn+'down.weight'],
    #                 z[ffn+'gateup.weight.qstate'],
    #                 z[ffn+'down.weight.qstate'],
    #                 self.ebits, self.mbits,
    #                 x
    #             )

    #         x = T5RMSNorm(x, z['ln_out.weight'], variance_epsilon=self.rms_norm_eps)
    #         x = fpx_matmul(x, z['head.weight'], z.get('head.weight.qstate', None), self.ebits, self.mbits)
    #         if not full_output: 
    #             x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持
    #         pos_cache += T
            
    #         # Calculate and print timing statistics
    #         if timemix_times:
    #             avg_timemix = sum(timemix_times) / len(timemix_times)
    #             total_timemix = sum(timemix_times)
    #             print(f"\nTimeMix Statistics:")
    #             print(f"  Total layers: {len(timemix_times)}")
    #             print(f"  Average time per layer: {avg_timemix*1000:.3f} ms")
    #             print(f"  Total time: {total_timemix*1000:.3f} ms")
                
    #         if attention_times:
    #             avg_attention = sum(attention_times) / len(attention_times)
    #             total_attention = sum(attention_times)
    #             print(f"\nGQA Attention Statistics:")
    #             print(f"  Total layers: {len(attention_times)}")
    #             print(f"  Average time per layer: {avg_attention*1000:.3f} ms")
    #             print(f"  Total time: {total_attention*1000:.3f} ms")
            
    #         # Store timing data in model for later analysis
    #         if not hasattr(self, 'layer_timings'):
    #             self.layer_timings = defaultdict(list)
            
    #         self.layer_timings['timemix'].extend(timemix_times)
    #         self.layer_timings['attention'].extend(attention_times)

    #         return x, None, last_wkv_states, kv_cache, pos_cache

