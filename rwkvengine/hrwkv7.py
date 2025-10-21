# HRWKV-7 RNN-HYBRID Transformer "hxa078"
# 2025 OpenMOSE

import os
import os, torch as th
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "True"
os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "True"

from torch.nn.attention.flex_attention import flex_attention  # PyTorch 2.5+

import time
import torch
from collections import defaultdict
#from flash_attn import flash_attn_varlen_func, flash_attn_func

from torch.nn.attention import SDPBackend, sdpa_kernel

import torch
import torch.nn.functional as F


import torch._dynamo

torch._dynamo.reset()  # Dynamo の内部キャッシュを全消去
torch._dynamo.config.cache_size_limit = 512  # 例えば32に拡張
torch._dynamo.config.capture_scalar_outputs = True
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
#from rwkvengine.fla.ops.rwkv7 import fused_recurrent_rwkv7
#from rwkvengine.cuda.wkv7triton import rwkv7_attn_triton
from rwkvengine.fla.ops.rwkv7 import fused_recurrent_rwkv7


from rwkvengine.flashattention_triton import triton_attention


MyStatic = torch.jit.script
import triton
import triton.language as tl


def _evict_cache_inplace_(layer_kv: torch.Tensor,
                          starts: torch.Tensor,
                          excess_tokens: torch.Tensor,
                          Lmax: int) -> torch.Tensor:
    if not bool((excess_tokens > 0).any()):
        return starts
    B = layer_kv.shape[0]
    with torch.no_grad():
        for b in range(B):
            e = int(excess_tokens[b].item())
            if e <= 0:
                continue
            for kv_idx in (0, 1):
                src = layer_kv[b, kv_idx, e:Lmax]
                dst = layer_kv[b, kv_idx, 0:Lmax - e]
                tmp = src.clone()
                dst.copy_(tmp)
                layer_kv[b, kv_idx, Lmax - e:Lmax].zero_()
        starts = (starts - excess_tokens).clamp_(min=0)
    return starts

def _apply_rope_(q: torch.Tensor, k_all: torch.Tensor, rope_theta, starts: torch.Tensor):
    # 必要に応じて移植してください（ダミー）
    return q, k_all

# @triton.autotune(
#     configs=[
#         triton.Config({'BV': BV}, num_warps=nw, num_stages=ns)
#         for BV in [32, 64, 128, 256]
#         for nw in [2, 4, 8]
#         for ns in [2, 3]
#     ],
#     key=['K','V','USE_INITIAL_STATE','STORE_FINAL_STATE'],
# )
# @triton.jit
# def dplr_decode_t1_kernel(
#     q, k, v, a, b, gk,
#     o,           # [B,T=1,H,V]
#     h0, ht,      # [N,H,K,V] or None
#     B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
#     BV: tl.constexpr,
#     USE_INITIAL_STATE: tl.constexpr,
#     STORE_FINAL_STATE: tl.constexpr,
# ):
#     # program ids
#     i_vblk = tl.program_id(0).to(tl.int64)           # along V
#     i_nh   = tl.program_id(1).to(tl.int64)           # along N*H (here B==N, T=1)
#     i_n, i_h = i_nh // H, i_nh % H

#     o_v = i_vblk * BV + tl.arange(0, BV)
#     mask_v = o_v < V

#     # pointers (T=1, no reverse/varlen)
#     p_q  = q  + (i_n * 1 + 0) * H*K + i_h*K + tl.arange(0, K)
#     p_k  = k  + (i_n * 1 + 0) * H*K + i_h*K + tl.arange(0, K)
#     p_a  = a  + (i_n * 1 + 0) * H*K + i_h*K + tl.arange(0, K)
#     p_b  = b  + (i_n * 1 + 0) * H*K + i_h*K + tl.arange(0, K)
#     p_gk = gk + (i_n * 1 + 0) * H*K + i_h*K + tl.arange(0, K)
#     p_v  = v  + (i_n * 1 + 0) * H*V + i_h*V + o_v
#     p_o  = o  + (i_n * 1 + 0) * H*V + i_h*V + o_v

#     # load vectors (K) and block (V)
#     b_q  = tl.load(p_q).to(tl.float32)
#     b_kv = tl.load(p_k).to(tl.float32)
#     b_av = tl.load(p_a).to(tl.float32)
#     b_bv = tl.load(p_b).to(tl.float32)
#     b_g  = tl.exp(tl.load(p_gk).to(tl.float32))      # fastmath exp once
#     b_v  = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

#     # precompute scalars for rank-1 terms
#     alpha = tl.sum(b_q * b_bv)     # q^T b
#     beta  = tl.sum(b_q * b_kv)     # q^T k

#     # reductions over rows of H: s = a^T H, t = (q ⊙ e^g)^T H
#     s = tl.zeros([BV], dtype=tl.float32)
#     t = tl.zeros([BV], dtype=tl.float32)

#     p_h0 = h0 + i_nh * (K*V) + (tl.arange(0, K)[:, None]) * V + o_v[None, :]
#     # read H once for both s and t
#     Hblk = tl.load(p_h0, mask=(o_v[None, :] < V), other=0).to(tl.float32)
#     # s = a^T H
#     s += tl.sum((b_av[:, None]) * Hblk, axis=0)
#     # t = (q ⊙ e^g)^T H
#     t += tl.sum(((b_q * b_g)[:, None]) * Hblk, axis=0)

#     # output: o = t + alpha*s + beta*v
#     out = t + alpha * s + beta * b_v
#     tl.store(p_o, out.to(o.dtype.element_ty), mask=mask_v, eviction_policy='evict_last')

#     # optionally write next state H'
 
#     # s is ready; (re)load H (keeps register pressure low)
#     Hprev = tl.zeros([K, BV], dtype=tl.float32)

#     p_h0 = h0 + i_nh * (K*V) + (tl.arange(0, K)[:, None]) * V + o_v[None, :]
#     Hprev = tl.load(p_h0, mask=(o_v[None, :] < V), other=0).to(tl.float32)

#     Hnext = (b_g[:, None]) * Hprev \
#             + (b_bv[:, None]) * s[None, :] \
#             + (b_kv[:, None]) * b_v[None, :]

#     p_ht = ht + i_nh * (K*V) + (tl.arange(0, K)[:, None]) * V + o_v[None, :]
#     tl.store(p_ht, Hnext.to(ht.dtype.element_ty), mask=(o_v[None, :] < V))

# def fused_recurrent_dplr_delta_rule_decode_t1(
#     q,k,v,a,b,gk,
#     initial_state=None,      # [B,H,K,V] あるいは [N,H,K,V]
#     output_final_state=False
# ):
#     B, T, H, K = q.shape
#     assert T == 1
#     V = v.shape[-1]
#     N = B

#     o  = torch.empty_like(v)
#     ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

#     def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)

#     dplr_decode_t1_kernel[grid](
#         q, k, v, a, b, gk,
#         o, initial_state, ht,
#         B=B, H=H, K=K, V=V,
#         USE_INITIAL_STATE = initial_state is not None,
#         STORE_FINAL_STATE = output_final_state,
#     )
#     return o, ht

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BV': BV, 'BK': BK}, num_warps=nw, num_stages=ns)
        for BV in [32, 64, 128]          # まずは256を外すのがおすすめ
        for BK in [32, 64, 128]          # K タイルサイズ
        for nw in [2, 4, 8]
        for ns in [2, 3]
    ],
    key=['K','V','USE_INITIAL_STATE','STORE_FINAL_STATE'],
)
@triton.jit
def dplr_decode_t1_kernel_streamK(
    q, k, v, a, b, gk,
    o,
    h0, ht,
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    # program ids
    i_vblk = tl.program_id(0).to(tl.int64)            # along V
    i_nh   = tl.program_id(1).to(tl.int64)            # along N*H (B==N, T=1)
    i_n, i_h = i_nh // H, i_nh % H

    o_v = i_vblk * BV + tl.arange(0, BV)
    mask_v = o_v < V

    # base pointers (T=1 固定なので index は簡略化)
    p_q  = q  + (i_n * 1 + 0) * H*K + i_h*K
    p_k  = k  + (i_n * 1 + 0) * H*K + i_h*K
    p_a  = a  + (i_n * 1 + 0) * H*K + i_h*K
    p_b  = b  + (i_n * 1 + 0) * H*K + i_h*K
    p_gk = gk + (i_n * 1 + 0) * H*K + i_h*K

    p_v  = v  + (i_n * 1 + 0) * H*V + i_h*V + o_v
    p_o  = o  + (i_n * 1 + 0) * H*V + i_h*V + o_v

    # load v-block once (cast to f32 for math)
    b_v  = tl.load(p_v, mask=mask_v, other=0, cache_modifier='.cg').to(tl.float32)

    # streaming reductions over K
    s = tl.zeros([BV], dtype=tl.float32)
    t = tl.zeros([BV], dtype=tl.float32)
    alpha = tl.zeros([],  dtype=tl.float32)  # sum(q * b)
    beta  = tl.zeros([],  dtype=tl.float32)  # sum(q * k)

    # K タイルで s, t, alpha, beta をまず集計
    k0 = tl.arange(0, BK)
    for k_start in range(0, K, BK):
        k_idx  = k_start + k0
        k_mask = k_idx < K

        q_sl  = tl.load(p_q  + k_idx, mask=k_mask, other=0).to(tl.float32)
        k_sl  = tl.load(p_k  + k_idx, mask=k_mask, other=0).to(tl.float32)
        a_sl  = tl.load(p_a  + k_idx, mask=k_mask, other=0).to(tl.float32)
        b_sl  = tl.load(p_b  + k_idx, mask=k_mask, other=0).to(tl.float32)
        g_sl  = tl.exp(tl.load(p_gk + k_idx, mask=k_mask, other=0).to(tl.float32))

        # Htile: [BK, BV]
        if USE_INITIAL_STATE:
            p_h0_tile = h0 + i_nh*(K*V) + k_idx[:, None]*V + o_v[None, :]
            Htile = tl.load(p_h0_tile,
                            mask=(k_mask[:, None] & mask_v[None, :]),
                            other=0,
                            cache_modifier='.cg').to(tl.float32)
        else:
            Htile = tl.zeros([BK, BV], dtype=tl.float32)

        s += tl.sum(a_sl[:, None] * Htile, axis=0)
        t += tl.sum((q_sl * g_sl)[:, None] * Htile, axis=0)

        alpha += tl.sum(q_sl * b_sl)
        beta  += tl.sum(q_sl * k_sl)

    # write output
    out = t + alpha * s + beta * b_v
    tl.store(p_o, out.to(o.dtype.element_ty), mask=mask_v, eviction_policy='evict_last')

    # 状態更新（必要な場合のみ）
    if STORE_FINAL_STATE:
        # 再び K タイルで H を更新（s と b_v は既にあるので再ロード不要）
        for k_start in range(0, K, BK):
            k_idx  = k_start + k0
            k_mask = k_idx < K

            if USE_INITIAL_STATE:
                p_h0_tile = h0 + i_nh*(K*V) + k_idx[:, None]*V + o_v[None, :]
                Hprev = tl.load(p_h0_tile,
                                mask=(k_mask[:, None] & mask_v[None, :]),
                                other=0,
                                cache_modifier='.cg').to(tl.float32)
            else:
                Hprev = tl.zeros([BK, BV], dtype=tl.float32)

            b_sl = tl.load(p_b + k_idx, mask=k_mask, other=0).to(tl.float32)
            k_sl = tl.load(p_k + k_idx, mask=k_mask, other=0).to(tl.float32)
            g_sl = tl.exp(tl.load(p_gk + k_idx, mask=k_mask, other=0).to(tl.float32))

            Hnext = (g_sl[:, None]) * Hprev \
                    + (b_sl[:, None]) * s[None, :] \
                    + (k_sl[:, None]) * b_v[None, :]

            p_ht_tile = ht + i_nh*(K*V) + k_idx[:, None]*V + o_v[None, :]
            tl.store(p_ht_tile,
                     Hnext.to(ht.dtype.element_ty),
                     mask=(k_mask[:, None] & mask_v[None, :]),
                     eviction_policy='evict_first')
def fused_recurrent_dplr_delta_rule_decode_t1(
    q,k,v,a,b,gk,
    initial_state=None,      # [B,H,K,V] or [N,H,K,V]  (省略可)
    output_final_state=False,
    state_dtype=None,        # 例: torch.bfloat16 / torch.float16 / torch.float32
):
    B, T, H, K = q.shape
    assert T == 1
    V = v.shape[-1]
    N = B

    o  = torch.empty_like(v)  # v と同 dtype
    if output_final_state:
        if state_dtype is None:
            state_dtype = torch.bfloat16  # まずは bf16 推奨（常時メモリ半減）
        ht = q.new_empty(N, H, K, V, dtype=state_dtype)
    else:
        ht = None

    # None を直接渡さない（constexpr ガードしても Triton は None を嫌う）
    dummy_ptr = q if initial_state is not None else o
    h0 = initial_state if initial_state is not None else dummy_ptr
    ht_ptr = ht if output_final_state else dummy_ptr

    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)

    dplr_decode_t1_kernel_streamK[grid](
        q, k, v, a, b, gk,
        o, h0, ht_ptr,
        B=B, H=H, K=K, V=V,
        USE_INITIAL_STATE = initial_state is not None,
        STORE_FINAL_STATE = output_final_state,
    )
    return o, (ht if output_final_state else None)



#@torch.jit.script
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
#@torch.jit.script
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
#@torch.compile
@torch.jit.script
def T5RMSNorm(hidden_states,weight,variance_epsilon:float=1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)
 

def get_batch_rope_cache_old(cos_cache, sin_cache, batch_pos, seq_len,device):
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
   # device = cos_cache.device
    dtype = cos_cache.dtype
    
    # 各バッチの位置インデックスを作成 [B, T]
    positions = batch_pos + torch.arange(seq_len, device=device, dtype=torch.long)
    
    # cos_cache と sin_cache から必要な部分を取得
    # cos_cache/sin_cache の shape は [1, max_seq_len, rotary_dim]
    # positions を使ってインデックシング
    cos_cache_squeezed = cos_cache.squeeze(0).to(device=device)  # [max_seq_len, rotary_dim]
    sin_cache_squeezed = sin_cache.squeeze(0).to(device=device)  # [max_seq_len, rotary_dim]
    
    # バッチごとに異なる位置から取得
    cos = cos_cache_squeezed[positions]  # [B, T, rotary_dim]
    sin = sin_cache_squeezed[positions]  # [B, T, rotary_dim]
    
    return cos, sin


def get_batch_rope_cache(cos_cache, sin_cache, batch_pos, seq_len, device):
    """
    事前計算されたcos/sin cacheから、バッチごとの開始位置に基づいて
    必要な長さ分のcos/sinを取得する
    足りない場合は最後の値でfillする
    """
    B = batch_pos.shape[0]
    rotary_dim = cos_cache.shape[-1]
    dtype = cos_cache.dtype

    max_seq_len = cos_cache.shape[1]

    # 各バッチの位置インデックスを作成 [B, T]
    positions = batch_pos + torch.arange(seq_len, device=device, dtype=torch.long)

    # clampして範囲外アクセスを防ぐ
    positions = positions.clamp(max=max_seq_len - 1)

    # cos_cache/sin_cache の shape は [1, max_seq_len, rotary_dim]
    cos_cache_squeezed = cos_cache.squeeze(0).to(device=device)  # [max_seq_len, rotary_dim]
    sin_cache_squeezed = sin_cache.squeeze(0).to(device=device)  # [max_seq_len, rotary_dim]

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


 

def repeat_kv_expand(x, kv_repeat: int):
    # x: [B, T, kv_per_head, N] -> [B, T, H, N] (実体コピー無し)
    B, T, kv_per_head, N = x.shape
    # [B,T,kv,1,N] -> expand -> [B,T,kv,kv_repeat,N] -> reshape
    x = x.view(B, T, kv_per_head, 1, N).expand(B, T, kv_per_head, kv_repeat, N)
    return x.reshape(B, T, kv_per_head * kv_repeat, N)

def l2_norm_lastdim(x, eps: float = 1e-6):
    # x: [..., N]
    return x * (x.square().sum(dim=-1, keepdim=True).add(eps).rsqrt())

# 改善案: カスタムfused kernelまたはtorch.jit.scriptで統合
@torch.jit.script
def fused_activation_block(xa, a2, a0, xw, w2, w0):
    a = torch.sigmoid(xa @ a2 + a0)
    w = torch.tanh(xw) @ w2 + w0
    w = -F.softplus(-w) - 0.5
    return a, w

class HRWKV_7(nn.Module):
    #@torch.compile()
    #@torch.compile(mode="max-autotune-no-cudagraphs")
    @torch.compile
    def hxa079_TimeMix_(layer_id: int, H: int, N: int,
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
        

        
        x = T5RMSNorm(x_in,ln1,variance_epsilon=rmsnorm_epsilon)
        xw,xa,xv,xg,xk = (x @ wavgk1).split(wavgk_split_list, dim=-1)

        B, T, HN = x.shape

        rkv = fpx_matmul(x, RKV_,RKV_state, ebits, mbits)
        r,k,v = rkv.split(rkv_split_list,dim=-1)

        kv_dim = k.shape[2] # B,T,kv_dim
        kv_repeat = H*N // kv_dim
        kv_per_head = kv_dim // N

        r = r.add_(R_bias)
        k = k.add_(K_bias).view(B, T, kv_per_head, N)
        v = v.add_(V_bias).view(B, T, kv_per_head, N)

        #ln_r = None

        if ln_r is None:
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
        # if T == 1:
        #     xx, state = fused_recurrent_dplr_delta_rule_decode_t1(r_,k_,v_,aa_,bb_,w_,initial_state=state,output_final_state=True)
        # else:
        xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)

        
        xx = xx.view(B,T,-1) * (float(N) ** -0.5)

        xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,N)*k.view(B,T,H,N)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)

        output = fpx_matmul((xx * (torch.sigmoid(xg) @ g2)), O_, O_state,ebits,mbits)

        x_in = x_in + output
        output = T5RMSNorm(x_in,ln2,variance_epsilon=rmsnorm_epsilon)

        return  output, x[:,-1], state, v_first, k_first, x_in
    

    @torch.compile(mode="max-autotune-no-cudagraphs")
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
        

        
        x = T5RMSNorm(x_in,ln1,variance_epsilon=rmsnorm_epsilon)
        xw,xa,xv,xg,xk = (x @ wavgk1).split(wavgk_split_list, dim=-1)

        B, T, HN = x.shape



        #while True:
        rkv = fpx_matmul(x, RKV_,RKV_state, ebits, mbits)
        r,k,v = rkv.split(rkv_split_list,dim=-1)

        kv_dim = k.shape[2] # B,T,kv_dim
        kv_repeat = H*N // kv_dim
        kv_per_head = kv_dim // N

        r = r.add_(R_bias)
        k = k.add_(K_bias).view(B, T, kv_per_head, N)
        v = v.add_(V_bias).view(B, T, kv_per_head, N)

        #ln_r = None

        if ln_r is None:
            r = r.view(B,T,H,N)
            k = k
        else:
            r = T5RMSNorm(r.view(B,T,H,N), ln_r, variance_epsilon=rmsnorm_epsilon)
            k = T5RMSNorm(k, ln_k, variance_epsilon=rmsnorm_epsilon)

        # Rotary embedding
        cos, sin = calc_cos.to(k.dtype), calc_sin.to(k.dtype)
        r, k = apply_rotary_pos_emb(r, k, cos, sin,unsqueeze_dim=2)

        if v_first is None:
            v_first = v
            k_first = k 
        else:

            v = v + (v_first - v) * torch.sigmoid(xv@v2 + v0).view(B, T, kv_per_head, N)
            k = k + (k_first - k) * torch.sigmoid(xk@k2 + k0).view(B, T, kv_per_head, N)

 

        k = repeat_kv(k, kv_repeat).view(B, T, -1)
        v = repeat_kv(v, kv_repeat).view(B, T, -1)

  
        #a = torch.sigmoid(xa @ a2 + a0)
        
        kk = F.normalize(k.view(B, T, H, N), p=2.0, dim=-1).view(B, T, H*N)
        #w = torch.tanh(xw) @ w2 + w0
        #w = -F.softplus(-w) - 0.5
        a, w = fused_activation_block(xa, a2, a0, xw, w2, w0)
        k = k * (1.0 - w + a)

        aa = -kk
        bb = kk * a

        w = -w.exp()
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,N) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        if T == 1:
            xx, state = fused_recurrent_dplr_delta_rule_decode_t1(r_,k_,v_,aa_,bb_,w_,initial_state=state,output_final_state=True)
        else:
            xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)

        
        xx = xx.view(B,T,-1) * (float(N) ** -0.5)

        xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,N)*k.view(B,T,H,N)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)

        output = fpx_matmul((xx * (torch.sigmoid(xg) @ g2)), O_, O_state,ebits,mbits)

        x_in = x_in + output
        output = T5RMSNorm(x_in,ln2,variance_epsilon=rmsnorm_epsilon)

        return  output, x[:,-1], state, v_first, k_first, x_in
    




    

    #@torch.compile()
    @torch.compile(mode="max-autotune-no-cudagraphs")
    def hxa07A_TimeMix(layer_id: int, H: int, N: int,
                        x_in, x_prev,state,cache_position,
                        calc_cos,calc_sin,
                        wag1,wag_split_list,
                        w0, w2, a0, a2,
                        g2,
                        r_k, 
                        RKV_,rkv_split_list,
                        O_,
                        RKV_state,
                        O_state,
                        R_bias, K_bias, V_bias, O_bias,
                        ln_r,ln_k,rmsnorm_epsilon:float,
                        
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

        r = r.add_(R_bias)
        k = k.add_(K_bias).view(B, T, kv_per_head, N)
        v = v.add_(V_bias).view(B, T, kv_per_head, N)

        if ln_r is None:
            r = r.view(B,T,H,N)
            k = k
        else:
            r = T5RMSNorm(r.view(B,T,H,N), ln_r, variance_epsilon=rmsnorm_epsilon)
            k = T5RMSNorm(k, ln_k, variance_epsilon=rmsnorm_epsilon)

        # Rotary embedding
        cos, sin = calc_cos.to(k.dtype), calc_sin.to(k.dtype)
        r, k = apply_rotary_pos_emb(r, k, cos, sin,unsqueeze_dim=2)

        k = repeat_kv(k.view(B, T, kv_per_head, N), kv_repeat).view(B, T, -1)#.contiguous()
        v = repeat_kv(v.view(B, T, kv_per_head, N), kv_repeat).view(B, T, -1)#.contiguous()

        a = torch.sigmoid(xa @ a2 + a0)
        
        kk = (k).view(B,T,H,-1).float()
        kk = (kk / (torch.norm(kk, dim=-1, keepdim=True) + 1e-12)).view(B,T,-1).to(k.dtype)
        w = torch.tanh(xw) @ w2 + w0
        w = -F.softplus(-w) - 0.5
        k = k * (1.0 - w + a)

        aa = -kk
        bb = kk * a
        w = -w.exp()
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,N) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        xx = xx.to(dtype=x.dtype)
        
        xx = xx.view(B,T,-1) * (float(N) ** -0.5)

        output = fpx_matmul((xx * (torch.sigmoid(xg) @ g2)), O_, O_state,ebits,mbits)

        x_in = x_in + output
        output = T5RMSNorm(x_in,ln2,variance_epsilon=rmsnorm_epsilon)

        return  output, x[:,-1], state, x_in
    
  

 
    @torch._dynamo.disable
    def evict_cache_inplace(
        past_key_value: torch.Tensor,
        gqa_layer_id: int,
        starts: torch.Tensor,
        excess_tokens: torch.Tensor,
        batches_to_evict: torch.Tensor,
        Lmax: int,
    ):
        """
        In-place eviction（PyTorch の重複 copy_ 禁止に対応するため、ソースを clone() して memmove 相当を実現）
        past_key_value: [GQA_L, B, 2, Lmax, KVd] （0:K, 1:V）
        """
        # 推論用途前提なら勾配不要
        with torch.no_grad():
            B = starts.numel()
            for b in range(B):
                if not bool(batches_to_evict[b]):
                    continue

                # starts の健全化
                current_len = int(starts[b].item())
                current_len = 0 if current_len < 0 else (Lmax if current_len > Lmax else current_len)
                starts[b] = current_len

                evict_count = int(excess_tokens[b].item())
                if evict_count <= 0:
                    continue
                if evict_count > current_len:
                    evict_count = current_len

                remaining_len = current_len - evict_count  # >= 0

                k_row = past_key_value[gqa_layer_id, b, 0]  # [Lmax, KVd]
                v_row = past_key_value[gqa_layer_id, b, 1]

                if remaining_len > 0:
                    # ★ ここがポイント：ソースを clone() してから copy_（重複メモリ回避）
                    src_k = k_row.narrow(0, evict_count, remaining_len).clone()
                    src_v = v_row.narrow(0, evict_count, remaining_len).clone()
                    k_row.narrow(0, 0, remaining_len).copy_(src_k)
                    v_row.narrow(0, 0, remaining_len).copy_(src_v)

                    # テールをゼロ（evict_count 分）
                    tail = evict_count  # == current_len - remaining_len
                    if tail > 0:
                        k_row.narrow(0, remaining_len, tail).zero_()
                        v_row.narrow(0, remaining_len, tail).zero_()

                    starts[b] = remaining_len
                else:
                    if current_len > 0:
                        k_row.narrow(0, 0, current_len).zero_()
                        v_row.narrow(0, 0, current_len).zero_()
                    starts[b] = 0

        return past_key_value, starts

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def GQA_Attention_Flash_(
        layer_id: int, gqa_layer_id: int, H: int, N: int,
        x_in: torch.Tensor,
        past_key_value: torch.Tensor,
        cache_position: torch.Tensor,  # 関数外で更新される前提（ここでは読み取りのみ）
        calc_cos, calc_sin,            # 未使用（必要ならRoPE適用部を追加）
        QKV_: torch.Tensor, qkv_split_list,
        O_: torch.Tensor,
        QKV_state, O_state,
        Q_bias: torch.Tensor, K_bias: torch.Tensor, V_bias: torch.Tensor, O_bias: torch.Tensor | None,
        ln_r, ln_k, rmsnorm_epsilon: float,
        ln1, ln2, rope_theta,
        ebits: int, mbits: int,
    ):
        """
        できる限り in-place / 再割当ゼロで動くGQA + Flash-Attention 前処理。
        - cache_position は外部管理：関数内では clamp した「読み値」を使うだけで更新しない
        - eviction と KV 書き込みは in-place（narrow + copy_ + zero_）
        - K/V 連結は cat 不使用：総長で一度だけ確保し順次 copy_

        期待形状:
        x_in           : [B, T, C]
        past_key_value : [GQA_L, B, 2, Lmax, KVd]  (KVd = kv_heads * N)
        cache_position : [B, 1]  （各バッチの現使用長）
        Q_bias         : [H*N],  K_bias/V_bias: [kv_heads*N], O_bias: [C] or None
        QKV_           : [C, (H*N + kv_heads*N + kv_heads*N)]
        O_             : [HN, C]
        """
        B, T, C = x_in.shape
        device = x_in.device
        Lmax = past_key_value.shape[3]

        # ---- cache_position は読み取り専用。関数内だけで矯正して使う（外は絶対に書き換えない）
        starts = cache_position[:, 0].clone()
        starts.clamp_(min=0, max=Lmax)

        # ---- 入力Norm（T5RMSNorm は in-place不可のため最小限の一時テンソル）
        x = T5RMSNorm(x_in, ln1, rmsnorm_epsilon)  # 実装は外部

        HN = H * N

        # ---- QKV投影（実装上の一時テンソルは不可避）
        qkv = fpx_matmul(x, QKV_, QKV_state, ebits, mbits)  # [B,T, QHN + KVd + KVd]
        q, k, v = qkv.split(qkv_split_list, dim=-1)

        # ---- Bias 加算（in-place）
        q.add_(Q_bias)
        k.add_(K_bias)
        v.add_(V_bias)

        # ---- 形状整備（viewは再割当なし）
        q = q.view(B, T, H, N)  # [B,T,H,N]
        kv_dim_flat = k.shape[-1]     # kv_heads * N
        kv_heads = kv_dim_flat // N
        k = k.view(B, T, kv_heads, N) # [B,T,kv_h,N]
        v = v.view(B, T, kv_heads, N)

        # ---- 任意の追加Norm（in-place不可だが最小限）
        if ln_r is not None:
            q = T5RMSNorm(q, ln_r, rmsnorm_epsilon)
            k = T5RMSNorm(k, ln_k, rmsnorm_epsilon)

        # ---- KV 書き込み（in-place）
        # [B,T,kv_h,N] → [B,T, kv_h*N] へ view し、連続スライスに copy_
        k_write = k.view(B, T, -1)
        v_write = v.view(B, T, -1)

        # 溢れ量を計算
        new_lengths = starts + T
        excess_tokens = torch.clamp(new_lengths - Lmax, min=0)
        batches_to_evict = excess_tokens > 0

        # Evict（in-place）
        if bool(batches_to_evict.any()):
            past_key_value, starts = HRWKV_7.evict_cache_inplace(
                past_key_value, gqa_layer_id, starts, excess_tokens, batches_to_evict, Lmax
            )

        # 追記先: [starts[b] : starts[b]+T]（はみ出しは切り詰め）
        for b in range(B):
            s = int(starts[b].item())
            write_T = min(T, Lmax - s)
            if write_T > 0:
                past_key_value[gqa_layer_id, b, 0].narrow(0, s, write_T).copy_(k_write[b, :write_T])
                past_key_value[gqa_layer_id, b, 1].narrow(0, s, write_T).copy_(v_write[b, :write_T])
            # write_T < T の超過分はスキップ（ここまでにエビクト済みなら通常発生しない）

        # このステップでの有効長（外の cache_position はここでは更新しない）
        valid_lengths = torch.minimum(starts + T, torch.tensor(Lmax, device=device))

        # ---- Flash-Attention 入力（in-placeフレンドリ構築）
        # Q: view のみ
        q_flash = q.view(B * T, H, N)  # [B*T,H,N]

        # K/V: cat せず総長で確保 → copy_ で詰める
        total_k = int(valid_lengths.sum().item())
        if total_k == 0:
            # ガード：有効KVが無い（理論上ほぼ無い）
            o_flash = torch.empty_like(q_flash)
            attn_output = o_flash
            cu_seqlens_q = torch.arange(0, (B + 1) * T, step=T, device=device, dtype=torch.int32)
            cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device=device)
            max_seqlen_q = T
            max_seqlen_k = 0
        else:
            k_flash = torch.empty((total_k, kv_heads, N), dtype=k.dtype, device=device)
            v_flash = torch.empty((total_k, kv_heads, N), dtype=v.dtype, device=device)

            # cu_seqlens（Q は等長なので arange で一発）
            cu_seqlens_q = torch.arange(0, (B + 1) * T, step=T, device=device, dtype=torch.int32)
            cu_seqlens_k = torch.empty(B + 1, dtype=torch.int32, device=device)
            cu_seqlens_k[0] = 0

            # 各バッチの有効KVを順次 copy_
            offset = 0
            max_seqlen_k = 0
            for b in range(B):
                vlen = int(valid_lengths[b].item())
                if vlen > 0:
                    # [vlen, KVd] -> [vlen, kv_h, N] に view して copy_
                    srcK = past_key_value[gqa_layer_id, b, 0].narrow(0, 0, vlen).view(vlen, kv_heads, N)
                    srcV = past_key_value[gqa_layer_id, b, 1].narrow(0, 0, vlen).view(vlen, kv_heads, N)
                    k_flash.narrow(0, offset, vlen).copy_(srcK)
                    v_flash.narrow(0, offset, vlen).copy_(srcV)
                    offset += vlen
                    if vlen > max_seqlen_k:
                        max_seqlen_k = vlen
                cu_seqlens_k[b + 1] = cu_seqlens_k[b] + vlen

            # 出力ワーク領域（最小）
            o_flash = torch.empty_like(q_flash)
            sm_scale = 1.0 / (N ** 0.5)

            # Triton Flash-Attention 呼び出し（実装は外部）
            attn_output, _ = triton_attention(
                q_flash,
                k_flash,
                v_flash,
                o_flash,
                cu_seqlens_q,
                cu_seqlens_k,
                T,                   # max_seqlen_q
                int(max_seqlen_k),   # max_seqlen_k
                True,                # causal
                sm_scale,
                None,                # bias
            )

        # ---- 出力投影
        attn_output = attn_output.view(B, T, HN)
        out = fpx_matmul(attn_output, O_, O_state, ebits, mbits)
        if O_bias is not None:
            out.add_(O_bias)

        # 残差
        x_out = x_in + out

        # 後段Norm（返り値は従来互換）
        y = T5RMSNorm(x_out, ln2, rmsnorm_epsilon)
        return y, x_out, past_key_value




    @torch.compile  # 必要に応じて(backend="inductor", dynamic=True)等に
    def GQA_Attention_Flash_F(
        layer_id: int, gqa_layer_id: int, H: int, N: int,
        x_in: torch.Tensor,                                   # [B,T,C]
        past_key_value: torch.Tensor,                         # [GQA_L, B, 2, Lmax, KVd], KVd=kv_h*N
        cache_position: torch.Tensor,                         # [B,1]
        calc_cos, calc_sin,                                   # 未使用（必要ならRoPEへ）
        QKV_: torch.Tensor, qkv_split_list,                   # [C, H*N + kv_h*N + kv_h*N]
        O_: torch.Tensor,                                     # [H*N, C]
        QKV_state, O_state,
        Q_bias: torch.Tensor, K_bias: torch.Tensor, V_bias: torch.Tensor, O_bias: torch.Tensor | None,
        ln_r, ln_k, rmsnorm_epsilon: float,
        ln1, ln2, rope_theta,
        ebits: int, mbits: int,
    ):
        B, T, C = x_in.shape
        device = x_in.device
        dtype  = x_in.dtype

        Lmax = past_key_value.shape[3]
        HN   = H * N

        # ---- cache_position は読み取り専用で矯正
        starts = cache_position[:, 0].clone().clamp_(min=0, max=Lmax)  # [B]

        # ---- 入力Norm
        x = T5RMSNorm(x_in, ln1, rmsnorm_epsilon)  # [B,T,C]

        # ---- QKV 射影（量子化Fused）
        qkv = fpx_matmul(x, QKV_, QKV_state, ebits, mbits)  # [B,T, HN + kv_h*N + kv_h*N]
        q, k, v = qkv.split(qkv_split_list, dim=-1)

        # ---- Bias（in-place）
        q.add_(Q_bias); k.add_(K_bias); v.add_(V_bias)

        # ---- 形状整備
        q = q.view(B, T, H, N).transpose(1, 2).contiguous()      # [B,H,T,N]
        kv_dim_flat = k.shape[-1]
        assert kv_dim_flat % N == 0, "KV dim must be multiple of head dim N"
        kv_h = kv_dim_flat // N
        k = k.view(B, T, kv_h, N).transpose(1, 2).contiguous()   # [B,kv_h,T,N]
        v = v.view(B, T, kv_h, N).transpose(1, 2).contiguous()   # [B,kv_h,T,N]

        # ---- 追加Norm
        if ln_r is not None:
            q = T5RMSNorm(q, ln_r, rmsnorm_epsilon)
            k = T5RMSNorm(k, ln_k, rmsnorm_epsilon)

        # ---- このレイヤのキャッシュ参照
        layer_kv = past_key_value[gqa_layer_id]                  # [B,2,Lmax,KVd]
        new_lengths   = starts + T
        excess_tokens = torch.clamp(new_lengths - Lmax, min=0)
        starts = _evict_cache_inplace_(layer_kv, starts, excess_tokens, Lmax)

        # ---- K/V をキャッシュへ in-place 追記
        k_write = k.transpose(1, 2).reshape(B, T, kv_h * N).contiguous()
        v_write = v.transpose(1, 2).reshape(B, T, kv_h * N).contiguous()
        with torch.no_grad():
            for b in range(B):
                s = int(starts[b].item())
                write_T = min(T, Lmax - s)
                if write_T > 0:
                    layer_kv[b, 0].narrow(0, s, write_T).copy_(k_write[b, :write_T])
                    layer_kv[b, 1].narrow(0, s, write_T).copy_(v_write[b, :write_T])

        # ---- 有効長
        valid_lengths = torch.minimum(starts + T, torch.tensor(Lmax, device=device))  # [B]

        # ---- FlexAttention 入力（キャッシュからビュー）
        K_all = layer_kv[:, 0].view(B, Lmax, kv_h, N).permute(0, 2, 1, 3).contiguous()  # [B,kv_h,Lmax,N]
        V_all = layer_kv[:, 1].view(B, Lmax, kv_h, N).permute(0, 2, 1, 3).contiguous()  # [B,kv_h,Lmax,N]

        # ---- RoPE（必要なら）
        #q, K_all = _apply_rope_(q, K_all, rope_theta, starts)

        # ---- スケール
        scale = 1.0 / (N ** 0.5)

        # ---- スコア改変（因果 + パディング）: -inf でマスク
        #   kv_idx < valid_lengths[b] かつ kv_idx <= starts[b] + q_idx のみ許可
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
        starts_i32 = starts.to(torch.int32)
        valid_i32  = valid_lengths.to(torch.int32)

        def score_mod(score, b, h, q_idx, kv_idx):
            keep = (kv_idx < valid_i32[b]) & (kv_idx <= (starts_i32[b] + q_idx))
            return torch.where(keep, score, neg_inf)

        # ---- FlexAttention（GQA有効）
        attn = flex_attention(
            query=q,             # [B,H,T,N]
            key=K_all,           # [B,kv_h,Lmax,N]
            value=V_all,         # [B,kv_h,Lmax,N]
            score_mod=score_mod,
            scale=scale,
            enable_gqa=True,     # K/V を H に自動ブロードキャスト
            # kernel_options={"ROWS_GUARANTEED_SAFE": True, "BLOCKS_ARE_CONTIGUOUS": True},  # 任意の最適化ヒント
        )  # → [B,H,T,N]

        # ---- 出力投影
        attn_output = attn.transpose(1, 2).reshape(B, T, HN).contiguous()
        out = fpx_matmul(attn_output, O_, O_state, ebits, mbits)
        if O_bias is not None:
            out.add_(O_bias)

        x_out = x_in + out
        y = T5RMSNorm(x_out, ln2, rmsnorm_epsilon)
        return y, x_out, past_key_value



 
    




    #from flash_attn import flash_attn_varlen_func, flash_attn_func
    





    
    # def SwiGLU_MLP_forward_fpx_w_add(x,gate_,down_,up_,gate_state,down_state,up_state,ebits,mbits,x_in):
    #     step1 = F.silu(fpx_matmul(x,gate_,gate_state,ebits,mbits)) * fpx_matmul(x,up_,up_state,ebits,mbits)
    #     xx = fpx_matmul(step1,down_,down_state,ebits,mbits)
    #     x_in = x_in + xx
    #     return xx, x_in
    #@torch.compile
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
    #@torch.compile(mode="max-autotune-no-cudagraphs")
    #@torch.compile
    #@torch.compile(mode="max-autotune-no-cudagraphs")
    def hxa079r_forward(self, idx, 
                last_wkv_states: List[torch.Tensor], kv_cache: List[torch.Tensor],pos_cache,  full_output:bool=False):
        
        with torch.no_grad(): 
            z = self.z

            if z['model.embed_tokens.weight'].device.type == 'cpu':
                x = z['model.embed_tokens.weight'][idx.cpu()].to(device=self.device,dtype=self.base_precision)
            else:
                x = z['model.embed_tokens.weight'][idx].to(dtype=self.base_precision)

            

            v_first = None#torch.empty_like(x)
            k_first = None#torch.empty_like(x)

            cache_pos = pos_cache

            B, T, C = x.shape

            calc_cos, calc_sin = get_batch_rope_cache(self.cos, self.sin, cache_pos, T,device=self.device)

            dummytensor = self.dummytensor

            # ln_r =  z.get(f'model.layers.{0}.self_attn.r_norm.weight',None)
            # rk_normmode = False
            # if ln_r is not None:
            #     rk_normmode = True


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
                    #t0 = time.perf_counter()

                #while True:
                    #print('Using HRWKV Attention Block!')
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

            # RWKV_avg = RWKV / max(RWKVBlockCount,1)
            # GQA_avg = GQA / max(GQABlockCount,1)
            # print(f"RWKV avg time per block: {RWKV_avg*1000:.3f} ms over {RWKVBlockCount} blocks")
            # print(f"GQA avg time per block: {GQA_avg*1000:.3f} ms over {GQABlockCount} blocks")
            return x, None, last_wkv_states, kv_cache, pos_cache
        




   # @torch.compile
    def hxa079r_forward_offload_(self, idx, 
                last_wkv_states: List[torch.Tensor], kv_cache: List[torch.Tensor], pos_cache, full_output: bool=False):

        import torch

        @torch.no_grad()
        def _ensure_pinned_cpu(t: torch.Tensor | None):
            if t is None or t.device.type != 'cpu':
                return t
            # pin + contiguous をできる範囲で保証（non_blocking H2D を確実化）
            try:
                if not t.is_pinned():
                    t = t.pin_memory()
            except Exception:
                pass
            return t.contiguous() if not t.is_contiguous() else t

        @torch.no_grad()
        def _set_Wq_tensor(mod, t_gpu: torch.Tensor):
            # Parameter は使わず、単純に Tensor 属性として持たせる
            try:
                setattr(mod, 'W_q', t_gpu)
            except TypeError:
                # 稀に __setattr__ 制限がある実装への保険
                if hasattr(mod, '_buffers') and 'W_q' in mod._buffers:
                    mod._buffers['W_q'] = t_gpu
                else:
                    object.__setattr__(mod, 'W_q', t_gpu)

        @torch.no_grad()
        def _drop_Wq_tensor(mod):
            # MLP 直後に参照を切る（staging 再利用可に）
            if hasattr(mod, 'W_q'):
                try:
                    setattr(mod, 'W_q', None)
                except TypeError:
                    if hasattr(mod, '_buffers') and 'W_q' in mod._buffers:
                        try:
                            del mod._buffers['W_q']
                        except Exception:
                            mod._buffers['W_q'] = None
                    try:
                        setattr(mod, 'W_q', None)
                    except Exception:
                        pass

        @torch.no_grad()
        def _alloc_staging_for(off_layers, key_prefix):
            # Offload 対象の中で最大 numel に合わせ staging 1枚だけ確保
            max_elems, dtype = 0, None
            for lid in off_layers:
                mod = z[f'model.layers.{lid}.mlp.' + key_prefix]
                src = _get_src_cpu(mod)
                if src is None:
                    continue
                n = src.numel()
                if n > max_elems:
                    max_elems = n
                    dtype = src.dtype
            if max_elems == 0:
                return None
            return torch.empty(max_elems, device=device, dtype=dtype)

        @torch.no_grad()
        def _view_like(base_flat: torch.Tensor, like_cpu: torch.Tensor):
            # staging 先頭から必要分だけを view（追加確保なし）
            return base_flat.narrow(0, 0, like_cpu.numel()).view(like_cpu.shape)

        @torch.no_grad()
        def _get_src_cpu(mod):
            # W_q_cpu を基本に、フォールバックも用意（cpu の W_q を拾う）
            cand = getattr(mod, 'W_q_cpu', None)
            if cand is None:
                cand = getattr(mod, 'W_q_host', None)
            if cand is None:
                wq = getattr(mod, 'W_q', None)
                if isinstance(wq, torch.Tensor) and wq.device.type == 'cpu':
                    cand = wq
            return _ensure_pinned_cpu(cand) if isinstance(cand, torch.Tensor) else None

        @torch.no_grad()
        def _schedule_prefetch(layer_id: int):
            """ 次に使う層の W_q を、prefetch_stream 上で H2D copy_（非同期）。
                staging 再利用の前に、必ず最新の mlp_done_ev を待機する。 """
            if device.type != 'cuda':
                return None

            ffn = f'model.layers.{layer_id}.mlp.'
            gu_mod = z[ffn + 'gateup.weight']
            dn_mod = z[ffn + 'down_proj.weight']

            gu_src = _get_src_cpu(gu_mod)
            dn_src = _get_src_cpu(dn_mod)

            if gu_src is None and dn_src is None:
                return None

            # 既にセット済みならスキップ
            if (gu_src is None or getattr(gu_mod, 'W_q', None) is not None) and \
            (dn_src is None or getattr(dn_mod, 'W_q', None) is not None):
                return None

            with torch.cuda.stream(prefetch_stream):
                # 計算ストリームでの直近 MLP 完了を待機（staging 上書きレース防止）
                prefetch_stream.wait_event(staging_free_ev_holder[0])
                compute_stream = torch.cuda.current_stream(self.device)

                if gu_src is not None and getattr(gu_mod, 'W_q', None) is None and gateup_staging is not None:
                    v = _view_like(gateup_staging, gu_src)
                    v.copy_(gu_src, non_blocking=True)  # pinned -> GPU
                    _set_Wq_tensor(gu_mod, v)
                    v.record_stream(compute_stream)

                if dn_src is not None and getattr(dn_mod, 'W_q', None) is not None:
                    pass  # すでにある
                elif dn_src is not None and down_staging is not None:
                    v = _view_like(down_staging, dn_src)
                    v.copy_(dn_src, non_blocking=True)
                    _set_Wq_tensor(dn_mod, v)
                    v.record_stream(compute_stream)

            ev = torch.cuda.Event()
            prefetch_stream.record_event(ev)
            return ev

        @torch.no_grad()
        def _ensure_ready_for_mlp(layer_id: int):
            """ MLP 開始直前：prefetch 完了を待つ。未ロードなら同期 copy_。
                同期 copy_ でも staging 再利用ハザードを避けるため mlp_done_ev を待ってから。 """
            ffn = f'model.layers.{layer_id}.mlp.'
            gu_mod = z[ffn + 'gateup.weight']
            dn_mod = z[ffn + 'down_proj.weight']

            if device.type == 'cuda':
                ev = ready_events.pop(layer_id, None)
                if ev is not None:
                    torch.cuda.current_stream(device).wait_event(ev)

            # まだなら、計算側の直近完了イベントを待ってから同期コピー
            if getattr(gu_mod, 'W_q', None) is None:
                gu_src = _get_src_cpu(gu_mod)
                if gu_src is not None and gateup_staging is not None:
                    torch.cuda.current_stream(device).wait_event(staging_free_ev_holder[0])
                    v = _view_like(gateup_staging, gu_src)
                    v.copy_(gu_src, non_blocking=False)  # 同期
                    _set_Wq_tensor(gu_mod, v)

            if getattr(dn_mod, 'W_q', None) is None:
                dn_src = _get_src_cpu(dn_mod)
                if dn_src is not None and down_staging is not None:
                    torch.cuda.current_stream(device).wait_event(staging_free_ev_holder[0])
                    v = _view_like(down_staging, dn_src)
                    v.copy_(dn_src, non_blocking=False)
                    _set_Wq_tensor(dn_mod, v)

        with torch.no_grad():
            z = self.z
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)

            # === Embedding ===
            if z['model.embed_tokens.weight'].device.type == 'cpu':
                x = z['model.embed_tokens.weight'][idx.cpu()].to(device=device, dtype=self.base_precision)
            else:
                x = z['model.embed_tokens.weight'][idx].to(dtype=self.base_precision)

            v_first = None
            k_first = None

            cache_pos = pos_cache
            B, T, C = x.shape
            calc_cos, calc_sin = get_batch_rope_cache(self.cos, self.sin, cache_pos, T)
            dummytensor = self.dummytensor

            # === Offload 準備（最大1層 in-flight）===
            off_layers = list(getattr(self, 'Offload_layers', []))
            use_offload = (device.type == 'cuda') and len(off_layers) > 0
            ready_events = {}

            if use_offload:
                # 高優先度 prefetch stream
                prefetch_stream = torch.cuda.Stream(device=device, priority=-1)
                gateup_staging  = _alloc_staging_for(off_layers, 'gateup.weight')
                down_staging    = _alloc_staging_for(off_layers, 'down_proj.weight')

                # 「staging 再利用可」を示す初期イベント（現行ストリームで即記録＝待つ必要なし）
                staging_free_ev_holder = [torch.cuda.Event()]
                torch.cuda.current_stream(device).record_event(staging_free_ev_holder[0])

                # 最初の1層を予約
                next_idx = 0
                first = off_layers[next_idx]
                ev0 = _schedule_prefetch(first)
                if ev0 is not None:
                    ready_events[first] = ev0
                next_idx += 1
            else:
                prefetch_stream = None
                gateup_staging = None
                down_staging = None
                staging_free_ev_holder = [torch.cuda.Event()]  # ダミー
                torch.cuda.current_stream(device).record_event(staging_free_ev_holder[0])
                next_idx = 0

            # ========= Main Forward =========
            for i in range(self.n_layer):
                bbb = f'model.layers.{i}.'
                att = f'model.layers.{i}.self_attn.'
                ffn = f'model.layers.{i}.mlp.'

                time_mix_shift = dummytensor
                channel_mix_state = dummytensor

                # ---- Attention / RWKV ----
                if self.HRWKV_Block_Mode[i][0] == 0:
                    time_mix_state = last_wkv_states[self.HRWKV_Block_Mode[i][2]]
                    xx, time_mix_shift, time_mix_state, v_first, k_first, x = HRWKV_7.hxa079_TimeMix(
                        self.HRWKV_Block_Mode[i][2], self.n_head, self.head_size, x, time_mix_shift, v_first, k_first,
                        time_mix_state, cache_pos, calc_cos, calc_sin,
                        z[att+'wavgk_fused'], self.HRWKV_Misc[att+'wavgk_split_list'],
                        z[att+'w0'], z[att+'w2'], z[att+'a0'], z[att+'a2'], z[att+'v0'], z[att+'v2'], z[att+'g2'],
                        z[att+'k0'], z[att+'k2'],
                        z[att+'r_k'],
                        z[att+'rkv_fused.weight'], self.HRWKV_Misc[att+'rkv_split_list'],
                        z[att+'output.weight'],
                        z[att+'rkv_fused.weight.qstate'],
                        z[att+'output.weight.qstate'],
                        z[att+'receptance.bias'], z[att+'key.bias'], z[att+'value.bias'], dummytensor,
                        z.get(att+'r_norm.weight', None), z.get(att+'k_norm.weight', None), self.rms_norm_eps,
                        z[bbb+'input_layernorm.weight'], z[bbb+'post_attention_layernorm.weight'], self.rope_theta,
                        self.attn_ebits, self.attn_mbits
                    )
                    last_wkv_states[self.HRWKV_Block_Mode[i][2]] = time_mix_state
                else:
                    xx, x, kv_cache = HRWKV_7.GQA_Attention_Flash_(
                        i, self.HRWKV_Block_Mode[i][2], self.n_head, self.head_size, x, kv_cache, cache_pos,
                        calc_cos, calc_sin,
                        z[att+'qkv_fused.weight'], self.HRWKV_Misc[att+'qkv_split_list'],
                        z[att+'o_proj.weight'],
                        z[att+'qkv_fused.weight.qstate'],
                        z[att+'o_proj.weight.qstate'],
                        z[att+'q_proj.bias'], z[att+'k_proj.bias'], z[att+'v_proj.bias'], dummytensor,
                        z.get(att+'q_norm.weight', None), z.get(att+'k_norm.weight', None), self.rms_norm_eps,
                        z[bbb+'input_layernorm.weight'], z[bbb+'post_attention_layernorm.weight'], self.rope_theta,
                        self.attn_ebits, self.attn_mbits
                    )

                # ---- MLP 前：prefetch 完了 or 同期 fallback を保証 ----
                if use_offload and (i in off_layers):
                    torch.cuda.synchronize()
                    _ensure_ready_for_mlp(i)

                # ---- MLP ----
                xx, x = HRWKV_7.SwiGLU_MLP_forward_fpx_w_add(
                    xx,
                    z[ffn+'gateup.weight'], self.HRWKV_Misc[ffn+'gateup_split_list'],
                    z[ffn+'down_proj.weight'],
                    z[ffn+'gateup.weight.qstate'],
                    z[ffn+'down_proj.weight.qstate'],
                    self.mlp_ebits, self.mlp_mbits,
                    x
                )

                # ---- 直後：計算ストリームで「staging 解放イベント」を記録してから W_q を破棄 ----
                if use_offload and (i in off_layers):
                    # ここで「この層の MLP が staging を使い終えた」ことを明示
                    new_free_ev = torch.cuda.Event()
                    torch.cuda.current_stream(device).record_event(new_free_ev)
                    staging_free_ev_holder[0] = new_free_ev

                    _drop_Wq_tensor(z[ffn+'gateup.weight'])
                    _drop_Wq_tensor(z[ffn+'down_proj.weight'])

                    # 次の 1 層だけを非同期プリフェッチ（prefetch は必ず new_free_ev を待つ）
                    if next_idx < len(off_layers):
                        nxt = off_layers[next_idx]
                        evn = _schedule_prefetch(nxt)
                        if evn is not None:
                            ready_events[nxt] = evn
                        next_idx += 1

            # ========= Head =========
            x = T5RMSNorm(x, z['model.norm.weight'], variance_epsilon=self.rms_norm_eps)
            x = fpx_matmul(x, z['lm_head.weight'], z.get('lm_head.weight.qstate', None),
                        self.head_ebits, self.head_mbits)
            if not full_output:
                x = x[:, -1, :]
            pos_cache += T
            return x, None, last_wkv_states, kv_cache, pos_cache




    #@torch.compile
    def hxa079r_forward_offload(self, idx, 
                last_wkv_states: List[torch.Tensor], kv_cache: List[torch.Tensor], pos_cache, full_output: bool=False):

        import torch

        # ========== Utils (軽量・必要最小限) ==========
        @torch.no_grad()
        def _ensure_pinned_cpu(t: torch.Tensor | None):
            if t is None or t.device.type != 'cpu':
                return t
            try:
                if not t.is_pinned():
                    t = t.pin_memory()
            except Exception:
                pass
            return t.contiguous() if not t.is_contiguous() else t

        @torch.no_grad()
        def _get_src_cpu(mod):
            # 優先: W_q_cpu -> W_q_host -> (W_q が CPU 上ならそれ)
            cand = getattr(mod, 'W_q_cpu', None)
            if cand is None:
                cand = getattr(mod, 'W_q_host', None)
            if cand is None:
                wq = getattr(mod, 'W_q', None)
                if isinstance(wq, torch.Tensor) and wq.device.type == 'cpu':
                    cand = wq
            return _ensure_pinned_cpu(cand) if isinstance(cand, torch.Tensor) else None

        @torch.no_grad()
        def _set_Wq_tensor(mod, t_gpu: torch.Tensor):
            # Parameter を使わず Tensor 属性として持つ（低コスト）
            try:
                setattr(mod, 'W_q', t_gpu)
            except TypeError:
                if hasattr(mod, '_buffers') and 'W_q' in mod._buffers:
                    mod._buffers['W_q'] = t_gpu
                else:
                    object.__setattr__(mod, 'W_q', t_gpu)

        @torch.no_grad()
        def _drop_Wq_tensor(mod):
            # MLP 直後に参照を切る（staging 再利用を即許可）
            if hasattr(mod, 'W_q'):
                try:
                    setattr(mod, 'W_q', None)
                except TypeError:
                    if hasattr(mod, '_buffers') and 'W_q' in mod._buffers:
                        try:
                            del mod._buffers['W_q']
                        except Exception:
                            mod._buffers['W_q'] = None
                    try:
                        setattr(mod, 'W_q', None)
                    except Exception:
                        pass

        @torch.no_grad()
        def _alloc_staging_for(off_layers, key_prefix):
            # Offload 対象中の最大 numel / dtype に合わせて 1 枚だけ先行確保（再確保しない）
            max_elems, dtype = 0, None
            for lid in off_layers:
                mod = z[f'model.layers.{lid}.mlp.' + key_prefix]
                src = _get_src_cpu(mod)
                if src is None:
                    continue
                n = src.numel()
                if n > max_elems:
                    max_elems = n
                    dtype = src.dtype
            if max_elems == 0:
                return None
            return torch.empty(max_elems, device=device, dtype=dtype)

        @torch.no_grad()
        def _view_like(base_flat: torch.Tensor, like_cpu: torch.Tensor):
            # staging 先頭から必要分だけ view（追加確保なし）
            return base_flat.narrow(0, 0, like_cpu.numel()).view(like_cpu.shape)

        # ========== Prefetch / Fallback（最小限の同期で高速化） ==========
        @torch.no_grad()
        def _schedule_prefetch(layer_id: int):
            """ 次に使う層の W_q を prefetch_stream で非同期 H2D。staging は 1 枚固定。 """
            if device.type != 'cuda':
                return None

            ffn = f'model.layers.{layer_id}.mlp.'
            gu_mod = z[ffn + 'gateup.weight']
            dn_mod = z[ffn + 'down_proj.weight']

            gu_src = _get_src_cpu(gu_mod)
            dn_src = _get_src_cpu(dn_mod)

            # どちらも不要、または既にロード済みなら何もしない
            if (gu_src is None or getattr(gu_mod, 'W_q', None) is not None) and \
            (dn_src is None or getattr(dn_mod, 'W_q', None) is not None):
                return None

            with torch.cuda.stream(prefetch_stream):
                # 前回の MLP が staging を使い終えてから上書き（レース防止）
                prefetch_stream.wait_event(staging_free_ev_holder[0])

                if gu_src is not None and getattr(gu_mod, 'W_q', None) is None and gateup_staging is not None:
                    v = _view_like(gateup_staging, gu_src)
                    v.copy_(gu_src, non_blocking=True)  # pinned -> GPU 非同期
                    _set_Wq_tensor(gu_mod, v)
                    # コピー完了後、そのビューを計算ストリームのライフタイムに紐付け
                    v.record_stream(compute_stream)

                if dn_src is not None and getattr(dn_mod, 'W_q', None) is None and down_staging is not None:
                    v = _view_like(down_staging, dn_src)
                    v.copy_(dn_src, non_blocking=True)
                    _set_Wq_tensor(dn_mod, v)
                    v.record_stream(compute_stream)

            ev = torch.cuda.Event()
            prefetch_stream.record_event(ev)  # このレイヤのプリフェッチ完了
            return ev

        @torch.no_grad()
        def _ensure_ready_for_mlp(layer_id: int):
            """ MLP 直前：当該レイヤのプリフェッチ完了を“計算ストリームで”待つ。 """
            if device.type != 'cuda':
                return
            ev = ready_events.pop(layer_id, None)
            if ev is not None:
                torch.cuda.current_stream(device).wait_event(ev)
            # （基本はこれで十分。プリフェッチが無かった場合のみ、下の同期コピーが発動）

            # プリフェッチが間に合わず W_q が無いなら、同期で埋める（頻度は稀）
            ffn = f'model.layers.{layer_id}.mlp.'
            gu_mod = z[ffn + 'gateup.weight']
            dn_mod = z[ffn + 'down_proj.weight']

            if getattr(gu_mod, 'W_q', None) is None:
                gu_src = _get_src_cpu(gu_mod)
                if gu_src is not None and gateup_staging is not None:
                    v = _view_like(gateup_staging, gu_src)
                    v.copy_(gu_src, non_blocking=False)  # 同期コピーで救済
                    _set_Wq_tensor(gu_mod, v)

            if getattr(dn_mod, 'W_q', None) is None:
                dn_src = _get_src_cpu(dn_mod)
                if dn_src is not None and down_staging is not None:
                    v = _view_like(down_staging, dn_src)
                    v.copy_(dn_src, non_blocking=False)
                    _set_Wq_tensor(dn_mod, v)

        # ========== Forward ==========
        with torch.no_grad():
            z = self.z
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)

            # === Embedding ===
            if z['model.embed_tokens.weight'].device.type == 'cpu':
                x = z['model.embed_tokens.weight'][idx.cpu()].to(device=device, dtype=self.base_precision)
            else:
                x = z['model.embed_tokens.weight'][idx].to(dtype=self.base_precision)

            v_first = None
            k_first = None

            cache_pos = pos_cache
            B, T, C = x.shape
            calc_cos, calc_sin = get_batch_rope_cache(self.cos, self.sin, cache_pos, T,device=self.device)
            dummytensor = self.dummytensor

            # === Offload 準備（最大1層 in-flight・高速寄り）===
            off_layers = list(getattr(self, 'Offload_layers', []))
            use_offload = (device.type == 'cuda') and len(off_layers) > 0
            ready_events = {}

            # 計算ストリームを固定参照（prefetch 内で record_stream に使う）
            compute_stream = torch.cuda.current_stream(device)

            if use_offload:
                prefetch_stream = torch.cuda.Stream(device=device, priority=-1)

                # staging を“最大サイズで一発確保”（再確保しない＝ピーク一定・速い）
                gateup_staging = _alloc_staging_for(off_layers, 'gateup.weight')
                down_staging   = _alloc_staging_for(off_layers, 'down_proj.weight')

                # 「staging が空いている」ことを示す最初のイベント（現行=計算ストリームで即 ready）
                staging_free_ev_holder = [torch.cuda.Event()]
                compute_stream.record_event(staging_free_ev_holder[0])

                # 最初の1層を予約（距離=1）
                next_idx = 0
                first = off_layers[next_idx]
                ev0 = _schedule_prefetch(first)
                if ev0 is not None:
                    ready_events[first] = ev0
                next_idx += 1
            else:
                prefetch_stream = None
                gateup_staging = None
                down_staging = None
                staging_free_ev_holder = [torch.cuda.Event()]
                compute_stream.record_event(staging_free_ev_holder[0])
                next_idx = 0

            # ========= Main Forward =========
            for i in range(self.n_layer):
                bbb = f'model.layers.{i}.'
                att = f'model.layers.{i}.self_attn.'
                ffn = f'model.layers.{i}.mlp.'

                time_mix_shift = dummytensor
                channel_mix_state = dummytensor

                # ---- Attention / RWKV ----
                if self.HRWKV_Block_Mode[i][0] == 0:
                    time_mix_state = last_wkv_states[self.HRWKV_Block_Mode[i][2]]
                    xx, time_mix_shift, time_mix_state, v_first, k_first, x = HRWKV_7.hxa079_TimeMix(
                        self.HRWKV_Block_Mode[i][2], self.n_head, self.head_size, x, time_mix_shift, v_first, k_first,
                        time_mix_state, cache_pos, calc_cos, calc_sin,
                        z[att+'wavgk_fused'], self.HRWKV_Misc[att+'wavgk_split_list'],
                        z[att+'w0'], z[att+'w2'], z[att+'a0'], z[att+'a2'], z[att+'v0'], z[att+'v2'], z[att+'g2'],
                        z[att+'k0'], z[att+'k2'],
                        z[att+'r_k'],
                        z[att+'rkv_fused.weight'], self.HRWKV_Misc[att+'rkv_split_list'],
                        z[att+'output.weight'],
                        z[att+'rkv_fused.weight.qstate'],
                        z[att+'output.weight.qstate'],
                        z[att+'receptance.bias'], z[att+'key.bias'], z[att+'value.bias'], dummytensor,
                        z.get(att+'r_norm.weight', None), z.get(att+'k_norm.weight', None), self.rms_norm_eps,
                        z[bbb+'input_layernorm.weight'], z[bbb+'post_attention_layernorm.weight'], self.rope_theta,
                        self.attn_ebits, self.attn_mbits
                    )
                    last_wkv_states[self.HRWKV_Block_Mode[i][2]] = time_mix_state
                else:
                    xx, x, kv_cache = HRWKV_7.GQA_Attention_Flash_(
                        i, self.HRWKV_Block_Mode[i][2], self.n_head, self.head_size, x, kv_cache, cache_pos,
                        calc_cos, calc_sin,
                        z[att+'qkv_fused.weight'], self.HRWKV_Misc[att+'qkv_split_list'],
                        z[att+'o_proj.weight'],
                        z[att+'qkv_fused.weight.qstate'],
                        z[att+'o_proj.weight.qstate'],
                        z[att+'q_proj.bias'], z[att+'k_proj.bias'], z[att+'v_proj.bias'], dummytensor,
                        z.get(att+'q_norm.weight', None), z.get(att+'k_norm.weight', None), self.rms_norm_eps,
                        z[bbb+'input_layernorm.weight'], z[bbb+'post_attention_layernorm.weight'], self.rope_theta,
                        self.attn_ebits, self.attn_mbits
                    )

                # ---- MLP 前：その層のプリフェッチ完了だけ待つ（グローバル同期なし）----
                if use_offload and (i in off_layers):
                    _ensure_ready_for_mlp(i)

                # ---- MLP ----
                xx, x = HRWKV_7.SwiGLU_MLP_forward_fpx_w_add(
                    xx,
                    z[ffn+'gateup.weight'], self.HRWKV_Misc[ffn+'gateup_split_list'],
                    z[ffn+'down_proj.weight'],
                    z[ffn+'gateup.weight.qstate'],
                    z[ffn+'down_proj.weight.qstate'],
                    self.mlp_ebits, self.mlp_mbits,
                    x
                )

                # ---- 直後：staging 解放イベントを記録 → W_q を破棄 → 次の1層を即プリフェッチ ----
                if use_offload and (i in off_layers):
                    new_free_ev = torch.cuda.Event()
                    compute_stream.record_event(new_free_ev)
                    staging_free_ev_holder[0] = new_free_ev

                    _drop_Wq_tensor(z[ffn+'gateup.weight'])
                    _drop_Wq_tensor(z[ffn+'down_proj.weight'])

                    if next_idx < len(off_layers):
                        nxt = off_layers[next_idx]
                        evn = _schedule_prefetch(nxt)  # 非同期プリフェッチ（距離=1）
                        if evn is not None:
                            ready_events[nxt] = evn
                        next_idx += 1

            # ========= Head =========
            

            if not full_output:
                x = x[:, -1, :].unsqueeze(1) 

            x = T5RMSNorm(x, z['model.norm.weight'], variance_epsilon=self.rms_norm_eps)
            x = fpx_matmul(x, z['lm_head.weight'], z.get('lm_head.weight.qstate', None),
                        self.head_ebits, self.head_mbits)
            if not full_output:
                x = x[:, -1, :]
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

 

            calc_cos, calc_sin = get_batch_rope_cache(self.cos, self.sin, cache_pos, T,device=self.device)

            dummytensor = self.dummytensor

            ln_r =  z.get(f'model.layers.{0}.self_attn.r_norm.weight',None)
            rk_normmode = False
            if ln_r is not None:
                rk_normmode = True

       

            for i in range(self.n_layer):
                bbb = f'model.layers.{i}.'
                att = f'model.layers.{i}.self_attn.'
                ffn = f'model.layers.{i}.mlp.'

                time_mix_shift = dummytensor


                if self.HRWKV_Block_Mode[i][0] == 0:
                    time_mix_state = last_wkv_states[self.HRWKV_Block_Mode[i][2]]

                    """
                    def hxa07A_TimeMix(layer_id: int, H: int, N: int,
                        x_in, x_prev,state,cache_position,
                        calc_cos,calc_sin,
                        wag1,wag_split_list,
                        w0, w2, a0, a2,
                        g2,
                        r_k, 
                        RKV_,rkv_split_list,
                        O_,
                        RKV_state,
                        O_state,
                        R_bias, K_bias, V_bias, O_bias,
                        ln_r,ln_k,rmsnorm_epsilon:float,
                        
                        ln1,ln2,
                        ebits:int, mbits:int
                        ):
                    """

                    xx, time_mix_shift, time_mix_state, x = HRWKV_7.hxa07A_TimeMix(self.HRWKV_Block_Mode[i][2], self.rwkv_n_head, self.rwkv_head_size, x, time_mix_shift, time_mix_state,cache_pos,
                                                                        calc_cos, calc_sin,
                                                 
                                                                        z[att+'wag_fused'],self.HRWKV_Misc[att+'wag_split_list'],
                                                                        z[att+'w0'], z[att+'w2'], z[att+'a0'], z[att+'a2'], 
                                                                        z[att+'g2'],
                                                                     
                                                                        None,
                                                                        z[att+'rkv_fused.weight'],self.HRWKV_Misc[att+'rkv_split_list'],
                                                                        z[att+'output.weight'],
                                                                        z[att+'rkv_fused.weight.qstate'],
                                                                        z[att+'output.weight.qstate'],
                                                                        z[att+'receptance.bias'], z[att+'key.bias'], z[att+'value.bias'], dummytensor,
                                                                        z.get(att+'r_norm.weight',None),z.get(att+'k_norm.weight',None),self.rms_norm_eps,
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

     