# HRWKV-7 RNN-HYBRID Transformer "hxa078"
# 2025 OpenMOSE

import os

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "True"
os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "True"

#from flash_attn import flash_attn_varlen_func, flash_attn_func

from torch.nn.attention import SDPBackend, sdpa_kernel

import torchao
from torchao.dtypes.floatx import to_scaled_tc_floatx
from torchao.ops import quant_llm_linear
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # 例えば32に拡張


try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
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
from rwkvengine.fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
from rwkvengine.fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
from rwkvengine.fla.ops.rwkv7 import chunk_rwkv7,fused_recurrent_rwkv7
from rwkvengine.cuda.wkv7triton import rwkv7_attn_triton
from rwkvengine.matmulhell import fused_dequant_gemm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(True)


MyStatic = torch.jit.script



@torch.compiler.disable
def bnb_nf4_matmul(x,weight,weight_state):
    batch, T, hiddensize = x.shape
    if T == 1 and batch == 1:
        x = x.view(batch,hiddensize)
        #print(f'trying {x.shape}' )
        o = bnb.functional.gemv_4bit(A=x,
                                    B=weight,
                                    state=weight_state,
                                    #transposed_A = True,
                                   # transposed_B = True
                                    )
        #print(o.shape)
        o = o.view(batch,T,-1)
        return o

    else:
        w = bnb.functional.dequantize_nf4(weight,quant_state=weight_state).to(torch.bfloat16)
        return x @ w.t()

@torch.compile
def fp8_matmul(x,weight,weight_state):
    xg = x
    b = weight
    dtype = x.dtype
    if len(xg.shape) == 2:   
        S0=xg.shape[0]
        if xg.dtype != torch.float8_e4m3fn:
            xscale = 448.0 / torch.max(torch.abs(xg)) + 1e-6
            #xg = xg.float() * xscale
            xg = xg.to(dtype=torch.float32) * xscale
            xg = torch.clamp(xg, -448.0, 448.0).to(dtype=torch.float8_e4m3fn)#.contiguous()
        else:
            xscale = torch.tensor(1.0, device='cuda')

        x = torch._scaled_mm(
            xg.view(S0,xg.shape[1]).to(torch.float8_e4m3fn),#,.contiguous(),
            b.t(),
            bias=None,
            out_dtype=dtype,
            scale_a=1.0 / xscale.to(dtype=torch.float32),
            scale_b=1.0 / weight_state,
            use_fast_accum = True
        )
        return x.view(S0, -1)
    else:

        S0=xg.shape[0]
        S1=xg.shape[1]
        
        if xg.dtype != torch.float8_e4m3fn:
            xscale = 448.0 / torch.max(torch.abs(xg)) + 1e-6
            xg = xg.to(dtype=torch.float32) * xscale
            xg = torch.clamp(xg, -448.0, 448.0).to(dtype=torch.float8_e4m3fn).contiguous()
        else:
            xscale = torch.tensor(1.0, device='cuda')
        
        x = torch._scaled_mm(
            xg.view(S0*S1,xg.shape[2]).to(torch.float8_e4m3fn),
            b.t(),
            bias=None,
            out_dtype=dtype,
            scale_a=1.0 / xscale.to(dtype=torch.float32),
            scale_b=1.0 / weight_state,
            use_fast_accum = True
        )
        return x.view(S0, S1, -1)
    
@torch.compile()
def fpx_matmul(x,weight,weight_state,ebits:int=3,mbits:int=2):
    if ebits == -4 and mbits == -4:
        return bnb_nf4_matmul(x,weight,weight_state)
    elif ebits == -44 and mbits == -44: #hqq
        S0=x.shape[0]
        S1=x.shape[1]
        dtype = x.dtype
        x = x.to(dtype=torch.float16).view(-1,x.shape[2])  
        out = weight(x).view(S0,S1,-1).to(dtype=dtype)
        return out

    elif weight.dtype == torch.uint8:
        S0=x.shape[0]
        S1=x.shape[1]
        dtype = x.dtype
        x = x.to(dtype=torch.float16).view(-1,x.shape[2])  
        d = quant_llm_linear(ebits, mbits, x, weight, weight_state).view(S0,S1,-1).to(dtype=dtype)# * 2.0
        return d
    elif weight.dtype == torch.int8:
        return fused_dequant_gemm(x,weight,weight_state)
    elif weight.dtype == torch.float8_e4m3fn: 
        return fp8_matmul(x,weight,weight_state)
    else:
        if weight.device == torch.device('cpu'):
            xdtype = x.dtype
            xdevice = x.device
            return (x.to(device=torch.device('cpu'),dtype=weight.dtype) @ weight).to(device=xdevice,dtype=xdtype)
            #return x @ weight.to(device=x.device).t()
        else:
            return x @ weight.t()
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
import triton
import triton.language as tl
@torch.compile
def fused_2gemm(xx, a0, a1, a2):
    # 第一段階：xx @ a1
    tmp = xx @ a1               # [B,T,H2] --- ここがメモリ食うから注意
    # 第二段階：tmp @ a2 + a0
    out = F.linear(tmp, a2.t(), a0)
    return out

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

def compute_rope_cache_range_ntk(cache_pos, seq_len, rotary_dim, device, dtype, rope_theta, ntk_alpha=10.0):
    """
    RoPEのcos/sinをバッチごとに cache_pos から始まる Tトークン分生成（NTKスケーリング対応）

    Args:
        cache_pos: [B, 1]  - 各バッチの開始位置
        seq_len: int       - Tステップ分のRoPE計算を行う
        rotary_dim: int    - 回転に使う次元数（通常は Head_dim）
        rope_theta: float  - RoPEのthetaパラメータ（例: 10000.0）
        ntk_alpha: float   - NTK拡張倍率（例: 10.0〜16.0）

    Returns:
        cos: [B, T, rotary_dim]
        sin: [B, T, rotary_dim]
        inv_freq: [half_dim]  # 使用したinv_freq（デバッグや可視化用）
    """
    B = cache_pos.shape[0]
    half_dim = rotary_dim // 2

    # 周波数ベーススケーリング（NTK RoPE）
    freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (rope_theta * (ntk_alpha ** (freq_seq / half_dim)))  # NTKスケーリング適用

    # 相対位置 [T]
    rel_pos = torch.arange(seq_len, dtype=dtype, device=device)

    # 絶対位置 [B, T] = cache_pos[b, 0] + rel_pos[t]
    pos = cache_pos.to(dtype) + rel_pos.view(1, -1)

    # 周波数乗算 [B, T, half_dim]
    freqs = torch.einsum("bt,d->btd", pos, inv_freq)

    # rotary_dimへの変換 [B, T, rotary_dim]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin, inv_freq

#Yarn Version
def compute_rope_cache_range_YaRN(cache_pos, seq_len, rotary_dim, device, dtype, rope_theta,
                                  ntk_alpha=10.0, yarn_interp_max_pos=4096.0):
    """
    YaRN: 短距離RoPEと長距離RoPEの補間によるRoPE安定化。
    cache_pos から始まる Tトークン分のcos/sinを生成。

    Args:
        cache_pos: [B, 1]  - 各バッチの開始位置
        seq_len: int       - Tトークン数
        rotary_dim: int    - 回転次元数（通常はHead_dim）
        device, dtype      - 計算用設定
        rope_theta: float  - 標準RoPEのtheta（通常10000）
        ntk_alpha: float   - 長距離RoPE拡張係数（例: 10.0〜16.0）
        yarn_interp_max_pos: float - 補間のスイッチ位置（推奨4096）

    Returns:
        cos, sin: [B, T, rotary_dim]
        inv_freq_short: 短距離RoPEの周波数（参考用）
    """
    B = cache_pos.shape[0]
    half_dim = rotary_dim // 2

    freq_seq = torch.arange(half_dim, dtype=dtype, device=device)

    # RoPE 周波数（短距離 / 長距離）
    inv_freq_short = 1.0 / (rope_theta ** (freq_seq / half_dim))
    inv_freq_long  = 1.0 / (rope_theta * (ntk_alpha ** (freq_seq / half_dim)))

    # 相対位置 [T]
    rel_pos = torch.arange(seq_len, dtype=dtype, device=device)

    # 絶対位置 [B, T]
    pos = cache_pos.to(dtype) + rel_pos.view(1, -1)

    # 周波数×位置 → [B, T, half_dim]
    freqs_short = torch.einsum("bt,d->btd", pos, inv_freq_short)
    freqs_long  = torch.einsum("bt,d->btd", pos, inv_freq_long)

    # 線形補間係数（位置が大きいほど長距離RoPEに寄る）
    mix_ratio = torch.clamp(pos / yarn_interp_max_pos, 0.0, 1.0).unsqueeze(-1)  # [B, T, 1]

    freqs = (1.0 - mix_ratio) * freqs_short + mix_ratio * freqs_long  # 補間

    # rotary次元に展開
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin, inv_freq_short

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


@torch.compile  # PyTorch 2.0 の最適化コンパイル
def fast_block(xx, w0, w1, w2):
    # 1) xx @ w1 → [B, T, H2]
    t = xx.matmul(w1)
    # 2) in-place tanh
    t.tanh_()
    # 3) fused matmul + bias: F.linear は内部で (t @ w2.T + w0) を一発で実行
    out = F.linear(t, w2.t(), w0)   # → [B, T, O]
    # 4) in-place negation
    #out.neg_()
    return out



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
    @torch.compile
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

        B, T, HN = x.shape

        

        # r = fpx_matmul(x, R_, R_state, ebits, mbits).add_(R_bias)
        # k = fpx_matmul(x, K_, K_state, ebits, mbits).add_(K_bias)
        # v = fpx_matmul(x, V_, V_state, ebits, mbits).add_(V_bias)

        rkv = fpx_matmul(x, RKV_,RKV_state, ebits, mbits)
        r,k,v = rkv.split(rkv_split_list,dim=-1)

        xw,xa,xv,xg,xk = (x @ wavgk1).split(wavgk_split_list, dim=-1)


        r = r.add_(R_bias)
        k = k.add_(K_bias)
        v = v.add_(V_bias)

        kv_dim = k.shape[2] # B,T,kv_dim
        kv_repeat = HN // kv_dim
        kv_per_head = kv_dim // N

        if ln_r == None:
            r = r.view(B,T,H,N)
            k = k.view(B,T,kv_per_head,N)
        else:
            r = T5RMSNorm(r.view(B,T,H,N), ln_r, variance_epsilon=rmsnorm_epsilon)
            k = T5RMSNorm(k.view(B,T,kv_per_head,N), ln_k, variance_epsilon=rmsnorm_epsilon)

        # Rotary embedding
        #cos, sin, _ = compute_rope_cache_range(cache_position, T, N, k.device, torch.float32, rope_theta)
        cos, sin = calc_cos.to(k.dtype), calc_sin.to(k.dtype)
        r, k = apply_rotary_pos_emb(r, k, cos, sin,unsqueeze_dim=2)


        k = k.view(B, T, kv_per_head, N)
        v = v.view(B, T, kv_per_head, N)

        if layer_id == 0:
            v_first = v
            k_first = k 
        else:
            #tmp = x @ v1
            tmp = F.linear(xv, v2.t(), v0)
            gate = torch.sigmoid(tmp).view(B, T, kv_per_head, N)
            v = v + (v_first - v) * gate

            #tmp2 = x @ k1
            tmp2 = F.linear(xk, k2.t(), k0)
            gate2 = torch.sigmoid(tmp2).view(B, T, kv_per_head, N)
            k = k + (k_first - k) * gate2

        #tmp = x @ w1             # → [B, T, H2]
        tmp = torch.tanh(xw)
        w = F.linear(tmp, w2.t(), w0)  # → [B, T, O]
        w = -F.softplus(-w) - 0.5

        

        k = repeat_kv(k, kv_repeat)
        v = repeat_kv(v, kv_repeat)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #tmp = x @ a1
        tmp = F.linear(xa, a2.t(), a0)
        a = torch.sigmoid(tmp)
        g = torch.sigmoid(xg) @ g2

        kk = F.normalize(k.view(B, T, H, N), p=2.0, dim=-1).view(B, T, H*N)
        k = k * (1.0 - w + a)

        aa = -kk
        bb = kk * a

        


        w = -w.to(dtype=torch.float32).exp()
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,N) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)

        
        xx = xx.view(B,T,-1)
        xx = xx * (float(N) ** -0.5)

        xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,N)*k.view(B,T,H,N)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,HN)
        output = fpx_matmul((xx * g), O_, O_state,ebits,mbits)

        x_in = x_in + output
        output = T5RMSNorm(x_in,ln2,variance_epsilon=rmsnorm_epsilon)

        return  output, x[:,-1], state, v_first, k_first, x_in
    
    
    # @torch.compile
    # def GQA_Attention_old(layer_id: int, gqa_layer_id: int, H: int, N: int,
    #               x_in, past_key_value, cache_position,
    #               calc_cos,calc_sin,
    #               Q_, K_, V_, O_,
    #               Q_state, K_state, V_state, O_state,
    #               Q_bias, K_bias, V_bias, O_bias,
    #               ln_r, ln_k, rmsnorm_epsilon: float,
    #               ln1, ln2, rope_theta,
    #               ebits: int, mbits: int):

    #     B, T, C = x_in.size()
    #     x = T5RMSNorm(x_in, ln1, rmsnorm_epsilon)

    #     HN = H * N
    #     #kv_dim = K_.shape[0] if K_.dtype != torch.int8 else K_.shape[1]
    #     # kv_per_head = kv_dim // N
    #     # kv_repeat = HN // kv_dim
    #     # kv_cache_size = past_key_value.shape[2]

    #     # Projections
    #     q = fpx_matmul(x, Q_, Q_state, ebits, mbits).add_(Q_bias).view(B, T, H, N)
    #     k = fpx_matmul(x, K_, K_state, ebits, mbits).add_(K_bias)
    #     v = fpx_matmul(x, V_, V_state, ebits, mbits).add_(V_bias)

    #     kv_dim = k.shape[2] # B,T,kv_dim
    #     kv_per_head = kv_dim // N
    #     kv_repeat = HN // kv_dim
    #     kv_cache_size = past_key_value.shape[2]

    #     #print(F'GQA VIEW KV')

    #     k = k.view(B, T, kv_per_head, N)
    #     v = v.view(B, T, kv_per_head, N)

    #     #print(F'GQA VIEW end')


    #     if ln_r is not None:
    #         q = T5RMSNorm(q, ln_r, rmsnorm_epsilon)
    #         k = T5RMSNorm(k, ln_k, rmsnorm_epsilon)

    #     # Rotary embedding
    #     #cos, sin, _ = compute_rope_cache_range(cache_position, T, N, k.device, torch.float32, rope_theta)
    #     cos, sin = calc_cos.to(k.dtype), calc_sin.to(k.dtype)
    #     #q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

    #     # === KV write準備 ===
    #     k_write = k.view(B, T, kv_per_head * N)
    #     v_write = v.view(B, T, kv_per_head * N)

    #     for b in range(B):
    #         start = cache_position[b, 0].item()
    #         end = start + T

    #         # if end > kv_cache_size:
    #         #     # 溢れる分だけ左詰め（truncate）
    #         #     shift = end - kv_cache_size
    #         #     past_key_value[gqa_layer_id, b, :-shift] = past_key_value[gqa_layer_id, b, shift:].clone()
    #         #     start -= shift
    #         #     end = start + T
    #         #     #cache_position[b, 0] = kv_cache_size

    #         #print(f'start {start} end {end} k_write[b] {k_write[b].shape}')

    #         # 連続領域に安全にcopy_
    #         past_key_value[gqa_layer_id, b, start:end, 0].copy_(k_write[b])
    #         past_key_value[gqa_layer_id, b, start:end, 1].copy_(v_write[b])
    #         #cache_position[b, 0] = end

    #     # === 読み出し ===
    #     max_seq_len = cache_position.max().item()
    #     k_all = past_key_value[gqa_layer_id, :, :max_seq_len, 0].view(B, max_seq_len, kv_per_head, N).transpose(1, 2).contiguous()
    #     v_all = past_key_value[gqa_layer_id, :, :max_seq_len, 1].view(B, max_seq_len, kv_per_head, N).transpose(1, 2).contiguous()

    #     k = repeat_kv_original(k_all, kv_repeat)
    #     v = repeat_kv_original(v_all, kv_repeat)

    #     # Attention mask
    #     S = k.shape[2]
    #     attn_mask = torch.full((B, 1, T, S), float('-inf'), dtype=q.dtype, device=q.device)
    #     for b in range(B):
    #         valid_len = cache_position[b, 0].item()
    #         attn_mask[b, 0, :, :valid_len] = 0.0

    #     attn_output = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask, is_causal=False)
    #     attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H * N)

    #     out = fpx_matmul(attn_output, O_, O_state, ebits, mbits)
    #     x_out = x_in + out

    #     return T5RMSNorm(x_out, ln2, rmsnorm_epsilon), x_out, past_key_value
    #@torch.compile
    def GQA_Attention(layer_id: int, gqa_layer_id: int, H: int, N: int,
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

        # Projections
        # q = fpx_matmul(x, Q_, Q_state, ebits, mbits).add_(Q_bias).view(B, T, H, N)
        # k = fpx_matmul(x, K_, K_state, ebits, mbits).add_(K_bias)
        # v = fpx_matmul(x, V_, V_state, ebits, mbits).add_(V_bias)
        qkv = fpx_matmul(x, QKV_,QKV_state, ebits, mbits)
        q,k,v = qkv.split(qkv_split_list,dim=-1)

        print(f'q = {q.shape} k = {k.shape} v = {v.shape}')

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

        # Rotary embedding (if needed)
        cos, sin = calc_cos.to(k.dtype), calc_sin.to(k.dtype)
        # q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # === Optimized KV Cache Write ===
        # Detect cache layout more reliably
        cache_shape = past_key_value.shape
        
        # Check if dimension 3 is 2 (old layout) or dimension 2 is 2 (new layout)
        # Old: [layer, B, seq, 2, dim] -> shape[3] == 2
        # New: [layer, B, 2, seq, dim] -> shape[2] == 2 and shape[3] != 2
        is_new_layout = (cache_shape[2] == 2) and (cache_shape[3] != 2)
        
        k_write = k.view(B, T, -1)
        v_write = v.view(B, T, -1)
        starts = cache_position[:, 0]
        
        if is_new_layout:
            # New optimized layout: [layer, B, 2, seq, dim]
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
                
            # Optimized read
            max_seq_len = cache_position.max().int()
            k_cache = past_key_value[gqa_layer_id, :, 0, :max_seq_len]
            v_cache = past_key_value[gqa_layer_id, :, 1, :max_seq_len]
            
        else:
            # Old layout: [layer, B, seq, 2, dim]
            if B == 1 or (starts == starts[0]).all():
                start = starts[0].int()
                end = start + T
                past_key_value[gqa_layer_id, :, start:end, 0] = k_write
                past_key_value[gqa_layer_id, :, start:end, 1] = v_write
            else:
                # For old layout with different positions, use loop (compile will optimize)
                for b in range(B):
                    start = cache_position[b, 0].int()
                    end = start + T
                    past_key_value[gqa_layer_id, b, start:end, 0] = k_write[b]
                    past_key_value[gqa_layer_id, b, start:end, 1] = v_write[b]
                    
            # Read with old layout
            max_seq_len = cache_position.max().int()
            k_cache = past_key_value[gqa_layer_id, :, :max_seq_len, 0]
            v_cache = past_key_value[gqa_layer_id, :, :max_seq_len, 1]

        # === Common reshape logic ===
        kv_heads = kv_dim // N
        k_all = k_cache.view(B, -1, kv_heads, N).transpose(1, 2)
        v_all = v_cache.view(B, -1, kv_heads, N).transpose(1, 2)

        # Repeat KV if needed (GQA)
        if kv_repeat > 1:
            k = repeat_kv_original(k_all, kv_repeat)
            v = repeat_kv_original(v_all, kv_repeat)
        else:
            k = k_all
            v = v_all

        # === Optimized Attention Mask ===
        S = k.shape[2]
        
        # Create mask using broadcasting instead of loops
        seq_indices = torch.arange(S, device=q.device, dtype=torch.int32)
        valid_lens = cache_position[:, 0].unsqueeze(1).unsqueeze(2)  # B, 1, 1
        
        # Create mask: True where seq_indices < valid_lens
        mask = seq_indices.unsqueeze(0).unsqueeze(1) < valid_lens  # B, 1, S
        mask = mask.unsqueeze(2).expand(B, 1, T, S)  # B, 1, T, S
        
        # Convert to attention mask values
        attn_mask = torch.where(mask, 0.0, float('-inf')).to(q.dtype)

        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k, v, attn_mask, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, HN)

        # Output projection
        out = fpx_matmul(attn_output, O_, O_state, ebits, mbits)
        x_out = x_in + out

        return T5RMSNorm(x_out, ln2, rmsnorm_epsilon), x_out, past_key_value
    @torch.compile
    def GQA_Attention_Nested(layer_id: int, gqa_layer_id: int, H: int, N: int,
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

        # Projections
        # q = fpx_matmul(x, Q_, Q_state, ebits, mbits).add_(Q_bias).view(B, T, H, N)
        # k = fpx_matmul(x, K_, K_state, ebits, mbits).add_(K_bias)
        # v = fpx_matmul(x, V_, V_state, ebits, mbits).add_(V_bias)
        qkv = fpx_matmul(x, QKV_,QKV_state, ebits, mbits)
        q,k,v = qkv.split(qkv_split_list,dim=-1)

        #print(f'q = {q.shape} k = {k.shape} v = {v.shape}')

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

        # Rotary embedding (if needed)
        cos, sin = calc_cos.to(k.dtype), calc_sin.to(k.dtype)
        # q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # === Optimized KV Cache Write (same as before) ===
        cache_shape = past_key_value.shape
        #print(f'cache shape = {cache_shape}')
        is_new_layout = (cache_shape[2] == 2) and (cache_shape[3] != 2)
        
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
        # else:
        #     if B == 1 or (starts == starts[0]).all():
        #         start = starts[0].int()
        #         end = start + T
        #         past_key_value[gqa_layer_id, :, start:end, 0] = k_write
        #         past_key_value[gqa_layer_id, :, start:end, 1] = v_write
        #     else:
        #         for b in range(B):
        #             start = cache_position[b, 0].int()
        #             end = start + T
        #             past_key_value[gqa_layer_id, b, start:end, 0] = k_write[b]
        #             past_key_value[gqa_layer_id, b, start:end, 1] = v_write[b]

        # === Create Nested Tensors for valid KV only ===
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
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
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
        gate,up = fpx_matmul(x,gateup_,gateup_state,ebits,mbits).split(gateup_split_list,dim=-1)
        xx = fpx_matmul((F.silu(gate) * up),down_,down_state,ebits,mbits)
        x_in = x_in + xx
        return xx, x_in
    


    #@torch.compile
    @torch.compile
    def hxa079r_forward(self, idx, 
                last_wkv_states: List[torch.Tensor], kv_cache,pos_cache,  full_output:bool=False):
        
        with torch.no_grad(): 
            z = self.z

            t0 = time.perf_counter()

            if self.emboncpu:
                x = z['emb.weight'][idx.cpu()].to(device=self.device,dtype=self.dtype)
            else:
                x = z['emb.weight'][idx]

            

            v_first = torch.empty_like(x)
            k_first = torch.empty_like(x)

            cache_pos = pos_cache#[i]
            #print(f'before = {cache_pos}')

            B, T, C = x.shape

            #cos,sin,_ = compute_qwen3_rope_cache(self.max_ctxlen,self.head_size,self.device,torch.float32,self.rope_theta)

            calc_cos, calc_sin = get_batch_rope_cache(self.cos, self.sin, cache_pos, T)

            dummytensor = self.dummytensor

            ln_r =  z.get(f'blocks.{0}.att.ln_r.weight',None)
            rk_normmode = False
            if ln_r is not None:
                rk_normmode = True


            #t1 = time.perf_counter()


            RWKV = 0.0
            GQA = 0.0

            RWKVBlockCount = 0    
            GQABlockCount = 0       

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = dummytensor#last_shift_states[i*2]
                channel_mix_state = dummytensor#last_shift_states[i*2+1]


                if self.HRWKV_Block_Mode[i][0] == 0:
                    time_mix_state = last_wkv_states[self.HRWKV_Block_Mode[i][2]]
                    RWKVBlockCount = RWKVBlockCount + 1
                    #print('rwkv')
                    #t2 = time.perf_counter()
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
                                                                        z[att+'ln_r.weight'],z[att+'ln_k.weight'],self.rms_norm_eps,
                                                                        z[bbb+'ln1.weight'],z[bbb+'ln2.weight'],self.rope_theta,
                                                                        self.ebits,self.mbits
                                                                        )
                    last_wkv_states[self.HRWKV_Block_Mode[i][2]] = time_mix_state
                    #t3 = time.perf_counter()
                    #RWKV+=t3-t2
                else:
                    #print('gqa')
                    GQABlockCount = GQABlockCount + 1
                    #t2 = time.perf_counter()
                    xx, x, kv_cache= HRWKV_7.GQA_Attention_Nested(i,self.HRWKV_Block_Mode[i][2],self.n_head,self.head_size,x,kv_cache,cache_pos,
                                                                        calc_cos,calc_sin,
                                                                        #z[att+'q_proj.weight'], z[att+'k_proj.weight'], z[att+'v_proj.weight'],
                                                                        z[att+'qkv_fused.weight'],self.HRWKV_Misc[att+'qkv_split_list'],
                                                                        z[att+'o_proj.weight'],
                                                                        #z[att+'q_proj.weight.qstate'], z[att+'k_proj.weight.qstate'], z[att+'v_proj.weight.qstate'],
                                                                        z[att+'qkv_fused.weight.qstate'],
                                                                        z[att+'o_proj.weight.qstate'],
                                                                        z[att+'q_proj.bias'], z[att+'k_proj.bias'], z[att+'v_proj.bias'], dummytensor,
                                                                        z[att+'ln_r.weight'],z[att+'ln_k.weight'],self.rms_norm_eps,
                                                                        z[bbb+'ln1.weight'],z[bbb+'ln2.weight'],self.rope_theta,
                                                                        self.ebits,self.mbits )
                    #t3 = time.perf_counter()
                    #GQA +=t3-t2


                xx,x = HRWKV_7.SwiGLU_MLP_forward_fpx_w_add(xx,
                                                            z[ffn+'gateup.weight'],self.HRWKV_Misc[ffn+'gateup_split_list'],
                                                            z[ffn+'down.weight'],

                                                            z[ffn+'gateup.weight.qstate'],
                                                            z[ffn+'down.weight.qstate'],
                                                            self.ebits,self.mbits,
                                                            x
                                                            )
      

                #x = x + xx

                #last_shift_states[i*2] = time_mix_shift
                #last_shift_states[i*2+1] = channel_mix_state


            #GQA = GQA / ( 1 ) * 1000

           #RWKV = RWKV / (1 ) * 1000

            #init = (t1 - t0) * 1000


            #t4 = time.perf_counter()


            
                
                

       

            x = T5RMSNorm(x,z['ln_out.weight'],variance_epsilon=self.rms_norm_eps)
            x = fpx_matmul(x , z['head.weight'],z.get('head.weight.qstate',None),self.ebits,self.mbits)
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持
            pos_cache += T


            # t5 = time.perf_counter()

            # post = (t5-t4) * 1000
            # total = (t5-t0) * 1000 
            #print(f'idx {idx.shape} RWKV = {RWKV}ms GQA = {GQA}ms init = {init}ms post = {post}ms  total = {total}')
            return x, None, last_wkv_states, kv_cache, pos_cache

