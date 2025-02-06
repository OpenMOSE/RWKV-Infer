import torchao
from torchao.dtypes.floatx import to_scaled_tc_floatx
from torchao.ops import quant_llm_linear


import torch
import torch.nn as nn
from typing import Optional,List
import types, gc, os, time, re
from torch.nn import functional as F
import numpy as np
import os, sys
import time
import bitsandbytes as bnb
import functools
from einops import rearrange
from torch.nn import functional as F

#from rwkvengine.misc import PIPELINE
from rwkvengine.misc import PIPELINE, TimeMixState, ChannelMixState,BlockState,BlockStateList
from rwkvengine.matmularena import hybrid_matmul
from rwkvengine.fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
from rwkvengine.fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
from rwkvengine.fla.ops.rwkv7 import chunk_rwkv7,fused_recurrent_rwkv7
from rwkvengine.cuda.wkv7triton import rwkv7_attn_triton

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)


MyStatic = torch.jit.script

@MyStatic
def x070_ChannelMix_Experts_LoRA(x,K_ref,V_ref,K_lora_a,K_lora_b,V_lora_a,V_lora_b,scaling:float=2.0):
    #print('Channel Mix MoE')
    k = torch.relu(hybrid_matmul(x,K_ref) + scaling * F.linear(F.linear(x, K_lora_a), K_lora_b) ) ** 2
    v = hybrid_matmul(k,V_ref) + scaling * F.linear(F.linear(k, V_lora_a), V_lora_b)
    return v

def x070_ChannelMix_Experts_Bone2(x,K_ref,V_ref,K_bone,V_bone):
    #print('Channel Mix MoE')
    temporalweight = K_ref.to(dtype=x.dtype)#.t()
    # print(f'kbone shape = {K_bone.shape}')
    # print(f'vbone shape = {V_bone.shape}')
    # print(f'temporalweight shape = {temporalweight.shape}')
    w = rearrange(temporalweight, '(a r1) (b r2) -> a b r1 r2', r1 = K_bone.shape[1], r2 = K_bone.shape[1]) @ K_bone + K_bone
    w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
    k = torch.relu(x @ (w + temporalweight).t()) ** 2

    #del temporalweight
    #del w

    temporalweight = V_ref.to(dtype=x.dtype)
    w = rearrange(temporalweight, '(a r1) (b r2) -> a b r1 r2', r1 = V_bone.shape[1], r2 = V_bone.shape[1]) @ V_bone + V_bone
    w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')

    v = k @ (w + temporalweight).t()

    return v
@MyStatic
def x070_ChannelMix_Experts_Bone(x,K_ref,V_ref,K_bone,V_bone):
    temporalweight = K_ref.to(dtype=x.dtype)
    # w = rearrange(temporalweight, '(a r1) (b r2) -> a b r1 r2', r1 = K_bone.shape[1], r2 = K_bone.shape[1]) @ K_bone + K_bone
    # w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
    # k = torch.relu(x @ (w + temporalweight).t()) ** 2




    # weight の形状は (a*r, b*r) を想定
    r = K_bone.shape[1]
    a = temporalweight.size(0) // r
    b = temporalweight.size(1) // r

    # 1. (a*r, b*r) -> (a, r, b, r)
    # 2. (a, r, b, r) -> (a, b, r, r) に permute で変形（einops でいう 'a b r1 r2' と同等）
    w_4d = temporalweight.view(a, r, b, r).permute(0, 2, 1, 3)
    # w_4d の形状: [a, b, r, r]

    # 3. バッチ行列積 @ bone + bone （最後の2次元が (r, r) のためブロードキャスト加算が可能）
    w_4d = torch.matmul(w_4d, K_bone) + K_bone  # shape: [a, b, r, r]

    # 4. 再び元の形状 (a*r, b*r) に戻す (permuteを元に戻してから view)
    w_delta = w_4d.permute(0, 2, 1, 3).reshape(a*r, b*r)

    # 5. 最後に (weight + 変形した w_delta) を使って F.linear
    k = torch.relu(x @ (w_delta + temporalweight).t()) ** 2


    # temporalweight = V_ref.to(dtype=x.dtype)
    # w = rearrange(temporalweight, '(a r1) (b r2) -> a b r1 r2', r1 = V_bone.shape[1], r2 = V_bone.shape[1]) @ V_bone + V_bone
    # w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')

    # v = k @ (w + temporalweight).t()
    temporalweight = V_ref.to(dtype=x.dtype)
    r = V_bone.shape[1]
    a = temporalweight.size(0) // r
    b = temporalweight.size(1) // r

    # 1. (a*r, b*r) -> (a, r, b, r)
    # 2. (a, r, b, r) -> (a, b, r, r) に permute で変形（einops でいう 'a b r1 r2' と同等）
    w_4d = temporalweight.view(a, r, b, r).permute(0, 2, 1, 3)
    # w_4d の形状: [a, b, r, r]

    # 3. バッチ行列積 @ bone + bone （最後の2次元が (r, r) のためブロードキャスト加算が可能）
    w_4d = torch.matmul(w_4d, V_bone) + V_bone  # shape: [a, b, r, r]

    # 4. 再び元の形状 (a*r, b*r) に戻す (permuteを元に戻してから view)
    w_delta = w_4d.permute(0, 2, 1, 3).reshape(a*r, b*r)

    v = k @ (w_delta + temporalweight).t()

    return v


class RWKV_7(nn.Module):
    # x070 Multi batch Implementation
    # modified from RWKV-LM v7 demo_fast code @ BlinkDL
    # Now fully supported flash-linear-attention
    # Unofficial MoE Support 

    @MyStatic
    def x070_TimeMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
        B, _ ,_= x.shape  # B, T, H*N
        #xx = x_prev - x
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x 

        #print(f'xx shape = {xx.shape}')
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = torch.tanh(xw @ w1) @ w2
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        #print(f'k shape = {k.shape} k_k shape = {k_k.shape} (k * k_k) shape = {(k * k_k).shape}')

        kk = torch.nn.functional.normalize((k * k_k).view(B,H,N), dim=-1, p=2.0).view(B,1,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

        vk = v.view(B,H,N,1) @ k.view(B,H,1,N)
        #print(f'a shape = {a.shape} kk = {kk.shape}')
        ab = (-kk).view(B,H,N,1) @ (kk*a).view(B,H,1,N)
        #state = state * w.view(B,H,1,N) + state @ ab.float() + vk.float()
        state = state * w.view(B,H,1,N) + state @ ab + vk
        xx = (state.to(dtype=x.dtype) @ r.view(B,H,N,1))

        xx = torch.nn.functional.group_norm(xx.view(B,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,H*N)    
        xx = (xx + ((r * k * r_k).view(B,H,N).sum(dim=-1, keepdim=True) * v.view(B,H,N)).view(B,H*N)).view(B,1,H*N)
        #print(f'TimeMix Before Return XX shape ={ xx.shape}')
        return (xx * g) @ O_, x, state, v_first
    
    
    @MyStatic
    def x070_TimeMix_one_hybrid(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):

        #dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 

        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

       

        #r = xr @ R_
        r = hybrid_matmul(xr,R_)
        w = torch.tanh(xw @ w1) @ w2
        k = hybrid_matmul(xk,K_)

        #v = xv @ V_
        v = hybrid_matmul(xv,V_)

        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        ######## cuda-free method 
        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        #w = torch.exp(-0.606531 * torch.sigmoid((w0 + w)))

        state = state.transpose(-1, -2).contiguous().float()

        t=0
        r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
        vk = v_.view(B,H,N,1) @ k_.view(B,H,1,N)
        ab = (-kk_).view(B,H,N,1) @ (kk_*a_).view(B,H,1,N)
        state = state * w_.view(B,H,1,N) + state @ ab.float() + vk.float()
        #state = state * w_.view(B,H,1,N) + state @ ab + vk
        xx[:,t] = (state.to(dtype=x.dtype) @ r_.view(B,H,N,1)).view(B,H*N)

        state = state.transpose(-1, -2).contiguous()

        #xx = xx.permute(0, 2, 1)  # (B,H*N,T)
        
        xx=xx.view(B,-1)
        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)

        # 元の形状 (B,T,H*N) に戻す
        #xx = xx.permute(0, 2, 1)
        xx=xx.view(B,1,-1)
        

        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        return hybrid_matmul((xx * g) , O_), x[:,-1], state, v_first
        #return (xx * g) @ O_, x[:,-1], state.float(), v_first
        

    
    @MyStatic
    def x070_TimeMix_fla_Step1(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):
        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        #r = xr @ R_
        r = hybrid_matmul(xr,R_)

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5

        #k = xk @ K_
        k = hybrid_matmul(xk,K_)

        #v = xv @ V_
        v = hybrid_matmul(xv,V_)

        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        

        aa=-kk
        bb=kk*a

        return r,w,k,v,g,aa,bb,xx,v_first
    
    def x070_TimeMix_fla_Step2(r, w, k, v, aa, bb,state,FullyFusedMode = True):

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)

        w=-torch.exp(w)
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,C) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        if T>128 and FullyFusedMode == False:
            xx, state = chunk_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state,cu_seqlens=None, output_final_state=True, head_first=False)
        else:
            xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        return xx, state
    @MyStatic
    def x070_TimeMix_fla_Step3(B:int,T:int,H:int,N:int,r,k,r_k,v,g,O_,x,xx,state,v_first,ln_w,ln_b):

        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx.permute(0, 2, 1)  # (B,H*N,T)

        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)#.view(B*T,-1)

        # 元の形状 (B,T,H*N) に戻す
        xx = xx.permute(0, 2, 1)
        #xx = xx.view(B,T,-1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        #return (xx * g) @ O_, x[:,-1], state.float(), v_first
        return hybrid_matmul((xx * g) , O_), x[:,-1], state.float(), v_first
        #return hybrid_matmul((xx * g) , O_), x[:,-1], state, v_first
        #hybrid_matmul


    #@MyStatic
    def x070_TimeMix_fla_combined(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):

        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        w=-torch.exp(w)

        aa=-kk
        bb=kk*a

        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,C) for i in [r,w,k,v,aa,bb]]
        #state = state.permute(0,1,3,2).contiguous()
        if T>128:
            xx, state = chunk_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state,cu_seqlens=None, output_final_state=True, head_first=False)
        else:
            xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        #state = state.permute(0,1,3,2).contiguous()


        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx.permute(0, 2, 1)  # (B,H*N,T)

        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)#.view(B*T,-1)

        # 元の形状 (B,T,H*N) に戻す
        xx = xx.permute(0, 2, 1)
        #xx = xx.view(B,T,-1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        return (xx * g) @ O_, x[:,-1], state.float(), v_first
    

    def x070_TimeMix_seq_fla(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):

        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        #w = torch.tanh(xw @ w1) @ w2
        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        w=-torch.exp(w)

        aa=-kk
        bb=kk*a

        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,C) for i in [r,w,k,v,aa,bb]]

        state = state.permute(0,1,3,2).contiguous()
        #xx, state = chunk_rwkv7(r_.float(), w_.float(), k_.float(), v_.float(), aa_.float(), bb_.float(), scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        xx, state = chunk_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)
        state = state.permute(0,1,3,2).contiguous()


        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx.permute(0, 2, 1)  # (B,H*N,T)

        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)#.view(B*T,-1)

        # 元の形状 (B,T,H*N) に戻す
        xx = xx.permute(0, 2, 1)
        #xx = xx.view(B,T,-1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        return (xx * g) @ O_, x[:,-1], state.float(), v_first


    
    @MyStatic
    def x070_TimeMix_seq(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):

        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 

        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = torch.tanh(xw @ w1) @ w2
        #w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        ######## cuda-free method 
        #w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w)))

        state = state.permute(0,1,3,2).contiguous()
        for t in range(T):
            r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
            vk = v_.view(B,H,N,1) @ k_.view(B,H,1,N)
            ab = (-kk_).view(B,H,N,1) @ (kk_*a_).view(B,H,1,N)
            #state = state * w_.view(B,H,1,N) + state @ ab.float() + vk.float()
            state = state * w_.view(B,H,1,N) + state @ ab + vk
            xx[:,t] = (state.to(dtype=x.dtype) @ r_.view(B,H,N,1)).view(B,H*N)

        state = state.permute(0,1,3,2).contiguous()

        xx = xx.permute(0, 2, 1)  # (B,H*N,T)

        # group_norm適用
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)

        # 元の形状 (B,T,H*N) に戻す
        xx = xx.permute(0, 2, 1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        return (xx * g) @ O_, x[:,-1], state, v_first
        #return (xx * g) @ O_, x[:,-1], state.float(), v_first
    


    @MyStatic
    def x070_ChannelMix_seq(x, x_prev, x_k, K_, V_):
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1, :]], dim=1) - x  # (B,T,H*N)
        k = x + xx * x_k
        #k = torch.relu(k @ K_) ** 2
        k = torch.relu(hybrid_matmul(k , K_)) ** 2

        #hybrid_matmul
        #return k @ V_, x[:,-1,:]
        return hybrid_matmul(k , V_), x[:,-1,:]
    




    @torch.compile
    def x070_ChannelMix_MoE(
        x, x_prev, x_k,
        Router_ref,
        K_ref, V_ref,
        Experts_K: List[List[torch.Tensor]],
        Experts_V: List[List[torch.Tensor]],
        MoETopk: int = 2,
        MoECount: int = 4,
        MoEMode: int = 0
    ):

        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1, :]], dim=1) - x
        hidden_with_tokenshift = x + xx * x_k

        # (B, S, n_embd) → (B*S, n_embd)
        flat_hidden = hidden_with_tokenshift.reshape(-1, hidden_with_tokenshift.size(-1))
        B = flat_hidden.size(0)


        flat_value = torch.zeros_like(flat_hidden)


        if MoEMode == 0:
            out_0 = x070_ChannelMix_Experts_LoRA(
                flat_hidden,
                K_ref, V_ref,
                Experts_K[0][0], Experts_K[0][1],
                Experts_V[0][0], Experts_V[0][1]
            )
        else:
            # 別の MoEMode 実装？
            raise NotImplementedError()

        flat_value += out_0

        router_scores = F.linear(flat_hidden, Router_ref)  # weight = Router_ref

        # topk
        AdaptiveActiveExperts = MoETopk - 1  # 例: 2 なら 1
        topk_values, topk_experts = torch.topk(router_scores, k=AdaptiveActiveExperts, dim=-1)
        gating = F.softmax(topk_values, dim=-1)  # shape [B, AdaptiveActiveExperts]


        topk_experts_flat = topk_experts.reshape(-1)
        gating_flat = gating.reshape(-1)

        source_indices = torch.arange(B, device=flat_hidden.device).unsqueeze(1)
        source_indices = source_indices.expand(B, AdaptiveActiveExperts).reshape(-1)

        for e in range(1, MoECount):
            mask_e = (topk_experts_flat == (e - 0))  # 例: 1番目expertを topk_experts_flat==1

            if not mask_e.any():
                continue

            indices_e = mask_e.nonzero(as_tuple=True)[0]
            input_e = flat_hidden[source_indices[indices_e]]
            # forward
            if MoEMode == 0:
                out_e = x070_ChannelMix_Experts_LoRA(
                    input_e,
                    K_ref, V_ref,
                    Experts_K[e][0], Experts_K[e][1],
                    Experts_V[e][0], Experts_V[e][1]
                )
            else:
                raise NotImplementedError()

            out_e = out_e * gating_flat[indices_e].unsqueeze(-1)
            flat_value.index_add_(0, source_indices[indices_e], out_e)

        # (B*S, n_embd) → (B, S, n_embd) に戻す
        kv = flat_value.view(x.size(0), x.size(1), x.size(2))
        return kv,  x[:,-1,:]

 
    
 