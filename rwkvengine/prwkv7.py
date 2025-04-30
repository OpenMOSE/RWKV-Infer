#ARWKV-7 Inference Implementation
#base architecture by RWKV-Red-Team.
#2025 OpenMOSE

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


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class Phi3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)
    
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    #@torch.compile
    @MyStatic
    def independent_forward(hidden_states,weight,variance_epsilon:float=1e-6):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return (weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
@MyStatic
def fp8_matmul(x,weight,weight_state):
    xg = x
    b = weight
    dtype = x.dtype
    if len(xg.shape) == 2:   
        S0=xg.shape[0]
        if xg.dtype != torch.float8_e4m3fn:
            xscale = 448.0 / torch.max(torch.abs(xg)) + 1e-6
            xg = xg.float() * xscale
            xg = torch.clamp(xg, -448.0, 448.0).to(dtype=torch.float8_e4m3fn).contiguous()
        else:
            xscale = torch.tensor(1.0, device='cuda')

        x = torch._scaled_mm(
            xg.view(S0,xg.shape[1]).to(torch.float8_e4m3fn).contiguous(),
            b.t(),
            bias=None,
            out_dtype=dtype,
            scale_a=1.0 / xscale.float(),
            scale_b=1.0 / weight_state,
            use_fast_accum = True
        )
        return x.view(S0, -1)
    else:

        S0=xg.shape[0]
        S1=xg.shape[1]
        
        if xg.dtype != torch.float8_e4m3fn:
            xscale = 448.0 / torch.max(torch.abs(xg)) + 1e-6
            xg = xg.float() * xscale
            xg = torch.clamp(xg, -448.0, 448.0).to(dtype=torch.float8_e4m3fn).contiguous()
        else:
            xscale = torch.tensor(1.0, device='cuda')
        
        x = torch._scaled_mm(
            xg.view(S0*S1,xg.shape[2]).to(torch.float8_e4m3fn),
            b.t(),
            bias=None,
            out_dtype=dtype,
            scale_a=1.0 / xscale.float(),
            scale_b=1.0 / weight_state,
            use_fast_accum = True
        )
        return x.view(S0, S1, -1)
    
@torch.compile
def fpx_matmul(x,weight,weight_state,ebits:int=3,mbits:int=2):
    if weight.dtype == torch.uint8:
        S0=x.shape[0]
        S1=x.shape[1]
        dtype = x.dtype
        x = x.to(dtype=torch.float16).view(-1,x.shape[2])  
        d = quant_llm_linear(ebits, mbits, x, weight, weight_state).view(S0,S1,-1).to(dtype=dtype)# * 2.0
        return d
    elif weight.dtype == torch.float8_e4m3fn: 
        return fp8_matmul(x,weight,weight_state)
    else:
        return x @ weight.t()
@MyStatic
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
@MyStatic
def T5RMSNorm(hidden_states,weight,variance_epsilon:float=1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)
class PRWKV_7(nn.Module):
    # x070 Multi batch Implementation
    # modified from RWKV-LM v7 demo_fast code @ BlinkDL
    # Now fully supported flash-linear-attention
    # Unofficial ARWKV Support

    
    @MyStatic
    def cxa073_TimeMix_fla_Step1(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2,
                        k_k, k_a, r_k, R_, K_, V_, O_,  R_bias, K_bias, V_bias, O_bias,
                        gate_enable: bool = True
                        ):
        dtype = x.dtype
        B, T, HN = x.shape  # B, T, H*N
        
        xx = x

        xr = xw = xk = xv = xa = xg = x


        r = hybrid_matmul(xr,R_) + R_bias

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.6

        k = hybrid_matmul(xk,K_) + K_bias
        v = hybrid_matmul(xv,V_) + V_bias

        kv_dim = K_.shape[0]

        k = k.view(B, T, int(kv_dim//N), N)
        v = v.view(B, T, int(kv_dim//N), N)

        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)


        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        
        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        

        aa=-kk
        bb=kk*a

        return r,w,k,v,aa,bb,xx,v_first
    

    @MyStatic
    def cxa076_TimeMix_fla_Step1(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2,
                        r_k, R_, K_, V_, O_,  R_bias, K_bias, V_bias, O_bias,
                        gate_enable: bool = True
                        ):
        dtype = x.dtype
        B, T, HN = x.shape  # B, T, H*N
        
        xx = x

        xr = xw = xk = xv = xa = xg = x


        r = hybrid_matmul(xr,R_) + R_bias

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.6

        k = hybrid_matmul(xk,K_) + K_bias
        v = hybrid_matmul(xv,V_) + V_bias

        kv_dim = K_.shape[0]

        k = k.view(B, T, int(kv_dim//N), N)
        v = v.view(B, T, int(kv_dim//N), N)

        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)


        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        
        kk = torch.nn.functional.normalize((k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)

        k = k * (1.0 - w + a)

        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        

        aa=-kk
        bb=kk*a

        return r,w,k,v,aa,bb,xx,v_first
    
    @MyStatic
    def cxa076r_TimeMix_fla_Step1(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2,
                        r_k, R_, K_, V_, O_,  R_bias, K_bias, V_bias, O_bias,
                        ln_r,ln_k,rmsnorm_epsilon:float
                        ):
        dtype = x.dtype
        B, T, HN = x.shape  # B, T, H*N
        
        xx = x

        xr = xw = xk = xv = xa = xg = x


        r = hybrid_matmul(xr,R_) + R_bias

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.6

        k = hybrid_matmul(xk,K_) + K_bias
        v = hybrid_matmul(xv,V_) + V_bias

        r = T5RMSNorm(r.view(B,T,-1,N),ln_r,variance_epsilon=rmsnorm_epsilon)
        k = T5RMSNorm(k.view(B,T,-1,N),ln_k,variance_epsilon=rmsnorm_epsilon)



        kv_dim = K_.shape[0]

        k = k.view(B, T, int(kv_dim//N), N)
        v = v.view(B, T, int(kv_dim//N), N)

        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)


        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        
        kk = torch.nn.functional.normalize((k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)

        k = k * (1.0 - w + a)

        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        

        aa=-kk
        bb=kk*a

        return r,w,k,v,aa,bb,xx,v_first
    



    @torch.compile
    def cxa073_TimeMix_fla_Step2(r, w, k, v, aa, bb,state,FullyFusedMode = True):

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        w = -w.float().exp()
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,C) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        if T>128 and FullyFusedMode == False:
            xx, state = chunk_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state,cu_seqlens=None, output_final_state=True, head_first=False)
        else:
            xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)

        #print(f'xx = {xx.shape}')
        return xx, state
    @MyStatic
    def cxa073_TimeMix_fla_Step3(B:int,T:int,H:int,N:int,r,k,r_k,v,O_,O_bias,x,xx,state,v_first,ln_w,ln_b):

        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx.permute(0, 2, 1)  # (B,H*N,T)
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)
        xx = xx.permute(0, 2, 1)

        xx = xx + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)

        xx=xx.to(dtype=r_k.dtype)

        output = hybrid_matmul((xx) , O_) + O_bias
        return output, x[:,-1], state.float(), v_first
    
    
    
    @MyStatic
    def cxa075_TimeMix_fla_Step3(B:int,T:int,H:int,N:int,r,k,r_k,v,O_,O_bias,x,xx,state,v_first,g1,g2):
      
        g_delta = torch.sigmoid(x @ g1) @ g2
        g = 1.0 + g_delta

        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)
        output = hybrid_matmul((xx*g) , O_) + O_bias
        return output, x[:,-1], state.float(), v_first

    @MyStatic
    def cxa076_TimeMix_fla_Step3(B:int,T:int,H:int,N:int,r,k,r_k,v,O_,O_bias,x,xx,state,v_first,g1,g2):

        g = torch.sigmoid(x @ g1) @ g2

        xx = xx.view(B,T,-1)

        xx = xx * (float(N) ** -0.5)

        xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)
        output = hybrid_matmul((xx*g) , O_) + O_bias
        return output, x[:,-1], state.float(), v_first
    



    ############################################################################################################################################################
    @torch.compile()
    def cxa073_TimeMix_fla_Step1_fpx(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, 
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        R_state,K_state,V_state,O_state,
                        R_bias, K_bias, V_bias, O_bias,
                        ebits:int, mbits:int
                        ):
        dtype = x.dtype
        B, T, HN = x.shape  # B, T, H*N
        
        xx = xr = xw = xk = xv = xa = xg = x



        r = fpx_matmul(xr,R_,R_state,ebits,mbits) + R_bias

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5

        k = fpx_matmul(xk,K_,K_state,ebits,mbits) + K_bias
        v = fpx_matmul(xv,V_,V_state,ebits,mbits) + V_bias

        kv_dim = K_.shape[0]

        k = k.view(B, T, int(kv_dim//N), N)
        v = v.view(B, T, int(kv_dim//N), N)

        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        
        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        

        aa=-kk
        bb=kk*a

        return r,w,k,v,aa,bb,v_first
    
    # @MyStatic 
    # def cxa076_Accelerate_Step1(layer_id: int, H: int, N: int,
    #                            kv_dim,xx,r,k,v,v_first,state,
    #                            w0, w1, w2, a0, a1, a2,
    #                             v0, v1, v2, 
    #                            ):
    #     dtype = xx.dtype
    #     B, T, HN = xx.shape  # B, T, H*N

    #     w = -F.softplus(-(w0 + torch.tanh(xx @ w1) @ w2)) - 0.5

    #     k = k.view(B, T, int(kv_dim//N), N)
    #     v = v.view(B, T, int(kv_dim//N), N)

    #     #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
    #     k = repeat_kv(k, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)
    #     v = repeat_kv(v, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)

    #     k = k.view(B, T, -1)
    #     v = v.view(B, T, -1)

    #     a = torch.sigmoid(a0 + (xx @ a1) @ a2)
        
    #     kk = torch.nn.functional.normalize((k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)

    #     k = k * (1.0 - w + a)

    #     if layer_id == 0: v_first = v
    #     else: v = v + (v_first - v) * torch.sigmoid(v0 + (xx @ v1) @ v2)
     
    #     B,T,HC = w.shape
    #     C = state.shape[3]#64
    #     H = int(HC//C)
        
    #     aa=-kk
    #     bb=kk*a

    #     return r,w,k,v,aa,bb,v_first
    
    @torch.compile()
    def cxa076_TimeMix_fla_Step1_fpx(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, 
                        r_k, R_, K_, V_, O_,
                        R_state,K_state,V_state,O_state,
                        R_bias, K_bias, V_bias, O_bias,
                        ebits:int, mbits:int
                        ):
        dtype = x.dtype
        B, T, HN = x.shape  # B, T, H*N
        kv_dim = K_.shape[0]
        xx = x #xr = xw = xk = xv = xa = xg = x



        r = fpx_matmul(xx,R_,R_state,ebits,mbits) + R_bias
        k = fpx_matmul(xx,K_,K_state,ebits,mbits) + K_bias
        v = fpx_matmul(xx,V_,V_state,ebits,mbits) + V_bias

        w = -F.softplus(-(w0 + torch.tanh(xx @ w1) @ w2)) - 0.5

        k = k.view(B, T, int(kv_dim//N), N)
        v = v.view(B, T, int(kv_dim//N), N)

        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        a = torch.sigmoid(a0 + (xx @ a1) @ a2)
        
        kk = torch.nn.functional.normalize((k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)

        k = k * (1.0 - w + a)

        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xx @ v1) @ v2)
     
        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        
        aa=-kk
        bb=kk*a

        return r,w,k,v,aa,bb,v_first
    
    @torch.compile()
    def cxa076r_TimeMix_fla_Step1_fpx(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, 
                        r_k, R_, K_, V_, O_,
                        R_state,K_state,V_state,O_state,
                        R_bias, K_bias, V_bias, O_bias,
                        ln_r,ln_k,rmsnorm_epsilon:float,
                        ebits:int, mbits:int
                        ):
        dtype = x.dtype
        B, T, HN = x.shape  # B, T, H*N
        kv_dim = K_.shape[0]
        xx = x #xr = xw = xk = xv = xa = xg = x



        r = fpx_matmul(xx,R_,R_state,ebits,mbits) + R_bias
        k = fpx_matmul(xx,K_,K_state,ebits,mbits) + K_bias
        v = fpx_matmul(xx,V_,V_state,ebits,mbits) + V_bias

        r = T5RMSNorm(r.view(B,T,-1,N),ln_r,variance_epsilon=rmsnorm_epsilon)
        k = T5RMSNorm(k.view(B,T,-1,N),ln_k,variance_epsilon=rmsnorm_epsilon)



        w = -F.softplus(-(w0 + torch.tanh(xx @ w1) @ w2)) - 0.5

        k = k.view(B, T, int(kv_dim//N), N)
        v = v.view(B, T, int(kv_dim//N), N)

        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, int(HN//kv_dim))#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        a = torch.sigmoid(a0 + (xx @ a1) @ a2)
        
        kk = torch.nn.functional.normalize((k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)

        k = k * (1.0 - w + a)

        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xx @ v1) @ v2)
     
        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        
        aa=-kk
        bb=kk*a

        return r,w,k,v,aa,bb,v_first

    

    @torch.compile
    def cxa073_TimeMix_fla_Step3_fpx(B:int,T:int,H:int,N:int,r,k,r_k,v,O_,x,xx,state,v_first,ln_w,ln_b,O_state,ebits,mbits):
        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        xx = xx.permute(0, 2, 1)  # (B,H*N,T)
        xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)
        xx = xx.permute(0, 2, 1)

        xx = xx + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)

        xx=xx.to(dtype=r_k.dtype)

        output = fpx_matmul((xx), O_, O_state,ebits,mbits)

        return  output, x[:,-1], state.float(), v_first
    @torch.compile
    def cxa075_TimeMix_fla_Step3_fpx(B:int,T:int,H:int,N:int,r,k,r_k,v,O_,x,xx,state,v_first,O_state,ebits,mbits,g1,g2):
        xx = xx.view(B,T,-1)#.to(dtype=r.dtype)
        
        g_delta = torch.sigmoid(x @ g1) @ g2
        g = 1.0 + g_delta

        xx = xx + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)
        xx=xx.to(dtype=r_k.dtype)
        output = fpx_matmul((xx * g), O_, O_state,ebits,mbits)

        return  output, x[:,-1], state.float(), v_first
    
    @torch.compile
    def cxa076_TimeMix_fla_Step3_fpx(B:int,T:int,H:int,N:int,r,k,r_k,v,O_,x,xx,state,v_first,O_state,ebits,mbits,g1,g2):

        g = torch.sigmoid(x @ g1) @ g2
        xx = xx.view(B,T,-1)#.to(dtype=r.dtype)
        xx = xx * (float(N) ** -0.5)

        xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)
        xx=xx.to(dtype=r_k.dtype)
        output = fpx_matmul((xx * g), O_, O_state,ebits,mbits)

        return  output, x[:,-1], state.float(), v_first



    ##############################################################################################################################################################

    


    '''
    up_states = self.gate_up_proj(hidden_states)

            gate, up_states = up_states.chunk(2, dim=-1)
            up_states = up_states * self.activation_fn(gate)

            return self.down_proj(up_states)
    '''

    @MyStatic
    def cxa073_MLP_PHI3_forward(x,gate_up,down_):
        
        up_states = hybrid_matmul(x,gate_up)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * F.silu(gate)
        return hybrid_matmul(up_states,down_)
    
    @torch.compile
    def cxa073_MLP_PHI3_forward_fpx(x,gate_up,down_,gate_up_state,down_state,ebits,mbits):
        
        up_states = fpx_matmul(x,gate_up,gate_up_state,ebits=ebits,mbits=mbits)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * F.silu(gate)
        return fpx_matmul(up_states,down_,down_state,ebits=ebits,mbits=mbits)


    @MyStatic
    def cxa073_MLP_forward(x,gate_,down_,up_):
        step1 = F.silu(hybrid_matmul(x,gate_)) * hybrid_matmul(x,up_)
        return hybrid_matmul(step1,down_)
    @torch.compile
    def cxa073_MLP_forward_fpx(x,gate_,down_,up_,gate_state,down_state,up_state,ebits,mbits):
        step1 = F.silu(fpx_matmul(x,gate_,gate_state,ebits,mbits)) * fpx_matmul(x,up_,up_state,ebits,mbits)
        return fpx_matmul(step1,down_,down_state,ebits,mbits)
    #@torch.compile
    def cxa073_forward_seq(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor],  full_output:bool=False, KernelMode:int=0, time_offset_state:torch.Tensor=None):
        
        if time_offset_state is None:
                time_offset_state = torch.zeros((self.n_layer,idx.shape[0],self.n_head,self.head_size,self.head_size),dtype=x.dtype, device=x.device)


        with torch.no_grad(): 
            z = self.z

            if self.emboncpu:
                x = z['emb.weight'][idx.cpu()].to(device=self.device,dtype=self.dtype)
            else:
                x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)

            StrategyMode = 0 # 0 is Fully BF16 or FP16 or FP8
            if self.bitfp6quant == True:
                StrategyMode = 3

            dummytensor = torch.tensor(0).to(dtype=x.dtype,device=x.device)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                if offset_tensor is None:
                    y_offset = torch.zeros((idx.shape[0],x.shape[2]),dtype=x.dtype, device=x.device)
                    out_offset = torch.zeros((idx.shape[0],x.shape[2]),dtype=x.dtype, device=x.device)

                #if fullbatch offset_tensor shape = B,n_layer,2,hidden_dim
                
                elif offset_tensor.shape[0] != idx.shape[0]:
                    #maybe got only batch0 offset
                    #so expand realbatch
                    #print('copied')
                    offset_tensor = offset_tensor.repeat(idx.shape[0], 1, 1, 1)

                if offset_tensor is not None:
                    #print(f'offset tensor have!!!')
                    y_offset = offset_tensor[:,i,0,:] # B,Hidden_dim
                    out_offset = offset_tensor[:,i,1,:] # B,Hidden_dim

                #print(f'y_offset = {y_offset}')
                #print(f'out_offset = {out_offset}')


                #print(f'x = {x.dtype}')

                xx = Qwen2RMSNorm.independent_forward(x,z[bbb+'ln1.weight'],variance_epsilon=1e-6)

                #print(f'xx norm = {xx.dtype}')

                B, T, X = xx.shape



                if StrategyMode == 0: 
                        r,w,k,v,aa,bb,xx_step1,v_first = PRWKV_7.cxa073_TimeMix_fla_Step1(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                      
                                                                            #z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                            z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                            z.get(att+'receptance.bias',dummytensor), z.get(att+'key.bias',dummytensor), z.get(att+'value.bias',dummytensor), torch.tensor(0.0,dtype=xx.dtype),
                                                                            self.gate_enable
                                                                            )
                
                        xx_step2, time_mix_state = PRWKV_7.cxa073_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = PRWKV_7.cxa073_TimeMix_fla_Step3(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,z[att+'output.weight'],torch.tensor(0.0,dtype=xx.dtype),
                                                                                                    xx,xx_step2,time_mix_state,v_first,z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                        
                if StrategyMode == 3:
                        r,w,k,v,aa,bb,v_first = PRWKV_7.cxa073_TimeMix_fla_Step1_fpx(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                      
                                                                           # z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                            z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                            z[att+'receptance.weight.qstate'], z[att+'key.weight.qstate'], z[att+'value.weight.qstate'], z[att+'output.weight.qstate'],
                                                                            self.ebits,self.mbits
                                                                            )
                
                        xx_step2, time_mix_state = PRWKV_7.cxa073_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = PRWKV_7.cxa073_TimeMix_fla_Step3_fpx(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,
                                                                                                    z[att+'output.weight'],
                                                                                                    xx,xx_step2,time_mix_state,v_first,
                                                                                                    z[att+'output.weight.qstate'],
                                                                                                    self.ebits,self.mbits,
                                                                                                    y_offset,out_offset
                                                                                                    )



                x = x + xx

                xx = Qwen2RMSNorm.independent_forward(x,z[bbb+'ln2.weight'],variance_epsilon=1e-6)


                if self.ARWKVMLPMode == 0: 
                    if StrategyMode == 0:
                        xx = PRWKV_7.cxa073_MLP_forward(xx,z[ffn+'gate.weight'],z[ffn+'down.weight'],z[ffn+'up.weight'])
                    if StrategyMode == 3:
                        xx = PRWKV_7.cxa073_MLP_forward_fpx(xx,z[ffn+'gate.weight'],z[ffn+'down.weight'],z[ffn+'up.weight'],
                                                        z[ffn+'gate.weight.qstate'],z[ffn+'down.weight.qstate'],z[ffn+'up.weight.qstate'],
                                                        self.ebits,self.mbits
                                                        )
                elif self.ARWKVMLPMode == 1:
                    if StrategyMode == 0:
                        xx = PRWKV_7.cxa073_MLP_PHI3_forward(xx,z[ffn+'gate_up.weight'],z[ffn+'down.weight'])
                        #exit()
                    if StrategyMode == 3:
                        xx = PRWKV_7.cxa073_MLP_PHI3_forward_fpx(xx,z[ffn+'gate_up.weight'],z[ffn+'down.weight'],
                                                        z[ffn+'gate_up.weight.qstate'],z[ffn+'down.weight.qstate'],
                                                        self.ebits,self.mbits
                                                        )

                x = x + xx

                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state

            
            
            #x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = Qwen2RMSNorm.independent_forward(x,z['ln_out.weight'],variance_epsilon=1e-6)
            if StrategyMode == 0:
                x = hybrid_matmul(x , z['head.weight'])
            if StrategyMode == 3:
                x = fpx_matmul(x , z['head.weight'],z['head.weight.qstate'],self.ebits,self.mbits)
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持

            return x, last_shift_states, last_wkv_states
        


    #@torch.compile
    def cxa075_forward_seq(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor],  full_output:bool=False, KernelMode:int=0, time_offset_state:torch.Tensor=None):
        

        with torch.no_grad(): 
            z = self.z

            if self.emboncpu:
                x = z['emb.weight'][idx.cpu()].to(device=self.device,dtype=self.dtype)
            else:
                x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)

            StrategyMode = 0 # 0 is Fully BF16 or FP16 or FP8
            if self.bitfp6quant == True:
                StrategyMode = 3

            dummytensor = torch.tensor(0).to(dtype=x.dtype,device=x.device)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]



                xx = Qwen2RMSNorm.independent_forward(x,z[bbb+'ln1.weight'],variance_epsilon=1e-6)

                B, T, X = xx.shape



                if StrategyMode == 0: 
                        r,w,k,v,aa,bb,xx_step1,v_first = PRWKV_7.cxa073_TimeMix_fla_Step1(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                      
                                                                            #z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                            z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                            z.get(att+'receptance.bias',dummytensor), z.get(att+'key.bias',dummytensor), z.get(att+'value.bias',dummytensor), dummytensor,
                                                                            self.gate_enable
                                                                            )
                
                        xx_step2, time_mix_state = PRWKV_7.cxa073_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = PRWKV_7.cxa075_TimeMix_fla_Step3(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,z[att+'output.weight'],dummytensor,
                                                                                                    xx,xx_step2,time_mix_state,v_first,z[att+'g1'],z[att+'g2'])
                        
                if StrategyMode == 3:
                        r,w,k,v,aa,bb,v_first = PRWKV_7.cxa073_TimeMix_fla_Step1_fpx(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                      
                                                                           # z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                            z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                            z[att+'receptance.weight.qstate'], z[att+'key.weight.qstate'], z[att+'value.weight.qstate'], z[att+'output.weight.qstate'],
                                                                            z.get(att+'receptance.bias',dummytensor), z.get(att+'key.bias',dummytensor), z.get(att+'value.bias',dummytensor), dummytensor,
                                                                            
                                                                            self.ebits,self.mbits
                                                                            )
                
                        xx_step2, time_mix_state = PRWKV_7.cxa073_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = PRWKV_7.cxa075_TimeMix_fla_Step3_fpx(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,
                                                                                                    z[att+'output.weight'],
                                                                                                    xx,xx_step2,time_mix_state,v_first,
                                                                                                    z[att+'output.weight.qstate'],
                                                                                                    self.ebits,self.mbits,
                                                                                                    z[att+'g1'],z[att+'g2']
                                                                                                    )
                x = x + xx

                xx = Qwen2RMSNorm.independent_forward(x,z[bbb+'ln2.weight'],variance_epsilon=1e-6)


                if self.ARWKVMLPMode == 0: 
                    if StrategyMode == 0:
                        xx = PRWKV_7.cxa073_MLP_forward(xx,z[ffn+'gate.weight'],z[ffn+'down.weight'],z[ffn+'up.weight'])
                    if StrategyMode == 3:
                        xx = PRWKV_7.cxa073_MLP_forward_fpx(xx,z[ffn+'gate.weight'],z[ffn+'down.weight'],z[ffn+'up.weight'],
                                                        z[ffn+'gate.weight.qstate'],z[ffn+'down.weight.qstate'],z[ffn+'up.weight.qstate'],
                                                        self.ebits,self.mbits
                                                        )
                elif self.ARWKVMLPMode == 1:
                    if StrategyMode == 0:
                        xx = PRWKV_7.cxa073_MLP_PHI3_forward(xx,z[ffn+'gate_up.weight'],z[ffn+'down.weight'])
                        #exit()
                    if StrategyMode == 3:
                        xx = PRWKV_7.cxa073_MLP_PHI3_forward_fpx(xx,z[ffn+'gate_up.weight'],z[ffn+'down.weight'],
                                                        z[ffn+'gate_up.weight.qstate'],z[ffn+'down.weight.qstate'],
                                                        self.ebits,self.mbits
                                                        )

                x = x + xx

                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state

            
            
            #x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = Qwen2RMSNorm.independent_forward(x,z['ln_out.weight'],variance_epsilon=1e-6)
            if StrategyMode == 0:
                x = hybrid_matmul(x , z['head.weight'])
            if StrategyMode == 3:
                x = fpx_matmul(x , z['head.weight'],z['head.weight.qstate'],self.ebits,self.mbits)
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持

            return x, last_shift_states, last_wkv_states
        


   #@torch.compile
    def cxa076_forward_seq(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor],  full_output:bool=False, KernelMode:int=0, time_offset_state:torch.Tensor=None):
        

        with torch.no_grad(): 
            z = self.z

            if self.emboncpu:
                x = z['emb.weight'][idx.cpu()].to(device=self.device,dtype=self.dtype)
            else:
                x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)

            StrategyMode = 0 # 0 is Fully BF16 or FP16 or FP8
            if self.bitfp6quant == True or self.bitfp8quant == True:
                StrategyMode = 3

            dummytensor = torch.tensor(0).to(dtype=x.dtype,device=x.device)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                ln_r =  z.get(f'blocks.{i}.att.ln_r.weight',None)
                rk_normmode = False
                if ln_r is not None:
                    rk_normmode = True



                xx = Qwen2RMSNorm.independent_forward(x,z[bbb+'ln1.weight'],variance_epsilon=1e-6)

                B, T, X = xx.shape



                if StrategyMode == 0: 
                        if rk_normmode:
                            r,w,k,v,aa,bb,xx_step1,v_first = PRWKV_7.cxa076r_TimeMix_fla_Step1(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                        
                                                                                #z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                                z[att+'r_k'],
                                                                                z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                                z.get(att+'receptance.bias',dummytensor), z.get(att+'key.bias',dummytensor), z.get(att+'value.bias',dummytensor), dummytensor,
                                                                                z[att+'ln_r.weight'],z[att+'ln_k.weight'],1e-6
                                                                                )
                        else:
                            r,w,k,v,aa,bb,xx_step1,v_first = PRWKV_7.cxa076_TimeMix_fla_Step1(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                        
                                                                                #z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                                z[att+'r_k'],
                                                                                z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                                z.get(att+'receptance.bias',dummytensor), z.get(att+'key.bias',dummytensor), z.get(att+'value.bias',dummytensor), dummytensor,
                                                                                self.gate_enable
                                                                                )
                
                        xx_step2, time_mix_state = PRWKV_7.cxa073_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = PRWKV_7.cxa076_TimeMix_fla_Step3(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,z[att+'output.weight'],dummytensor,
                                                                                                    xx,xx_step2,time_mix_state,v_first,z[att+'g1'],z[att+'g2'])
                        
                if StrategyMode == 3:
                        if rk_normmode:
                            r,w,k,v,aa,bb,v_first = PRWKV_7.cxa076r_TimeMix_fla_Step1_fpx(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                        
                                                                            # z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                                z[att+'r_k'],
                                                                                z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                                z[att+'receptance.weight.qstate'], z[att+'key.weight.qstate'], z[att+'value.weight.qstate'], z[att+'output.weight.qstate'],
                                                                                z.get(att+'receptance.bias',dummytensor), z.get(att+'key.bias',dummytensor), z.get(att+'value.bias',dummytensor), dummytensor,
                                                                                z[att+'ln_r.weight'],z[att+'ln_k.weight'],1e-6,
                                                                                self.ebits,self.mbits
                                                                                )
                        else:
                            r,w,k,v,aa,bb,v_first = PRWKV_7.cxa076_TimeMix_fla_Step1_fpx(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                        
                                                                            # z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                                z[att+'r_k'],
                                                                                z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                                z[att+'receptance.weight.qstate'], z[att+'key.weight.qstate'], z[att+'value.weight.qstate'], z[att+'output.weight.qstate'],
                                                                                z.get(att+'receptance.bias',dummytensor), z.get(att+'key.bias',dummytensor), z.get(att+'value.bias',dummytensor), dummytensor,
                                                                                self.ebits,self.mbits
                                                                                )
                
                        xx_step2, time_mix_state = PRWKV_7.cxa073_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = PRWKV_7.cxa076_TimeMix_fla_Step3_fpx(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,
                                                                                                    z[att+'output.weight'],
                                                                                                    xx,xx_step2,time_mix_state,v_first,
                                                                                                    z[att+'output.weight.qstate'],
                                                                                                    self.ebits,self.mbits,
                                                                                                    z[att+'g1'],z[att+'g2']
                                                                                                    )
                x = x + xx

                xx = Qwen2RMSNorm.independent_forward(x,z[bbb+'ln2.weight'],variance_epsilon=1e-6)


                if self.ARWKVMLPMode == 0: 
                    if StrategyMode == 0:
                        xx = PRWKV_7.cxa073_MLP_forward(xx,z[ffn+'gate.weight'],z[ffn+'down.weight'],z[ffn+'up.weight'])
                    if StrategyMode == 3:
                        xx = PRWKV_7.cxa073_MLP_forward_fpx(xx,z[ffn+'gate.weight'],z[ffn+'down.weight'],z[ffn+'up.weight'],
                                                        z[ffn+'gate.weight.qstate'],z[ffn+'down.weight.qstate'],z[ffn+'up.weight.qstate'],
                                                        self.ebits,self.mbits
                                                        )
                elif self.ARWKVMLPMode == 1:
                    if StrategyMode == 0:
                        xx = PRWKV_7.cxa073_MLP_PHI3_forward(xx,z[ffn+'gate_up.weight'],z[ffn+'down.weight'])
                        #exit()
                    if StrategyMode == 3:
                        xx = PRWKV_7.cxa073_MLP_PHI3_forward_fpx(xx,z[ffn+'gate_up.weight'],z[ffn+'down.weight'],
                                                        z[ffn+'gate_up.weight.qstate'],z[ffn+'down.weight.qstate'],
                                                        self.ebits,self.mbits
                                                        )

                x = x + xx

                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state

            
            
            #x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = Qwen2RMSNorm.independent_forward(x,z['ln_out.weight'],variance_epsilon=1e-6)
            if StrategyMode == 0:
                x = hybrid_matmul(x , z['head.weight'])
            if StrategyMode == 3:
                x = fpx_matmul(x , z['head.weight'],z['head.weight.qstate'],self.ebits,self.mbits)
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持

            return x, last_shift_states, last_wkv_states



    def PRWKV7_forward(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor], full_output=False,one_mode=False,KernelMode = 0,time_offset_state:torch.Tensor=None):
      
        isGroupNorm = self.z.get(f'blocks.0.att.ln_x.weight',None)
        isKK = self.z.get(f'blocks.0.att.k_k',None)
        #print(f'isKK = {isKK}')
        if isGroupNorm == None and isKK is not None:
            #print('prwkv 075')
            return PRWKV_7.cxa075_forward_seq(self,idx, last_shift_states,last_wkv_states, full_output,KernelMode,time_offset_state)
        elif isGroupNorm == None:
            #print('prwkv 076')
            return PRWKV_7.cxa076_forward_seq(self,idx, last_shift_states,last_wkv_states, full_output,KernelMode,time_offset_state)
        else:
            return PRWKV_7.cxa073_forward_seq(self,idx, last_shift_states,last_wkv_states, full_output,KernelMode,time_offset_state)
    


