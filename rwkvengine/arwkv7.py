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
    
@torch.compile
def fpx_matmul(x,weight,weight_state,ebits:int,mbits:int):
    S0=x.shape[0]
    S1=x.shape[1]
    dtype = x.dtype
    x = x.to(dtype=torch.float16).view(-1,x.shape[2])  
    d = quant_llm_linear(ebits, mbits, x, weight, weight_state).view(S0,S1,-1).to(dtype=dtype)
    return d



class ARWKV_7(nn.Module):
    # x070 Multi batch Implementation
    # modified from RWKV-LM v7 demo_fast code @ BlinkDL
    # Now fully supported flash-linear-attention
    # Unofficial ARWKV Support

    
    @MyStatic
    def ax070_TimeMix_fla_Step1(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,


                       x_r, x_w, x_k, x_v, x_a, x_g,



                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ):
        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 

        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g



        r = hybrid_matmul(xr,R_)

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5

        k = hybrid_matmul(xk,K_)
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
    
    def ax070_TimeMix_fla_Step2(r, w, k, v, aa, bb,state,FullyFusedMode = True):

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
    def ax070_TimeMix_fla_Step3(B:int,T:int,H:int,N:int,r,k,r_k,v,g,O_,x,xx,state,v_first):

        xx = xx.view(B,T,-1).to(dtype=r.dtype)

        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        return hybrid_matmul((xx * g) , O_), x[:,-1], state.float(), v_first
    



    ############################################################################################################################################################
    @torch.compile()
    def ax070_TimeMix_fla_Step1_fpx(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                       x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        R_state,K_state,V_state,O_state,
                        ebits:int, mbits:int
                        ):
        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 

        #print(f'x = {x.dtype}, xx = {xx.dtype} xw = {x_w.dtype}')

        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g



        r = fpx_matmul(xr,R_,R_state,ebits,mbits)

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5

        k = fpx_matmul(xk,K_,K_state,ebits,mbits)
        v = fpx_matmul(xv,V_,V_state,ebits,mbits)

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
    

    @torch.compile
    def ax070_TimeMix_fla_Step3_fpx(B:int,T:int,H:int,N:int,r,k,r_k,v,g,O_,x,xx,state,v_first,O_state,ebits,mbits):

        xx = xx.view(B,T,-1).to(dtype=r.dtype)

        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        return fpx_matmul((xx * g) , O_, O_state,ebits,mbits), x[:,-1], state.float(), v_first



    ##############################################################################################################################################################

    






    @MyStatic
    def ax070_MLP_forward(x,gate_,down_,up_):
        step1 = F.silu(hybrid_matmul(x,gate_)) * hybrid_matmul(x,up_)
        return hybrid_matmul(step1,down_)
    @torch.compile
    def ax070_MLP_forward_fpx(x,gate_,down_,up_,gate_state,down_state,up_state,ebits,mbits):
        step1 = F.silu(fpx_matmul(x,gate_,gate_state,ebits,mbits)) * fpx_matmul(x,up_,up_state,ebits,mbits)
        return fpx_matmul(step1,down_,down_state,ebits,mbits)
    
    def ax070_forward_seq(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor],  full_output:bool=False, KernelMode:int=0):
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



            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                #print(f'x = {x.dtype}')

                xx = Qwen2RMSNorm.independent_forward(x,z[bbb+'ln1.weight'])

                #print(f'xx norm = {xx.dtype}')

                B, T, X = xx.shape

                if StrategyMode == 0:
                        r,w,k,v,g,aa,bb,xx_step1,v_first = ARWKV_7.ax070_TimeMix_fla_Step1(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                      
                                                                            z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                            z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                            )
                
                        xx_step2, time_mix_state = ARWKV_7.ax070_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = ARWKV_7.ax070_TimeMix_fla_Step3(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,g,z[att+'output.weight'],
                                                                                                    xx,xx_step2,time_mix_state,v_first)
                        
                if StrategyMode == 3:
                        r,w,k,v,g,aa,bb,xx_step1,v_first = ARWKV_7.ax070_TimeMix_fla_Step1_fpx(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                      
                                                                            z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                            z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                            z[att+'receptance.weight.qstate'], z[att+'key.weight.qstate'], z[att+'value.weight.qstate'], z[att+'output.weight.qstate'],
                                                                            self.ebits,self.mbits
                                                                            )
                
                        xx_step2, time_mix_state = ARWKV_7.ax070_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                        xx, time_mix_shift, time_mix_state, v_first = ARWKV_7.ax070_TimeMix_fla_Step3_fpx(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,g,z[att+'output.weight'],
                                                                                                    xx,xx_step2,time_mix_state,v_first,
                                                                                                    z[att+'output.weight.qstate'],
                                                                                                    self.ebits,self.mbits
                                                                                                    )



                x = x + xx

                xx = Qwen2RMSNorm.independent_forward(x,z[bbb+'ln2.weight'])

                if StrategyMode == 0:
                    xx = ARWKV_7.ax070_MLP_forward(xx,z[ffn+'gate.weight'],z[ffn+'down.weight'],z[ffn+'up.weight'])
                if StrategyMode == 3:
                    xx = ARWKV_7.ax070_MLP_forward_fpx(xx,z[ffn+'gate.weight'],z[ffn+'down.weight'],z[ffn+'up.weight'],
                                                       z[ffn+'gate.weight.qstate'],z[ffn+'down.weight.qstate'],z[ffn+'up.weight.qstate'],
                                                       self.ebits,self.mbits
                                                       )

                x = x + xx

                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state

            
            
            #x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = Qwen2RMSNorm.independent_forward(x,z['ln_out.weight'])
            if StrategyMode == 0:
                x = hybrid_matmul(x , z['head.weight'])
            if StrategyMode == 3:
                x = fpx_matmul(x , z['head.weight'],z['head.weight.qstate'],self.ebits,self.mbits)
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持

            return x, last_shift_states, last_wkv_states



    def ax070_forward(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor], full_output=False,one_mode=False,KernelMode = 0):
        return ARWKV_7.ax070_forward_seq(self,idx, last_shift_states,last_wkv_states, full_output,KernelMode)
    


