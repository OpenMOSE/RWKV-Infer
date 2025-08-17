# RWKV x070 Multi Batch Implementation
# 2025 OpenMOSE

import os

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "True"
os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "True"

import time
import torch
from collections import defaultdict
#from flash_attn import flash_attn_varlen_func, flash_attn_func

from torch.nn.attention import SDPBackend, sdpa_kernel


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


from rwkvengine.misc import PIPELINE, TimeMixState, ChannelMixState,BlockState,BlockStateList
from rwkvengine.matmularena import hybrid_matmul
from rwkvengine.quantization import fpx_matmul

from rwkvengine.fla.ops.rwkv7 import fused_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7

class RWKV_7(nn.Module):
    @torch.compile
    def x070_TimeMix(layer_id: int, H: int, N: int,
                        x_in, x_prev, v_first,state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a,
                        r_k, 
                        R_, K_, V_,
                        O_,
                        R_state,K_state,V_state,
                        O_state,

                        ln1_w,ln1_b,
                        ln2_w,ln2_b,
                        ln_x_w,ln_x_b,
                        ebits:int, mbits:int
                        ):
        
        
        x = F.layer_norm(x_in, (x_in.shape[2],), weight=ln1_w, bias=ln1_b)

        xx = torch.cat([x_prev, x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        B, T, HN = x.shape
        r = fpx_matmul(xr, R_ ,R_state, ebits, mbits)
        k = fpx_matmul(xk, K_ ,K_state, ebits, mbits)
        v = fpx_matmul(xv, V_ ,V_state, ebits, mbits)

        if layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid((xv @ v1)@v2 + v0)

        a = torch.sigmoid((xa @ a1) @ a2 + a0)

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        k = fused_k_rwkv7(k, a, k_a)


        w = torch.tanh(xw @ w1) @ w2 + w0
        w = -F.softplus(-w) - 0.5

        aa = -kk
        bb = kk * a
        w = -w.to(dtype=torch.float32).exp()

        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,N) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)

        xx = xx.view(B * T, -1)
        xx = torch.nn.functional.group_norm(xx.to(dtype=ln_x_w.dtype),num_groups = H, weight=ln_x_w,bias=ln_x_b, eps= 64e-5).view(B, T, -1)

        xx = xx.to(dtype=r.dtype) + ((r.view(B,T,H,N)*k.view(B,T,H,N)*r_k.view(H,N)).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,HN)

        output = fpx_matmul((xx * (torch.sigmoid(xg @ g1) @ g2)), O_, O_state,ebits,mbits)

        x_in = x_in + output
         
        output = F.layer_norm(x_in, (x.shape[2],), weight=ln2_w, bias=ln2_b)

        return  output, x[:,-1], state, v_first, x_in


    @torch.compile
    def x070_ChannelMix(x, x_prev, x_k,
                             K_, V_,
                             K_State,V_State,
                             ebits:int, mbits:int,x_in
                             ):
        xx = torch.cat([x_prev, x[:, :-1]], dim=1) - x  # (B,T,H*N)
        k = x + xx * x_k
        k = torch.relu(fpx_matmul(k , K_, K_State,ebits,mbits)) ** 2
        xx = fpx_matmul(k , V_, V_State,ebits,mbits)
        x_in = x_in + xx
        return xx, x[:,-1:], x_in

    #@torch.compile
    def x070_forward(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor], full_output:bool=False):
        
        with torch.no_grad(): 
            z = self.z

            if z['model.embed_tokens.weight'].device.type == 'cpu':
                x = z['model.embed_tokens.weight'][idx.cpu()].to(device=self.device,dtype=self.base_precision)
            else:
                x = z['model.embed_tokens.weight'][idx]

            v_first = torch.empty_like(x)
            B, T, C = x.shape
            dummytensor = self.dummytensor

            for i in range(self.n_layer):
                bbb = f'model.layers.{i}.'
                att = f'model.layers.{i}.self_attn.'
                ffn = f'model.layers.{i}.mlp.'
                ffn1 = f'model.layers.{i+1}.mlp.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

           

                xx, time_mix_shift, time_mix_state, v_first, x = RWKV_7.x070_TimeMix(layer_id=i, H=self.n_head, N=self.head_size,
                                                                                    x_in = x, x_prev=time_mix_shift, v_first = v_first,state= time_mix_state,
                                                                                    x_r=z[att+'x_r'], x_w=z[att+'x_w'], x_k=z[att+'x_k'], x_v=z[att+'x_v'], x_a=z[att+'x_a'], x_g=z[att+'x_g'],
                                                                                    w0=z[att+'w0'], w1=z[att+'w1'], w2=z[att+'w2'], a0=z[att+'a0'], a1=z[att+'a1'], a2=z[att+'a2'],
                                                                                    v0=z[att+'v0'], v1=z[att+'v1'], v2=z[att+'v2'], g1=z[att+'g1'], g2=z[att+'g2'],
                                                                                    
                                                                                    k_k=z[att+'k_k'], k_a=z[att+'k_a'],
                                                                                    r_k=z[att+'r_k'], 
                                                                                    R_=z[att+'receptance.weight'], K_=z[att+'key.weight'], V_=z[att+'value.weight'],
                                                                                    O_=z[att+'output.weight'],
                                                                                    R_state=z[att+'receptance.weight.qstate'],K_state=z[att+'key.weight.qstate'],V_state=z[att+'value.weight.qstate'],
                                                                                    O_state=z[att+'output.weight.qstate'],

                                                                                    ln1_w=z[bbb+'input_layernorm.weight'] ,ln1_b=z[bbb+'input_layernorm.bias'],
                                                                                    ln2_w=z[bbb+'post_attention_layernorm.weight'],ln2_b=z[bbb+'post_attention_layernorm.bias'],
                                                                                    ln_x_w=z[att+'ln_x.weight'],ln_x_b=z[att+'ln_x.bias'],
                                                                                    ebits=self.attn_ebits, mbits=self.attn_mbits
                                                                                    )


                xx,channel_mix_state,x = RWKV_7.x070_ChannelMix(x=xx, x_prev=channel_mix_state, x_k=z[ffn+'x_k'],
                                                                K_=z[ffn+'key.weight'], V_=z[ffn+'value.weight'],
                                                                K_State=z[ffn+'key.weight.qstate'],V_State=z[ffn+'value.weight.qstate'],
                                                                ebits=self.mlp_ebits, mbits=self.mlp_mbits,
                                                                x_in = x
                                                                )
 
                # xx,x = HRWKV_7.SwiGLU_MLP_forward_fpx_w_add(xx,
                #                                             z[ffn+'gateup.weight'],self.HRWKV_Misc[ffn+'gateup_split_list'],
                #                                             z[ffn+'down_proj.weight'],

                #                                             z[ffn+'gateup.weight.qstate'],
                #                                             z[ffn+'down_proj.weight.qstate'],
                #                                             self.mlp_ebits,self.mlp_mbits,
                #                                             x
                #                                             )
       
                last_shift_states[i*2] = time_mix_shift.unsqueeze(1)
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state


            #x = T5RMSNorm(x,z['model.norm.weight'],variance_epsilon=self.rms_norm_eps)
            x = F.layer_norm(x, (self.n_embd,), weight=z['model.norm.weight'], bias=z['model.norm.bias'])
            x = fpx_matmul(x , z['lm_head.weight'],z.get('lm_head.weight.qstate',None),self.head_ebits,self.head_mbits)

            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持


            return x, last_shift_states, last_wkv_states