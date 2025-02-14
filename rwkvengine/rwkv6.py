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


#from rwkvengine.misc import PIPELINE
from rwkvengine.misc import PIPELINE, TimeMixState, ChannelMixState,BlockState,BlockStateList
from rwkvengine.matmularena import hybrid_matmul
from rwkvengine.fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
from rwkvengine.fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
from rwkvengine.fla.ops.rwkv7 import chunk_rwkv7
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
def fused_recurrent_rwkv6_torch(
        r: torch.Tensor,      # [B, H, T, K]
        k: torch.Tensor,      # [B, H, T, K]
        v: torch.Tensor,      # [B, H, T, V]
        w: torch.Tensor,      # [B, H, T, K]
        u: torch.Tensor,      # [H, K]
        initial_state: torch.Tensor,  # [B, H, K, V]
        #output_final_state: bool = False,
        #causal: bool = True
    ):
        scale = 1.0 # hardcoded lol
        if scale == -1:
            scale = r.shape[-1] ** -0.5

        q = r
        B, H, T, K = q.shape
        V = v.shape[-1]

        if scale == -1:
            scale = K ** -0.5

        o = torch.zeros(B, H, T, V, dtype=q.dtype, device=q.device)

        if initial_state is not None:
            b_h = initial_state.clone()
        else:
            b_h = torch.zeros(B, H, K, V, dtype=q.dtype, device=q.device)

        idx = 0

        b_k = k[:, :, idx, :]                   # [B, H, K]
        b_v = v[:, :, idx, :]                   # [B, H, V]
        b_q = q[:, :, idx, :] * scale           # [B, H, K]
        b_w = w[:, :, idx, :]                   # [B, H, K]
        b_w = torch.exp(b_w.float()).to(b_w.dtype)  # [B, H, K]

        b_kv = b_k.unsqueeze(-1) * b_v.unsqueeze(-2)  # [B, H, K, V]
        b_u = u.unsqueeze(0).unsqueeze(-1)            # [1, H, K, 1]

        b_o = (b_h + b_kv * b_u) * b_q.unsqueeze(-1)  # [B, H, K, V]
        b_o = b_o.sum(dim=2)                          # [B, H, V]

        b_h = b_h * b_w.unsqueeze(-1) + b_kv          # [B, H, K, V]

        o[:, :, idx, :] = b_o

        final_state = b_h.detach()# if output_final_state else None

        return o, final_state

class RWKV_6(nn.Module):
    @MyStatic
    def x060_First(emb,idx,n_embd:int,ln0_weight,ln0_bias,cpumode:bool):

        #x = F.embedding(idx, emb)
        if cpumode:
                x = emb[idx.cpu()].to(device=idx.device)
        else:
                x = emb[idx]

        x = F.layer_norm(x.to(dtype=ln0_weight.dtype), (n_embd,), weight=ln0_weight, bias=ln0_bias)
        return x
    
    
    @MyStatic
    def x060_TimeMix_FC_Step1(B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
                              time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                              time_decay_w1, time_decay_w2,time_decay,
                              receptance_weight, key_weight, value_weight, gate_weight,
                              ln1_weight,ln1_bias
                              ):

        xx = F.layer_norm(x, (embd,), weight=ln1_weight, bias=ln1_bias)

        x = xx

        #B, T, C = x.size()

        output = torch.concat((last_state_shift, x[:, :-1]), dim=1).to(dtype=time_maa_x.dtype)
        last_state_shift[:] = x[:,-1:]

        xx = output - x

        xxx = torch.addcmul(x, xx, time_maa_x)
        xxx = torch.tanh(xxx @ time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, time_maa_w2).view(5, B, T, -1)

        combined = (xxx + time_wkvrg) * xx + x
        
        xw, xk, xv, xr, xg = combined.unbind(dim=0)
  
        ww = torch.tanh(xw @ time_decay_w1) @ time_decay_w2

        w = (time_decay + ww).exp().neg()

        #print(f'receptance_weight.dtype = {receptance_weight.dtype}')
        

        if receptance_weight.dtype == torch.float8_e4m3fn:
            S0=xr.shape[0]
            S1=xr.shape[1]
            r = torch._scaled_mm(
                xr.view(-1,xr.shape[2]).to(torch.float8_e4m3fn),
                receptance_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
            )
            r = r.view(S0, S1, -1)
        else:
            r = (xr.to(dtype=time_maa_x.dtype) @ receptance_weight.t())

        if key_weight.dtype == torch.float8_e4m3fn:
            S0=xk.shape[0]
            S1=xk.shape[1]
            k = torch._scaled_mm(
                xk.view(-1,xk.shape[2]).to(torch.float8_e4m3fn),
                key_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
            )
            k = k.view(S0, S1, -1)
        else:
            k = (xk.to(dtype=time_maa_x.dtype) @ key_weight.t())

        if value_weight.dtype == torch.float8_e4m3fn:
            S0=xv.shape[0]
            S1=xv.shape[1]
            v = torch._scaled_mm(
                xv.view(-1,xv.shape[2]).to(torch.float8_e4m3fn),
                value_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
            )
            v = v.view(S0, S1, -1)
        else:
            v = (xv.to(dtype=time_maa_x.dtype) @ value_weight.t())

        if gate_weight.dtype == torch.float8_e4m3fn:
            S0=xg.shape[0]
            S1=xg.shape[1]
            g = torch._scaled_mm(
                xg.view(-1,xg.shape[2]).to(torch.float8_e4m3fn),
                gate_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
            )
            g = torch.nn.functional.silu(g.view(S0, S1, -1))
        else:
            g = torch.nn.functional.silu((xg.to(dtype=time_maa_x.dtype) @ gate_weight.t()))
        
        return r, k, v, g, w, xx
    

    @torch.compile#(mode="reduce-overhead")
    def x060_TimeMix_FC_FP6_Step1(B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
                              time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                              time_decay_w1, time_decay_w2,time_decay,
                              receptance_weight, 
                              receptance_qstate,
                              key_weight,
                              key_qstate,
                              value_weight,
                              value_qstate,
                              gate_weight,
                              gate_qstate,
                              ln1_weight,ln1_bias,
                              ebits:int,mbits:int
                              ):

        xx = F.layer_norm(x.to(dtype=ln1_weight.dtype), (embd,), weight=ln1_weight, bias=ln1_bias)

        x = xx

        #B, T, C = x.size()

        output = torch.concat((last_state_shift, x[:, :-1]), dim=1).to(dtype=time_maa_x.dtype)
        last_state_shift[:] = x[:,-1:]

        xx = output - x

        xxx = torch.addcmul(x, xx, time_maa_x)
        xxx = torch.tanh(xxx @ time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, time_maa_w2).view(5, B, T, -1)

        combined = (xxx + time_wkvrg) * xx + x
        
        xw, xk, xv, xr, xg = combined.unbind(dim=0)
  
        ww = torch.tanh(xw @ time_decay_w1) @ time_decay_w2

        w = (time_decay + ww).exp().neg()


        if receptance_weight.dtype == torch.uint8:
            S0=xr.shape[0]
            S1=xr.shape[1]
            xr = xr.to(dtype=torch.float16).view(-1,xr.shape[2])#.cuda().half()
            #print(f'xr = {xr.shape}')
            r = quant_llm_linear(ebits, mbits, xr, receptance_weight, receptance_qstate).view(S0,S1,-1)
        else:
            r = (xr.to(dtype=time_maa_x.dtype) @ receptance_weight.t())


        if key_weight.dtype == torch.uint8:
            S0=xk.shape[0]
            S1=xk.shape[1]
            xk = xk.to(dtype=torch.float16).view(-1,xk.shape[2])#.cuda().half()
            k = quant_llm_linear(ebits, mbits, xk, key_weight, key_qstate).view(S0,S1,-1)
        else:
            k = (xk.to(dtype=time_maa_x.dtype) @ key_weight.t())


        if value_weight.dtype == torch.uint8:
            S0=xv.shape[0]
            S1=xv.shape[1]
            xv = xv.to(dtype=torch.float16).view(-1,xv.shape[2])
            v = quant_llm_linear(ebits, mbits, xv, value_weight, value_qstate).view(S0,S1,-1)
        else:
            v = (xv.to(dtype=time_maa_x.dtype) @ value_weight.t())


        if gate_weight.dtype == torch.uint8:
            S0=xg.shape[0]
            S1=xg.shape[1]
            xg = xg.to(dtype=torch.float16).view(-1,xg.shape[2])
            g = torch.nn.functional.silu(quant_llm_linear(ebits, mbits, xg, gate_weight, gate_qstate).view(S0,S1,-1))
        else:
            g = torch.nn.functional.silu((xg.to(dtype=time_maa_x.dtype) @ gate_weight.t()))
        
        return r, k, v, g, w, xx
    

    
    @MyStatic
    def x060_TimeMix_FC_NF4_Step0(B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
                              time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                              time_decay_w1, time_decay_w2,time_decay,
                              ln1_weight,ln1_bias,
                              ):
        xx = F.layer_norm(x.to(dtype=ln1_weight.dtype), (embd,), weight=ln1_weight, bias=ln1_bias)#.to(dtype=time_maa_x.dtype)

        x = xx

        B, T, C = x.size()
        x = x.contiguous()
        output = torch.concat((last_state_shift, x[:, :-1]), dim=1)#.to(dtype=ln1_weight.dtype)
        last_state_shift[:] = x[:,-1:]

        xx = output - x

        xxx = torch.addcmul(x, xx, time_maa_x).to(dtype=time_maa_x.dtype)
        xxx = torch.tanh(xxx @ time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, time_maa_w2).view(5, B, T, -1)

        combined = (xxx + time_wkvrg) * xx + x
        xw, xk, xv, xr, xg = combined.to(dtype=time_maa_x.dtype).unbind(dim=0)

        ww = torch.tanh(xw @ time_decay_w1) @ time_decay_w2
        w = (time_decay + ww).exp().neg()

        return xw,xk,xv,xr,xg,w,xx
    
    def matmul_4bit_(a,b,b_state):
        outlist = []
        for i in range(a.shape[0]):
            outlist.append(bnb.matmul_4bit(a,b,b_state))
        outlist = torch.cat(outlist, dim=0)

        return outlist

    

    def x060_TimeMix_FC_Step2_Seq(
                         B:int,T:int, C:int, H:int,ctx,
                         x,last_state_wkv,
                         r,w,k,v,g,
                         time_faaaa,
                         ):
        r= r.to(dtype=torch.bfloat16)
        k= k.to(dtype=torch.bfloat16)
        v= v.to(dtype=torch.bfloat16)
        g= g.to(dtype=torch.bfloat16)
        
        x,last_state_wkv[:] = ChunkRWKV6Function.forward(ctx,
            r.view(B,T,H,-1).transpose(1,2),
            k.view(B,T,H,-1).transpose(1,2),
            v.view(B,T,H,-1).transpose(1,2),
            w.view(B,T,H,-1).transpose(1,2),
            time_faaaa.view(H,-1),1.0,
            last_state_wkv,True,
            None,
            True)
        x =x.transpose(1,2)
        x = x.reshape(B,T,C)
        return x, last_state_wkv

    
    @MyStatic
    def x060_TimeMix_FC_Step2_One(B:int,T:int, C:int, H:int,ctx:int,
                         x,last_state_wkv,
                         r,w,k,v,g,
                         time_faaaa,
                         ):
        r= r.to(dtype=torch.bfloat16)
        k= k.to(dtype=torch.bfloat16)
        v= v.to(dtype=torch.bfloat16)
        g= g.to(dtype=torch.bfloat16)

               
        x,last_state_wkv = fused_recurrent_rwkv6_torch(
            r.view(B,T,H,-1).transpose(1,2),
            k.view(B,T,H,-1).transpose(1,2),
            v.view(B,T,H,-1).transpose(1,2),
            w.view(B,T,H,-1).transpose(1,2),
            time_faaaa.view(H,-1),
            last_state_wkv,
            )
          
        x = x.view(B,T,C)
        return x, last_state_wkv
    
    def x060_TimeMix_FC_Step2_One_HighBatch(B:int,T:int, C:int, H:int,ctx:int,
                         x,last_state_wkv,
                         r,w,k,v,g,
                         time_faaaa,
                         ):
        r= r.to(dtype=torch.bfloat16)
        k= k.to(dtype=torch.bfloat16)
        v= v.to(dtype=torch.bfloat16)
        g= g.to(dtype=torch.bfloat16)

        x, last_state_wkv = fused_recurrent_rwkv6(
                r.view(B,T,H,-1).transpose(1,2),
                k.view(B,T,H,-1).transpose(1,2),
                v.view(B,T,H,-1).transpose(1,2),
                w.view(B,T,H,-1).transpose(1,2),
                time_faaaa.view(H,-1),
                1.0,
                last_state_wkv,True, 0)
             
         
        x = x.view(B,T,C)
        return x, last_state_wkv

    @MyStatic
    def x060_TimeMix_FC_Step3(B:int,T:int,C:int,x,g,dim_head:int,
                              ln_x_weight,ln_x_bias,
                              output_weight,
                           ):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = torch.nn.functional.group_norm(x.to(dtype=ln_x_weight.dtype),num_groups = dim_head, weight=ln_x_weight,bias=ln_x_bias, eps= 64e-5).view(B, T, C)
        if output_weight.dtype == torch.float8_e4m3fn:
            xg = x * g
            S0=xg.shape[0]
            S1=xg.shape[1]

            xg = torch.clamp(xg, min=-448.0, max=448.0)
            
            x = torch._scaled_mm(
                xg.view(-1,xg.shape[2]).to(torch.float8_e4m3fn),
                output_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
            )
            return x.view(S0, S1, -1)
        else:
            return (x * g).to(dtype=output_weight.dtype) @ output_weight.t()
        
    #@MyStatic
    #@torch.compile
    @torch.compile(mode="reduce-overhead")
    def x060_TimeMix_FC_FP6_Step3(B:int,T:int,C:int,x,g,dim_head:int,
                              ln_x_weight,ln_x_bias,
                              output_weight,
                              output_qstate,
                              ebits:int,mbits:int
                           ):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = torch.nn.functional.group_norm(x.to(dtype=ln_x_weight.dtype),num_groups = dim_head, weight=ln_x_weight,bias=ln_x_bias, eps= 64e-5).view(B, T, C)
   
        if output_weight.dtype == torch.uint8:
            xg = ( x * g )
            S0=xg.shape[0]
            S1=xg.shape[1]
            xg = xg.to(dtype=torch.float16).view(-1,xg.shape[2])
            o = quant_llm_linear(ebits, mbits, xg, output_weight, output_qstate).view(S0,S1,-1).to(dtype=ln_x_weight.dtype)
        else:
            return (x * g).to(dtype=output_weight.dtype) @ output_weight.t()
        return o
        


    
    @MyStatic
    def x060_ChannelMix_FC_Step1(x,last_state,
                            ln2_weight,
                            ln2_bias,
                            n_embd:int,
                            time_maa_k,
                            time_maa_r,
                            receptance_weight,
                            key_weight,
                            value_weight
                            ):
        #transfered ln2 norm here 
        x = F.layer_norm(x.to(dtype=ln2_weight.dtype), (n_embd,), weight=ln2_weight, bias=ln2_bias)

        xx = torch.concat((last_state, x[:, :-1]),
                          dim=1).to(dtype=time_maa_k.dtype)
        last_state[:] = x[:, -1:]
        
        
        xk = xx * time_maa_k + x * (1 - time_maa_k)
        xr = xx * time_maa_r + x * (1 - time_maa_r)
        

        
        if key_weight.dtype == torch.float8_e4m3fn:
            S0=xk.shape[0] 
            S1=xk.shape[1]
            xkg = torch._scaled_mm(
                xk.view(-1,xk.shape[2]).to(torch.float8_e4m3fn),
                key_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
                )
            xkg = xkg.view(S0, S1, -1)
            xkg = (torch.relu(xkg) ** 2)
        else:
            xkg = (torch.relu(xk.to(dtype=key_weight.dtype) @ key_weight.t()) ** 2)
        if value_weight.dtype == torch.float8_e4m3fn:
            S0=xkg.shape[0] 
            S1=xkg.shape[1]
            #xkg = xkg * 0.333
            xkg = torch.clamp(xkg, min=-448.0, max=448.0)
            xkv = torch._scaled_mm(
                xkg.view(-1,xkg.shape[2]).to(torch.float8_e4m3fn),
                value_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
                )
            kv = xkv.view(S0, S1, -1)# * 2
        else:
            kv = xkg @ value_weight.t()

        if receptance_weight.dtype == torch.float8_e4m3fn:
            S0=xr.shape[0] 
            S1=xr.shape[1]
            xkr = torch._scaled_mm(
                xr.view(-1,xr.shape[2]).to(torch.float8_e4m3fn),
                receptance_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
                )
            xkr = xkr.view(S0, S1, -1)
            return torch.sigmoid(xkr) * kv, last_state
        else:
            return torch.sigmoid(
                                xr.to(dtype=receptance_weight.dtype) @ receptance_weight.t() 
                                ) * kv, last_state
        

    @torch.compile
    def x060_ChannelMix_FC_FP6_Step1(x,last_state,
                            ln2_weight,
                            ln2_bias,
                            n_embd:int,
                            time_maa_k,
                            time_maa_r,
                            receptance_weight,
                            receptance_qstate,
                            key_weight,
                            key_qstate,
                            value_weight,
                            value_qstate,
                            ebits:int,mbits:int
                            ):
        #transfered ln2 norm here 
        x = F.layer_norm(x.to(dtype=ln2_weight.dtype), (n_embd,), weight=ln2_weight, bias=ln2_bias)

        xx = torch.concat((last_state, x[:, :-1]),
                          dim=1).to(dtype=time_maa_k.dtype)
        last_state[:] = x[:, -1:]
        
        
        xk = xx * time_maa_k + x * (1 - time_maa_k)
        xr = xx * time_maa_r + x * (1 - time_maa_r)

        if key_weight.dtype == torch.uint8:  
            S0=xk.shape[0]
            S1=xk.shape[1]
            xk = xk.to(dtype=torch.float16).view(-1,xk.shape[2])      
            xkg = quant_llm_linear(ebits, mbits, xk, key_weight, key_qstate).view(S0,S1,-1)#.to(dtype=ln_x_weight.dtype)
            xkg = torch.relu(xkg) ** 2
        else:
            xkg = (torch.relu(xk.to(dtype=key_weight.dtype) @ key_weight.t()) ** 2)

        if value_weight.dtype == torch.uint8:
            S0=xkg.shape[0]
            S1=xkg.shape[1]
            xkg = xkg.to(dtype=torch.float16).view(-1,xkg.shape[2])  
            kv = quant_llm_linear(ebits, mbits, xkg, value_weight, value_qstate).view(S0,S1,-1)
        else:
            kv = xkg @ value_weight.t()

        if receptance_weight.dtype == torch.uint8:
            S0=xr.shape[0]
            S1=xr.shape[1]
            xr = xr.to(dtype=torch.float16).view(-1,xr.shape[2])  
            xrr =  quant_llm_linear(ebits, mbits, xr, receptance_weight, receptance_qstate).view(S0,S1,-1) 
            return torch.sigmoid(   xrr
                                    #xr.to(dtype=receptance_weight.dtype) @ receptance_weight.t() 
                                    ) * kv, last_state
        else:
            return torch.sigmoid(   
                                    xr.to(dtype=receptance_weight.dtype) @ receptance_weight.t() 
                                    ) * kv, last_state

    

    @MyStatic
    def x060_Final(x,head_weight,n_embd:int,ln_out_weight,ln_out_bias):
        x = F.layer_norm(x.to(dtype=ln_out_weight.dtype), (n_embd,), weight=ln_out_weight, bias=ln_out_bias)
        
        if head_weight.dtype == torch.float8_e4m3fn:

            S0=x.shape[0]
            S1=x.shape[1]

            x = torch.clamp(x, min=-448.0, max=448.0)

            x = torch._scaled_mm(
                x.view(-1,x.shape[2]).to(torch.float8_e4m3fn),
                head_weight.t(),
                bias=None,
                out_dtype=torch.bfloat16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
            )
            x = x.view(S0, S1, -1)
            return x
        else:
            x = x.to(dtype=head_weight.dtype)
            x = x @ head_weight.t()
            return x

    
    @torch.compile
    def x060_Final_FP6(x,head_weight,head_qstate,n_embd:int,ln_out_weight,ln_out_bias,ebits:int,mbits:int):
        x = F.layer_norm(x.to(dtype=ln_out_weight.dtype), (n_embd,), weight=ln_out_weight, bias=ln_out_bias)
        if head_weight.dtype == torch.uint8:
            S0=x.shape[0]
            S1=x.shape[1]
            x = x.to(dtype=torch.float16).view(-1,x.shape[2])
            x = quant_llm_linear(ebits, mbits, x, head_weight, head_qstate).view(S0,S1,-1)
        else:
            x = x.to(dtype=head_weight.dtype)
            x = x @ head_weight.t()

        return x
    

    #@torch.compile
    def x060_forward(self, idx: torch.Tensor, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor]):
        StrategyMode = 0 # 0 is Fully BF16 or FP16 or FP8
        if self.bit4quant == True:
            StrategyMode = 2
        elif self.bitfp6quant == True:
            StrategyMode = 3
            
        with torch.no_grad():
            #
            z = self.z
            H = self.n_head

            x = RWKV_6.x060_First(z['emb.weight'],idx,self.n_embd,z['blocks.0.ln0.weight'],z['blocks.0.ln0.bias'],self.emboncpu)
            x=x.to(dtype=self.base_precision)
            B, T, C = x.size()

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'



                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                if StrategyMode == 0:
                    r,k,v,g,w,xx = RWKV_6.x060_TimeMix_FC_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],
                                                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'],z[att+'gate.weight'],
                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias']
                                                    )
                    if T>1:
                        att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        if B < 16:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = RWKV_6.x060_TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight']
                                               )
                    

                    x = x + att1

                    ffn1, channel_mix_state = RWKV_6.x060_ChannelMix_FC_Step1(x,channel_mix_state,
                                                                     z[bbb+'ln2.weight'],
                                                                     z[bbb+'ln2.bias'],
                                                                     int(self.n_embd),
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     z[ffn+'receptance.weight'],
                                                                     z[ffn+'key.weight'],
                                                                     z[ffn+'value.weight']
                                                                     )
                    
                    x = x + ffn1



                elif StrategyMode == 3: #FP6
                    r,k,v,g,w,xx = RWKV_6.x060_TimeMix_FC_FP6_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],
                                                    z[att+'receptance.weight'].to(device='cuda'),
                                                    z[att+'receptance.weight.qstate'].to(device='cuda'),
                                                    z[att+'key.weight'].to(device='cuda'),
                                                    z[att+'key.weight.qstate'].to(device='cuda'),
                                                    z[att+'value.weight'].to(device='cuda'),
                                                    z[att+'value.weight.qstate'].to(device='cuda'),
                                                    z[att+'gate.weight'].to(device='cuda'),
                                                    z[att+'gate.weight.qstate'].to(device='cuda'),
                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias'],
                                                    self.ebits, self.mbits
                                                    )
                    if T>1:
                        att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        if B < 16:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = RWKV_6.x060_TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = RWKV_6.x060_TimeMix_FC_FP6_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight'].to(device='cuda'),
                                               z[att+'output.weight.qstate'].to(device='cuda'),
                                               self.ebits, self.mbits
                                               )

                    

                    x = x + att1

                    ffn1, channel_mix_state = RWKV_6.x060_ChannelMix_FC_FP6_Step1(x,channel_mix_state,
                                                                     z[bbb+'ln2.weight'],
                                                                     z[bbb+'ln2.bias'],
                                                                     int(self.n_embd),
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     z[ffn+'receptance.weight'].to(device='cuda'),
                                                                     z[ffn+'receptance.weight.qstate'].to(device='cuda'),
                                                                     z[ffn+'key.weight'].to(device='cuda'),
                                                                     z[ffn+'key.weight.qstate'].to(device='cuda'),
                                                                     z[ffn+'value.weight'].to(device='cuda'),
                                                                     z[ffn+'value.weight.qstate'].to(device='cuda'),
                                                                     self.ebits, self.mbits
                                                                     )

                    
                    x = x + ffn1
                
                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state



            if self.bitfp6quant:
                x = RWKV_6.x060_Final_FP6(x,z['head.weight'].to(device='cuda'),
                                   z['head.weight.qstate'].to(device='cuda'),
                            self.n_embd,z['ln_out.weight'],z['ln_out.bias'],
                            self.ebits, self.mbits
                            )
            else:
                x = RWKV_6.x060_Final(x,z['head.weight'],self.n_embd,z['ln_out.weight'],z['ln_out.bias'])            

            return x, last_shift_states, last_wkv_states