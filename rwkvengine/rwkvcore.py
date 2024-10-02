#Refactoring RWKV x060 Inference Engine with Flash Linear Attention
#2024 OpenMOSE

from fla.ops.gla import fused_chunk_gla
from fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6


import torch
import torch.nn as nn
from typing import Optional
import types, gc, os, time, re
from typing import List
from torch.nn import functional as F
import numpy as np
import os, sys
import time
import bitsandbytes as bnb
import functools
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity
mode = 0
#from rwkvengine.misc import PIPELINE
from misc import PIPELINE
from matmularena import custom_matmul

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

torch._dynamo.config.suppress_errors = True


MyStatic = torch.jit.script

class TimeMixState:
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state
class BlockState:
    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state
class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, H, device, dtype):
        result = BlockStateList.empty(N, B, C, H, device, dtype)
        #print(f'dtype = {dtype}')
        result.wkv_states[:] = 0
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        wkv_states = torch.empty((N, B, H, C//H, C//H),
                                 device=device,
                                 dtype=torch.bfloat16)
        shift_states = torch.empty((N*2,B,1, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state
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

        #output_final_state = True
        q = r
        B, H, T, K = q.shape
        V = v.shape[-1]

        if scale == -1:
            scale = K ** -0.5

        # 出力テンソルを初期化
        o = torch.zeros(B, H, T, V, dtype=q.dtype, device=q.device)

        # 初期状態を設定
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
    def __init__(self,load_model: str,quantize:bool = False,base_precision: str = 'int8'):
        print('Helloworld RWKV v060 :) Initializing')
        super().__init__()
        #GANBATTE CODE KAKOU
        self.time_debug = False
        self.bit8quant = False
        self.bit4quant = False
        self.bitfp8quant = False

        #base_precision = ''

        if base_precision == 'fp16':
            self.base_precision = torch.float16
        elif base_precision == 'int8':
            self.base_precision = torch.bfloat16
            self.bit8quant = True
            self.bit4quant = False
        elif base_precision == 'fp16int8':
            self.base_precision = torch.float16
            self.bit8quant = True
            self.bit4quant = False
        elif base_precision == 'nf4':
            self.base_precision = torch.float16
            self.bit8quant = False
            self.bit4quant = True
        elif base_precision == 'fp8':
            self.base_precision = torch.float16
            self.bit8quant = False
            self.bit4quant = False
            self.bitfp8quant = True
        else:
            self.base_precision = torch.bfloat16
        
        modelpath = load_model

        z = torch.load(modelpath,map_location="cpu",mmap=True)
        keys = list(z.keys())
        print("keys", keys)

        keys = list(z.keys())

        # detect model details
        vocab_size, n_embd = z["emb.weight"].shape
        n_embd = n_embd
        vocab_size = vocab_size
        n_layer = 0
        for key in keys:
            if key.startswith("blocks."):
                layer = int(key.split(".")[1])
                if layer > n_layer:
                    n_layer = layer
        n_layer = n_layer + 1
        print("n_layer", n_layer)
        dim_ffn = z[f"blocks.0.ffn.value.weight"].shape[1]
        print(f'dim_ffn = {dim_ffn}')
        n_head = z[f"blocks.0.att.time_faaaa"].shape[0]
        print("n_head", n_head)

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.dim_ffn = dim_ffn
        self.ctx = 1024 #FLA Window

        keys = list(z.keys())

        self.requires_grad_(False)

        QuantList = ['.receptance.weight','.key.weight','.value.weight','.gate.weight','.output.weight','head.weight']

        QuantListFP8 = ['att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight'] #, ,
 
        # 4bit Quantize Mode via Bitsandbytes NF4
        if self.bit4quant == True:
            for k in keys:
                QuantKeyFound = False
                for QuantKey in QuantList:
                    if k.endswith(QuantKey):
                        print(f'Quant {k} to NF4')
                        QuantKeyFound = True
                        z[k] = z[k].to(device='cuda',dtype=torch.float16) 
                        z[k], z[k+'.qstate']= bnb.functional.quantize_nf4(z[k])
                        

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if k.endswith('.time_decay'): z[k] = z[k].float()
                    elif k.endswith('.time_faaaa'): z[k] = z[k].float()
                    #elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    else:
                        z[k] = z[k].to(dtype = self.base_precision)

        # FP8 Transformer Engine Quantize Mode 
        elif self.bitfp8quant == True:
            for k in keys:
                QuantKeyFound = False
                for QuantKey in QuantListFP8:
                    if k.endswith(QuantKey):
                        print(f'Quant {k} to torch.float8_e4m3fn')
                        QuantKeyFound = True
                        #if 'ffn.value.weight' in k:
                        #    print(f'{k} is halfed')
                        #    #z[k] = z[k] * 0.5
                        z[k] = z[k].to(device='cuda',dtype=torch.float8_e4m3fn) 
                        #z[k], z[k+'.qstate']= bnb.functional.quantize_nf4(z[k])
                if QuantKeyFound == False:
                    for QuantKey in QuantList:
                        if k.endswith(QuantKey):
                            print(f'Quant {k} PassThrough')
                            QuantKeyFound = True
                            z[k] = z[k].to(device='cuda',dtype = self.base_precision).t()
                        

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if k.endswith('.time_decay'): z[k] = z[k].float()
                    elif k.endswith('.time_faaaa'): z[k] = z[k].float()
                    #elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    else:
                        z[k] = z[k].to(dtype = self.base_precision)

            time.sleep(2)
            #exit()

        # 8bit Quantize Mode via Custom Kernel
        elif self.bit8quant == True:
            for k in keys:
                QuantKeyFound = False
                for QuantKey in QuantList:
                    if k.endswith(QuantKey):
                        print(f'Quant {k} to Int8')
                        QuantKeyFound = True
                        z[k] = z[k].t().to(device='cuda').float() 
                        if z[k].shape[0] > z[k].shape[1]:
                            z[k+'.my'] = torch.amin(z[k], dim=1).unsqueeze(1)
                            z[k] = z[k] - z[k+'.my']
                            z[k+'.mx'] = torch.amin(z[k], dim=0)
                            z[k] = z[k] - z[k+'.mx']
                            z[k+'.rx'] = torch.amax(z[k], dim=0)
                            z[k] = z[k] / z[k+'.rx']
                            z[k+'.ry'] = torch.amax(z[k], dim=1).unsqueeze(1)
                            z[k] = z[k] / z[k+'.ry']
                        else:
                            z[k+'.mx'] = torch.amin(z[k], dim=0)
                            z[k] = z[k] - z[k+'.mx']
                            z[k+'.my'] = torch.amin(z[k], dim=1).unsqueeze(1)
                            z[k] = z[k] - z[k+'.my']
                            z[k+'.rx'] = torch.amax(z[k], dim=0)
                            z[k] = z[k] / z[k+'.rx']
                            z[k+'.ry'] = torch.amax(z[k], dim=1).unsqueeze(1)
                            z[k] = z[k] / z[k+'.ry']
                        z[k] = torch.clip(torch.floor(z[k] * 256), min=0, max=255).to(dtype=torch.uint8).contiguous()
                        z[k+'.my'] = z[k+'.my'].to(dtype=torch.float16,device='cuda').contiguous()
                        z[k+'.mx'] = z[k+'.mx'].to(dtype=torch.float16,device='cuda').contiguous()
                        z[k+'.rx'] = (z[k+'.rx']/16).to(dtype=torch.float16,device='cuda').contiguous()
                        z[k+'.ry'] = (z[k+'.ry']/16).to(dtype=torch.float16,device='cuda').contiguous()

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if k.endswith('.time_decay'): z[k] = z[k].float()
                    elif k.endswith('.time_faaaa'): z[k] = z[k].float()
                    #elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    #elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    else:
                        z[k] = z[k].to(dtype = self.base_precision)
            #exit()

        # Non Quantize Mode FP16 or BF16
        else:
            for k in keys:
                #if             '.time_' in k: z[k] = z[k].squeeze()
                z[k] = z[k].to(device='cuda')
                if k.endswith('.time_decay'): z[k] = z[k].float()
                elif k.endswith('.time_faaaa'): z[k] = z[k].float()
                elif k.endswith('.receptance.weight'): z[k] = z[k].t().to(dtype = self.base_precision)
                elif k.endswith('.key.weight'): z[k] = z[k].t().to(dtype = self.base_precision)
                elif k.endswith('.value.weight'): z[k] = z[k].t().to(dtype = self.base_precision)
                elif k.endswith('.gate.weight'): z[k] = z[k].t().to(dtype = self.base_precision)
                elif k.endswith('.output.weight'): z[k] = z[k].t().to(dtype = self.base_precision)
                elif k.endswith('head.weight'): z[k] = z[k].t().to(dtype = self.base_precision)

                #elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                #elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                #elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                #elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)

                elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                else:
                    z[k] = z[k].to(dtype = self.base_precision)
        for i in range(n_layer):
            z[f'blocks.{i}.att.time_maa_wkvrg'] = torch.stack([z[f'blocks.{i}.att.time_maa_w'], z[f'blocks.{i}.att.time_maa_k'], z[f'blocks.{i}.att.time_maa_v'], z[f'blocks.{i}.att.time_maa_r'], z[f'blocks.{i}.att.time_maa_g']], dim=0).contiguous()

        self.emb = nn.Embedding(vocab_size, n_embd)
        self.emb.weight.data = z['emb.weight']
        self.z = z
        

        gc.collect()
        torch.cuda.empty_cache()

        #with torch.no_grad():
    #@MyStatic
    def new_state(self, B):
         return BlockStateList.create(
                 self.n_layer, B, self.n_embd, 
                 self.n_head,# self.head_size,
                 self.emb.weight.device, self.emb.weight.dtype
             )
    
    @MyStatic
    #@torch.compile(dynamic=True)
    def TimeMix_FC_Step1(B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
                              time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                              time_decay_w1, time_decay_w2,time_decay,
                              receptance_weight, key_weight, value_weight, gate_weight,
                              ln1_weight,ln1_bias
                              ):
        xx = F.layer_norm(x.to(dtype=ln1_weight.dtype), (embd,), weight=ln1_weight, bias=ln1_bias)

        x = xx

        B, T, C = x.size()
        x = x.contiguous()
        output = torch.concat((last_state_shift, x[:, :-1]), dim=1)
        last_state_shift[:] = x[:,-1:]

        xx = output - x

        xxx = torch.addcmul(x, xx, time_maa_x).to(dtype=time_maa_x.dtype)
        xxx = torch.tanh(xxx @ time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, time_maa_w2).view(5, B, T, -1)

        combined = (xxx + time_wkvrg) * xx + x
        xw, xk, xv, xr, xg = combined.to(dtype=time_maa_x.dtype).unbind(dim=0)

        ww = torch.tanh(xw @ time_decay_w1) @ time_decay_w2
        w = (time_decay + ww).exp().neg()

        if receptance_weight.dtype == torch.float8_e4m3fn:
            #print('fp8')
            #print(f'xr shape = {xr.shape}')
            S0=xr.shape[0]
            S1=xr.shape[1]
            #A_x = torch._get_scale(xr)
            r, output_amax = torch._scaled_mm(
                xr.view(S0*S1,xr.shape[2]).to(torch.float8_e4m3fn),
                receptance_weight.t(),
                bias=None,
                out_dtype=torch.float8_e4m3fn,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda')
            )
            r = r.view(S0, S1, receptance_weight.shape[-1])
        else:
            r = (xr.to(dtype=time_maa_x.dtype) @ receptance_weight)

        if key_weight.dtype == torch.float8_e4m3fn:
            #print('fp8')
            #print(f'xr shape = {xr.shape}')
            S0=xk.shape[0]
            S1=xk.shape[1]
            k, output_amax = torch._scaled_mm(
                xk.view(S0*S1,xk.shape[2]).to(torch.float8_e4m3fn),
                key_weight.t(),
                bias=None,
                out_dtype=torch.float8_e4m3fn,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda')
            )
            k = k.view(S0, S1, key_weight.shape[-1])
        else:
            k = (xk.to(dtype=time_maa_x.dtype) @ key_weight)

        if value_weight.dtype == torch.float8_e4m3fn:
            #print('fp8')
            #print(f'xr shape = {xr.shape}')
            S0=xv.shape[0]
            S1=xv.shape[1]
            v, output_amax = torch._scaled_mm(
                xv.view(S0*S1,xv.shape[2]).to(torch.float8_e4m3fn),
                value_weight.t(),
                bias=None,
                out_dtype=torch.float8_e4m3fn,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda')
            )
            v = v.view(S0, S1, value_weight.shape[-1])
        else:
            v = (xv.to(dtype=time_maa_x.dtype) @ value_weight)

        if gate_weight.dtype == torch.float8_e4m3fn:
            #print('fp8')
            #print(f'xr shape = {xr.shape}')
            S0=xg.shape[0]
            S1=xg.shape[1]
            g, output_amax = torch._scaled_mm(
                xg.view(S0*S1,xg.shape[2]).to(torch.float8_e4m3fn),
                gate_weight.t(),
                bias=None,
                out_dtype=torch.float8_e4m3fn,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda')
            )
            g = g.view(S0, S1, gate_weight.shape[-1])    
            g = torch.nn.functional.silu(g.to(dtype=time_maa_x.dtype))
        else:
            g = torch.nn.functional.silu((xg.to(dtype=time_maa_x.dtype) @ gate_weight))
        
        return r, k, v, g, w, xx
    
    @MyStatic
    def TimeMix_FC_Quant8_Step1(B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
                              time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                              time_decay_w1, time_decay_w2,time_decay,
                              receptance_weight, key_weight, value_weight, gate_weight,
                              ln1_weight,ln1_bias,
                              rmx,rmy,rrx,rry,
                              kmx,kmy,krx,kry,
                              vmx,vmy,vrx,vry,
                              gmx,gmy,grx,gry
                              ):
        #print(f'ln1_weight.dtype = {ln1_weight.dtype}')
        xx = F.layer_norm(x.to(dtype=ln1_weight.dtype), (embd,), weight=ln1_weight, bias=ln1_bias)#.to(dtype=time_maa_x.dtype)

        x = xx#.to(dtype=time_maa_x.dtype)

        B, T, C = x.size()
        x = x.contiguous()
        #print(f'last_state_shift = {last_state_shift.dtype}, x = {x.dtype}')
        output = torch.concat((last_state_shift, x[:, :-1]), dim=1)#.to(dtype=ln1_weight.dtype)
        last_state_shift[:] = x[:,-1:]

        xx = output - x

        #print(f'output = {output.dtype}, x = {x.dtype}')

        xxx = torch.addcmul(x, xx, time_maa_x).to(dtype=time_maa_x.dtype)
        xxx = torch.tanh(xxx @ time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, time_maa_w2).view(5, B, T, -1)

        combined = (xxx + time_wkvrg) * xx + x
        xw, xk, xv, xr, xg = combined.to(dtype=time_maa_x.dtype).unbind(dim=0)

        #print(f'xw = {xw.dtype}, xxx = {xxx.dtype} time_wkvrg = {time_wkvrg.dtype} xx = {xx.dtype} x = {x.dtype}')

        ww = torch.tanh(xw @ time_decay_w1) @ time_decay_w2
        w = (time_decay + ww).exp().neg()

        #r = (xr.to(dtype=torch.bfloat16) @ receptance_weight)
        r = custom_matmul(xr.to(dtype=time_maa_x.dtype), receptance_weight,
                          rmx,rrx,rmy,rry
                          )
        #k = (xk.to(dtype=torch.bfloat16) @ key_weight)
        k = custom_matmul(xk.to(dtype=time_maa_x.dtype),key_weight,
                          kmx,krx,kmy,kry
                          )
        #v = (xv.to(dtype=torch.bfloat16) @ value_weight)
        v = custom_matmul(xv.to(dtype=time_maa_x.dtype),value_weight,
                          vmx,vrx,vmy,vry
                          )


        #g = torch.nn.functional.silu((xg.to(dtype=torch.bfloat16) @ gate_weight))
        g = torch.nn.functional.silu(
            custom_matmul(xg.to(dtype=time_maa_x.dtype),gate_weight,
                          gmx,grx,gmy,gry
                          )
        )

        
        
        return r, k, v, g, w, xx

    #maybe cannot jit?
    #@torch.compile(dynamic=True)
    def TimeMix_FC_Step2_Seq(self,
                         B:int,T:int, C:int, H:int,ctx,
                         x,last_state_wkv,
                         r,w,k,v,g,
                         time_faaaa,
                         #ln_x_weight,
                         #ln_x_bias
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
            0)
        x =x.transpose(1,2)
        x = x.reshape(B,T,C)
        return x, last_state_wkv

    
    @MyStatic
    #@torch.compile(dynamic=True)
    def TimeMix_FC_Step2_One(B:int,T:int, C:int, H:int,ctx:int,
                         x,last_state_wkv,
                         r,w,k,v,g,
                         time_faaaa,
                         #ln_x_weight,
                         #ln_x_bias
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
        
        # else:
        #     x, last_state_wkv = fused_recurrent_rwkv6(
        #             r.view(B,T,H,-1).transpose(1,2),
        #             k.view(B,T,H,-1).transpose(1,2),
        #             v.view(B,T,H,-1).transpose(1,2),
        #             w.view(B,T,H,-1).transpose(1,2),
        #             time_faaaa.view(H,-1),
        #             1.0,
        #             last_state_wkv,True, 0)
        #     mode = 0
        #print(f'x2 - x1 = {torch.sum(x2)-torch.sum(x)}')
        x = x.reshape(B,T,C)
        return x, last_state_wkv

    @MyStatic
    def TimeMix_FC_Step3(B:int,T:int,C:int,x,g,dim_head:int,
                              ln_x_weight,ln_x_bias,
                              output_weight,
                           ):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = torch.nn.functional.group_norm(x.to(dtype=ln_x_weight.dtype),num_groups = dim_head, weight=ln_x_weight,bias=ln_x_bias, eps= 64e-5).view(B, T, C)
        if output_weight.dtype == torch.float8_e4m3fn:
            #print('fp8')
            #print(f'xr shape = {xr.shape}')
            xg = x * g
            S0=xg.shape[0]
            S1=xg.shape[1]

            xg = torch.clamp(xg, min=-448.0, max=448.0)

            #print(f'xg max = {xg.abs().max()}')
            
            x, output_amax = torch._scaled_mm(
                xg.view(S0*S1,xg.shape[2]).to(torch.float8_e4m3fn),
                output_weight.t(),
                bias=None,
                out_dtype=torch.float16,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda')
            )
            x = x.view(S0, S1, output_weight.shape[-1])
        else:
            x = (x * g).to(dtype=output_weight.dtype) @ output_weight
        return x
    
    @MyStatic
    def TimeMix_FC_Quant8_Step3(B:int,T:int,C:int,x,g,dim_head:int,
                              ln_x_weight,ln_x_bias,
                              output_weight,
                              omx,omy,orx,ory
                           ):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = torch.nn.functional.group_norm(x.to(dtype=ln_x_weight.dtype),num_groups = dim_head, weight=ln_x_weight,bias=ln_x_bias, eps= 64e-5).view(B, T, C)
        #x = (x * g.to(dtype=torch.bfloat16)) @ output_weight
        x = custom_matmul((x * g.to(dtype=torch.bfloat16)),output_weight,
                          omx,orx,omy,ory
                          )
        #print(f'r={torch.sum(x)} ')
        #exit()
        return x
    







    
    @MyStatic
    def ChannelMix_FC_Step1(x,last_state,
                            time_maa_k,
                            time_maa_r,
                            receptance_weight,
                            key_weight,
                            value_weight

                            ):
        xx = torch.concat((last_state, x[:, :-1]),
                          dim=1).to(dtype=time_maa_k.dtype)
        last_state[:] = x[:, -1:]
        
        
        xk = xx * time_maa_k + x * (1 - time_maa_k)
        xr = xx * time_maa_r + x * (1 - time_maa_r)

        if key_weight.dtype == torch.float8_e4m3fn or value_weight.dtype == torch.float8_e4m3fn or receptance_weight.dtype == torch.float8_e4m3fn:
            if key_weight.dtype == torch.float8_e4m3fn:
                S0=xk.shape[0] 
                S1=xk.shape[1]
                xkg, output_amax = torch._scaled_mm(
                    xk.view(S0*S1,xk.shape[2]).to(torch.float8_e4m3fn),
                    key_weight.t(),
                    bias=None,
                    out_dtype=torch.float16,
                    scale_a=torch.tensor(1.0, device='cuda'),
                    scale_b=torch.tensor(1.0, device='cuda')
                    )
                #print(f's0 = {S0} s1 = {S1} kshape = {key_weight.shape[-1]}')
                #print(f'xkg shape {xkg.shape}')
                xkg = xkg.view(S0, S1, -1)
                xkg = (torch.relu(xkg) ** 2)
            else:
                xkg = (torch.relu(xk.to(dtype=key_weight.dtype) @ key_weight) ** 2)
            #print(f'xkg = {xkg}')
            if value_weight.dtype == torch.float8_e4m3fn:
                S0=xkg.shape[0] 
                S1=xkg.shape[1]
                xkg = xkg * 0.333
                xkg = torch.clamp(xkg, min=-448.0, max=448.0)
                xkv, output_amax = torch._scaled_mm(
                    xkg.view(S0*S1,xkg.shape[2]).to(torch.float8_e4m3fn),
                    value_weight.t(),
                    bias=None,
                    out_dtype=torch.float16,
                    scale_a=torch.tensor(3.333, device='cuda'),
                    scale_b=torch.tensor(1.0, device='cuda')
                    )
                kv = xkv.view(S0, S1, -1)# * 2
                #print(f'kv = {kv}')
                #if torch.isnan(kv.mean()):
                #sample = kv[::100]  # 100要素ごとにサンプリング
                #if torch.isnan(sample).any():
                #    print('NaN')
                #    kv = xkg @ value_weight.to(dtype=torch.float16).t()
                #kv = xkg @ value_weight.to(dtype=torch.float16).t()
            else:
                kv = xkg @ value_weight

            if receptance_weight.dtype == torch.float8_e4m3fn:
                S0=xr.shape[0] 
                S1=xr.shape[1]
                xkr, output_amax = torch._scaled_mm(
                    xr.view(S0*S1,xr.shape[2]).to(torch.float8_e4m3fn),
                    receptance_weight.t(),
                    bias=None,
                    out_dtype=torch.float16,
                    scale_a=torch.tensor(1.0, device='cuda'),
                    scale_b=torch.tensor(1.0, device='cuda')
                    )
                xkr = xkr.view(S0, S1, -1)
                return torch.sigmoid(xkr) * kv, last_state
            else:
                return torch.sigmoid(
                                    xr.to(dtype=receptance_weight.dtype) @ receptance_weight 
                                    ) * kv, last_state

            
        
        else:
            kv = (torch.relu(xk.to(dtype=key_weight.dtype) @ key_weight) ** 2) @ value_weight

            return torch.sigmoid(
                                    xr.to(dtype=receptance_weight.dtype) @ receptance_weight 
                                    ) * kv, last_state
    
    @MyStatic
    def ChannelMix_FC_Quant8_Step1(x,last_state,
                            time_maa_k,
                            time_maa_r,
                            receptance_weight,
                            key_weight,
                            value_weight,
                            rmx,rmy,rrx,rry,
                            kmx,kmy,krx,kry,
                            vmx,vmy,vrx,vry,

                            ):
        xx = torch.concat((last_state, x[:, :-1]),
                          dim=1).to(dtype=time_maa_k.dtype)
        last_state[:] = x[:, -1:]
        
        
        xk = xx * time_maa_k + x * (1 - time_maa_k)
        xr = xx * time_maa_r + x * (1 - time_maa_r)

        #kv = (torch.relu(xk.to(dtype=torch.bfloat16) @ key_weight) ** 2) @ value_weight

        kv = custom_matmul(torch.relu(
                                    custom_matmul(
                                                    xk.to(dtype=time_maa_k.dtype),key_weight,
                                                            kmx,krx,kmy,kry
                                                )            
                                    )** 2 ,value_weight,
                           vmx,vrx,vmy,vry
                           )
        
        

        #return torch.sigmoid(
        #                          xr.to(dtype=torch.bfloat16) @ receptance_weight 
        #                        ) * kv, last_state
        return torch.sigmoid(
                                  custom_matmul(
                                      xr.to(dtype=time_maa_k.dtype), receptance_weight,
                                      rmx,rrx,rmy,rry
                                  )
                            ) * kv, last_state
     
    def forward(self, idx: torch.Tensor, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor]):
        with torch.no_grad():
            #
            z = self.z
            x = self.emb(idx)

            x = F.layer_norm(x.to(dtype=z['blocks.0.ln0.weight'].dtype), (self.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
            x=x.to(dtype=self.base_precision)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                #time_mix_shift,channel_mix_shift,time_mix_state
                #x,last_shift_states[i*2],last_shift_states[i*2+1], last_wkv_states[i]
                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                #xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                B, T, C = x.size()
                H = self.n_head

                StrategyMode = 0 # 0 is Fully BF16

                if self.bit8quant == True:
                    StrategyMode = 1
                elif self.bit4quant == True:
                    StrategyMode = 2

                if StrategyMode == 0:
                    # B:int,T:int, C:int, H:int,x, last_state_shift, 
                    # time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                    # time_decay_w1, time_decay_w2,time_decay,
                    # receptance_weight, key_weight, value_weight, gate_weight,
                    r,k,v,g,w,xx = self.TimeMix_FC_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],
                                                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'],z[att+'gate.weight'],
                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias']
                                                    )
                    if T>1:
                        #  B,T,C,H,ctx,
                        #  x,last_state_wkv,
                        #  r,w,k,v,g,
                        #  time_faaaa,
                        att1, time_mix_state = self.TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        #  B,T,C,H,ctx,
                        #  x,last_state_wkv,
                        #  r,w,k,v,g,
                        #  time_faaaa,
                        att1, time_mix_state = self.TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    #B:int,T:int,C:int,x,g,dim_head:int,
                    # ln_x_weight,ln_x_bias,
                    # output_weight,
                    att1 = self.TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight']
                                               )
                    
                    #Finished TimeMix xx,time_mix_shift,time_mix_state
                    #x,last_state,
                    # time_maa_k,
                    # time_maa_r,
                    # receptance_weight,
                    # key_weight,
                    # value_weight
                    x = x + att1
                    #print(f'att1={torch.sum(att1)} i={i}')
                    xx2 = F.layer_norm(x.to(dtype=z[bbb+'ln2.weight'].dtype), (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                    ffn1, channel_mix_state = self.ChannelMix_FC_Step1(xx2,channel_mix_state,
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     z[ffn+'receptance.weight'],
                                                                     z[ffn+'key.weight'],
                                                                     z[ffn+'value.weight']
                                                                     )
                    
                    x = x + ffn1
                    #print(f'ffn1={torch.sum(ffn1)} i={i}')

                elif StrategyMode == 1: # 8bit Quantize
                    # B:int,T:int, C:int, H:int,x, last_state_shift, 
                    # time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                    # time_decay_w1, time_decay_w2,time_decay,
                    # receptance_weight, key_weight, value_weight, gate_weight,

                    #rmx,rmy,rrx,rry,

                    r,k,v,g,w,xx = self.TimeMix_FC_Quant8_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],
                                                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'],z[att+'gate.weight'],
                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias'],

                                                    z[att+'receptance.weight'+'.mx'],z[att+'receptance.weight'+'.my'],z[att+'receptance.weight'+'.rx'],z[att+'receptance.weight'+'.ry'],
                                                    z[att+'key.weight'+'.mx'],z[att+'key.weight'+'.my'],z[att+'key.weight'+'.rx'],z[att+'key.weight'+'.ry'],
                                                    z[att+'value.weight'+'.mx'],z[att+'value.weight'+'.my'],z[att+'value.weight'+'.rx'],z[att+'value.weight'+'.ry'],
                                                    z[att+'gate.weight'+'.mx'],z[att+'gate.weight'+'.my'],z[att+'gate.weight'+'.rx'],z[att+'gate.weight'+'.ry'],
                                                    )
                    if T>1:
                        #  B,T,C,H,ctx,
                        #  x,last_state_wkv,
                        #  r,w,k,v,g,
                        #  time_faaaa,
                        att1, time_mix_state = self.TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        #  B,T,C,H,ctx,
                        #  x,last_state_wkv,
                        #  r,w,k,v,g,
                        #  time_faaaa,
                        att1, time_mix_state = self.TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    #B:int,T:int,C:int,x,g,dim_head:int,
                    # ln_x_weight,ln_x_bias,
                    # output_weight,
                    att1 = self.TimeMix_FC_Quant8_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight'],
                                               z[att+'output.weight'+'.mx'],z[att+'output.weight'+'.my'],z[att+'output.weight'+'.rx'],z[att+'output.weight'+'.ry'],
                                               )
                    
                    #Finished TimeMix xx,time_mix_shift,time_mix_state
                    #x,last_state,
                    # time_maa_k,
                    # time_maa_r,
                    # receptance_weight,
                    # key_weight,
                    # value_weight
                    x = x + att1
                    xx2 = F.layer_norm(x.to(dtype=z[bbb+'ln2.weight'].dtype), (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                    #xx2 = xx2.to(dtype = x.dtype)
                    ffn1, channel_mix_state = self.ChannelMix_FC_Quant8_Step1(xx2,channel_mix_state,
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     z[ffn+'receptance.weight'],
                                                                     z[ffn+'key.weight'],
                                                                     z[ffn+'value.weight'],
                                                                     z[ffn+'receptance.weight'+'.mx'],z[ffn+'receptance.weight'+'.my'],z[ffn+'receptance.weight'+'.rx'],z[ffn+'receptance.weight'+'.ry'],
                                                                     z[ffn+'key.weight'+'.mx'],z[ffn+'key.weight'+'.my'],z[ffn+'key.weight'+'.rx'],z[ffn+'key.weight'+'.ry'],
                                                                     z[ffn+'value.weight'+'.mx'],z[ffn+'value.weight'+'.my'],z[ffn+'value.weight'+'.rx'],z[ffn+'value.weight'+'.ry'],
                                                                     )
                    
                    
                    
                    x = x + ffn1

                    #print(f'ffn1={torch.sum(ffn1)} i={i}')
                    #exit()

                if StrategyMode == 2: #NF4 Mode
                    # B:int,T:int, C:int, H:int,x, last_state_shift, 
                    # time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                    # time_decay_w1, time_decay_w2,time_decay,
                    # receptance_weight, key_weight, value_weight, gate_weight,
                    r,k,v,g,w,xx = self.TimeMix_FC_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],


                                                    #z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'],z[att+'gate.weight'],
                                                    bnb.functional.dequantize_4bit(z[att+'receptance.weight'],
                                                                                   quant_state=z[att+'receptance.weight.qstate']).to(dtype=torch.float16).t(),
                                                    bnb.functional.dequantize_4bit(z[att+'key.weight'],
                                                                                   quant_state=z[att+'key.weight.qstate']).to(dtype=torch.float16).t(),
                                                    bnb.functional.dequantize_4bit(z[att+'value.weight'],
                                                                                   quant_state=z[att+'value.weight.qstate']).to(dtype=torch.float16).t(),
                                                    bnb.functional.dequantize_4bit(z[att+'gate.weight'],
                                                                                   quant_state=z[att+'gate.weight.qstate']).to(dtype=torch.float16).t(),



                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias']
                                                    )
                    if T>1:
                        #  B,T,C,H,ctx,
                        #  x,last_state_wkv,
                        #  r,w,k,v,g,
                        #  time_faaaa,
                        att1, time_mix_state = self.TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        #  B,T,C,H,ctx,
                        #  x,last_state_wkv,
                        #  r,w,k,v,g,
                        #  time_faaaa,
                        att1, time_mix_state = self.TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    #B:int,T:int,C:int,x,g,dim_head:int,
                    # ln_x_weight,ln_x_bias,
                    # output_weight,
                    att1 = self.TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               #z[att+'output.weight']
                                               bnb.functional.dequantize_4bit(z[att+'output.weight'],
                                                                                   quant_state=z[att+'output.weight.qstate']).to(dtype=torch.float16).t()
                                               )
                    
                    #Finished TimeMix xx,time_mix_shift,time_mix_state
                    #x,last_state,
                    # time_maa_k,
                    # time_maa_r,
                    # receptance_weight,
                    # key_weight,
                    # value_weight
                    x = x + att1
                    #print(f'att1={torch.sum(att1)} i={i}')
                    xx2 = F.layer_norm(x.to(dtype=z[bbb+'ln2.weight'].dtype), (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                    ffn1, channel_mix_state = self.ChannelMix_FC_Step1(xx2,channel_mix_state,
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     #z[ffn+'receptance.weight'],
                                                                     #z[ffn+'key.weight'],
                                                                     #z[ffn+'value.weight']
                                                                     bnb.functional.dequantize_4bit(z[ffn+'receptance.weight'],
                                                                                   quant_state=z[ffn+'receptance.weight.qstate']).to(dtype=torch.float16).t(),
                                                                     bnb.functional.dequantize_4bit(z[ffn+'key.weight'],
                                                                                   quant_state=z[ffn+'key.weight.qstate']).to(dtype=torch.float16).t(),
                                                                     bnb.functional.dequantize_4bit(z[ffn+'value.weight'],
                                                                                   quant_state=z[ffn+'value.weight.qstate']).to(dtype=torch.float16).t(),
                                                                     )
                    
                    x = x + ffn1
                
                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state

                        
            x = F.layer_norm(x.to(dtype=z['ln_out.weight'].dtype), (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x.to(dtype=self.base_precision)

            if self.bit8quant:
                x = custom_matmul(x,z['head.weight'],
                                   z['head.weight'+'.mx'],z['head.weight'+'.rx'],z['head.weight'+'.my'],z['head.weight'+'.ry'],
                                   )#.float()
            elif self.bit4quant:
                x = x @ bnb.functional.dequantize_4bit(z['head.weight'],quant_state=z['head.weight.qstate']).to(dtype=torch.float16).t()
            else:
                x = x @ z['head.weight']

            #print(f'x = {x}')
            #exit()

            

            return x, last_shift_states, last_wkv_states
    def load_state(self,state_filename):
        try:
            state_raw = torch.load(state_filename, map_location="cpu")
        except Exception as e:
            print(e)
            return "error"
        state_raw_shape = next(iter(state_raw.values())).shape

        #args = model.args
        self.debug = 1
        if self.debug:
            print(f"{len(state_raw)} != {self.n_layer}")
            print(f"{state_raw_shape[0] * state_raw_shape[1]} != {self.n_embd}")

        if (
            len(state_raw) != self.n_layer
            or state_raw_shape[0] * state_raw_shape[1] != self.n_embd
        ):
            print("state failed to load")
            return "error"

        #strategy = model.strategy

        model_current_statetuned = [None] * self.n_layer * 3

        dev = 'cpu'

        for i in range(self.n_layer):
            #dd = strategy[i]
            #dd.device
            atype = torch.bfloat16 #dd.atype
            model_current_statetuned[i * 3 + 0] = torch.zeros(
                self.n_embd, dtype=atype, requires_grad=False, device=dev
            ).contiguous()
            model_current_statetuned[i * 3 + 1] = (
                state_raw[f"blocks.{i}.att.time_state"]
                .transpose(1, 2)
                .to(dtype=torch.float, device=dev)
                .requires_grad_(False)
                .contiguous()
            )
            model_current_statetuned[i * 3 + 2] = torch.zeros(
                self.n_embd, dtype=atype, requires_grad=False, device=dev
            ).contiguous()

        wkv_states = torch.empty((self.n_layer, self.n_head, self.n_embd//self.n_head, self.n_embd//self.n_head),
                                 device=dev,
                                 dtype=torch.bfloat16)
        
        for i in range(self.n_layer):
            wkv_states[i] = model_current_statetuned[i*3 + 1]

        return wkv_states#.to(dtype=torch.float16)
        







if __name__ == '__main__':
    print('RWKV x060Core with FLA Test')

    pipeline = PIPELINE()
    model = RWKV_6('../models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth',False,'fp8')
    Target_batch = 256

    States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    context =  'User: What is advantage of C++?\n\nAssistant:'
    context2 = 'User: What is advantage of C++?\n\nAssistant:'

    #model.load_state('states/ojousama2.pth')

    

    shift_states = States.shift_states
    wkv_states = States.wkv_states

    def print_tensor_shapes(tensor_list):
        for i, tensor in enumerate(tensor_list):
            if isinstance(tensor, torch.Tensor):
                print(f"Tensor {i}: Shape = {tensor.shape}")
            else:
                print(f"Item {i} is not a Tensor")

    #print_tensor_shapes(model.model_current_statetuned )

    #print(f'state-tune-file = {model.model_current_statetuned    }')

    print('////////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    print(f'wkv_states = {wkv_states.shape    }')
    print(f'shift_states = {shift_states.shape    }')

    #wkv_states[0] = model.model_current_statetuned

    #for i in range(model.n_layer):
    #    wkv_states[i][0] = model.model_current_statetuned[i*3 + 1]



    #exit()

    tokens = pipeline.encode(context)
    tokens2 = pipeline.encode(context2)
    prompts = []
    for i in range(Target_batch):
        if i%2 == 0:
            prompts.append(torch.tensor(tokens).unsqueeze(0).to('cuda'))
        else:
            prompts.append(torch.tensor(tokens2).unsqueeze(0).to('cuda'))
    #idx = torch.cat([torch.tensor(tokens).unsqueeze(0).to('cuda')], dim=0)

    idx = torch.cat(prompts, dim=0)

    #shift_states = States.shift_states
    #wkv_states = States.wkv_states

    #exit()

    print(f'{idx.shape}')

    x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)

    print(f'x = {x}')

    print(context)
    out_tokens = [[] for _ in range(Target_batch)]
    out_last = [0 for _ in range(Target_batch)]
    output_text = ['' for _ in range(Target_batch)]







    

    FirstTime = 1

    t000 = time.perf_counter()
    min_time = 1e10
    min_time_all = 1e10
    min_time_all_single = 1e10

    maxtoken= 1000

    for i in range(maxtoken):
        t00 = time.perf_counter()
        #x[0][0] -= 1e10
        #if FirstTime:
        #    token = pipeline.sample_logits_mose2(x[0][0], temperature=1, top_p=0.3)
        #else:
        #    token = pipeline.sample_logits_mose2(x[0], temperature=1, top_p=0.3)

        # otokens = []
        # for j in range(Target_batch):
        #     x[j][0][0] -= 1e10
        #     token = pipeline.sample_logits_blink(x[j][0], temperature=1.0, top_p=0.3)
        #     otokens.append(token)
        
        x[:, 0, 0] -= 1e10
        # 2. sample_logits_blink をバッチ全体に適用
        otokens = pipeline.improved_nucleus_sampling_multi(x[:, 0], temperature=1.0, top_p=0.3)

        tokens = []
        for j in range(Target_batch):
            tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0).to('cuda'))
        
        #idx = torch.cat([torch.tensor(token).unsqueeze(0).unsqueeze(0).to('cuda')], dim=0)
        idx = torch.cat(tokens, dim=0)

        for j in range(Target_batch):
            out_tokens[j] += [otokens[j]]
            try:
                tmp = pipeline.decode(out_tokens[j][out_last[j]:])
                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                        #yield tmp
                        if j == Target_batch - 1:
                            print(tmp,end="", flush=True)
                        output_text[j] = output_text[j] + tmp
                        out_last[j] = i + 1
            except:
                pass
        t0 = time.perf_counter()

        x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)

        t1 = time.perf_counter()
        min_time = min(min_time, (t1 - t0)/Target_batch)
        min_time_all = min(min_time_all, (t1 - t00)/Target_batch)
        min_time_all_single = min(min_time_all_single, (t1 - t00))

    t001 = time.perf_counter()

    print(output_text)
    print('RWKV-Infer FLA Refactor')

    tokensec = maxtoken / (t001-t000)
    print(f'TargetBatch = {Target_batch} Total token/s = {round(tokensec*Target_batch,2)} Single token/s = {round(tokensec,2)}')

    #print(f'\n[ {round(1/min_time_all,2)} (real) {round(1/min_time_all_single,2)} (singlereal) / {round(1/min_time,2)} (ignore sampling & tokenizer) token/s = {round(time.perf_counter()-t000,3)}s ]', end='')






                    

    


 


        

