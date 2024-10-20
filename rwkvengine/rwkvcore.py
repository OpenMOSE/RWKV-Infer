#Refactoring RWKV x060 Inference Engine with Flash Linear Attention
#2024 OpenMOSE

#Test Torchao
import torchao
from torchao.dtypes.fpx import to_scaled_tc_fpx
from torchao.ops import quant_llm_linear


from fla.ops.gla import fused_chunk_gla
from fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6


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

#from rwkvengine.misc import PIPELINE
from rwkvengine.misc import PIPELINE
from rwkvengine.matmularena import custom_matmul, fp8_hybrod_matmul

torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
#torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
#torch._C._jit_set_autocast_mode(False)

#torch._dynamo.config.suppress_errors = True


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

    def __init__(self,load_model: str,base_precision: str = 'int8'):

        print('Helloworld RWKV v060 :) Initializing')

        super().__init__()

        #GANBATTE CODE KAKOU
        self.bit8quant = False
        self.bit4quant = False
        self.bitfp8quant = False
        self.bitfp6quant = False

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
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = True
        elif base_precision == 'fp8':
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = False
            self.bitfp8quant = True
        elif base_precision == 'fp6':
            self.base_precision = torch.bfloat16
            self.bit8quant = False
            self.bit4quant = False
            self.bitfp8quant = False
            self.bitfp6quant = True
        else:
            self.base_precision = torch.bfloat16
        
        modelpath = load_model

        z = torch.load(modelpath,map_location="cpu",mmap=True)
        keys = list(z.keys())
        print("keys", keys)

        keys = list(z.keys())

        # detect model details
        vocab_size, n_embd = z["emb.weight"].shape

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

        QuantListFP8 = ['att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight','head.weight'] #, ,
        QuantListFP6 = ['att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight','head.weight'] #, ,
 
        # 4bit Quantize Mode via Bitsandbytes NF4
        if self.bit4quant == True:
            for k in keys:
                QuantKeyFound = False
                for QuantKey in QuantList:
                    if k.endswith(QuantKey):
                        print(f'Quant {k} to NF4')
                        QuantKeyFound = True
                        z[k] = z[k].to(device='cuda',dtype=torch.bfloat16) 
                        z[k], z[k+'.qstate']= bnb.functional.quantize_nf4(z[k])
                        

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if k.endswith('.time_decay'): z[k] = z[k].float()
                    elif k.endswith('.time_faaaa'): z[k] = z[k].float()
                    elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
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
                print(f' k = {k} shape = {z[k].shape}' )
                QuantKeyFound = False
                for QuantKey in QuantListFP8:
                    if k.endswith(QuantKey):
                        print(f'Quant {k} to torch.float8_e4m3fn')
                        QuantKeyFound = True
                        z[k] = z[k].to(device='cuda',dtype=torch.float8_e4m3fn).contiguous() 
                       
                if QuantKeyFound == False:
                    for QuantKey in QuantList:
                        if k.endswith(QuantKey):
                            print(f'Quant {k} PassThrough')
                            QuantKeyFound = True
                            z[k] = z[k].to(device='cuda',dtype = self.base_precision).contiguous() 
                        

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if k.endswith('.time_decay'): z[k] = z[k].float().contiguous() 
                    elif k.endswith('.time_faaaa'): z[k] = z[k].float().contiguous() 
                    elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                    else:
                        z[k] = z[k].to(dtype = self.base_precision).contiguous() 

        # FP6 Quantize Mode via Torch.AO
        elif self.bitfp6quant == True:
            self.ebits, self.mbits = 3, 2
            for k in keys:
                QuantKeyFound = False
                for QuantKey in QuantListFP6:
                    if k.endswith(QuantKey):
                        print(f'Quant {k} to FP6 shape = {z[k].shape}' )
                        QuantKeyFound = True
                        z[k] = z[k].to(device='cuda',dtype=torch.float16)#.t() 

                        # pre-process the weight. this will quantize the weight to FP6 and pack it in a special
                        # layout for tensor cores. refer to paper for more details.
                        z[k], z[k+'.qstate'] = to_scaled_tc_fpx(z[k], 3, 2)
                        #z[k], z[k+'.qstate']= bnb.functional.quantize_nf4(z[k])

                if QuantKeyFound == False:
                    for QuantKey in QuantList:
                        if k.endswith(QuantKey):
                            print(f'Quant {k} PassThrough')
                            QuantKeyFound = True
                            z[k] = z[k].to(device='cuda',dtype = self.base_precision).contiguous() 
                            z[k+'.qstate'] = torch.randn(1)
                        

                if QuantKeyFound == False:
                    z[k] = z[k].to(device='cuda')
                    if k.endswith('.time_decay'): z[k] = z[k].float()
                    elif k.endswith('.time_faaaa'): z[k] = z[k].float()
                    elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    else:
                        z[k] = z[k].to(dtype = self.base_precision)


        # int 8bit Quantize Mode via Custom Kernel
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
                    elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                    elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
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
                #if k.endswith('.time_decay'): z[k] = z[k].float()
                if k.endswith('.time_faaaa'): z[k] = z[k].float()
                elif k.endswith('.receptance.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                elif k.endswith('.key.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                elif k.endswith('.value.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                elif k.endswith('.gate.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                elif k.endswith('.output.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                elif k.endswith('head.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 

                #elif k.endswith('.ln1.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                #elif k.endswith('.ln1.bias'): z[k] = z[k].to(dtype=torch.bfloat16)
                #elif k.endswith('.ln2.weight'): z[k] = z[k].to(dtype=torch.bfloat16)
                #elif k.endswith('.ln2.bias'): z[k] = z[k].to(dtype=torch.bfloat16)

                elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                else:
                    z[k] = z[k].to(dtype = self.base_precision).contiguous() 
        for i in range(n_layer):
            z[f'blocks.{i}.att.time_maa_wkvrg'] = torch.stack([z[f'blocks.{i}.att.time_maa_w'], z[f'blocks.{i}.att.time_maa_k'], z[f'blocks.{i}.att.time_maa_v'], z[f'blocks.{i}.att.time_maa_r'], z[f'blocks.{i}.att.time_maa_g']], dim=0).contiguous()

        self.z = z
        self.device = z['emb.weight'].device
        self.dtype = z['emb.weight'].dtype
    
        gc.collect()
        torch.cuda.empty_cache()


    def new_state(self, B):
         return BlockStateList.create(
                 self.n_layer, B, self.n_embd, 
                 self.n_head,# self.head_size,
                 self.device, self.dtype
             )
    
    @MyStatic
    def First(emb,idx,n_embd:int,ln0_weight,ln0_bias):
        x = F.embedding(idx, emb)
        x = F.layer_norm(x.to(dtype=ln0_weight.dtype), (n_embd,), weight=ln0_weight, bias=ln0_bias)
        return x
    
    
    @MyStatic
    def TimeMix_FC_Step1(B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
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
    

    #@torch.compile#(mode="reduce-overhead", fullgraph=True)
    def TimeMix_FC_FP6_Step1(self,B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
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

        #print(f'receptance_weight.dtype = {receptance_weight.dtype}')
        

        # if receptance_weight.dtype == torch.float8_e4m3fn:
        #     S0=xr.shape[0]
        #     S1=xr.shape[1]
        #     r = torch._scaled_mm(
        #         xr.view(-1,xr.shape[2]).to(torch.float8_e4m3fn),
        #         receptance_weight.t(),
        #         bias=None,
        #         out_dtype=torch.bfloat16,
        #         scale_a=torch.tensor(1.0, device='cuda'),
        #         scale_b=torch.tensor(1.0, device='cuda'),
        #         use_fast_accum = True
        #     )
        #     r = r.view(S0, S1, -1)
        # else:
        if receptance_weight.dtype == torch.uint8:
            S0=xr.shape[0]
            S1=xr.shape[1]
            xr = xr.to(dtype=torch.float16).view(-1,xr.shape[2])#.cuda().half()
            #print(f'xr = {xr.shape}')
            r = quant_llm_linear(ebits, mbits, xr, receptance_weight, receptance_qstate).view(S0,S1,-1)
        else:
            r = (xr.to(dtype=time_maa_x.dtype) @ receptance_weight.t())




        







        # if key_weight.dtype == torch.float8_e4m3fn:
        #     S0=xk.shape[0]
        #     S1=xk.shape[1]
        #     k = torch._scaled_mm(
        #         xk.view(-1,xk.shape[2]).to(torch.float8_e4m3fn),
        #         key_weight.t(),
        #         bias=None,
        #         out_dtype=torch.bfloat16,
        #         scale_a=torch.tensor(1.0, device='cuda'),
        #         scale_b=torch.tensor(1.0, device='cuda'),
        #         use_fast_accum = True
        #     )
        #     k = k.view(S0, S1, -1)
        # else:
        
        #k = (xk.to(dtype=time_maa_x.dtype) @ key_weight.t())
        if key_weight.dtype == torch.uint8:
            S0=xk.shape[0]
            S1=xk.shape[1]
            xk = xk.to(dtype=torch.float16).view(-1,xk.shape[2])#.cuda().half()
            k = quant_llm_linear(ebits, mbits, xk, key_weight, key_qstate).view(S0,S1,-1)
        else:
            k = (xk.to(dtype=time_maa_x.dtype) @ key_weight.t())


        # if value_weight.dtype == torch.float8_e4m3fn:
        #     S0=xv.shape[0]
        #     S1=xv.shape[1]
        #     v = torch._scaled_mm(
        #         xv.view(-1,xv.shape[2]).to(torch.float8_e4m3fn),
        #         value_weight.t(),
        #         bias=None,
        #         out_dtype=torch.bfloat16,
        #         scale_a=torch.tensor(1.0, device='cuda'),
        #         scale_b=torch.tensor(1.0, device='cuda'),
        #         use_fast_accum = True
        #     )
        #     v = v.view(S0, S1, -1)
        # else:
        #     v = (xv.to(dtype=time_maa_x.dtype) @ value_weight.t())
        if value_weight.dtype == torch.uint8:
            S0=xv.shape[0]
            S1=xv.shape[1]
            xv = xv.to(dtype=torch.float16).view(-1,xv.shape[2])
            v = quant_llm_linear(ebits, mbits, xv, value_weight, key_qstate).view(S0,S1,-1)
        else:
            v = (xv.to(dtype=time_maa_x.dtype) @ value_weight.t())

        # if gate_weight.dtype == torch.float8_e4m3fn:
        #     S0=xg.shape[0]
        #     S1=xg.shape[1]
        #     g = torch._scaled_mm(
        #         xg.view(-1,xg.shape[2]).to(torch.float8_e4m3fn),
        #         gate_weight.t(),
        #         bias=None,
        #         out_dtype=torch.bfloat16,
        #         scale_a=torch.tensor(1.0, device='cuda'),
        #         scale_b=torch.tensor(1.0, device='cuda'),
        #         use_fast_accum = True
        #     )
        #     g = torch.nn.functional.silu(g.view(S0, S1, -1))
        # else:
        #     g = torch.nn.functional.silu((xg.to(dtype=time_maa_x.dtype) @ gate_weight.t()))
        if gate_weight.dtype == torch.uint8:
            S0=xg.shape[0]
            S1=xg.shape[1]
            xg = xg.to(dtype=torch.float16).view(-1,xg.shape[2])
            g = torch.nn.functional.silu(quant_llm_linear(ebits, mbits, xg, gate_weight, gate_qstate).view(S0,S1,-1))
        else:
            g = torch.nn.functional.silu((xg.to(dtype=time_maa_x.dtype) @ gate_weight.t()))
        
        return r, k, v, g, w, xx
    
    @MyStatic
    def TimeMix_FC_Int8_Step1(B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
                              time_maa_x, time_wkvrg, time_maa_w1, time_maa_w2,
                              time_decay_w1, time_decay_w2,time_decay,
                              receptance_weight, key_weight, value_weight, gate_weight,
                              ln1_weight,ln1_bias,
                              rmx,rmy,rrx,rry,
                              kmx,kmy,krx,kry,
                              vmx,vmy,vrx,vry,
                              gmx,gmy,grx,gry
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

        r = custom_matmul(xr.to(dtype=time_maa_x.dtype), receptance_weight,
                          rmx,rrx,rmy,rry
                          )
        k = custom_matmul(xk.to(dtype=time_maa_x.dtype),key_weight,
                          kmx,krx,kmy,kry
                          )
        v = custom_matmul(xv.to(dtype=time_maa_x.dtype),value_weight,
                          vmx,vrx,vmy,vry
                          )

        g = torch.nn.functional.silu(
            custom_matmul(xg.to(dtype=time_maa_x.dtype),gate_weight,
                          gmx,grx,gmy,gry
                          )
                        )
        
        return r, k, v, g, w, xx
    
    @MyStatic
    def TimeMix_FC_NF4_Step0(B:int,T:int, C:int, H:int, embd:int,x, last_state_shift, 
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
    
    def matmul_4bit_(self,a,b,b_state):
        outlist = []
        for i in range(a.shape[0]):
            outlist.append(bnb.matmul_4bit(a,b,b_state))
        outlist = torch.cat(outlist, dim=0)
        # S0 = a.shape[0]
        # S1 = a.shape[1]
        # S2 = a.shape[2]
        # a = a.view(-1,S2)
        # print(f'a shape = {a.shape}')
        # print(f'b shape = {b.shape}')
        # r = bnb.functional.gemv_4bit(A=a,B=b,out=None,state=b_state)
        # r = r.view(S0,S1,-1)
        return outlist

    def TimeMix_FC_NF4_Step1(self,xr,xk,xv,xg,
                              receptance_weight,
                              receptance_qstate,
                              key_weight,
                              key_qstate,
                              value_weight,
                              value_qstate,
                              gate_weight,
                              gate_qstate

                              ):
        
        #Direct 4bit Matmul with Bitsandbytes
        
        r = bnb.matmul_4bit(xr.to(dtype=torch.float16),receptance_weight.t(),receptance_qstate)

        k = bnb.matmul_4bit(xk.to(dtype=torch.float16),key_weight.t(),key_qstate)

        v = bnb.matmul_4bit(xv.to(dtype=torch.float16),value_weight.t(),value_qstate)

        g = torch.nn.functional.silu(
            bnb.matmul_4bit(xg.to(dtype=torch.float16),gate_weight.t(),gate_qstate)
                          )
       
        return r, k, v, g

    def TimeMix_FC_Step2_Seq(self,
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
            0)
        x =x.transpose(1,2)
        x = x.reshape(B,T,C)
        return x, last_state_wkv

    
    @MyStatic
    def TimeMix_FC_Step2_One(B:int,T:int, C:int, H:int,ctx:int,
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
    
    def TimeMix_FC_Step2_One_HighBatch(self,B:int,T:int, C:int, H:int,ctx:int,
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
    def TimeMix_FC_Step3(B:int,T:int,C:int,x,g,dim_head:int,
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
    @torch.compile
    def TimeMix_FC_FP6_Step3(self,B:int,T:int,C:int,x,g,dim_head:int,
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
        
    
    def TimeMix_FC_NF4_Step3(self,B:int,T:int,C:int,x,g,dim_head:int,
                              ln_x_weight,ln_x_bias,
                              output_weight,
                              output_qstate,
                           ):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = torch.nn.functional.group_norm(x.to(dtype=ln_x_weight.dtype),num_groups = dim_head, weight=ln_x_weight,bias=ln_x_bias, eps= 64e-5).view(B, T, C)
        return bnb.matmul_4bit((x * g).to(dtype=torch.float16),
                               output_weight.t(),
                               output_qstate
                               )
    
    @MyStatic
    def TimeMix_FC_Int8_Step3(B:int,T:int,C:int,x,g,dim_head:int,
                              ln_x_weight,ln_x_bias,
                              output_weight,
                              omx,omy,orx,ory
                           ):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = torch.nn.functional.group_norm(x.to(dtype=ln_x_weight.dtype),num_groups = dim_head, weight=ln_x_weight,bias=ln_x_bias, eps= 64e-5).view(B, T, C)
        x = custom_matmul((x * g.to(dtype=torch.bfloat16)),output_weight,
                          omx,orx,omy,ory
                          )
        return x
    


    @MyStatic
    def ChannelMix_FC_NF4_Step0(x,last_state,
                                ln2_weight,
                                ln2_bias,
                                n_embd:int,
                                time_maa_k,
                                time_maa_r):
        #transfered ln2 norm here 
        x = F.layer_norm(x.to(dtype=ln2_weight.dtype), (n_embd,), weight=ln2_weight, bias=ln2_bias)

        xx = torch.concat((last_state, x[:, :-1]),
                          dim=1).to(dtype=time_maa_k.dtype)
        last_state[:] = x[:, -1:]
        
        
        xk = xx * time_maa_k + x * (1 - time_maa_k)
        xr = xx * time_maa_r + x * (1 - time_maa_r)

        return xk,xr,last_state
    





    
    @MyStatic
    def ChannelMix_FC_Step1(x,last_state,
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
        
    def ChannelMix_FC_NF4_Step1(self,xk,xr,
                            receptance_weight,
                            receptance_qstate,
                            key_weight,
                            key_qstate,
                            value_weight,
                            value_qstate
                            ):
     
            xkg = (torch.relu(bnb.matmul_4bit(xk.to(dtype=torch.float16),key_weight.t(),key_qstate)) ** 2)

            #xkg = (torch.relu(xk.to(dtype=key_weight.dtype) @ key_weight.t()) ** 2)

            kv = bnb.matmul_4bit(xkg,value_weight.t(),value_qstate)
 
            #kv = xkg @ value_weight.t()

            return torch.sigmoid(
                                    bnb.matmul_4bit(xr.to(dtype=torch.float16),receptance_weight.t(),receptance_qstate)
                                ) * kv

            # return torch.sigmoid(
            #                     xr.to(dtype=receptance_weight.dtype) @ receptance_weight.t() 
            #                     ) * kv
    @torch.compile
    def ChannelMix_FC_FP6_Step1(self,x,last_state,
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
    def ChannelMix_FC_Int8_Step1(x,last_state,
                                    ln2_weight,
                                    ln2_bias,
                                    n_embd:int,
                                    time_maa_k,
                                    time_maa_r,
                                    receptance_weight,
                                    key_weight,
                                    value_weight,
                                    rmx,rmy,rrx,rry,
                                    kmx,kmy,krx,kry,
                                    vmx,vmy,vrx,vry,

                            ):
        #transfered ln2 norm here 
        x = F.layer_norm(x.to(dtype=ln2_weight.dtype), (n_embd,), weight=ln2_weight, bias=ln2_bias)

        xx = torch.concat((last_state, x[:, :-1]),
                          dim=1).to(dtype=time_maa_k.dtype)
        last_state[:] = x[:, -1:]
        
        
        xk = xx * time_maa_k + x * (1 - time_maa_k)
        xr = xx * time_maa_r + x * (1 - time_maa_r)

        kv = custom_matmul(torch.relu(
                                    custom_matmul(
                                                    xk.to(dtype=time_maa_k.dtype),key_weight,
                                                            kmx,krx,kmy,kry
                                                )            
                                    )** 2 ,value_weight,
                           vmx,vrx,vmy,vry
                           )

        return torch.sigmoid(
                                  custom_matmul(
                                      xr.to(dtype=time_maa_k.dtype), receptance_weight,
                                      rmx,rrx,rmy,rry
                                  )
                            ) * kv, last_state
    @MyStatic
    def Final(x,head_weight,n_embd:int,ln_out_weight,ln_out_bias):
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
        
    def Final_NF4(self,x,head_weight,head_qstate,n_embd:int,ln_out_weight,ln_out_bias):
        x = F.layer_norm(x.to(dtype=ln_out_weight.dtype), (n_embd,), weight=ln_out_weight, bias=ln_out_bias)
        x = bnb.matmul_4bit(x.to(dtype=torch.float16),head_weight.t(),head_qstate)
        return x
    
    #@torch.compile
    def Final_FP6(self,x,head_weight,head_qstate,n_embd:int,ln_out_weight,ln_out_bias,ebits:int,mbits:int):
        x = F.layer_norm(x.to(dtype=ln_out_weight.dtype), (n_embd,), weight=ln_out_weight, bias=ln_out_bias)
        #x = bnb.matmul_4bit(x.to(dtype=torch.float16),head_weight.t(),head_qstate)
        if head_weight.dtype == torch.uint8:
            S0=x.shape[0]
            S1=x.shape[1]
            x = x.to(dtype=torch.float16).view(-1,x.shape[2])
            x = quant_llm_linear(ebits, mbits, x, head_weight, head_qstate).view(S0,S1,-1)
        else:
            x = x.to(dtype=head_weight.dtype)
            x = x @ head_weight.t()

        return x
        
    @MyStatic
    def Final_int8(x,head_weight,head_mx,head_rx,head_my,head_ry,n_embd:int,ln_out_weight,ln_out_bias):
        x = F.layer_norm(x.to(dtype=ln_out_weight.dtype), (n_embd,), weight=ln_out_weight, bias=ln_out_bias)
        x = custom_matmul(x,head_weight,head_mx,head_rx,head_my,head_ry)
        return x

    def forward(self, idx: torch.Tensor, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor]):
        StrategyMode = 0 # 0 is Fully BF16 or FP16 or FP8
        if self.bit8quant == True:
            StrategyMode = 1
        elif self.bit4quant == True:
            StrategyMode = 2
        elif self.bitfp6quant == True:
            StrategyMode = 3
            
        with torch.no_grad():
            #
            z = self.z
            H = self.n_head

            x = self.First(z['emb.weight'],idx,self.n_embd,z['blocks.0.ln0.weight'],z['blocks.0.ln0.bias'])
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
                    r,k,v,g,w,xx = self.TimeMix_FC_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],
                                                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'],z[att+'gate.weight'],
                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias']
                                                    )
                    if T>1:
                        att1, time_mix_state = self.TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        if B < 16:
                            att1, time_mix_state = self.TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = self.TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = self.TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight']
                                               )
                    

                    x = x + att1

                    ffn1, channel_mix_state = self.ChannelMix_FC_Step1(x,channel_mix_state,
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


                elif StrategyMode == 1: # int 8bit Quantize

                    r,k,v,g,w,xx = self.TimeMix_FC_Int8_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
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
                        att1, time_mix_state = self.TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        if B < 16:
                            att1, time_mix_state = self.TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = self.TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = self.TimeMix_FC_Int8_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight'],
                                               z[att+'output.weight'+'.mx'],z[att+'output.weight'+'.my'],z[att+'output.weight'+'.rx'],z[att+'output.weight'+'.ry'],
                                               )
                    
                    x = x + att1

                    ffn1, channel_mix_state = self.ChannelMix_FC_Int8_Step1(x,channel_mix_state,
                                                                     z[bbb+'ln2.weight'],
                                                                     z[bbb+'ln2.bias'],
                                                                     int(self.n_embd),
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



                elif StrategyMode == 2: #NF4 Mode

                    r,k,v,g,w,xx = self.TimeMix_FC_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],

                                                    bnb.functional.dequantize_4bit(z[att+'receptance.weight'],
                                                                                   quant_state=z[att+'receptance.weight.qstate']) ,
                                                    bnb.functional.dequantize_4bit(z[att+'key.weight'],
                                                                                   quant_state=z[att+'key.weight.qstate']) ,
                                                    bnb.functional.dequantize_4bit(z[att+'value.weight'],
                                                                                   quant_state=z[att+'value.weight.qstate']) ,
                                                    bnb.functional.dequantize_4bit(z[att+'gate.weight'],
                                                                                   quant_state=z[att+'gate.weight.qstate']) ,



                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias']
                                                    )
                    if T>1:

                        att1, time_mix_state = self.TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:

                        if B < 16:
                            att1, time_mix_state = self.TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = self.TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = self.TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               bnb.functional.dequantize_4bit(z[att+'output.weight'],
                                                                                   quant_state=z[att+'output.weight.qstate'])
                                               )
                    
                    x = x + att1

                    ffn1, channel_mix_state = self.ChannelMix_FC_Step1(x,channel_mix_state,
                                                                     z[bbb+'ln2.weight'],
                                                                     z[bbb+'ln2.bias'],
                                                                     int(self.n_embd),
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     bnb.functional.dequantize_4bit(z[ffn+'receptance.weight'],
                                                                                   quant_state=z[ffn+'receptance.weight.qstate']),
                                                                     bnb.functional.dequantize_4bit(z[ffn+'key.weight'],
                                                                                   quant_state=z[ffn+'key.weight.qstate']),
                                                                     bnb.functional.dequantize_4bit(z[ffn+'value.weight'],
                                                                                   quant_state=z[ffn+'value.weight.qstate']),
                                                                     )
                    
                    x = x + ffn1


                elif StrategyMode == 3: #FP6
                    r,k,v,g,w,xx = self.TimeMix_FC_FP6_Step1(B,T,C,H,self.n_embd,x,time_mix_shift,
                                                    z[att+'time_maa_x'], z[att+'time_maa_wkvrg'], z[att+'time_maa_w1'], z[att+'time_maa_w2'],
                                                    z[att+'time_decay_w1'], z[att+'time_decay_w2'],z[att+'time_decay'],
                                                    z[att+'receptance.weight'],
                                                    z[att+'receptance.weight.qstate'],
                                                    z[att+'key.weight'],
                                                    z[att+'key.weight.qstate'],
                                                    z[att+'value.weight'],
                                                    z[att+'value.weight.qstate'],
                                                    z[att+'gate.weight'],
                                                    z[att+'gate.weight.qstate'],
                                                    z[bbb+'ln1.weight'],z[bbb+'ln1.bias'],
                                                    self.ebits, self.mbits
                                                    )
                    if T>1:
                        att1, time_mix_state = self.TimeMix_FC_Step2_Seq(B,T,C,H,self.ctx,
                                                                      xx,time_mix_state,
                                                                      r,w,k,v,g,
                                                                      z[att+'time_faaaa']
                                                                      )
                    else:
                        if B < 16:
                            att1, time_mix_state = self.TimeMix_FC_Step2_One(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )
                        else:
                            att1, time_mix_state = self.TimeMix_FC_Step2_One_HighBatch(B,T,C,H,self.ctx,
                                                                        xx,time_mix_state,
                                                                        r,w,k,v,g,
                                                                        z[att+'time_faaaa']
                                                                        )

                    att1 = self.TimeMix_FC_FP6_Step3(B,T,C,att1,g,self.n_head,
                                               z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                                               z[att+'output.weight'],
                                               z[att+'output.weight.qstate'],
                                               self.ebits, self.mbits
                                               )
                    # att1 = self.TimeMix_FC_Step3(B,T,C,att1,g,self.n_head,
                    #                            z[att+'ln_x.weight'],z[att+'ln_x.bias'],
                    #                            z[att+'output.weight']
                    #                            )
                    

                    x = x + att1

                    ffn1, channel_mix_state = self.ChannelMix_FC_FP6_Step1(x,channel_mix_state,
                                                                     z[bbb+'ln2.weight'],
                                                                     z[bbb+'ln2.bias'],
                                                                     int(self.n_embd),
                                                                     z[ffn+'time_maa_k'],
                                                                     z[ffn+'time_maa_r'],
                                                                     z[ffn+'receptance.weight'],
                                                                     z[ffn+'receptance.weight.qstate'],
                                                                     z[ffn+'key.weight'],
                                                                     z[ffn+'key.weight.qstate'],
                                                                     z[ffn+'value.weight'],
                                                                     z[ffn+'value.weight.qstate'],
                                                                     self.ebits, self.mbits
                                                                     )
                    # ffn1, channel_mix_state = self.ChannelMix_FC_Step1(x,channel_mix_state,
                    #                                                  z[bbb+'ln2.weight'],
                    #                                                  z[bbb+'ln2.bias'],
                    #                                                  int(self.n_embd),
                    #                                                  z[ffn+'time_maa_k'],
                    #                                                  z[ffn+'time_maa_r'],
                    #                                                  z[ffn+'receptance.weight'],
                    #                                                  z[ffn+'key.weight'],
                    #                                                  z[ffn+'value.weight']
                    #                                                  )
                    
                    x = x + ffn1
                
                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state

            if self.bit8quant:
                x = self.Final_int8(x,z['head.weight'],z['head.weight'+'.mx'],z['head.weight'+'.rx'],z['head.weight'+'.my'],z['head.weight'+'.ry'],
                                    self.n_embd,z['ln_out.weight'],z['ln_out.bias'])
            elif self.bit4quant:
                x = self.Final(x,bnb.functional.dequantize_4bit(z['head.weight'],quant_state=z['head.weight.qstate']),
                               self.n_embd,z['ln_out.weight'],z['ln_out.bias'])
                # x = self.Final_NF4(x,z['head.weight'],z['head.weight.qstate'],
                #                self.n_embd,z['ln_out.weight'],z['ln_out.bias'])
            elif self.bitfp6quant:
                x = self.Final_FP6(x,z['head.weight'],z['head.weight.qstate'],
                            self.n_embd,z['ln_out.weight'],z['ln_out.bias'],
                            self.ebits, self.mbits
                            )
            else:
                x = self.Final(x,z['head.weight'],self.n_embd,z['ln_out.weight'],z['ln_out.bias'])            

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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tb", default="1", type=int) #batch size 
    args = parser.parse_args()
    print('RWKV x060Core with FLA Test')

    pipeline = PIPELINE()
    model = RWKV_6('../models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth','fp6')
    Target_batch = args.tb#16

    States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    context =  'User: Tell me advantage of C++.\n\nAssistant:'
    context2 = 'User: Tell me advantage of C++.\n\nAssistant:'

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


    idx = torch.cat(prompts, dim=0)

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

    temperature = torch.full((Target_batch,), 1.0)
    top_p = torch.full((Target_batch,), 0.7)


    SamplingSum = 0
    ForwardSum = 0
    DecodeSum = 0

    for i in range(maxtoken):
        
        t0 = time.perf_counter()
        x[:, 0, 0] -= 1e10

        otokens = pipeline.improved_nucleus_sampling_multi_static(x[:, 0], temperature=temperature, top_p=top_p).tolist()

        tokens = []
        for j in range(Target_batch):
            tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0).to('cuda'))

        idx = torch.cat(tokens, dim=0)
        t1 = time.perf_counter()
        for j in range(Target_batch):
            out_tokens[j] += [otokens[j]]
            try:
                tmp = pipeline.decode(out_tokens[j][out_last[j]:])
                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                        #yield tmp
                        #if j == Target_batch - 1:
                        #    print(tmp,end="", flush=True)
                        output_text[j] = output_text[j] + tmp
                        out_last[j] = i + 1
            except:
                pass
        t2 = time.perf_counter()

        x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)
        t3 = time.perf_counter()
        ForwardSum += (t3 - t2)
        DecodeSum += (t2 - t1)
        SamplingSum += (t1 - t0)

    ForwardSum = ForwardSum / (float(maxtoken)) * 1000
    DecodeSum = DecodeSum / (float(maxtoken)) * 1000
    SamplingSum = SamplingSum / (float(maxtoken)) * 1000

    print('performance')
    print(f'ForwardAverage= {round(ForwardSum,4)} ms')
    print(f'DecodeSum= {round(DecodeSum,4)} ms')
    print(f'SamplingSum= {round(SamplingSum,4)} ms')



    t001 = time.perf_counter()

    print(output_text)
    print('RWKV-Infer FLA Refactor')

    tokensec = maxtoken / (t001-t000)
    print(f'TargetBatch = {Target_batch} Total token/s = {round(tokensec*Target_batch,2)} Single token/s = {round(tokensec,2)}')

    