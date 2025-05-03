#Refactoring RWKV x060,x070 Inference Engine with Flash Linear Attention
# Experimental Implement x070
#2024 OpenMOSE
from safetensors import safe_open
from safetensors.torch import load_file
#Test Torchao
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

from rwkvengine.matmulhell import quantize_weight, fused_dequant_gemm


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
#torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
#torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)


MyStatic = torch.jit.script


from rwkvengine.rwkv6 import RWKV_6, fused_recurrent_rwkv6_torch
from rwkvengine.rwkv7 import RWKV_7
from rwkvengine.arwkv7 import ARWKV_7
from rwkvengine.prwkv7 import PRWKV_7



class RWKV_x(nn.Module):

    def __init__(self,load_model: str,base_precision: str = 'int8',adapter_model:str = '', adapter_mode:str = '', adapter_scale:float=2.0,fully_fusedrecurrent:bool=True, tokenizer=''):

        #print('Helloworld RWKV v060 :) Initializing')
        print('RWKV-Infer RWKVCore Initializing')

        super().__init__()
        self.transfer_stream = torch.cuda.Stream()

        #GANBATTE CODE KAKOU
        self.bit8quant = False
        self.bit4quant = False
        self.bitfp8quant = False
        self.bitfp6quant = False

        self.fully_fusedrecurrent = fully_fusedrecurrent

        self.ExtremeCPUOffload = False

        self.debug = False

        self.eval()

        with torch.no_grad(): 

            if base_precision == 'fp16':
                self.base_precision = torch.half
            elif base_precision == 'int8':
                print('This is experimental fused matmul mode. HOSHOUSHIMASENN. ')
                self.base_precision = torch.float16
                self.bit8quant = True
                #self.bit4quant = True
            # elif base_precision == 'fp16int8':
            #     print('int8 Duplicated Automatic Change to NF4')
            #     self.base_precision = torch.bfloat16
            #     self.bit8quant = False
            #     self.bit4quant = True
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
                self.bitfp5quant = False
            elif base_precision == 'fp5':
                self.base_precision = torch.bfloat16
                self.bit8quant = False
                self.bit4quant = False
                self.bitfp8quant = False
                self.bitfp6quant = True
                self.bitfp5quant = True
            else:
                self.base_precision = torch.bfloat16
            
            modelpath = load_model
            self.SafeTensorMode = False

            def load_split_safetensors(model_dir, device="cpu"):
                """
                分割されたSafeTensorファイルを読み込む関数
                
                Args:
                    model_dir: 分割されたSafeTensorファイルが格納されているディレクトリパス
                    device: ロード先のデバイス（デフォルトは"cpu"）
                
                Returns:
                    dict: 統合されたモデルの状態辞書
                """
                # ディレクトリから全てのsafetensorファイルを取得
                files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
                
                if not files:
                    raise ValueError(f"No safetensors files found in {model_dir}")
                
                # 状態辞書を初期化
                state_dict = {}
                
                # 各ファイルを読み込んで統合
                for file in files:
                    file_path = os.path.join(model_dir, file)
                    # load_fileはtorch形式で読み込む
                    file_state_dict = load_file(file_path, device=device)
                    state_dict.update(file_state_dict)
                
                return state_dict
            if '.pth' in modelpath:
                z = torch.load(modelpath,map_location="cpu",mmap=True)
            else:
                self.SafeTensorMode = True
                # safetensor_files = sorted([f for f in os.listdir(modelpath) if f.endswith('.safetensors')])
                # self.modelpath = modelpath
                # self.safetensor_files = safetensor_files
                # if not safetensor_files:
                #     raise ValueError(f"No safetensors files found in {modelpath}")
                z = load_split_safetensors(modelpath)


            z_adapter_keys = None
            self.ModeMode = 'standard'
            if adapter_model != '' and adapter_mode != '':
                print('Adapter LoadMode')
                if 'lora' in adapter_mode or 'bone' in adapter_mode or 'hybrid' in adapter_mode:
                    print('LoRA Mode Lets Merge!')
                    self.ModeMode = adapter_mode
                    z_adapter = torch.load(adapter_model,map_location="cpu",mmap=True)
                    z_adapter_keys = list(z_adapter.keys())
                    for zkeys in z_adapter_keys:
                        z[zkeys] = z_adapter[zkeys]

            def Attach_Adapter(keyname,weight,adapter,mode,scaling=2.0,device='cuda'): #from JL-er lora merge inspired
                
                print(f'AttachAdapter = {keyname}')
                if keyname.endswith('.weight') or keyname.endswith('head'):
                    adapterkeys = list(adapter.keys())
                    #print(adapterkeys)
                    #exit()

                    if mode != '':
                        #print(f'scaling = {scaling}')
                        prefix = keyname[:-len('.weight')]
                        lora_A = prefix + '.lora_A'
                        lora_B = prefix + '.lora_B'
                        lora_M = prefix + '.lora_M'
                        gbmm = prefix + '.bone'
                        if lora_A in adapterkeys:
                            w=adapter
                            assert lora_B in adapterkeys

                            if lora_M in adapterkeys:
                                print('dora merging {lora_A} and {lora_B} and {lora_M} into {k}')
                                assert w[lora_B].shape[1] == w[lora_A].shape[0]

                                w[lora_A] = w[lora_A].to(device=device)
                                w[lora_B] = w[lora_B].to(device=device)
                                w[lora_M] = w[lora_M].to(device=device)
                                weight = weight + w[lora_B] @ w[lora_A] * scaling
                                norm = weight.norm(dim=0, keepdim=True) + 1e-6
                                weight = (w[lora_M] * weight) / norm  

                                del w[lora_A]
                                del w[lora_B]
                                del w[lora_M]
                                return weight
                            
                            else:
                                print(f'lora merging {lora_A} and {lora_B} into {k}')
                                
                                assert w[lora_B].shape[1] == w[lora_A].shape[0]

                                w[lora_A] = w[lora_A].to(device=device)
                                w[lora_B] = w[lora_B].to(device=device)
                                weight = weight + w[lora_B] @ w[lora_A] * scaling
                                del w[lora_A]
                                del w[lora_B]
                                return weight

                        if gbmm in adapterkeys :
                            w=adapter
                            print(f'bone merging {gbmm} into {k}')
                            w[gbmm] = w[gbmm].to(device=device)
                            b,r,_ = w[gbmm].shape
                            bone = rearrange(weight, '(a r1) (b r2) -> a b r1 r2', r1 = r, r2 = r)@w[gbmm]+w[gbmm]
                            weight += rearrange(bone, 'a b r1 r2 ->(a r1) (b r2) ')
                            print(weight)
                            del w[gbmm]
                            return weight
                        for key in adapterkeys:
                            if key == keyname:
                                weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                                print(f'key = {key} is swapped from Adapter')
                        return weight
                    else:
                        return weight
                else:
                    adapterkeys = list(adapter.keys())
                    for key in adapterkeys:
                        if key == keyname:
                            weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                            print(f'key = {key} is swapped from Adapter')

                    return weight
                
            # if self.SafeTensorMode:
            #     keys = []
            #     for file in self.safetensor_files:
            #         file_path = os.path.join(self.modelpath, file)
            #         # safe_openを使ってメモリマッピングでファイルを開く
            #         with safe_open(file_path, framework="pt", device='cpu') as f:
            #             for key in f.keys():
            #                 keys.append(key)

            # else:
            keys = list(z.keys())
            print("keys", keys)


            self.emboncpu = False
            RWKVMode = 6 #default RWKV 6
            self.gate_enable = False

            

            self.MoE = 0
            for key in keys:
                if 'blocks.0.att.r_k' in key and RWKVMode != 7:
                    print("RWKV x070 Mode :) with Native Pytorch Implementation")
                    RWKVMode = 7
                    #break
                elif 'router' in key and self.MoE != 1:
                    self.MoE = 1
                    print('Shared Mixture of Experts Mode!')
                elif ('x_g' in key or 'g1' in key ) and self.gate_enable != True:
                    self.gate_enable = True
                    print('Gate Enabled')

            
            if z_adapter_keys is not None:
                for key in z_adapter_keys :
                    if 'blocks.0.att.r_k' in key and RWKVMode != 7:
                        print("RWKV x070 Mode :) with Native Pytorch Implementation")
                        RWKVMode = 7
                    elif 'router' in key and self.MoE != 1:
                        self.MoE = 1
                        print('Shared Mixture of Experts Mode!')
                        #exit()
                    elif  ('x_g' in key or 'g1' in key ) and self.gate_enable != True:
                        self.gate_enable = True
                        print('Gate Enabled')


            ARWKVMode = 0

            self.ARWKVMLPMode = 0
            self.TokenshiftMode = 1


            for key in keys:
                if '.down.weight' in key and ARWKVMode != 1:
                    print("ARWKV-7 Mode :) Powered by RWKV-Red-Team.")
                    ARWKVMode = 1
                if '.x_a' in key and self.TokenshiftMode != 0:
                    print("Tokenshift Found")
                    self.TokenshiftMode = 0
            if z_adapter_keys is not None:
                for key in z_adapter_keys:
                    if '.down.weight' in key and ARWKVMode != 1:
                        print("ARWKV-7 Mode. Powered by RWKV-Red-Team.")
                        ARWKVMode = 1
                    if '.x_a' in key and self.TokenshiftMode != 0:
                        print("Tokenshift Found")
                        self.TokenshiftMode = 0

            self.ARWKVMode = ARWKVMode


            for key in keys:
                if '.gate_up.weight' in key and ARWKVMode == 1:
                    print("ARWKV Phi3.5 MLP Mode")
                    self.ARWKVMLPMode = 1
                    break
            if z_adapter_keys is not None:
                for key in z_adapter_keys:
                    if '.gate_up.weight' in key and ARWKVMode == 1:
                        print("ARWKV Phi3.5 MLP Mode")
                        self.ARWKVMLPMode = 1
                        break



            
            

            if RWKVMode == 6:
                print('RWKV x060 Mode :) with Flash-Linear-Attention')

            self.RWKVMode = RWKVMode
        

            

            # detect model details
            vocab_size, n_embd = z["emb.weight"].shape
            print(f'vocab = {vocab_size}')
            print(f'n_embd = {n_embd}')

            self.n_embd = n_embd
            self.vocab_size = vocab_size

            

            n_layer = 0
            for key in keys:
                if key.startswith("blocks."):
                    layer = int(key.split(".")[1])
                    if layer > n_layer:
                        n_layer = layer

            n_layer = n_layer + 1
            print("n_layer", n_layer)

            if self.RWKVMode == 7:
                self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
                print(self.head_size)
                if self.ARWKVMode == 1:
                    z['emb.weight'] = z['emb.weight']#.float() 
                else:
                    z['emb.weight'] = F.layer_norm(z['emb.weight'], (self.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])

                z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
                z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
                z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

                if self.gate_enable == False:
                    for i in range(n_layer):
                        z[f'blocks.{i}.att.x_g'] = torch.tensor(0)
                        z[f'blocks.{i}.att.g1'] = torch.tensor(0)
                        z[f'blocks.{i}.att.g2'] = torch.tensor(0)
            else:
                dim_ffn = z[f"blocks.0.ffn.value.weight"].shape[1]
                print(f'dim_ffn = {dim_ffn}')
                n_head = z[f"blocks.0.att.time_faaaa"].shape[0]
                print("n_head", n_head)
                self.head_size = n_embd // n_head
                self.dim_ffn = dim_ffn
                self.n_head = n_head
                self.ctx = 1024 #FLA Window


            self.dim_hidden = z['emb.weight'].shape[1]

            
            self.n_layer = n_layer

            keys = list(z.keys())

            self.requires_grad_(False)


            if self.MoE:
                print('will get MoE Configuration')

                self.MoELayerMode = [] #LoRA:0 Bone:1

                for i in range(self.n_layer):
                    bonetext = f"blocks.{i}.ffn.expert_0.key.bone_expert_0"
                    loratext = f"blocks.{i}.ffn.expert_0.key.lora_A_expert_0"
                    Found = False
                    for key in keys:
                        if bonetext in key:
                            self.MoELayerMode.append(1)
                            Found=True
                            break
                        elif loratext in key:
                            self.MoELayerMode.append(0)
                            Found=True
                            break
                    if Found == False:
                        assert "MoE FFN Layer is incorrect. please check file."
                print(self.MoELayerMode)
                print(f'total MoE Layers = {len(self.MoELayerMode)}')

                self.MoEExperts = 0

                #moe_info
                for i in range(128):
                    Found = False
                    for key in keys:
                        if f'blocks.0.ffn.expert_{i}' in key:
                            self.MoEExperts += 1
                            Found = True
                            break
                    if Found == False:
                        break


                #Search ActiveMoE
                self.ActiveMoEs = 0
                Found = False
                for key in keys:
                    if f'moe_info' in key:
                        self.ActiveMoEs = int(z[key][1])
                        Found = True
                        break
                
                print(f'MoE Experts Count : {self.MoEExperts}')
                print(f'MoE Active Count : {self.ActiveMoEs}')
                #exit()






            QuantList = ['.receptance.weight','.key.weight','.value.weight','.gate.weight','.output.weight','head.weight','.down.weight','up.weight','gate_up.weight']
            QuantListFP8 = ['att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight','head.weight','ffn.down.weight','ffn.up.weight','ffn.gate.weight','gate_up.weight'] #, ,
            QuantListFP6 = ['att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight','ffn.down.weight','ffn.up.weight','ffn.gate.weight','gate_up.weight'] #, ,
    
            

            # FP8 Transformer Engine Quantize Mode 
            # or Int8 FusedMatmul Engine
            if self.bitfp8quant == True or self.bit8quant:
                emboncpu = False
                self.ebits, self.mbits = 4, 3
                for k in keys:
                    print(f' k = {k} shape = {z[k].shape}' )
                    if self.ModeMode != 'standard':
                        z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                        z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda')

                    QuantKeyFound = False

                    if 'emb' in k and emboncpu:
                        z['emb.weight'] = z['emb.weight'].cpu()
                        QuantKeyFound = True 
                        gc.collect()
                        torch.cuda.empty_cache() 
                        self.emboncpu = True

                    def bf16_to_fp8(tensor):
                        FP8_MAX = 448.0
                        tensor=tensor.to(device='cuda')
                        scale = FP8_MAX / torch.max(torch.abs(tensor)) + 1e-6
                        tensor_scaled = tensor.float() * scale
                        tensor_clipped = torch.clamp(tensor_scaled, -FP8_MAX, FP8_MAX)
                        tensor_fp8 = tensor_clipped.to(dtype=torch.float8_e4m3fn ).contiguous()
                        return tensor_fp8, scale.float()




                    for QuantKey in QuantListFP8:
                        if k.endswith(QuantKey):
                            
                            QuantKeyFound = True
                            if self.bitfp8quant:
                                print(f'Quant {k} to torch.float8_e4m3fn')
                                z[k], z[k+'.qstate'] = bf16_to_fp8(z[k])
                            else:
                                print(f'Quant {k} to torch.int8 OpenMOSE SillyMatmul')
                                z[k], z[k+'.qstate'] = quantize_weight(z[k].to(device='cuda',dtype = torch.float16).t()) # quant to int8
              
                        
                    if QuantKeyFound == False:
                        for QuantKey in QuantList:
                            if k.endswith(QuantKey):
                                print(f'Quant {k} PassThrough')
                                QuantKeyFound = True
                                z[k] = z[k].to(device='cuda',dtype = self.base_precision).contiguous() 
                            

                    if QuantKeyFound == False:
                        z[k] = z[k].to(device='cuda')
                        if self.RWKVMode == 6:
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
                        elif self.RWKVMode == 7:
                            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                                print(f'target = {k} shape = {z[k].shape}')
                                z[k] = z[k].t()
                            if k.endswith('att.r_k'): z[k] = z[k].flatten()
                            z[k] = z[k].squeeze().to(dtype=self.base_precision)

            # FP6 Quantize Mode via Torch.AO
            elif self.bitfp6quant == True:
                emboncpu = False
                if self.bitfp5quant:
                    self.ebits, self.mbits = 2, 2
                else:
                    self.ebits, self.mbits = 3, 2
                count = 0
                for k in keys:
                    count = count + 1
                    #if count % 10:
                        #gc.collect()
                        #torch.cuda.empty_cache()
                    if self.ModeMode != 'standard':
                        z[k] = z[k].to(device='cuda', dtype=(self.base_precision))
                        z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda')
                    QuantKeyFound = False

                    if 'emb' in k and emboncpu:
                        z['emb.weight'] = z['emb.weight'].to(self.base_precision).cpu()
                        QuantKeyFound = True 
                        gc.collect()
                        torch.cuda.empty_cache() 
                        self.emboncpu = True


                    for QuantKey in QuantListFP6:
                        if k.endswith(QuantKey):
                            if self.bitfp5quant:
                                print(f'Quant {k} to FP5 shape = {z[k].shape}' )
                            else:
                                print(f'Quant {k} to FP6 shape = {z[k].shape}' )
                            QuantKeyFound = True
                            z[k] = z[k].to(device='cuda',dtype=torch.float16)#.t() 

                            # pre-process the weight. this will quantize the weight to FP6 and pack it in a special
                            # layout for tensor cores. refer to paper for more details.
                            if 'head' in k:
                                z[k], z[k+'.qstate'] = to_scaled_tc_floatx(z[k].cpu(), self.ebits, self.mbits) 
                                z[k]=z[k].to(device='cuda')
                                z[k+'.qstate']=z[k+'.qstate'].to(device='cuda')
                                gc.collect()
                                torch.cuda.empty_cache() 
                            # elif 'gate.weight' in k or  'up.weight' in k:
                            #     z[k], z[k+'.qstate'] = to_scaled_tc_floatx(z[k].cpu(), self.ebits, self.mbits)
                            #     z[k]=z[k].to(device='cuda')
                            #     z[k+'.qstate']=z[k+'.qstate'].to(device='cuda')
                            #     gc.collect()
                            #     torch.cuda.empty_cache() 
                            else:
                                z[k], z[k+'.qstate'] = to_scaled_tc_floatx(z[k], self.ebits, self.mbits)

                            if self.ExtremeCPUOffload:
                                z[k] = z[k].to(device='cpu')
                                z[k+'.qstate'] = z[k+'.qstate'].to(device='cpu')


                    if QuantKeyFound == False:
                        for QuantKey in QuantList:
                            if k.endswith(QuantKey):
                                print(f'Quant {k} PassThrough')
                                QuantKeyFound = True
                                z[k] = z[k].to(device='cuda',dtype = self.base_precision).contiguous() 
                                z[k+'.qstate'] = torch.randn(1)
                            

                    if QuantKeyFound == False:
                        z[k] = z[k].to(device='cuda')
                        if self.RWKVMode == 6:
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
                        elif self.RWKVMode == 7:
                            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                                print(f'target = {k} shape = {z[k].shape}')
                                z[k] = z[k].t()
                            if k.endswith('att.r_k'): z[k] = z[k].flatten()
                            z[k] = z[k].squeeze().to(dtype=self.base_precision)


            # Non Quantize Mode FP16 or BF16
            else:
                for k in keys:
                    if self.ModeMode != 'standard':
                        z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                        z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda')
                    z[k] = z[k].to(device='cuda')

                    if self.RWKVMode == 6:
                        if k.endswith('.time_decay'): z[k] = z[k].float()
                        if k.endswith('.time_faaaa'): z[k] = z[k].float()
                        elif k.endswith('.receptance.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                        elif k.endswith('.key.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                        elif k.endswith('.value.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                        elif k.endswith('.gate.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                        elif k.endswith('.output.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                        elif k.endswith('head.weight'): z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                        elif k.endswith('.ln_x.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('.ln_x.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('blocks.0.ln0.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('blocks.0.ln0.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('ln_out.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('ln_out.bias'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        elif k.endswith('emb.weight'): z[k] = z[k].to(dtype=torch.bfloat16).contiguous() 
                        else:
                            z[k] = z[k].to(dtype = self.base_precision).contiguous() 
                    elif self.RWKVMode == 7:
                        if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                            print(f'target = {k} shape = {z[k].shape}')
                            z[k] = z[k]#.t()
                        if k.endswith('att.r_k'): z[k] = z[k].flatten()

                        z[k] = z[k].squeeze().to(dtype=self.base_precision)

            if self.RWKVMode == 6:
                for i in range(n_layer):
                    z[f'blocks.{i}.att.time_maa_wkvrg'] = torch.stack([z[f'blocks.{i}.att.time_maa_w'], z[f'blocks.{i}.att.time_maa_k'], z[f'blocks.{i}.att.time_maa_v'], z[f'blocks.{i}.att.time_maa_r'], z[f'blocks.{i}.att.time_maa_g']], dim=0).contiguous()
            # if self.RWKVMode == 7:
            #     for i in range(n_layer):
            #         z[f'blocks.{i}.att.time_maa_wkvrg'] = torch.stack([z[f'blocks.{i}.att.x_r'], z[f'blocks.{i}.att.x_w'], z[f'blocks.{i}.att.x_k'], z[f'blocks.{i}.att.x_v'], z[f'blocks.{i}.att.x_a'], z[f'blocks.{i}.att.x_g']], dim=0).contiguous()
            #         z[f'blocks.{i}.att.x_r'] = None
            #         z[f'blocks.{i}.att.x_w'] = None
            #         z[f'blocks.{i}.att.x_k'] = None
            #         z[f'blocks.{i}.att.x_v'] = None
            #         z[f'blocks.{i}.att.x_a'] = None
            #         z[f'blocks.{i}.att.x_g'] = None
            self.z = z
            self.device = z['blocks.0.att.receptance.weight'].device
            self.dtype = z['emb.weight'].dtype

            self.dummytensor = torch.tensor(0).to(dtype=self.dtype,device=self.device)

            if self.ModeMode != 'standard':
                del z_adapter

            keys = list(z.keys())
            for key in keys:
                if z[key] is not None:
                    print(f'{key} {z[key].shape} {z[key].dtype}')
                    if ('.bone' in key or '.lora' in key) and 'expert' not in key:
                        z[key] = None
                        print(f'{key} deleted')

            
        
            gc.collect()
            torch.cuda.empty_cache()


    def new_state(self, B):
         if self.RWKVMode == 6:
            return BlockStateList.create(
                    self.n_layer,
                    B,
                    self.n_embd, 
                    self.n_head,
                    self.device, self.dtype
                )
         elif self.RWKVMode == 7:
            return BlockStateList.x070_create(self.n_layer,
                                              B,
                                              self.n_embd,
                                              self.head_size,
                                              self.device,
                                              self.dtype)
            
    
        
    def load_state(self,state_filename,EnableOffset = False):
        try:
            state_raw = torch.load(state_filename, map_location="cpu")
        except Exception as e:
            print(e)
            if EnableOffset:
                return "error", None
            return "error"
        state_raw_shape = state_raw[f"blocks.0.att.time_state"].shape #next(iter(state_raw.values())).shape

        state_keys = list(state_raw.keys())

        state_count = 0
        state_time_offset_mode = 0
        for key in state_keys:
            #print(f'{key}')
            if 'time_state' in key and 'time_offset' not in key:
                state_count = state_count + 1
            if 'time_offset' in key:
                if state_raw[key].shape == state_raw_shape:
                    state_time_offset_mode = 1

        #exit()

        #args = model.args
        self.debug = 1
        if self.debug:
            print(f"{state_count} != {self.n_layer}")
            print(f"{state_raw_shape[0] * state_raw_shape[1]} != {self.n_embd}")

        if (
            state_count != self.n_layer
            or state_raw_shape[0] * state_raw_shape[1] != self.n_embd
        ):
            print("state failed to load")
            if EnableOffset:
                return "error", None
            return "error"

        #strategy = model.strategy
        
        atype = torch.bfloat16 #dd.atype
        dev = 'cpu'
        model_current_statetuned = [None] * self.n_layer * 3

        model_current_offset = torch.zeros((self.n_layer,self.dim_hidden), dtype=atype, requires_grad=False, device=dev).contiguous()
   

        for i in range(self.n_layer):
            #dd = strategy[i]
            #dd.device
            
            model_current_statetuned[i * 3 + 0] = torch.zeros(
                self.n_embd, dtype=atype, requires_grad=False, device=dev
            ).contiguous()

            #self.RWKVMode
            tempstate = state_raw[f"blocks.{i}.att.time_state"]
            
            #Offset Tuning
            if EnableOffset:
                    try:
                        time_state_offset = state_raw[f"blocks.{i}.att.time_offset"]
                        #time_state_offset=time_state_offset.transpose(1, 2)
                        
                    except:
                        time_state_offset = torch.zeros((self.dim_hidden),
                                                        dtype=atype, requires_grad=False, device=dev
                                            )

                    model_current_offset[i]=time_state_offset



            if self.RWKVMode == 7:
               tempstate=tempstate.transpose(1, 2)
            model_current_statetuned[i * 3 + 1] = (
                tempstate
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
            wkv_states[i] = model_current_statetuned[i*3 + 1]#.permute(0,2,1)

        print(wkv_states)

        if EnableOffset:
            return wkv_states, model_current_offset

        return wkv_states#.to(dtype=torch.float16)
    


    
    
    
    

    def forward(self, idx, last_shift_states , last_wkv_states, one_mode = False, KernelMode = 0,time_offset_state:torch.Tensor=None):
        if self.RWKVMode == 6:
            return RWKV_6.x060_forward(self,idx,last_shift_states,last_wkv_states)
        elif self.RWKVMode == 7 and self.ARWKVMode == 0:
            return RWKV_7.x070_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=False,time_offset_state=time_offset_state)
        elif self.RWKVMode == 7 and self.ARWKVMode == 1 and self.TokenshiftMode == 0:
            #print('ARWKV')
            return ARWKV_7.ax070_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=False )
        
        elif self.RWKVMode == 7 and self.ARWKVMode == 1 and self.TokenshiftMode == 1:
            #print('PRWKV')
            return PRWKV_7.PRWKV7_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=False,time_offset_state=time_offset_state)
    













    