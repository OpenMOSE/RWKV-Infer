#Refactoring RWKV x060,x070 Inference Engine with Flash Linear Attention
#Experimental Implement x070
#HXA079 Hybrid
#2025 OpenMOSE
import os
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "True"
os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "True"

import torch
import json
#Test Torchao
torch.compiler.reset()


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
from rwkvengine.tensormagic import Attach_Adapter
from rwkvengine.matmulhell import quantize_weight, fused_dequant_gemm


#from rwkvengine.misc import PIPELINE
from rwkvengine.misc import PIPELINE, TimeMixState, ChannelMixState,BlockState,BlockStateList
from rwkvengine.matmularena import hybrid_matmul
# from rwkvengine.fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
# from rwkvengine.fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
from rwkvengine.fla.ops.rwkv7 import chunk_rwkv7
from rwkvengine.cuda.wkv7triton import rwkv7_attn_triton
from rwkvengine.quantization import CleanQuantizationMode,DoQuantizationIfPossible
from rwkvengine.tensormagic import load_split_safetensors, RenameToHFStyle, pad_weight_tensor_to_256
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

MyStatic = torch.jit.script


# from rwkvengine.rwkv6 import RWKV_6, fused_recurrent_rwkv6_torch
from rwkvengine.rwkv7 import RWKV_7
# from rwkvengine.arwkv7 import ARWKV_7
from rwkvengine.prwkv7 import PRWKV_7
from rwkvengine.hrwkv7 import HRWKV_7, compute_qwen3_rope_cache




# class RWKV_x(nn.Module):
#     def __init__(self,load_model: str,base_precision: str = 'int8',adapter_model:str = '', adapter_mode:str = '', adapter_scale:float=2.0,fully_fusedrecurrent:bool=True, tokenizer='',rope_theta=1000000.0,rms_norm_eps=1e-6,max_ctxlen=8192):
#         print('RWKV-Infer RWKVCore2 Initializing')


class RWKV_x(nn.Module):

    def __init__(self,load_model: str,base_precision: str = 'int8',adapter_model:str = '', adapter_mode:str = '', adapter_scale:float=2.0,fully_fusedrecurrent:bool=True, tokenizer='',rope_theta=1000000.0,rms_norm_eps=1e-6,max_ctxlen=8192,device='cuda'):

        #print('Helloworld RWKV v060 :) Initializing')
        print('RWKV-Infer RWKVCore Initializing')
        self.max_ctxlen = max_ctxlen

        super().__init__()
 
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        #GANBATTE CODE KAKOU
        # self.bit8quant = False
        # self.bit4quant = False 
        # self.bitfp8quant = False
        # self.bitfp6quant = False

        # self.fully_fusedrecurrent = fully_fusedrecurrent
        # self.ExtremeCPUOffload = False
        self.debug = False

        self.eval()

        #Dummy
        self.ebits, self.mbits = 4, 3

        with torch.no_grad(): 

            
            
            modelpath = load_model
            self.SafeTensorMode = False

            
            if '.pth' in modelpath:
                z = torch.load(modelpath,map_location="cpu",mmap=True)
                self.HFMode = False
            else:
                self.SafeTensorMode = True
                z = load_split_safetensors(modelpath)
                self.HFMode = True


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

            
            def natural_sort_key(text):
                """自然順ソート用のキー関数"""
                return [int(c) if c.isdigit() else c for c in re.split('(\d+)', text)]
       
            keys = sorted(list(z.keys()),key=natural_sort_key)#.sort()
            print("keys", keys)
            #exit()

            if self.HFMode:
                with open(f'{load_model}/config.json', 'r', encoding='utf-8') as file:
                    modelconfig = json.load(file)
                self.n_layer = modelconfig['num_hidden_layers']
                self.n_kv = modelconfig['num_key_value_heads']#//modelconfig['head_dim']
                self.head_size = modelconfig['head_dim']
                self.n_head=modelconfig['num_attention_heads']
                self.rms_norm_eps = modelconfig['rms_norm_eps']
                self.rope_theta = modelconfig['rope_theta']

            else:
                modelconfig = {}
                #modelconfig['']
                print('Checking n_layers')
                n_layer = 0
                for key in keys:
                    if key.startswith("blocks."):
                        layer = int(key.split(".")[1])
                        if layer > n_layer:
                            n_layer = layer
                n_layer = n_layer + 1
                modelconfig['num_hidden_layers'] = n_layer

                print('Checking Headsize, num_attention_heads')
                rk = None
                for key in keys:
                    if '.r_k' in key:
                        rk = z[key]
                        break
                
                if rk is not None:
                    self.n_head, self.head_size = rk.shape
                    modelconfig['num_attention_heads'] = self.n_head
                    modelconfig['head_dim'] = self.head_size
                    modelconfig['weight_dtype'] = rk.dtype
                    modelconfig['torch_dtype'] = rk.dtype

                else:
                    raise 'Cannot detect Headsize and num_attention_heads. \ncurrently deprecated rwkv6. basically im focusing on latest architectures. thank you :)'
                

                print(f'Checking KV Heads')
                self.n_kv = 1

                key_shape = None
                for key in keys:
                    if '.key' in key:
                        key_shape = z[key].shape
                        break

                self.n_kv = key_shape[0]//self.head_size

                #     for key in keys:
                #         if 'k_proj' in key:
                #             k_shape = z[key].shape
                #             print(k_shape)
                #             self.n_kv = k_shape[0]//self.head_size
                #             print(self.n_kv)
                #             break
                    #
                modelconfig['rope_theta'] = rope_theta
                modelconfig['tokenizer'] = tokenizer
                modelconfig['rms_norm_eps'] = rms_norm_eps
                modelconfig['max_position_embeddings'] = max_ctxlen

                print('Checking r(q), k norm')
                modelconfig['enable_qk_norm'] = False
                for key in keys:
                    if '.ln_k' in key:
                        modelconfig['enable_qk_norm'] = True
                        break

                print('Checking Model Architecture.')
                DefaultArchitecture = 'x070'
                modelconfig['rwkv_architecture'] = DefaultArchitecture

                for key in keys:
                    if '.k0' in key:
                        modelconfig['rwkv_architecture'] = 'hxa079'
                        break

                if modelconfig['rwkv_architecture'] == DefaultArchitecture:
                    for key in keys:
                        if 'down.weight' in key:
                            modelconfig['rwkv_architecture'] = 'cxa076'
                            break

                if modelconfig['rwkv_architecture'] == 'hxa079':
                    print('Check RWKV Layers and Transformer Layers.')









            self.n_layer = modelconfig['num_hidden_layers']

            self.HRWKV_Block_Mode = [] #0: RWKV, 1: Attention
            self.HRWKV_Misc = {}
            self.GQALayers = 0
            self.RWKVLayers = 0
            self.device = device

            self.lora_rank_decay = 0
            self.lora_rank_iclr = 0
            self.lora_rank_value_residual_mix = 0
            self.lora_rank_key_residual_mix = 0
            self.lora_rank_gate = 0

            first_rwkv_layer = -1

            for i in range(self.n_layer):
                Found = False
                for key in keys:
                    if ( f'blocks.{i}.' in key or f'layers.{i}.' in key) and 'q_proj' in key:
                        Found = True
                        self.HRWKV_Block_Mode.append([1,i,self.GQALayers])
                        self.GQALayers = self.GQALayers + 1
                if Found == False:
                    self.HRWKV_Block_Mode.append([0,i,self.RWKVLayers])
                    self.RWKVLayers = self.RWKVLayers + 1
                    if first_rwkv_layer == -1:
                        first_rwkv_layer = i
            print(f'Layer info = {self.HRWKV_Block_Mode}')
            print(f'RWKV Layer = {self.RWKVLayers}')
            print(f'GQA Layer = {self.GQALayers}')


            # if first_rwkv_layer != -1 and self.HFMode:
            #     self.lora_rank_decay = z[f'model.layers.{first_rwkv_layer}.self_attn.w1'].shape[1]
            #     self.lora_rank_iclr = z[f'model.layers.{first_rwkv_layer}.self_attn.a1'].shape[1]
            #     self.lora_rank_value_residual_mix = z[f'model.layers.{first_rwkv_layer}.self_attn.v1'].shape[1]
            #     self.lora_rank_key_residual_mix = z[f'model.layers.{first_rwkv_layer}.self_attn.k1'].shape[1]
            #     self.lora_rank_gate = z[f'model.layers.{first_rwkv_layer}.self_attn.g1'].shape[1]
            #     print(f'lora_rank_decay = {self.lora_rank_decay}')
            #     print(f'lora_rank_iclr = {self.lora_rank_iclr}')
            #     print(f'lora_rank_value_residual_mix = {self.lora_rank_value_residual_mix}')
            #     print(f'lora_rank_key_residual_mix = {self.lora_rank_key_residual_mix}')
            #     print(f'lora_rank_gate = {self.lora_rank_gate}')
            #     #exit()



            


            self.MoEMode = False
            for key in keys:
                if '.experts.' in key:
                    self.MoEMode = True
                    self.num_experts = modelconfig['num_experts'] 
                    self.num_experts_per_tok = modelconfig['num_experts_per_tok'] 
                    self.norm_topk_prob = modelconfig['norm_topk_prob']
                    break
            



            self.attn_quant,self.ffn_quant,self.base_precision,self.attn_ebits, self.attn_mbits,self.mlp_ebits, self.mlp_mbits = CleanQuantizationMode(base_precision)
            self.head_ebits,self.head_mbits = -8,-8
            






            print(modelconfig)

            if modelconfig['rwkv_architecture'] == 'hxa079':
                print('hxa079 architecture initializing')
            elif modelconfig['rwkv_architecture'] == 'cxa076':
                print('cxa076 architecture initializing')
            elif modelconfig['rwkv_architecture'] == 'x070':
                print('x070 architecture initializing')

            if self.HFMode == False:
                for k in keys:
                    if f'head.' in k:
                        if self.ModeMode != 'standard':
                            z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                            z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda').to(device='cpu')
                            #exit()
                for key in keys:
                    if f'emb.' in key:
                        z = RenameToHFStyle(z,key,0)
                    elif f'head.' in key:
                        z = RenameToHFStyle(z,key,0)
                    elif f'ln_out.' in key:
                        z = RenameToHFStyle(z,key,0)

            self.RWKVMode = 7
            self.HRWKV_Mode = 0


            


            for i in range(self.n_layer):
                keys = sorted(list(z.keys()),key=natural_sort_key)

                if modelconfig['rwkv_architecture'] == 'hxa079':
                    self.HRWKV_Mode = 1
                    if self.HFMode == False:
                        for key in keys:
                            if f'blocks.{i}.' in key:
                                z = RenameToHFStyle(z,key,self.HRWKV_Block_Mode[i][0])
                        keys = sorted(list(z.keys()),key=natural_sort_key)

                    bbb = f'model.layers.{i}.'
                    att = f'model.layers.{i}.self_attn.'
                    ffn = f'model.layers.{i}.mlp.'
                    if self.HRWKV_Block_Mode[i][0] == 0:
                        print(f'r_k flatten')
                        z[att+'r_k'] = z[att+'r_k'].flatten()

                        print(f'Layer:{i} RWKV hxa079 block r,k,v try to fusion')
                        #
                        print(f"r shape = {z[att+'receptance.weight'].shape}")
                        print(f"k shape = {z[att+'key.weight'].shape}")
                        print(f"v shape = {z[att+'value.weight'].shape}")
                        z[att + 'rkv_fused.weight'] = torch.cat([z[att+'receptance.weight'],
                                                                 z[att+'key.weight'],
                                                                 z[att+'value.weight'],
                                                                ],dim=0)

                        self.HRWKV_Misc[att + 'rkv_split_list'] = [z[att+'receptance.weight'].shape[0],
                                                     z[att+'key.weight'].shape[0],
                                                     z[att+'value.weight'].shape[0],
                                                    ]


                        print(f"rkv shape = {z[att + 'rkv_fused.weight'].shape}")
                        #exit()

                        print(f'wavgk 1 fuse')
                        z[att + 'wavgk_fused'] = torch.cat([z[att+'w1'],
                                                                 z[att+'a1'],
                                                                 z[att+'v1'],
                                                                 z[att+'g1'],
                                                                 z[att+'k1'],
                                                                ],dim=1)

                        self.HRWKV_Misc[att + 'wavgk_split_list'] = [z[att+'w1'].shape[1],
                                                     z[att+'a1'].shape[1],
                                                     z[att+'v1'].shape[1],
                                                     z[att+'g1'].shape[1],
                                                     z[att+'k1'].shape[1],
                                                    ]

                        print(self.HRWKV_Misc[att + 'wavgk_split_list'])

                        print(f"wavgk shape = {z[att + 'wavgk_fused'].shape}")

                       
                        z[att+'receptance.weight'] = None
                        z[att+'key.weight'] = None
                        z[att+'value.weight'] = None
                        z[att+'w1'] = None
                        z[att+'a1'] = None
                        z[att+'v1'] = None
                        z[att+'g1'] = None
                        z[att+'k1'] = None
                        del z[att+'receptance.weight']
                        del z[att+'key.weight']
                        del z[att+'value.weight']

                        del z[att+'w1']
                        del z[att+'a1']
                        del z[att+'v1']
                        del z[att+'g1']
                        del z[att+'k1']
                    else:
                        print(f'Layer:{i} Attention block q,k,v try to fusion')
                        z[att + 'qkv_fused.weight'] = torch.cat([z[att+'q_proj.weight'],
                                                                 z[att+'k_proj.weight'],
                                                                 z[att+'v_proj.weight']
                                                                ],dim=0)

                        self.HRWKV_Misc[att + 'qkv_split_list'] = [z[att+'q_proj.weight'].shape[0],
                                                     z[att+'k_proj.weight'].shape[0],
                                                     z[att+'v_proj.weight'].shape[0],
                                                    ]
                        
                        z[att+'q_proj.weight'] = None
                        z[att+'k_proj.weight'] = None
                        z[att+'v_proj.weight'] = None
                        del z[att+'q_proj.weight']
                        del z[att+'k_proj.weight']
                        del z[att+'v_proj.weight']
                    
                    if self.MoEMode == False:
                        print('SwiGLU MLP Layer Fusing')
                        z[ffn+'gate_proj.weight'] = pad_weight_tensor_to_256(z[ffn+'gate_proj.weight'])
                        z[ffn+'up_proj.weight'] = pad_weight_tensor_to_256(z[ffn+'up_proj.weight'])
                        z[ffn + 'gateup.weight'] = torch.cat([z[ffn+'gate_proj.weight'],
                                                                    z[ffn+'up_proj.weight']
                                                                    ],dim=0)

                        z[ffn+'down_proj.weight'] = pad_weight_tensor_to_256(z[ffn+'down_proj.weight'])

                        self.HRWKV_Misc[ffn + 'gateup_split_list'] = [z[ffn+'gate_proj.weight'].shape[0],
                                                        z[ffn+'up_proj.weight'].shape[0],
                                                    ]
                        z[ffn+'gate_proj.weight'] = None
                        z[ffn+'up_proj.weight'] = None
                        del z[ffn+'gate_proj.weight']
                        del z[ffn+'up_proj.weight']
                    else:
                        print('SwiGLU MLP Experts Layer Fusing')
                        for j in range(self.num_experts):
                            addfix = f'experts.{j}.'
                            z[ffn+addfix+'gate_proj.weight'] = pad_weight_tensor_to_256(z[ffn+addfix+'gate_proj.weight'])
                            z[ffn+addfix+'up_proj.weight'] = pad_weight_tensor_to_256(z[ffn+addfix+'up_proj.weight'])
                            z[ffn+addfix + 'gateup.weight'] = torch.cat([z[ffn+addfix+'gate_proj.weight'],
                                                                        z[ffn+addfix+'up_proj.weight']
                                                                        ],dim=0)

                            self.HRWKV_Misc[ffn+addfix + 'gateup_split_list'] = [z[ffn+addfix+'gate_proj.weight'].shape[0],
                                                            z[ffn+addfix+'up_proj.weight'].shape[0],
                                                        ]
                            z[ffn+addfix+'gate_proj.weight'] = None
                            z[ffn+addfix+'up_proj.weight'] = None
                            del z[ffn+addfix+'gate_proj.weight']
                            del z[ffn+addfix+'up_proj.weight']
                        
                        # z[f'model.layer.{i}.mlp.experts.{j}.gate_proj.weight']





                elif modelconfig['rwkv_architecture'] == 'x070':
                    
                        
                    if self.HFMode == False:
                        for k in keys:
                            if f'blocks.{i}.' in k:
                                if self.ModeMode != 'standard':
                                    z[k] = z[k].to(device='cuda', dtype=torch.bfloat16)
                                    z[k] = Attach_Adapter(keyname=k,weight=z[k],adapter=z_adapter,mode=self.ModeMode,scaling=adapter_scale,device='cuda').to(device='cpu')

                        keys = sorted(list(z.keys()),key=natural_sort_key)

                        for key in keys:
                            if f'blocks.{i}.' in key:
                                z = RenameToHFStyle(z,key,0)
                        keys = sorted(list(z.keys()),key=natural_sort_key)

                    bbb = f'model.layers.{i}.'
                    att = f'model.layers.{i}.self_attn.'
                    ffn = f'model.layers.{i}.mlp.'
                    
                    

                keys = sorted(list(z.keys()),key=natural_sort_key)
                bbb = f'model.layers.{i}.'
                for key in keys:
                    if bbb in key:
                        z = DoQuantizationIfPossible(z,key,self.attn_quant,self.ffn_quant,self.base_precision,self.device) 

            z = DoQuantizationIfPossible(z,'lm_head.weight',self.attn_quant,self.ffn_quant,self.base_precision,self.device) 
            

            keys = sorted(list(z.keys()),key=natural_sort_key)

            offload_blocks = []
            # for k in range(self.n_layer-12):
            #     offload_blocks.append(f'layers.{k}.')

            #blocks = ['layers.0.','layers.1.','layers.2.','layers.3.','layers.4.','layers.5.','layers.6.','layers.7.','layers.8.','layers.9.','layers.10.','layers.11.','layers.12.','layers.13.','layers.14.']
            for k in keys:
                is_in_blocks = any(block in k for block in offload_blocks)
                
                if not k.endswith('qstate') and 'emb' not in k and 'ln0' not in k:# or is_in_blocks):  and ('mlp' not in k or is_in_blocks)
                    print(f'{k} move to device {device}')
                    z[k] = z[k].to(device=self.device)
            
         

            # detect model details
            vocab_size, n_embd = z["model.embed_tokens.weight"].shape
            print(f'vocab = {vocab_size}')
            print(f'n_embd = {n_embd}')

            self.n_embd = n_embd
            self.vocab_size = vocab_size

            if modelconfig['rwkv_architecture'] == 'x070':
                z = DoQuantizationIfPossible(z,'model.norm.weight',self.attn_quant,self.ffn_quant,self.base_precision,self.device) 
                z = DoQuantizationIfPossible(z,'model.norm.bias',self.attn_quant,self.ffn_quant,self.base_precision,self.device) 
                emb_device = z['model.embed_tokens.weight'].device
                z['model.embed_tokens.weight'] = F.layer_norm(z['model.embed_tokens.weight'].to(dtype=self.base_precision,device='cuda'), (self.n_embd,), weight=z['model.layers.0.ln0.weight'].to(device='cuda'), bias=z['model.layers.0.ln0.bias'].to(device='cuda')).to(device=emb_device)
                                 


            self.dim_hidden = z['model.embed_tokens.weight'].shape[1]

      

            self.dummytensor = torch.tensor(0).to(dtype=self.base_precision,device=self.device)

            if self.ModeMode != 'standard':
                del z_adapter

            keys = list(z.keys())
            for key in keys:
                if z[key] is not None:
                    #print(f'{key} {z[key].shape} {z[key].dtype}')
                    if ('.bone' in key or '.lora' in key) and 'expert' not in key:
                        z[key] = None
                        print(f'{key} deleted')


#Sliding

            
        
            gc.collect()
            torch.cuda.empty_cache()

            if modelconfig['rwkv_architecture'] == 'hxa079':
                self.cos, self.sin, _ = compute_qwen3_rope_cache(1048576,self.head_size,self.device,torch.float32,self.rope_theta)
                #
                DummyCheckList = ['receptance.weight.qstate','key.weight.qstate','value.weight.qstate','output.weight.qstate',
                             'receptance.bias','key.bias','value.bias','output.bias',
                             'q_proj.weight.qstate','k_proj.weight.qstate','v_proj.weight.qstate','o_proj.weight.qstate',
                             'q_proj.bias','k_proj.bias','v_proj.bias','o_proj.bias',
                             'rkv_fused.weight.qstate','qkv_fused.weight.qstate',
                                ]
                DummyCheckList2 = ['gate.weight.qstate','down.weight.qstate','up.weight.qstate','gateup.weight.qstate'
                                ]
                NoneCheckList = ['ln_r.weight','ln_k.weight']
                for i in range(self.n_layer):
                    bbb = f'model.layers.{i}.'
                    att = f'model.layers.{i}.self_attn.'
                    ffn = f'model.layers.{i}.mlp.'

                    for key in DummyCheckList:
                        z[att+key] = z.get(att+key,self.dummytensor)
                    for key in DummyCheckList2:
                        z[ffn+key] = z.get(ffn+key,self.dummytensor)

                    for key in NoneCheckList:
                        z[att+key] = z.get(att+key,None)

        self.modelconfig = modelconfig
        self.z = z



    def new_state(self, B, max_token=4096):
        #  if self.RWKVMode == 6:
        #     return BlockStateList.create(
        #             self.n_layer,
        #             B,
        #             self.n_embd, 
        #             self.n_head,
        #             self.device, self.base_precision
        #         )
        if self.modelconfig['rwkv_architecture'] == 'hxa079':
             return BlockStateList.hx078_create(self.RWKVLayers,self.GQALayers,
                                                B,
                                                self.n_head,
                                                self.n_kv,
                                                self.head_size,
                                                max_token,
                                                self.device,
                                                self.base_precision                                                
                                                )
        elif self.RWKVMode == 7:
            return BlockStateList.x070_create(self.n_layer,
                                              B,
                                              self.n_embd,
                                              self.head_size,
                                              self.device,
                                              self.base_precision)
            
    
        
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
    


    
    
    
    

    def forward(self, idx, last_shift_states , last_wkv_states,kv_cache=None,pos_cache=None,full_output=False, one_mode = False, KernelMode = 0,time_offset_state:torch.Tensor=None):
        
        if self.modelconfig['rwkv_architecture'] == 'hxa079' and self.MoEMode == False:
            return HRWKV_7.hxa079r_forward(self,idx,last_wkv_states,kv_cache,pos_cache,full_output)
        elif self.modelconfig['rwkv_architecture'] == 'hxa079' and self.MoEMode == True:
            return HRWKV_7.hxa079r_moe_forward(self,idx,last_wkv_states,kv_cache,pos_cache,full_output) 
        elif self.modelconfig['rwkv_architecture'] == 'x070':
            return RWKV_7.x070_forward(self,idx,last_shift_states,last_wkv_states,full_output=full_output) 

        
        # if self.RWKVMode == 6:
        #     return RWKV_6.x060_forward(self,idx,last_shift_states,last_wkv_states)
        # elif self.RWKVMode == 7 and self.ARWKVMode == 0:
        #     return RWKV_7.x070_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=full_output,time_offset_state=time_offset_state)
        # elif self.RWKVMode == 7 and self.HRWKV_Mode == 1:
        #     if self.HRWKV_Generation == 79:
        #         return HRWKV_7.hxa079r_forward(self,idx,last_wkv_states,kv_cache,pos_cache,full_output)
        #     else:
        #         return HRWKV_7.hxa078r_forward(self,idx,last_wkv_states,kv_cache,pos_cache,full_output)
        # elif self.RWKVMode == 7 and self.ARWKVMode == 1 and self.TokenshiftMode == 0:
        #     #print('ARWKV')
        #     return ARWKV_7.ax070_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=full_output )
        
        # elif self.RWKVMode == 7 and self.ARWKVMode == 1 and self.TokenshiftMode == 1:
        #     #print('PRWKV')
        #     return PRWKV_7.PRWKV7_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=full_output,time_offset_state=time_offset_state)
    













    