#Refactoring RWKV x060,x070 Inference Engine with Flash Linear Attention
#Experimental Implement x070
#HXA079 Hybrid
#2025 OpenMOSE


from safetensors import safe_open
from safetensors.torch import load_file
import torch
#Test Torchao

try:
    import torchao
    from torchao.dtypes.floatx import to_scaled_tc_floatx
    from torchao.ops import quant_llm_linear
    HAS_TORCHAO = True
except ImportError:
    print('torchao not found')
    HAS_TORCHAO = False
    bnb = None

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    print('Bitsandbytes not found')
    HAS_BITSANDBYTES = False
    bnb = None
try:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
    HAS_HQQ = True
except ImportError:
    print('hqq not found')
    HAS_HQQ = False
    HQQLinear = None


def create_hqq_module_from_weight(
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    nbits: int = 4,
    group_size: int = 64,
    compute_dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
    initialize: bool = True
) -> HQQLinear:
    """
    Linear Weightテンソルから直接HQQ Quant Moduleを作成する関数
    
    Args:
        weight: Linear layerのweightテンソル (shape: [out_features, in_features])
        bias: Linear layerのbiasテンソル (optional)
        nbits: 量子化ビット数 (default: 4)
        group_size: グループサイズ (default: 64)
        compute_dtype: 計算時のデータ型 (default: torch.float16)
        device: デバイス (default: 'cuda')
        initialize: 初期化フラグ (default: True)
    
    Returns:
        HQQLinear: 量子化されたLinear module
    """
    # Weightの形状から入出力の特徴量を取得
    out_features, in_features = weight.shape
    
    # ダミーのLinear layerを作成
    dummy_linear = torch.nn.Linear(in_features, out_features, bias=(bias is not None))
    
    # Weightとbiasをコピー
    with torch.no_grad():
        dummy_linear.weight.copy_(weight)
        if bias is not None:
            dummy_linear.bias.copy_(bias)
    
    # 量子化設定
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size)
    
    # HQQLinearを作成
    hqq_layer = HQQLinear(
        dummy_linear,
        quant_config=quant_config,
        compute_dtype=compute_dtype,
        device=device,
        initialize=initialize,
        del_orig=False  # ダミーレイヤーは削除
    )

    
    gemlite_dtype = TORCH_TO_DTYPE[compute_dtype]
    gemlite_linear = GemLiteLinearTriton(nbits, 
                                        group_size=group_size, 
                                        in_features=in_features, 
                                        out_features=out_features, 
                                        input_dtype=gemlite_dtype, 
                                        output_dtype=gemlite_dtype)

    orig_shape   = (out_features, in_features)

    W_q           = hqq_layer.unpack(dtype=torch.uint8).view(orig_shape)
    scales        = hqq_layer.meta['scale']
    zeros         = hqq_layer.meta['zero']
    gemlite_linear.pack(W_q, scales, zeros, None)
    
    return gemlite_linear


# 使用例1: weightのみの場合
def example_usage_weight_only():
    # サンプルのweight tensor
    weight = torch.randn(512, 1024, dtype=torch.float32)
    
    # HQQ moduleを作成
    hqq_module = create_hqq_module_from_weight(weight)
    
    # テスト入力
    x = torch.randn(1, 1024, dtype=torch.float16).cuda()
    output = hqq_module(x)
    print(f"Output shape: {output.shape}")
    
    return hqq_module


# 使用例2: weightとbiasがある場合
def example_usage_with_bias():
    # サンプルのweight/bias tensors
    weight = torch.randn(512, 1024, dtype=torch.float32)
    bias = torch.randn(512, dtype=torch.float32)
    
    # カスタム設定でHQQ moduleを作成
    hqq_module = create_hqq_module_from_weight(
        weight=weight,
        bias=bias,
        nbits=8,  # 8ビット量子化
        group_size=128,  # より大きなグループサイズ
        compute_dtype=torch.bfloat16  # bfloat16を使用
    )
    
    return hqq_module


# 使用例3: 既存のLinear layerからweightを抽出して使用
def example_from_existing_linear():
    # 既存のLinear layer
    original_linear = torch.nn.Linear(1024, 512)
    
    # Weightを抽出
    weight = original_linear.weight.data
    bias = original_linear.bias.data if original_linear.bias is not None else None
    
    # HQQ moduleを作成
    hqq_module = create_hqq_module_from_weight(weight, bias)
    
    # デクォンタイズして確認
    W_r = hqq_module.dequantize()
    print(f"Original weight shape: {weight.shape}")
    print(f"Dequantized weight shape: {W_r.shape}")
    
    return hqq_module


# より高度な使用例: バッチ処理
def batch_quantize_weights(weights_dict: dict, **kwargs) -> dict:
    """
    複数のweightを一括で量子化
    
    Args:
        weights_dict: {layer_name: weight_tensor} の辞書
        **kwargs: create_hqq_module_from_weightに渡す追加引数
    
    Returns:
        dict: {layer_name: hqq_module} の辞書
    """
    hqq_modules = {}
    
    for name, weight in weights_dict.items():
        hqq_modules[name] = create_hqq_module_from_weight(weight, **kwargs)
        print(f"Quantized layer: {name}, shape: {weight.shape}")
    
    return hqq_modules


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
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
#torch._C._jit_set_autocast_mode(False)


MyStatic = torch.jit.script


from rwkvengine.rwkv6 import RWKV_6, fused_recurrent_rwkv6_torch
from rwkvengine.rwkv7 import RWKV_7
from rwkvengine.arwkv7 import ARWKV_7
from rwkvengine.prwkv7 import PRWKV_7
from rwkvengine.hrwkv7 import HRWKV_7, compute_qwen3_rope_cache

def pad_weight_tensor_to_256(weight_tensor):
    """重みテンソルを256の倍数にゼロパディング"""
    # 現在のサイズを取得
    original_shape = weight_tensor.shape
    
    # 各次元を256の倍数に切り上げ
    padded_shape = []
    for dim in original_shape:
        padded_dim = ((dim + 255) // 256) * 256
        padded_shape.append(padded_dim)
    
    # すでに256の倍数の場合はそのまま返す
    if list(original_shape) == padded_shape:
        return weight_tensor
    
    # 新しいゼロテンソルを作成
    padded_tensor = torch.zeros(
        padded_shape,
        dtype=weight_tensor.dtype,
        device=weight_tensor.device,
        requires_grad=weight_tensor.requires_grad
    )
    
    # 元のデータをコピー
    if len(original_shape) == 2:  # Linear層の重み
        padded_tensor[:original_shape[0], :original_shape[1]] = weight_tensor
    elif len(original_shape) == 4:  # Conv2d層の重み
        padded_tensor[:original_shape[0], :original_shape[1], 
                     :original_shape[2], :original_shape[3]] = weight_tensor
    else:  # その他の次元
        slices = tuple(slice(0, dim) for dim in original_shape)
        padded_tensor[slices] = weight_tensor
    
    return padded_tensor

class RWKV_x(nn.Module):

    def __init__(self,load_model: str,base_precision: str = 'int8',adapter_model:str = '', adapter_mode:str = '', adapter_scale:float=2.0,fully_fusedrecurrent:bool=True, tokenizer='',rope_theta=1000000.0,rms_norm_eps=1e-6,max_ctxlen=8192):

        #print('Helloworld RWKV v060 :) Initializing')
        print('RWKV-Infer RWKVCore Initializing')
        self.max_ctxlen = max_ctxlen

        super().__init__()
        self.transfer_stream = torch.cuda.Stream()

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        #GANBATTE CODE KAKOU
        self.bit8quant = False
        self.bit4quant = False
        self.bitfp8quant = False
        self.bitfp6quant = False

        self.fully_fusedrecurrent = fully_fusedrecurrent

        self.ExtremeCPUOffload = False

        self.debug = False

        self.eval()

        #Dummy
        self.ebits, self.mbits = 4, 3

        with torch.no_grad(): 


            emboncpu = False
            head8bit = False

            if base_precision == 'fp16':
                self.base_precision = torch.half
            elif base_precision == 'int8':
                print('This is experimental fused matmul mode. HOSHOUSHIMASENN. ')
                self.base_precision = torch.float16
                self.bit8quant = True
            elif base_precision == 'nf4':
                self.base_precision = torch.bfloat16
                if HAS_HQQ:
                    self.base_precision = torch.float16
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
            elif base_precision == 'fp5c':
                self.base_precision = torch.bfloat16
                self.bit8quant = False
                self.bit4quant = False
                self.bitfp8quant = False
                self.bitfp6quant = True
                self.bitfp5quant = True
                emboncpu = True
            elif base_precision == 'fp5x':
                self.base_precision = torch.bfloat16
                self.bit8quant = False
                self.bit4quant = False
                self.bitfp8quant = False
                self.bitfp6quant = True
                self.bitfp5quant = True
                emboncpu = True
                head8bit = True
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



            self.HRWKV_Mode = 0
            self.HRWKV_StartLayers = 0
            self.HRWKV_Generation=78 # hxa078, hxa079 supported

            self.HRWKV_Block_Mode = [] #0: RWKV, 1: Attention

            self.GQALayers = 0
            self.RWKVLayers = 0

            for i in range(n_layer):
                t = f'blocks.{i}.'
                Found = False
                for key in keys:
                    if t in key and 'q_proj' in key:
                        Found = True
                        self.HRWKV_Block_Mode.append([1,i,self.GQALayers])
                        self.GQALayers = self.GQALayers + 1
                        if self.HRWKV_Mode == 0:
                            self.HRWKV_Mode = 1
                        break
                        # self.HRWKV_Mode = 1
                        # self.HRWKV_StartLayers = i
                        # print(i)
                        # Found = True
                        # break
                # if Found == True:
                #     break
                if Found == False:
                    self.HRWKV_Block_Mode.append([0,i,self.RWKVLayers])
                    self.RWKVLayers = self.RWKVLayers + 1

            for key in keys:
                if 'k0' in key:
                    self.HRWKV_Generation=79
                    if self.HRWKV_Mode == 0:
                        self.HRWKV_Mode = 1
                        #self.HRWKV_StartLayers=n_layer
            

            if self.HRWKV_Mode:
                print('HRWKV-7 Mode. Hybrid RWKV Mode. hxa078r, hxa079r')

                
                # for mode in self.HRWKV_Block_Mode:
                #     if mode == 0:
                #         self.RWKVLayers = self.RWKVLayers + 1
                #     else:
                #         self.GQALayers = self.GQALayers + 1
                print(f'RWKVLayers = {self.RWKVLayers}')
                print(f'GQALayers = {self.GQALayers}')
                #exit()

                self.n_kv = 1

                for key in keys:
                    if 'k_proj' in key:
                        k_shape = z[key].shape
                        print(k_shape)
                        self.n_kv = k_shape[0]//self.head_size
                        print(self.n_kv)
                        break

                #exit()


            
                

            
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
            QuantListBNB = ['q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight','head.weight','ffn.down.weight','ffn.up.weight','ffn.gate.weight','gate_up.weight']
            QuantListFP8 = ['q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight','head.weight','ffn.down.weight','ffn.up.weight','ffn.gate.weight','gate_up.weight'] #, ,
            QuantListFP6 = ['q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','att.receptance.weight','att.key.weight','att.value.weight','att.gate.weight','att.output.weight','ffn.key.weight','ffn.receptance.weight','ffn.value.weight','ffn.down.weight','ffn.up.weight','ffn.gate.weight','gate_up.weight'] #, ,
    
            

            # FP8 Transformer Engine Quantize Mode 
            # or Int8 FusedMatmul Engine
            if self.bitfp8quant == True or self.bit8quant:
                
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
                                print(f'before size {z[k].shape}')
                                z[k] = pad_weight_tensor_to_256(z[k])
                                print(f'after size {z[k].shape}')
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
                #emboncpu = False
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

                    if 'head' in k and head8bit:
                        z['head.weight'], z['head.weight.qstate'] = quantize_weight(z['head.weight'].to(device='cuda',dtype = torch.float16).t()) # quant to int8
                        #z['head.weight'] = z['head.weight'].to(dtype=torch.float32).cpu().t()
                        QuantKeyFound = True 
                        gc.collect()
                        torch.cuda.empty_cache() 
                        self.head8bit = True


                    for QuantKey in QuantListFP6:
                        if k.endswith(QuantKey):
                            if self.bitfp5quant:
                                print(f'Quant {k} to FP5 shape = {z[k].shape}' )
                            else:
                                print(f'Quant {k} to FP6 shape = {z[k].shape}' )
                            QuantKeyFound = True
                            z[k] = z[k].to(device='cuda',dtype=torch.float16)#.t() 
                            print(f'before size {z[k].shape}')
                            z[k] = pad_weight_tensor_to_256(z[k])
                            print(f'after size {z[k].shape}')

                          
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

            # BNB 4bit
            elif self.bit4quant == True:
                #emboncpu = False

                self.ebits, self.mbits = -4, -4
                if HAS_HQQ:
                    self.ebits, self.mbits = -44, -44

                count = 0
                for k in keys:
                    count = count + 1
                   
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

                    if 'head' in k and head8bit:
                        z['head.weight'], z['head.weight.qstate'] = quantize_weight(z['head.weight'].to(device='cuda',dtype = torch.float16).t()) # quant to int8
                        #z['head.weight'] = z['head.weight'].to(dtype=torch.float32).cpu().t()
                        QuantKeyFound = True 
                        gc.collect()
                        torch.cuda.empty_cache() 
                        self.head8bit = True


                    for QuantKey in QuantListBNB:
                        if k.endswith(QuantKey):
                            if HAS_HQQ:
                                print(f'Quant {k} to HQQ 4bit shape = {z[k].shape}' )
                            else:
                                print(f'Quant {k} to BNB NF4 shape = {z[k].shape}' )
                            QuantKeyFound = True
                            z[k] = z[k].to(device='cuda',dtype=torch.bfloat16)
                            print(f'before size {z[k].shape}')
                            z[k] = pad_weight_tensor_to_256(z[k])
                            print(f'after size {z[k].shape}')

                            if HAS_HQQ:
                                z[k], z[k+'.qstate'] = create_hqq_module_from_weight((z[k])), None
                            else:
                                z[k], z[k+'.qstate'] = bnb.functional.quantize_nf4((z[k]))


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
                    #print(f'{key} {z[key].shape} {z[key].dtype}')
                    if ('.bone' in key or '.lora' in key) and 'expert' not in key:
                        z[key] = None
                        print(f'{key} deleted')

            
        
            gc.collect()
            torch.cuda.empty_cache()

            if self.HRWKV_Generation == 79:
                self.cos, self.sin, _ = compute_qwen3_rope_cache(self.max_ctxlen,self.head_size,self.device,torch.float32,self.rope_theta)


    def new_state(self, B, max_token=4096):
         if self.RWKVMode == 6:
            return BlockStateList.create(
                    self.n_layer,
                    B,
                    self.n_embd, 
                    self.n_head,
                    self.device, self.dtype
                )
         elif self.RWKVMode == 7 and self.ARWKVMode == 1 and self.HRWKV_Mode == 1:
             return BlockStateList.hx078_create(self.RWKVLayers,self.GQALayers,
                                                B,
                                                self.n_embd,
                                                self.n_kv,
                                                self.head_size,
                                                max_token,
                                                self.device,
                                                self.dtype                                                
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
    


    
    
    
    

    def forward(self, idx, last_shift_states , last_wkv_states,kv_cache=None,pos_cache=None,full_output=False, one_mode = False, KernelMode = 0,time_offset_state:torch.Tensor=None):
        if self.RWKVMode == 6:
            return RWKV_6.x060_forward(self,idx,last_shift_states,last_wkv_states)
        elif self.RWKVMode == 7 and self.ARWKVMode == 0:
            return RWKV_7.x070_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=full_output,time_offset_state=time_offset_state)
        elif self.RWKVMode == 7 and self.HRWKV_Mode == 1:
            if self.HRWKV_Generation == 79:
                return HRWKV_7.hxa079r_forward(self,idx,last_wkv_states,kv_cache,pos_cache,full_output)
            else:
                return HRWKV_7.hxa078r_forward(self,idx,last_wkv_states,kv_cache,pos_cache,full_output)
        elif self.RWKVMode == 7 and self.ARWKVMode == 1 and self.TokenshiftMode == 0:
            #print('ARWKV')
            return ARWKV_7.ax070_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=full_output )
        
        elif self.RWKVMode == 7 and self.ARWKVMode == 1 and self.TokenshiftMode == 1:
            #print('PRWKV')
            return PRWKV_7.PRWKV7_forward(self,idx,last_shift_states,last_wkv_states,one_mode=one_mode,KernelMode=KernelMode,full_output=full_output,time_offset_state=time_offset_state)
    













    