import torch
from rwkvengine.matmulhell import quantize_weight
import torch
import time
import numpy as np
from torch.utils.cpp_extension import load

try:
    import torchao
    from torchao.dtypes.floatx import to_scaled_tc_floatx
    from torchao.ops import quant_llm_linear
    HAS_TORCHAO = True
except ImportError:
    print('torchao not found')
    HAS_TORCHAO = False
    bnb = None

# try:
#     import bitsandbytes as bnb
#     HAS_BITSANDBYTES = True
# except ImportError:
print('Bitsandbytes not found')
HAS_BITSANDBYTES = False
bnb = None
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
try:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
    HAS_HQQ = True
except ImportError:
    print('hqq not found')
    HAS_HQQ = False
    HQQLinear = None


# def load_turbo_kernel():
#     """ターボ版量子化カーネルをロード"""
#     return load(
#         name='quantized_linear_turbo',
#         sources=['rwkvengine/cpu_kernel_turbo/quantized_linear_turbo.cpp'],
#         extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx2', '-mfma', '-funroll-loops'],
#         extra_ldflags=['-fopenmp'],
#         verbose=False,
#         with_cuda=False
#     )

# int4cpukernel =  load_turbo_kernel()

# class HQQLinearKernel(torch.nn.Module):
#     """汎用HQQ互換線形層"""
    
#     def __init__(self, hqq_layer, kernel_type="turbo"):
#         super().__init__()
        
#         # カーネルタイプに応じてロード
#         if kernel_type == "original":
#             self.kernel = load_original_kernel()
#             self.forward_func = "forward"
#         elif kernel_type == "simple":
#             self.kernel = load_simple_kernel()
#             self.forward_func = "forward_simple"
#         elif kernel_type == "turbo":
#             self.kernel = load_turbo_kernel()
#             self.forward_func = "forward_turbo"
#         else:
#             raise ValueError(f"Unknown kernel type: {kernel_type}")
        
#         self.kernel_type = kernel_type
        
#         # HQQレイヤーから情報を抽出
#         self.in_features = hqq_layer.in_features
#         self.out_features = hqq_layer.out_features
#         self.nbits = hqq_layer.meta['nbits']
#         self.group_size = hqq_layer.meta['group_size']
#         self.compute_dtype = hqq_layer.compute_dtype
        
#         # メタデータを抽出
#         self.scale = hqq_layer.meta['scale'].cpu()
#         self.zero = hqq_layer.meta['zero'].cpu() if 'zero' in hqq_layer.meta else None
#         self.W_q = hqq_layer.W_q.cpu()
#         self.bias = hqq_layer.bias.cpu() if hasattr(hqq_layer, 'bias') and hqq_layer.bias is not None else None
#         self.packing = "4bit_u8"
#     @torch.jit.ignore    
#     def forward(self, x):
#         """汎用順伝播"""
#         original_device = x.device
#         x_cpu = x.cpu().to(torch.float32)
        
#         bias = self.bias if self.bias is not None else torch.empty(0)
#         zero = self.zero if self.zero is not None else torch.zeros_like(self.scale)
#         W_shape = [self.in_features, self.out_features]
        
#         # カーネル固有の処理
#         try:
#             if self.kernel_type == "original":
#                 # オリジナルカーネルはgroup_size=8のみサポート
#                 if self.group_size != 8:
#                     return torch.empty(0)
#                 output_cpu = getattr(self.kernel, self.forward_func)(
#                     x_cpu, bias, self.W_q, self.scale, zero,
#                     W_shape, self.group_size, self.nbits, 1, self.packing
#                 )
#             else:
#                 # シンプル・ターボはgroup_size=64固定
#                 if self.group_size != 64:
#                     return torch.empty(0)
#                 output_cpu = getattr(self.kernel, self.forward_func)(
#                     x_cpu, bias, self.W_q, self.scale, zero,
#                     W_shape, self.group_size, self.nbits, 1, self.packing
#                 )
            
#             if output_cpu.numel() == 0:
#                 return torch.empty(0)
                
#         except Exception as e:
#             return torch.empty(0)
        
#         return output_cpu.to(original_device)

#HAS_HQQ = False
#HQQLinear = None

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

    cpumode = False
    
    # ダミーのLinear layerを作成
    if out_features < 65536 and in_features < 65536:
        dummy_linear = torch.nn.Linear(in_features, out_features, bias=(bias is not None),device=device)
    else:
        print('on cpu ')
        cpumode = True
        dummy_linear = torch.nn.Linear(in_features, out_features, bias=(bias is not None))
    
    # Weightとbiasをコピー
    with torch.no_grad():
        dummy_linear.weight.copy_(weight)
        if bias is not None:
            dummy_linear.bias.copy_(bias)

    del weight
    del bias
    
    # 量子化設定
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size)
    
    # HQQLinearを作成
    if cpumode:
        hqq_layer = HQQLinear(
            dummy_linear,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
            initialize=initialize,
            del_orig=True  # ダミーレイヤーは削除
        )#.to(device=device)
    else:
        hqq_layer = HQQLinear(
            dummy_linear,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
            initialize=initialize,
            del_orig=True  # ダミーレイヤーは削除
        )

    
    gemlite_dtype = TORCH_TO_DTYPE[compute_dtype]
    gemlite_linear = GemLiteLinearTriton(nbits, 
                                        group_size=group_size, 
                                        in_features=in_features, 
                                        out_features=out_features, 
                                        input_dtype=gemlite_dtype, 
                                        output_dtype=gemlite_dtype)

    orig_shape   = (out_features, in_features)
    #gemlite_linear.cputensor = HQQLinearKernel(hqq_layer)
    W_q           = hqq_layer.unpack(dtype=torch.uint8).view(orig_shape).to(device=device)
    scales        = hqq_layer.meta['scale']
    zeros         = hqq_layer.meta['zero']
    gemlite_linear.pack(W_q, scales, zeros, None)
    
    return gemlite_linear#hqq_layer#gemlite_linear


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

def bf16_to_fp8(tensor):
    FP8_MAX = 448.0
    tensor=tensor.to(device='cuda')
    scale = FP8_MAX / torch.max(torch.abs(tensor)) + 1e-6
    tensor_scaled = tensor.float() * scale
    tensor_clipped = torch.clamp(tensor_scaled, -FP8_MAX, FP8_MAX)
    tensor_fp8 = tensor_clipped.to(dtype=torch.float8_e4m3fn ).contiguous()
    return tensor_fp8, scale.float()

QuantizationList = {}

QuantizationList['embedding'] = ['emb.weight','embed_tokens']
QuantizationList['head'] = ['head.weight','lm_head.weight']
QuantizationList['attention'] = ['.rkv_fused.weight','.qkv_fused.weight','q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','self_attn.receptance.weight','self_attn.key.weight','self_attn.value.weight','self_attn.gate.weight','self_attn.output.weight']
QuantizationList['mlp'] = ['.gateup.weight','.down_proj.weight','.up_proj.weight','.gate_proj.weight','mlp.gate_up_proj.weight','mlp.key.weight','mlp.value.weight','mlp.receptance.weight']

QuantizationModeList = ['hqq_int4','bnb_int4','ao_fp5','ao_fp6','op_int8']

def CleanQuantizationMode(quantname:str):
    if 'int4' == quantname or 'nf4' == quantname:
        return 'hqq_int4','hqq_int4', torch.float16,-44,-44,  -44,-44
    if 'bnb_int4' == quantname:
        return 'bnb_int4','bnb_int4', torch.float16,-4,-4,-4,-4
    
    if 'attn_int8_ffn_int4' in quantname:
        return 'op_int8','hqq_int4', torch.float16,-8,-8,-44,-44
    if 'attn_int8_ffn_fp5' in quantname:
        return 'op_int8','ao_fp5', torch.float16,-8,-8, 2,2
    if 'attn_int8_ffn_fp6' in quantname:
        return 'op_int8','ao_fp6', torch.float16,-8,-8, 3,2
    if 'int4' in quantname:
        return 'hqq_int4','hqq_int4', torch.float16,-44,-44,  -44,-44
    if 'int8' in quantname:
        return 'op_int8','op_int8' , torch.float16,-8,-8, -8,-8
    if 'fp5' in quantname:
        return 'ao_fp5','ao_fp5',torch.float16,2,2, 2,2
    if 'fp6' in quantname:
        return 'ao_fp6','ao_fp6',torch.float16,3,2, 3,2
    return 'None','None', torch.bfloat16, -999,-999,-999,-999

def Quant(z,TensorKey,QuantMode,device):
    print(QuantMode)
    if QuantMode == 'hqq_int4':
        print(f'{TensorKey} quant to hqq 4bit')
        z[TensorKey], z[TensorKey+'.qstate'] = create_hqq_module_from_weight((z[TensorKey].to(dtype=torch.float16))), None
        z[TensorKey]=z[TensorKey].to(device='cpu')
        return z
    if QuantMode == 'bnb_int4':
        print(f'{TensorKey} quant to bnb 4bit')
        z[TensorKey], z[TensorKey+'.qstate'] = bnb.functional.quantize_nf4((z[TensorKey]))
        return z
    if QuantMode == 'ao_fp5':
        print(f'{TensorKey} quant to torchao fp5')
        ebits, mbits = 2, 2
        z[TensorKey], z[TensorKey+'.qstate'] = to_scaled_tc_floatx(z[TensorKey].to(dtype=torch.float16), ebits, mbits)
        z[TensorKey]=z[TensorKey].to(device='cpu')
        return z
    if QuantMode == 'ao_fp6':
        print(f'{TensorKey} quant to torchao fp6')
        ebits, mbits = 3, 2
        z[TensorKey], z[TensorKey+'.qstate'] = to_scaled_tc_floatx(z[TensorKey].to(dtype=torch.float16), ebits, mbits)
        z[TensorKey]=z[TensorKey].to(device='cpu')
        return z
    if QuantMode == 'op_int8':
        print(f'{TensorKey} quant to op_int8 bit')
        z[TensorKey], z[TensorKey+'.qstate'] = quantize_weight(z[TensorKey].to(device=device,dtype = torch.float16).t())
        z[TensorKey]=z[TensorKey].to(device='cpu')
        return z
    else:
        z[TensorKey+'.qstate'] = None
    
    return z



def DoQuantizationIfPossible(z,TensorKey,attention_quant,mlp_quant,base_precision=torch.float16,device='cuda'):
    print(f'Tensor Target {TensorKey}')
    for QuantKey in QuantizationList['attention']:
        if TensorKey.endswith(QuantKey):
            z = Quant(z,TensorKey,attention_quant,device)
            return z
    for QuantKey in QuantizationList['mlp']:
        if TensorKey.endswith(QuantKey):
            z = Quant(z,TensorKey,mlp_quant,device)
            return z
    for QuantKey in QuantizationList['head']:
        if TensorKey.endswith(QuantKey):
            z = Quant(z,TensorKey,'op_int8',device)
            return z
    if 'norm' not in TensorKey  and 'emb' not in TensorKey and TensorKey.endswith('weight'):
        print(f'{TensorKey} linear weight passthrough')
        z[TensorKey], z[TensorKey+'.qstate'] = z[TensorKey].t().to(device=device,dtype = base_precision), torch.zeros(1)
    else:
        print(f'{TensorKey} passthrough')
        z[TensorKey] = z[TensorKey].to(device=device,dtype = base_precision)
    
    return z
    
    

from rwkvengine.matmulhell import fused_dequant_gemm
import torchao
from torchao.dtypes.floatx import to_scaled_tc_floatx
from torchao.ops import quant_llm_linear

@torch.compiler.disable
def bnb_nf4_matmul(x,weight,weight_state):
    batch, T, hiddensize = x.shape
    if T == 1 and batch == 1:
        x = x.view(batch,hiddensize)
        #print(f'trying {x.shape}' )
        o = bnb.functional.gemv_4bit(A=x,
                                    B=weight,
                                    state=weight_state,
                                    #transposed_A = True,
                                   # transposed_B = True
                                    )
        #print(o.shape)
        o = o.view(batch,T,-1)
        return o

    else:
        w = bnb.functional.dequantize_nf4(weight,quant_state=weight_state).to(torch.bfloat16)
        return x @ w.t()

#@torch.compile
def fp8_matmul(x,weight,weight_state):
    xg = x
    b = weight
    dtype = x.dtype
    if len(xg.shape) == 2:   
        S0=xg.shape[0]
        if xg.dtype != torch.float8_e4m3fn:
            xscale = 448.0 / torch.max(torch.abs(xg)) + 1e-6
            #xg = xg.float() * xscale
            xg = xg.to(dtype=torch.float32) * xscale
            xg = torch.clamp(xg, -448.0, 448.0).to(dtype=torch.float8_e4m3fn)#.contiguous()
        else:
            xscale = torch.tensor(1.0, device='cuda')

        x = torch._scaled_mm(
            xg.view(S0,xg.shape[1]).to(torch.float8_e4m3fn),#,.contiguous(),
            b.t(),
            bias=None,
            out_dtype=dtype,
            scale_a=1.0 / xscale.to(dtype=torch.float32),
            scale_b=1.0 / weight_state,
            use_fast_accum = True
        )
        return x.view(S0, -1)
    else:

        S0=xg.shape[0]
        S1=xg.shape[1]
        
        if xg.dtype != torch.float8_e4m3fn:
            xscale = 448.0 / torch.max(torch.abs(xg)) + 1e-6
            xg = xg.to(dtype=torch.float32) * xscale
            xg = torch.clamp(xg, -448.0, 448.0).to(dtype=torch.float8_e4m3fn).contiguous()
        else:
            xscale = torch.tensor(1.0, device='cuda')
        
        x = torch._scaled_mm(
            xg.view(S0*S1,xg.shape[2]).to(torch.float8_e4m3fn),
            b.t(),
            bias=None,
            out_dtype=dtype,
            scale_a=1.0 / xscale.to(dtype=torch.float32),
            scale_b=1.0 / weight_state,
            use_fast_accum = True
        )
        return x.view(S0, S1, -1)
    
@torch.compile()
def fpx_matmul(x,weight,weight_state,ebits:int=3,mbits:int=2):
    if ebits == -44 and mbits == -44: #hqq
        need_view = False
        if len(x.shape) == 3: 
            S0=x.shape[0]
            S1=x.shape[1]
            dtype = x.dtype
            x = x.to(dtype=torch.float16).view(-1,x.shape[2])  
            need_view = True
        else:
            dtype = x.dtype
            x = x.to(dtype=torch.float16)
            
        #out = weight.forward_manual(x, matmul_type="GEMM").view(S0,S1,-1).to(dtype=dtype)

        if weight.W_q.device == torch.device('cpu'):
          
            weight = weight.to(device=x.device)
            out = weight(x).to(dtype=dtype)
            weight = weight.to(device='cpu')
        else:
        #     #print('gpu')
        #out = weight(x).view(S0,S1,-1).to(dtype=dtype)
            #out = weight(x*0.5).to(dtype=dtype)*2.0
            out = weight.forward_manual(x, matmul_type="GEMM").to(dtype=dtype)
        if need_view:
            return out.view(S0,S1,-1)
        return out

    elif weight.dtype == torch.uint8:
        S0=x.shape[0]
        S1=x.shape[1]
        dtype = x.dtype
        x = x.to(dtype=torch.float16).view(-1,x.shape[2])  
        d = quant_llm_linear(ebits, mbits, x, weight, weight_state).view(S0,S1,-1).to(dtype=dtype)# * 2.0
        return d
    elif weight.dtype == torch.int8:
        return fused_dequant_gemm(x,weight,weight_state)
    elif weight.dtype == torch.float8_e4m3fn: 
        return fp8_matmul(x,weight,weight_state)
    else:
        if weight.device == torch.device('cpu'):
            xdtype = x.dtype
            xdevice = x.device
            return (x.to(device=torch.device('cpu'),dtype=weight.dtype) @ weight).to(device=xdevice,dtype=xdtype)
            #return x @ weight.to(device=x.device).t()
        else:
            return x @ weight.t()