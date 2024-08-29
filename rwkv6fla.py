#FLA Inference
#Thank you harrisonvanderbyl for the code :)
#cuda mm8 kernels from ChatRWKV@BlinkDL

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
from torch.utils.cpp_extension import load

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_flush_denormal(True)

current_path = os.path.dirname(os.path.abspath(__file__))

from torch.utils.cpp_extension import load

try:
    load(
        name=f"wkv_cuda",
        sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu", f"{current_path}/cuda/gemm_fp16_cublas.cpp"],
        verbose=True,
        extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
        extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
        is_python_module=False)
    DISABLE_CUBLAS_GEMM = False
except:
    print("Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow.")
    load(
        name=f"wkv_cuda",
        sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
        verbose=True,
        extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
        extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
        is_python_module=False)
    DISABLE_CUBLAS_GEMM = True

MyStatic = torch.jit.script
MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyCompile = torch.compile()

#DISABLE_CUBLAS_GEMM = True

if DISABLE_CUBLAS_GEMM == False:
    @MyStatic
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        if output_dtype is None:
            output_dtype = a.dtype
        if a.dtype == b.dtype == torch.float16 and a.device.type == 'cuda' and 0:

            a_2d = a.reshape((a.shape[0] * a.shape[1]),a.shape[2]).contiguous()#x.view(-1, x.shape[-1])

            c_2d = torch.empty((a.shape[0] * a.shape[1],  b.shape[-1]), dtype=output_dtype, device=a.device)
    
            torch.ops.rwkv.gemm_fp16_cublas(a_2d, b, c_2d)

            c = c_2d.reshape(a.shape[0], a.shape[1], b.shape[-1]).to(dtype=a.dtype)

            return c
        else:
            return (a @ b.t()).to(output_dtype)

else:
    @MyStatic
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        #print(f'a = {a.shape} b = {b.shape}')
        return (a @ b.t()).to(output_dtype)


@MyStatic
def cuda_mm8_batch_one(Z:int,B: int, N: int, M: int, x, w, mx, rx, my, ry):
    if x.dtype != torch.float16:
        x = x.to(dtype=torch.float16)

    B = B * Z
    a_2d = x.reshape((x.shape[0] * x.shape[1]),x.shape[2]).contiguous()#x.view(-1, x.shape[-1])

    #a_2d = x.view(-1, x.shape[-1])

    y = torch.empty((x.shape[0] * x.shape[1],  w.shape[-1]), dtype=torch.float32, device=x.device)

    torch.ops.rwkv.mm8_seq(B, N, M, a_2d, w, mx, rx, my, ry, y)

    y = y.reshape(x.shape[0], x.shape[1], w.shape[-1]).to(dtype=x.dtype)
    #y = y.view(x.shape[0], x.shape[1], w.shape[-1]).to(dtype=x.dtype)


    return y


@MyStatic
def torch_mm8_seq(x, w, mx, rx, my, ry):
    if x.dtype != torch.float16:
        x = x.to(dtype=torch.float16)
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

@MyStatic
def mm8(x, w, mx, rx, my, ry):
    if w.device.type == 'cuda' and x.dtype == torch.float16 and x.shape[1] == 1:
        Z, B, N, M = x.shape[0], x.shape[1], w.shape[0], w.shape[1]
        return cuda_mm8_batch_one(Z, B, N, M, x, w, mx, rx, my, ry)
    else:
        return torch_mm8_seq(x, w, mx, rx, my, ry)

@MyStatic
def matmul(a, b, mx: Optional[torch.Tensor]=None, rx: Optional[torch.Tensor]=None, my: Optional[torch.Tensor]=None, ry: Optional[torch.Tensor]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = a.dtype
    if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        assert a.dtype == b.dtype
        return matmul_float(a, b, output_dtype=output_dtype)
    elif b.dtype == torch.uint8:
        assert mx is not None
        assert rx is not None
        assert my is not None
        assert ry is not None
        return mm8(a, b, mx, rx, my, ry).to(output_dtype)
    else:
        raise ValueError("Unsupported dtype")

@torch.jit.ignore
class QuantLinear(nn.Module): # inspired by RWKV-PEFT @JL-er Thanks! 
    def __init__(self, in_features: int, out_features: int, bias: bool, precision=torch.float16 ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.is_quant = False
        self.grad = None
        self.precision = precision

        self.mx = None
        self.my = None
        self.rx = None
        self.ry = None
    def shape(self):
        return self.weight.data.shape

    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        #self.dummy_tensor = nn.Parameter(torch.zeros(1))
        if self.quant_type=='4bit':
            self.weight.data, self.qstate= bnb.functional.quantize_4bit((self.weight.data.to(dtype=self.precision)).to('cuda'))
        elif self.quant_type=='nf4':
            self.weight.data, self.qstate= bnb.functional.quantize_nf4((self.weight.data.to(dtype=self.precision)).to('cuda'))
        elif self.quant_type=='fp4':
            self.weight.data, self.qstate= bnb.functional.quantize_fp4((self.weight.data.to(dtype=self.precision)).to('cuda'))
        elif self.quant_type=='int8':
            self.weight.data, self.qstate= bnb.functional.quantize((self.weight.data.to(dtype=self.precision)).to('cuda'))
    def forward(self, x):

        if self.is_quant:
            if self.quant_type=='4bit':
                return F.linear(x.to(self.precision), bnb.functional.dequantize_4bit(self.weight.data,quant_state=self.qstate).to(self.precision)) 
            elif self.quant_type=='nf4':
                return F.linear(x.to(self.precision), bnb.functional.dequantize_nf4(self.weight.data,quant_state=self.qstate).to(self.precision))
            elif self.quant_type=='fp4':
                return F.linear(x.to(self.precision), bnb.functional.dequantize_fp4(self.weight.data,quant_state=self.qstate).to(self.precision))
            elif self.quant_type=='int8':
                return F.linear(x.to(self.precision), bnb.functional.dequantize(self.weight.data,state=self.qstate).to(self.precision))
        else:
            #print('unquant forward')
            return F.linear(x, self.weight)
@functools.wraps(QuantLinear)
def HybridLinear(*args, **kwargs):
        #return nn.Linear(*args, **kwargs)
        return QuantLinear(*args, **kwargs)



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



class PIPELINE():
    def __init__(self, model='dummy'):
        self.model = model
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER_MOSE
        self.tokenizer = TRIE_TOKENIZER_MOSE(os.path.dirname(os.path.abspath(__file__)) + '/rwkv/rwkv_vocab_v20230424.txt')        

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        if 'Tokenizer' in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)
    
    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        if temperature == 0:
            temperature = 1.0
            top_p = 0
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
    

    
    def sample_logits_mose(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        if temperature == 0:
            temperature = 1.0
            top_p = 0

        probs = F.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff_value = sorted_probs[cutoff_index]
        probs[probs < cutoff_value] = 0
        if top_k > 0 and top_k < len(probs):
            probs[sorted_indices[top_k:]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / torch.sum(probs)
        out = torch.multinomial(probs, num_samples=1)[0]

        return int(out)
    

    def sample_logits_mose2(self,logits, temperature=1.0, top_p=0.85, top_k=0):

        if temperature == 0:
            temperature = 1.0
            top_p = 0

        probs = F.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff_value = sorted_probs[cutoff_index]
        probs = torch.where(probs < cutoff_value, torch.tensor(0.0, device=probs.device), probs)
        if top_k > 0 and top_k < len(probs):
            probs[sorted_indices[top_k:]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / torch.sum(probs)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
    @MyStatic
    def sample_logits_mose2_optimized(logits, temperature:float=1.0, top_p:float=0.85, top_k:int=0):
        if temperature == 0:
            return int(torch.argmax(logits))

        probs = F.softmax(logits.float(), dim=-1)

        if top_k > 0 and top_k < len(probs):
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
        else:
            top_k_probs = probs

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = torch.searchsorted(cumulative_probs, top_p)
            probs_above_cutoff = sorted_probs[:cutoff_index + 1]
            indices_above_cutoff = sorted_indices[:cutoff_index + 1]
            probs.zero_()
            probs.scatter_(0, indices_above_cutoff, probs_above_cutoff)

        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum()

        return int(torch.multinomial(probs, num_samples=1)[0])
    
    def improved_nucleus_sampling(self, logits, temperature=1.0, top_p=0.9):
       p = top_p
       probs = F.softmax(logits, dim=-1)
       sorted_probs, sorted_indices = torch.sort(probs, descending=True)
       cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
       sorted_indices_to_remove = cumulative_probs > p
       sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
       sorted_indices_to_remove[0] = False
       indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
       probs.masked_fill_(indices_to_remove, 0.0)
       return int(torch.multinomial(probs, num_samples=1)[0])



class RWKV6_ChannelMix(torch.nn.Module):
    
    def __init__(self, layer_id, n_layer, n_embd, dim_ffn,precision=torch.bfloat16):
        super().__init__()
        # self.layer_id = layer_id
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.precision = precision

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_maa_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = HybridLinear(n_embd, dim_ffn, bias=False)
        self.receptance = HybridLinear(n_embd, n_embd, bias=False)
        self.value = HybridLinear(dim_ffn, n_embd, bias=False)
        self.lastlayer = layer_id == n_layer - 1
    # forwarding channel mix given the model weights and the input tokens and states.
    #
    # Given:
    # - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
    # - Last shift states of the various batches [batch_size, state_size]
    #
    # Returns a pair 
    # - of output embedding of shape [batch_size, seq_len, embedding_size]
    # - and the last output state of shape [batch_size, state_size]
    # @torch.compile(backend="eager")
    
    def forward(self, x, last_state: torch.Tensor):
 
    
        xx = torch.concat((last_state, x[:, :-1]),
                          dim=1).to(dtype=self.precision)
        last_state[:] = x[:, -1:]
        
        # if(self.lastlayer):
        #     x = x[:,-1:]
        #     xx = xx[:,-1:]
        
        xk = xx * self.time_maa_k + x * (1 - self.time_maa_k)
        xr = xx * self.time_maa_r + x * (1 - self.time_maa_r)
        #kv = self.value( torch.relu( self.key(xk.to(dtype=self.precision)) ) ** 2 )
        if self.value.is_quant and self.key.is_quant:
            kv = self.value( torch.relu( self.key(xk.to(dtype=self.precision)) ) ** 2 )
        elif self.key.mx is None and self.value.mx is None:
            kv = (torch.relu(xk.to(dtype=self.precision) @ self.key.weight.t()) ** 2) @ self.value.weight.t()
        else:
            kv = matmul(torch.relu(matmul(xk.to(dtype=self.precision),self.key.weight,
                                      mx=self.key.mx,
                                      my=self.key.my,
                                      rx=self.key.rx,
                                      ry=self.key.ry,
                                      )) ** 2,self.value.weight,
                                      mx=self.value.mx,
                                      my=self.value.my,
                                      rx=self.value.rx,
                                      ry=self.value.ry)
        if self.receptance.is_quant:
            return (torch.sigmoid(self.receptance(xr.to(dtype=self.precision))) * kv), last_state
        elif self.receptance.mx is None:
            return torch.sigmoid(
                                  xr.to(dtype=self.precision) @ self.receptance.weight.t()  
                                ) * kv, last_state
    
        else:
            return (torch.sigmoid(matmul(xr.to(dtype=self.precision),self.receptance.weight,
                                     mx=self.receptance.mx,
                                      my=self.receptance.my,
                                      rx=self.receptance.rx,
                                      ry=self.receptance.ry,
                                     )) * kv), last_state
        #return (torch.sigmoid(self.receptance(xr.to(dtype=self.precision))) * kv), last_state
    


class Block6(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, dim_ffn,precision):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.precision = precision

        self.att = RWKV6_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att,precision=precision)
        self.ffn = RWKV6_ChannelMix(layer_id, n_layer, n_embd, dim_ffn,precision=precision)
        self.time_debug = False
        
    
        # Setup droupout at block level

    def forward(self, x, time_mix_shift, channel_mix_state, time_mix_state):

        if self.time_debug:
            start_time1 = time.time()

        att_out, time_mix_shift, time_mix_state = self.att(
                self.ln1(x.to(dtype=torch.bfloat16)),
                #self.ln1(x),
                time_mix_shift,
                time_mix_state
            )
        
        if self.time_debug:
            start_time2 = time.time()

        
            # Handle without dropout
        x = x + att_out
        ffn_out, channel_mix_state = self.ffn(
            self.ln2(x.to(dtype=torch.bfloat16)),
            #self.ln2(x),
            channel_mix_state,
        )
        x = x + ffn_out

        if self.time_debug:
            start_time3 = time.time()
            time_ffn = start_time3 - start_time2
            time_att = start_time2 - start_time1
            print(f'time_ffn = {time_ffn*1000*1000:0.3f}us')    
            print(f'time_att = {time_att*1000*1000:0.3f}us')  

        
        return x, time_mix_shift, channel_mix_state, time_mix_state

    



# RWKV TimeMix module
#@MyModule
class RWKV6_TimeMix(torch.nn.Module):
    #chunk_len:int = 128, precision:int = 64
    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, chunk_len:int = 1, precision=torch.bfloat16):
        super().__init__()
        
                
        self.layer_id = layer_id
        self.time_debug = False

        self.head_size = head_size
        self.precision = precision
        self.n_head = n_head
        head_size_divisor = 8
        assert dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32#n_head # generate TIME_MIX for w,k,v,r,g
            if n_embd==4096:
                D_MIX_LORA = D_MIX_LORA*2
            self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,dim_att))

            #D_DECAY_LORA = 2*n_head
            D_DECAY_LORA = 64
            if n_embd==4096:
                D_DECAY_LORA = D_DECAY_LORA*2
            self.time_decay_w1 = nn.Parameter(torch.zeros(n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(dim_att)
            for n in range(dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(torch.zeros(n_head, head_size))
            print(self.time_faaaa.shape)

        #self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance =HybridLinear(n_embd, dim_att, bias=False)
        self.key =HybridLinear(n_embd, dim_att, bias=False)
        self.value =HybridLinear(n_embd, dim_att, bias=False)

        self.output =HybridLinear(dim_att, n_embd, bias=False)

        self.gate =HybridLinear(n_embd, dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, dim_att, eps=(1e-5)*(head_size_divisor**2))
        self.ctx = 1024#saveBackDummy()

    #@torch.jit.script
    # def jit_func(self, x, last_state_shift):
    #     B, T, C = x.size()

    #     #print(f'last_state_shift = {last_state_shift.shape}')

    #     output = torch.concat((last_state_shift, x[:, :-1]), dim=1)
    #     last_state_shift[:] = x[:,-1:]

    #     xx = output - x

    #     xxx = x + xx * self.time_maa_x
    #     xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
    #     xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
    #     mw, mk, mv, mr, mg = xxx.unbind(dim=0)

    #     xw = x + xx * (self.time_maa_w + mw)
    #     xk = x + xx * (self.time_maa_k + mk)
    #     xv = x + xx * (self.time_maa_v + mv)
    #     xr = x + xx * (self.time_maa_r + mr)
    #     xg = x + xx * (self.time_maa_g + mg)

    #     r = self.receptance(xr)
    #     k = self.key(xk)
    #     v = self.value(xv)
    #     g = torch.nn.functional.silu(self.gate(xg))

    #     ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
    #     w = self.time_decay + ww
    #     w = w.exp().neg()
    #     return r, k, v, g, w
    
    @MyStatic  
    def jit_func_parts(x, last_state_shift,
                       time_maa_x,
                       time_maa_w1,
                       time_maa_w2,
                       time_maa_w,
                       time_maa_k,
                       time_maa_v,
                       time_maa_r,
                       time_maa_g,
                       time_decay_w1,
                       time_decay_w2,
                       time_decay
                       ):
        B, T, C = x.size()

        #print(f'last_state_shift = {last_state_shift.shape}')

        output = torch.concat((last_state_shift, x[:, :-1]), dim=1)
        last_state_shift[:] = x[:,-1:]

        xx = output - x

        xxx = x + xx * time_maa_x
        xxx = torch.tanh(xxx @ time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (time_maa_w + mw)
        xk = x + xx * (time_maa_k + mk)
        xv = x + xx * (time_maa_v + mv)
        xr = x + xx * (time_maa_r + mr)
        xg = x + xx * (time_maa_g + mg)

        ww = torch.tanh(xw @ time_decay_w1) @ time_decay_w2

        w = time_decay + ww
        w = w.exp().neg()

        return xw,xk,xv,xr,xg,ww,w
    
    def jit_func_parts2(self,xw,xk,xv,xr,xg):

        #print(f'xr shape = {xr.shape}')

        #r = self.receptance(xr.to(dtype=self.precision))#.to(dtype=torch.bfloat16)
        if self.receptance.is_quant:
            r = self.receptance(xr.to(dtype=self.precision))
        elif self.receptance.mx is None:
            r = (xr.to(dtype=self.precision) @ self.receptance.weight.t())
        else:
            r = matmul(xr.to(dtype=self.precision),
                    self.receptance.weight,
                    mx=self.receptance.mx,
                    my=self.receptance.my,
                    rx=self.receptance.rx,
                    ry=self.receptance.ry,
                    )#.to(dtype=torch.bfloat16)
        #r = matmul(xr,self.receptance.weight)
        #print(r)
        #k = self.key(xk.to(dtype=self.precision))
        if self.key.is_quant:
            k = self.key(xk.to(dtype=self.precision))
        elif self.key.mx is None:
            k = (xk.to(dtype=self.precision) @ self.key.weight.t())
        else:
            k = matmul(xk.to(dtype=self.precision),self.key.weight,
                   mx=self.key.mx,
                    my=self.key.my,
                    rx=self.key.rx,
                    ry=self.key.ry,
                   )
        #v = self.value(xv.to(dtype=self.precision))
        if self.value.is_quant:
            v = self.value(xv.to(dtype=self.precision))
        elif self.value.mx is None:
            v = (xv.to(dtype=self.precision) @ self.value.weight.t())
        else:
            v = matmul(xv.to(dtype=self.precision),self.value.weight,
                   mx=self.value.mx,
                    my=self.value.my,
                    rx=self.value.rx,
                    ry=self.value.ry,)
        #g = torch.nn.functional.silu(self.gate(xg.to(dtype=self.precision)))
        if self.gate.is_quant:
            g = torch.nn.functional.silu(self.gate(xg.to(dtype=self.precision)))
        elif self.gate.mx is None:
            g = torch.nn.functional.silu((xg.to(dtype=self.precision) @ self.gate.weight.t()))
        else:
            g = torch.nn.functional.silu(matmul(xg.to(dtype=self.precision),self.gate.weight,
                                            mx=self.gate.mx,
                                            my=self.gate.my,
                                            rx=self.gate.rx,
                                            ry=self.gate.ry,))#)self.gate(xg.to(dtype=self.precision)))
        return r, k, v, g
    
    
   # @torch.jit.scrip
    #@MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        #x = self.output(x * g)
        #x = matmul(x * g, self.output.weight.t())
        if self.gate.is_quant:
            x = self.output(x * g)
        elif self.output.mx is None:
            x = (x * g).to(dtype=self.precision) @ self.output.weight.t()
        else:
            x = matmul((x * g).to(dtype=self.precision),self.output.weight,
                   mx=self.output.mx,
                    my=self.output.my,
                    rx=self.output.rx,
                    ry=self.output.ry,)
        return x
 

    def forward(self, x, last_state_shift, last_state_wkv):
        #print("x shape:", x.shape)
        B, T, C = x.size()
        H = self.n_head
        if self.time_debug:
            start_time1 = time.time()

        xw,xk,xv,xr,xg,ww,w  = self.jit_func_parts(x=x, last_state_shift=last_state_shift,
                       time_maa_x=self.time_maa_x,
                       time_maa_w1=self.time_maa_w1,
                       time_maa_w2=self.time_maa_w2,
                       time_maa_w=self.time_maa_w,
                       time_maa_k=self.time_maa_k,
                       time_maa_v=self.time_maa_v,
                       time_maa_r=self.time_maa_r,
                       time_maa_g=self.time_maa_g,
                       time_decay_w1=self.time_decay_w1,
                       time_decay_w2=self.time_decay_w2,
                       time_decay=self.time_decay
                       )
        
        if self.time_debug:
            start_time2 = time.time()
        
        r, k, v, g = self.jit_func_parts2(xw,xk,xv,xr,xg)

        r= r.to(dtype=torch.bfloat16)
        k= k.to(dtype=torch.bfloat16)
        v= v.to(dtype=torch.bfloat16)
        g= g.to(dtype=torch.bfloat16)

        if self.time_debug:
            start_time3 = time.time()
        

        if T > 1:
            #print(f'T = {T}')
            x,last_state_wkv[:] = ChunkRWKV6Function.forward(self.ctx,
                r.view(B,T,H,-1).transpose(1,2),
                k.view(B,T,H,-1).transpose(1,2),
                v.view(B,T,H,-1).transpose(1,2),
                w.view(B,T,H,-1).transpose(1,2),
                self.time_faaaa.view(H,-1),1.0,
                last_state_wkv,True,
                0)
            x =x.transpose(1,2)
        else:
            x, last_state_wkv = fused_recurrent_rwkv6(
                r.view(B,T,H,-1).transpose(1,2),
                k.view(B,T,H,-1).transpose(1,2),
                v.view(B,T,H,-1).transpose(1,2),
                w.view(B,T,H,-1).transpose(1,2),
                self.time_faaaa.view(H,-1),
                1.0,
                last_state_wkv,True, 0)

        x = x.reshape(B,T,C)
        if self.time_debug:
            start_time4 = time.time()
            time_fla = start_time4 - start_time3
            time_jitfunc_parts2 = start_time3 - start_time2
            time_jitfunc_parts1 = start_time2 - start_time1

            print(f'time_jitfunc_parts1 = {time_jitfunc_parts1*1000*1000:0.3f}us')    
            print(f'time_jitfunc_parts2 = {time_jitfunc_parts2*1000*1000:0.3f}us')    
            print(f'time_fla = {time_fla*1000*1000:0.3f}us')   
            # print(f'time_block = {time_block*1000:0.3f}ms')   
            # print(f'time_lnout = {time_lnout*1000:0.3f}ms')   
            # print(f'time_head = {time_head*1000:0.3f}ms')   

        return self.jit_func_2(x, g), last_state_shift, last_state_wkv
 
    
### ---
# Core RWKV module
### ---
class RWKV6(nn.Module):

    def __init__(self,
                 # Model file path to load from
                 load_model: str,
                 # Model size settings, which we either
                 quantize: bool = False,

                 base_precision: str = 'int8'
             
                 ):

 
        # Setup the parent class
        super().__init__()

        self.time_debug = True
        self.bit8quant = False

        if base_precision == 'fp16':
            self.base_precision = torch.float16
        elif base_precision == 'int8':
            self.base_precision = torch.float16
            self.bit8quant = True
        else:
            self.base_precision = torch.bfloat16

           
        try:
            self.batches = micro_bsz
        except:
            self.batches = 1
            micro_bsz = 1

        try:
            grad_cp
        except:
            grad_cp = 0

        try:
            ctx_len
        except:
            ctx_len = 1024

        try:
            modelpath = load_model

        except:
            modelpath = None
        
        if modelpath:
            file = torch.load(modelpath,map_location="cpu", mmap=True)
            keys = list(file.keys())
            print("keys", keys)
            # remove _orig_mod from keys for compatibility with torch.compile
            newObj = {}
            for key in keys:
                if "_orig_mod." in key:
                    newKey = key.replace("_orig_mod.", "")
                    newObj[newKey] = file[key]
                else:
                    newObj[key] = file[key]
            file = newObj
            keys = list(file.keys())

            # detect model details
            vocab_size, n_embd = file["emb.weight"].shape
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
            # try:
            dim_ffn = file[f"blocks.0.ffn.value.weight"].shape[1]
            # except:
            #     dim_ffn = 2 * n_embd
            # model layers are model.2.x.yyy: find highest x
            
            try:
                n_head = file[f"blocks.0.att.time_faaaa"].shape[0]
                print("n_head", n_head)
            except:
                n_head = 64

           
        else:
            file = None

        try:
            dim_ffn = dim_ffn
        except:
            dim_ffn = int(3.5 * n_embd)
            
        
        self.n_embd = n_embd
        
        self.n_layer = n_layer
        
        self.n_head = n_head
        
        self.head_size = n_embd // n_head
        
        self.dim_ffn = dim_ffn
        
        self.emb = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.ModuleList([
        
            Block6(i, n_layer, n_embd, n_head, self.head_size, n_embd, dim_ffn,precision=self.base_precision) for i in range(n_layer)
        ])
        
        self.head = HybridLinear(n_embd, vocab_size,bias=False)
        
        file["ln_in.weight"] = file.pop("blocks.0.ln0.weight")
        file["ln_in.bias"] = file.pop("blocks.0.ln0.bias")
        
        self.ln_in = nn.LayerNorm(n_embd)

        self.ln_out = nn.LayerNorm(n_embd)

        self.requires_grad_(False)
        
        self.load_state_dict(file,strict=False)


        quant_layers = 0

        if quantize:
            quant_layers = 61


        for name, m in self.named_modules():
            ThroughFound = False
            if hasattr(m, "quant") and callable(getattr(m, "quant")):
                for i in range(self.n_layer):
                    if f'blocks.{i}.' in name:
                        if i < quant_layers:
                            m.quant('nf4')
                            print(f'Quant {name}')
                            ThroughFound = True



                
 
            

            for i in range(self.n_layer):
                if name == f'blocks.{i}':
                    ThroughFound = True

            if name == 'blocks' or name == '':# or name.endswith('.ffn') or name.endswith('.att'):
                ThroughFound = True

            #if '.att.' in name or '.ffn.' in name:
            #    ThroughFound = False

            if ThroughFound == False:

                #print('jikken')
                #if ('receptance' in name  or 'output' in name or 'key' in name or 'value' in name ) and ('.att' in name or '.ffn' in name ):
                #jikken
                if ((('receptance' in name or 'key' in name or 'value' in name or 'gate' in name or 'output' in name) and ('.att' in name or '.ffn' in name)) or 'head' == name) and self.bit8quant:
                #if ('receptance' in name ) and ('.att' in name ):
                    print(f'{name} is quant to int8')
                    m.weight.data = m.weight.data.t()
                    m.weight.data = m.weight.data.float()
                    if m.weight.data.shape[0] > m.weight.data.shape[1]:
                        print('ugyu')
                        m.my = torch.amin(m.weight.data, dim=1).unsqueeze(1)
                        m.weight.data = m.weight.data - m.my
                        m.mx = torch.amin(m.weight.data, dim=0)
                        m.weight.data = m.weight.data - m.mx
                        m.rx = torch.amax(m.weight.data, dim=0)
                        m.weight.data = m.weight.data / m.rx
                        m.ry = torch.amax(m.weight.data, dim=1).unsqueeze(1)
                        m.weight.data = m.weight.data / m.ry
                    else:
                        print('agi')
                        m.mx = torch.amin(m.weight.data, dim=0)
                        m.weight.data = m.weight.data - m.mx
                        m.my = torch.amin(m.weight.data, dim=1).unsqueeze(1)
                        m.weight.data = m.weight.data - m.my
                        m.rx = torch.amax(m.weight.data, dim=0)
                        m.weight.data = m.weight.data / m.rx
                        m.ry = torch.amax(m.weight.data, dim=1).unsqueeze(1)
                        m.weight.data = m.weight.data / m.ry
                    m.weight.data = torch.clip(torch.floor(m.weight.data * 256), min=0, max=255).to(dtype=torch.uint8)

                    m.my = m.my.to(dtype=torch.float16,device='cuda').contiguous()
                    m.mx = m.mx.to(dtype=torch.float16,device='cuda').contiguous()
                    m.rx = (m.rx/16).to(dtype=torch.float16,device='cuda').contiguous()
                    m.ry = (m.ry/16).to(dtype=torch.float16,device='cuda').contiguous()

                    m=m.to('cuda')#.contiguous()

                    m.weight.data = m.weight.data.contiguous()
                    


                    #self.base_precision
                else:
                    #if 'ln_in' in name or 'ln_out' in name or 'emb' in name or 'head' in name or 'ln1' in name or 'ln2' in name:
                    #    m=m.to('cuda',dtype=torch.bfloat16)
                    if (( 'receptance' in name or 'key' in name  or 'value' in name or  'value' in name or 'gate' in name or 'output' in name) and ('.att' in name or '.ffn' in name)) or 'head' == name:
                        m=m.to('cuda',dtype=self.base_precision)#.t()
                        m.weight.data = m.weight.data#.t().contiguous()
                        print(f'special mode {name}')
                    else:
                        m=m.to('cuda',dtype=torch.bfloat16)
                    print(f'Pass through to cuda:{name}')


            
            #elif name != 'blocks':
            #    m=m.to('cuda',dtype=torch.bfloat16)
            #    print(f'Pass through to cuda:{name}')


        del file
        gc.collect()
        
        
        self.eval()

        #self.to('cuda', dtype=torch.bfloat16)#.to(torch.float)
            

    def new_state(self, B):
         return BlockStateList.create(
                 self.n_layer, B, self.n_embd, 
                 self.n_head,# self.head_size,
                 self.emb.weight.device, self.emb.weight.dtype
             )
        
    #@TCompileBaseline
    def forward(self, idx: torch.Tensor, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor]):
        if self.time_debug:
            start_time = time.time()
        if idx.dtype != torch.int64:
            x = idx.requires_grad_(False).to(device=self.emb.weight.device)
        else:
            x = self.emb(idx)
        if self.time_debug:
            start_time1 = time.time()
        x = self.ln_in(x.to(dtype=torch.bfloat16))

        if self.time_debug:
            start_time2 = time.time()

        for i,b in enumerate(self.blocks):
            #print(i)
            x,last_shift_states[i*2],last_shift_states[i*2+1], last_wkv_states[i]  = b(x, last_shift_states[i*2],last_shift_states[i*2+1], last_wkv_states[i])

        if self.time_debug:
            start_time3 = time.time()

        x = self.ln_out(x.to(dtype=torch.bfloat16)).to(dtype=self.base_precision)
        if self.time_debug:
            start_time4 = time.time()




        if self.head.is_quant:
            x = self.head(x)
        else:
            #print(f'head device x={x.device} w = {self.head.weight.device} mx = {self.head.mx.device}')
            x = matmul(x.to(dtype=self.base_precision),self.head.weight,
                   mx=self.head.mx,
                    my=self.head.my,
                    rx=self.head.rx,
                    ry=self.head.ry,).to(dtype=torch.float32)



        if self.time_debug:
            start_time5 = time.time()

        if self.time_debug:
            time_head = start_time5 - start_time4
            time_lnout = start_time4 - start_time3
            time_block = start_time3 - start_time2
            time_lnin = start_time2 - start_time1
            time_emb = start_time1 - start_time

            # print(f'time_emb = {time_emb*1000:0.3f}ms')    
            # print(f'time_lnin = {time_lnin*1000:0.3f}ms')   
            # print(f'time_block = {time_block*1000:0.3f}ms')   
            # print(f'time_lnout = {time_lnout*1000:0.3f}ms')   
            # print(f'time_head = {time_head*1000:0.3f}ms')   

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

        for i in range(self.n_layer):
            #dd = strategy[i]
            dev = 'cpu'#dd.device
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
                                 device='cuda',
                                 dtype=torch.bfloat16)
        
        for i in range(self.n_layer):
            wkv_states[i] = model_current_statetuned[i*3 + 1]

        return wkv_states#.to(dtype=torch.float16)