#FLA Inference
#Thank you harrisonvanderbyl for the code :)

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

class QuantLinear(nn.Module): # inspired by RWKV-PEFT @JL-er Thanks! 
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.is_quant = False
        self.grad = None

    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        #self.dummy_tensor = nn.Parameter(torch.zeros(1))
        if self.quant_type=='4bit':
            self.weight.data, self.qstate= bnb.functional.quantize_4bit((self.weight.data).to('cuda'))
        elif self.quant_type=='nf4':
            self.weight.data, self.qstate= bnb.functional.quantize_nf4((self.weight.data).to('cuda'))
        elif self.quant_type=='fp4':
            self.weight.data, self.qstate= bnb.functional.quantize_fp4((self.weight.data).to('cuda'))
        elif self.quant_type=='int8':
            self.weight.data, self.qstate= bnb.functional.quantize((self.weight.data).to('cuda'))
    def forward(self, x):

        if self.is_quant:
            if self.quant_type=='4bit':
                return F.linear(x, bnb.functional.dequantize_4bit(self.weight.data,quant_state=self.qstate).to(torch.bfloat16)) 
            elif self.quant_type=='nf4':
                return F.linear(x, bnb.functional.dequantize_nf4(self.weight.data,quant_state=self.qstate).to(torch.bfloat16))
            elif self.quant_type=='fp4':
                return F.linear(x, bnb.functional.dequantize_fp4(self.weight.data,quant_state=self.qstate).to(torch.bfloat16))
            elif self.quant_type=='int8':
                return F.linear(x, bnb.functional.dequantize(self.weight.data,state=self.qstate).to(torch.bfloat16))
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



class RWKV6_ChannelMix(torch.nn.Module):
    
    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()
        # self.layer_id = layer_id
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

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
                          dim=1)
        last_state[:] = x[:, -1:]
        
        # if(self.lastlayer):
        #     x = x[:,-1:]
        #     xx = xx[:,-1:]
        
        xk = xx * self.time_maa_k + x * (1 - self.time_maa_k)
        xr = xx * self.time_maa_r + x * (1 - self.time_maa_r)
        kv = self.value( torch.relu( self.key(xk) ) ** 2 )
        return (torch.sigmoid(self.receptance(xr)) * kv), last_state
    


class Block6(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = RWKV6_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.ffn = RWKV6_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)
        
    
        # Setup droupout at block level

    def forward(self, x, time_mix_shift, channel_mix_state, time_mix_state):

        att_out, time_mix_shift, time_mix_state = self.att(
                self.ln1(x),
                time_mix_shift,
                time_mix_state
            )

        
            # Handle without dropout
        x = x + att_out
        ffn_out, channel_mix_state = self.ffn(
            self.ln2(x),
            channel_mix_state,
        )
        x = x + ffn_out
        
        return x, time_mix_shift, channel_mix_state, time_mix_state

    



# RWKV TimeMix module
class RWKV6_TimeMix(torch.nn.Module):
    #chunk_len:int = 128, precision:int = 64
    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, chunk_len:int = 1, precision:int = 64):
        super().__init__()
        
                
        self.layer_id = layer_id

        self.head_size = head_size
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

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance =HybridLinear(n_embd, dim_att, bias=False)
        self.key =HybridLinear(n_embd, dim_att, bias=False)
        self.value =HybridLinear(n_embd, dim_att, bias=False)
        self.output =HybridLinear(dim_att, n_embd, bias=False)
        self.gate =HybridLinear(n_embd, dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, dim_att, eps=(1e-5)*(head_size_divisor**2))
        self.ctx = 1024#saveBackDummy()

    def jit_func(self, x, last_state_shift):
        B, T, C = x.size()

        #print(f'last_state_shift = {last_state_shift.shape}')

        output = torch.concat((last_state_shift, x[:, :-1]), dim=1)
        last_state_shift[:] = x[:,-1:]

        xx = output - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = torch.nn.functional.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww
        w = w.exp().neg()
        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x, last_state_shift, last_state_wkv):
        #print("x shape:", x.shape)
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x, last_state_shift)

        if T > 1:
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
        # x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

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
             
                 ):

 
        # Setup the parent class
        super().__init__()
           
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
            ctx_len = 512

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

            #n_head = 32
           
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
        
        #with torch.no_grad():
        self.emb = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.ModuleList([
        
            Block6(i, n_layer, n_embd, n_head, self.head_size, n_embd, dim_ffn) for i in range(n_layer)
        ])
        
        self.head = nn.Linear(n_embd, vocab_size,bias=False)
        
        file["ln_in.weight"] = file.pop("blocks.0.ln0.weight")
        # lnb = file.pop("blocks.0.ln0.bias")
        file["ln_in.bias"] = file.pop("blocks.0.ln0.bias")
        
        self.ln_in = nn.LayerNorm(n_embd)
        # file["emb.weight"] = self.RMSNorm(file["emb.weight"].cpu().float()) * lnw.cpu().float() + lnb.cpu().float()
        
        # lnoa = file.pop("ln_out.weight")
        # lnob = file.pop("ln_out.bias")
        # file["head.bias"] = file["head.weight"] @ lnob
        # file["head.weight"] = file["head.weight"] * lnoa
        self.ln_out = nn.LayerNorm(n_embd)

        self.requires_grad_(False)
        
        self.load_state_dict(file,strict=False)


        quant_layers = 0

        if quantize:
            quant_layers = 32


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

            if name == 'blocks' or name == '':
                ThroughFound = True

            if ThroughFound == False:
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
        
    # @TCompileBaseline
   
    def forward(self, idx: torch.Tensor, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor]):
        if idx.dtype != torch.int64:
            x = idx.requires_grad_(False).to(device=self.emb.weight.device)
        else:
            x = self.emb(idx)
        x = self.ln_in(x)

        for i,b in enumerate(self.blocks):
            #print(i)
            x,last_shift_states[i*2],last_shift_states[i*2+1], last_wkv_states[i]  = b(x, last_shift_states[i*2],last_shift_states[i*2+1], last_wkv_states[i])

        x = self.ln_out(x)
        x = self.head(x)

        return x, last_shift_states, last_wkv_states
    

    def load_state(self,state_filename):
        try:
            state_raw = torch.load(state_filename, map_location="cpu")
        except Exception as e:
            print(e)
            return "state file failed to load"
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
            return "state shape mismatch"

        #strategy = model.strategy

        self.model_current_statetuned = [None] * self.n_layer * 3

        for i in range(self.n_layer):
            #dd = strategy[i]
            dev = 'cuda'#dd.device
            atype = torch.bfloat16 #dd.atype
            self.model_current_statetuned[i * 3 + 0] = torch.zeros(
                self.n_embd, dtype=atype, requires_grad=False, device=dev
            ).contiguous()
            self.model_current_statetuned[i * 3 + 1] = (
                state_raw[f"blocks.{i}.att.time_state"]
                .transpose(1, 2)
                .to(dtype=torch.float, device=dev)
                .requires_grad_(False)
                .contiguous()
            )
            self.model_current_statetuned[i * 3 + 2] = torch.zeros(
                self.n_embd, dtype=atype, requires_grad=False, device=dev
            ).contiguous()