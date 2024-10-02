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

MyStatic = torch.jit.script

class PIPELINE():
    def __init__(self, model='dummy'):
        self.model = model
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rwkv_tokenizer import TRIE_TOKENIZER_MOSE
        self.tokenizer = TRIE_TOKENIZER_MOSE(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')        

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
    @MyStatic    
    def improved_nucleus_sampling(logits, temperature:float=1.0, top_p:float=0.9):
       if temperature == 0.0:
           temperature = 1.0
       p = top_p
       probs = F.softmax(logits.float(), dim=-1)
       sorted_probs, sorted_indices = torch.sort(probs, descending=True)
       cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
       sorted_indices_to_remove = cumulative_probs > p
       sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
       sorted_indices_to_remove[0] = False
       indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
       probs.masked_fill_(indices_to_remove, 0.0)
       if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum()
       return int(torch.multinomial(probs, num_samples=1)[0])
    
    @MyStatic
    def sample_logits_mose2(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):

        if temperature == 0:
            temperature = 1.0
            top_p = 0.3

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
    def sample_logits_blink(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
        probs = F.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        
        if top_k > 0:
            probs[sorted_ids[top_k:]] = 0

        if top_p < 1:
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = torch.searchsorted(cumulative_probs, top_p)
            cutoff = sorted_probs[cutoff_index]
            probs[probs < cutoff] = 0

            if top_p > 0:
                idx = torch.where(probs == cutoff)[0]
                if len(idx) > 0:
                    probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                    # assert abs(torch.sum(probs).item() - top_p) < 1e-6
        
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)

        return torch.multinomial(probs, num_samples=1).item()
    
    def improved_nucleus_sampling_multi(self,logits, temperature=1.0, top_p=0.9):
        batch_size = logits.size(0)
        device = logits.device
        vocab_size = logits.size(-1)
        
        # temperature をテンソルに変換し、バッチサイズに対応
        if isinstance(temperature, (int, float)):
            temperature = torch.full((batch_size, 1), fill_value=temperature, device=device, dtype=logits.dtype)
        else:
            temperature = torch.tensor(temperature, device=device, dtype=logits.dtype).view(-1, 1)
        temperature = temperature.clone()
        temperature[temperature == 0.0] = 1.0

        # top_p をテンソルに変換し、バッチサイズに対応
        if isinstance(top_p, (int, float)):
            p = torch.full((batch_size, 1), fill_value=top_p, device=device, dtype=logits.dtype)
        else:
            p = torch.tensor(top_p, device=device, dtype=logits.dtype).view(-1, 1)

        # ソフトマックスを計算
        probs = F.softmax(logits.float(), dim=-1)
        
        # 確率を降順にソート
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 累積確率が top_p を超える部分をマスク
        sorted_indices_to_remove = cumulative_probs > p
        shifted = torch.zeros_like(sorted_indices_to_remove)
        shifted[:, 1:] = sorted_indices_to_remove[:, :-1]
        sorted_indices_to_remove = shifted
        sorted_indices_to_remove[:, 0] = False

        # 元のインデックスにマスクを適用
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove = indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        
        # 温度スケーリングを適用
        if not torch.all(temperature == 1.0):
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum(dim=-1, keepdim=True)
        
        # サンプリングを実行
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        #print(samples)
        
        return samples.tolist()
