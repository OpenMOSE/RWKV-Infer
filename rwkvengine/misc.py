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
from tokenizers import Tokenizer

MyStatic = torch.jit.script

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

class TRIE_TOKENIZER():
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()

class PIPELINE():
    def __init__(self, mode='world'):
        self.mode = mode
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        #from rwkv_tokenizer import TRIE_TOKENIZER_MOSE
        if mode == 'world':
            self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')  
        elif mode == 'pile':
            print(f'Pile Tokenizer')
            self.tokenizer = Tokenizer.from_file(os.path.dirname(os.path.abspath(__file__)) + "/20B_tokenizer.json")


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
        if self.mode == 'pile':
            print('pile')
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
    #@MyStatic
    def improved_nucleus_sampling_multi_static(self,logits, temperature, top_p):
        #batch_size = logits.size(0)
        device = logits.device
        #vocab_size = logits.size(-1)
        
        # temperature をテンソルに変換し、バッチサイズに対応
        # if isinstance(temperature, (int, float)):
        #     temperature = torch.full((batch_size, 1), fill_value=temperature, device=device, dtype=logits.dtype)
        # else:
        #     #temperature = torch.tensor(temperature, device=device, dtype=logits.dtype).view(-1, 1)
        #     temperature = temperature.view(-1, 1).to(device=device,dtype=logits.dtype)

        temperature = temperature.view(-1, 1).to(device=device,dtype=logits.dtype)
        #temperature = temperature.clone()
        temperature[temperature == 0.0] = 1.0

        # top_p をテンソルに変換し、バッチサイズに対応
        # if isinstance(top_p, (int, float)):
        #     p = torch.full((batch_size, 1), fill_value=top_p, device=device, dtype=logits.dtype)
        # else:
        #     #p = torch.tensor(top_p, device=device, dtype=logits.dtype).view(-1, 1)
        #     p = top_p.view(-1, 1).to(device=device,dtype=logits.dtype)

        p = top_p.view(-1, 1).to(device=device,dtype=logits.dtype)

        # ソフトマックスを計算
        probs = F.softmax(logits.to(dtype=torch.bfloat16), dim=-1)
        
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
        
        return samples#