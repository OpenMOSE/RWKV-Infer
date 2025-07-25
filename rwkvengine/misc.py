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
import json
# (標準ライブラリではない点に注意)
from jinja2 import Template


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
        self.hfmode = False
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        #from rwkv_tokenizer import TRIE_TOKENIZER_MOSE
        if mode == 'world':
            self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')  
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/world")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")
        elif mode == 'pile':
            print(f'Pile Tokenizer')
            self.tokenizer = Tokenizer.from_file(os.path.dirname(os.path.abspath(__file__)) + "/20B_tokenizer.json")
            self.default_eos_token = "\n\n"
        elif mode == 'qwen':
            print(f'Qwen Tokenizer')
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/qwen")
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/qwen")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")
        elif mode == 'qwen3':
            print(f'Qwen3 Tokenizer')
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/qwen3")
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/qwen3")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")
        elif mode == 'llmjp':
            print(f'llmjp llama Tokenizer')
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/llmjp")
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/llmjp")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")
        elif mode == 'phi3.5':
            print(f'phi3.5 Tokenizer')
            from transformers import AutoTokenizer
            self.hfmode = True
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/phi3.5", add_prefix_space=True)
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/phi3.5")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")

        elif mode == 'phi4mini':
            print(f'phi1 mini Tokenizer')
            from transformers import AutoTokenizer
            self.hfmode = True
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/phi4mini", add_prefix_space=True)
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/phi4mini")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")
        elif mode == 'phi4':
            print(f'phi4 Tokenizer')
            from transformers import AutoTokenizer
            self.hfmode = True
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/phi4", add_prefix_space=True)
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/phi4")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")
        elif mode == 'mistralsmall3':
            print(f'mistral small 3 Tokenizer')
            from transformers import AutoTokenizer
            self.hfmode = True
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/mistralsmall3") #, add_prefix_space=True
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/mistralsmall3")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")
        elif mode == 'rekaflash3':
            print(f'rekaflash3 Tokenizer')
            from transformers import AutoTokenizer
            self.hfmode = True
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + "/rekaflash3") #, add_prefix_space=True
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + "/rekaflash3")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")
        else:
            print(f'{mode} Tokenizer')
            from transformers import AutoTokenizer
            self.hfmode = True
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)) + f"/{mode}") #, add_prefix_space=True
            self.modeltemplate = self.load_tokenizer_config(os.path.dirname(os.path.abspath(__file__)) + f"/{mode}")
            self.default_eos_token = self.modeltemplate.get("eos_token", "<|endoftext|>")

    def load_tokenizer_config(self, config_path: str) -> dict:
        """
        TokenizerConfig (例: tokenizer_config.json) をロードして辞書として返す
        """
        with open(config_path + "/tokenizer_config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_prompt_from_config(self, tokenizer_config: dict, messages: list, add_generation_prompt: bool = False) -> str:
        """
        tokenizer_config 内の chat_template を元に、messages (role, content) をまとめた文字列を生成する
        """
        # chat_template を取り出す
        chat_template_str = tokenizer_config.get("chat_template", None)
        if not chat_template_str:
            raise ValueError("chat_template が TokenizerConfig 内に存在しません。")

        # eos_token など必要なトークンも取り出す
        eos_token = tokenizer_config.get("eos_token", "<|endoftext|>")

        # テンプレートを読み込む
        template = Template(chat_template_str)
        from datetime import datetime
        # カスタム関数の定義
        def strftime_now(format_string):
            return datetime.now().strftime(format_string)
        
        template.globals['strftime_now'] = strftime_now

        #print(chat_template_str)

        # テンプレート変数を指定してレンダリング
        rendered_text = template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            eos_token=eos_token,
            enable_thinking = False,

        )
        return rendered_text


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
        if self.mode == 'qwen':
            return self.tokenizer.encode(x)
        else:
            return self.tokenizer.encode(x)
    
    def decode(self, x):
        if self.hfmode:
            #tokens_str = self.tokenizer.convert_ids_to_tokens(x)
            #strs = self.tokenizer.convert_tokens_to_string(tokens_str)
            #print(f'str = "{strs}"')
            #return strs
            return self.tokenizer.decode(x,skip_special_tokens=False,    # 特殊トークンをスキップしない
    clean_up_tokenization_spaces=False)  # トークン化の際のスペースをクリーンアップしない)
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
    @torch.compile
    def improved_nucleus_sampling_multi_static(self,logits, temperature, top_p):
        device = logits.device

        temperature = temperature.view(-1, 1).to(device=device,dtype=logits.dtype)
        temperature[temperature == 0.0] = 1.0

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
        
        return samples#
    @torch.compile
    def nucleous_sample(
        self,
        logits: torch.Tensor,        # (B, V)
        temperature: torch.Tensor,   # (B,) または (B,1)
        top_p: torch.Tensor          # (B,) または (B,1)
    ) -> torch.Tensor:
        B, V = logits.shape
        device = logits.device
        dtype = logits.dtype

        # 温度と p を整形／転送
        temperature = temperature.view(-1, 1).to(device, dtype)
        # 0 は無効扱い → 1.0 にリセット
        temperature.masked_fill_(temperature == 0.0, 1.0)

        p = top_p.view(-1, 1).to(device, dtype)

        # ソフトマックス（float16 のほうが高速になる GPU も多い）
        probs = F.softmax(logits.to(torch.bfloat16), dim=-1)
        self.max_k = 1024
        # 部分ソート（top-k）
        k = min(self.max_k, V)
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1, largest=True, sorted=True)

        # 累積確率を計算
        cum_probs = torch.cumsum(topk_probs, dim=-1)

        # top-p を超えたトークンはマスク
        # まず超過箇所を True に
        mask = cum_probs > p
        # いちばん確率の高い単語は常に残す
        mask[..., 0] = False
        # 最初の True の位置以降を消すためにシフト
        mask[..., 1:] = mask[..., :-1].clone()

        # マスクされた確率を 0 に（in-place）
        topk_probs.masked_fill_(mask, 0.0)

        # 温度スケーリング＆リノーマライズ
        if not torch.all(temperature == 1.0):
            # float16→元 dtype に戻してからべき乗
            topk_probs = topk_probs.to(dtype)
            topk_probs.pow_(1.0 / temperature)
        topk_probs.div_(topk_probs.sum(dim=-1, keepdim=True))

        # 最終サンプリング
        # (B, k) → (B, 1) → (B,)
        idx_in_topk = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
        samples = topk_indices.gather(1, idx_in_topk.unsqueeze(-1)).squeeze(-1)

        return samples
    
    def improved_nucleus_sampling_multi_static_topk(self, logits, temperature, top_p, top_k=5):
        """
        logits       : [batch_size, vocab_size] のテンソル
        temperature  : スカラー もしくは [batch_size, 1] のテンソル
        top_p        : スカラー もしくは [batch_size, 1] のテンソル
        top_k        : None, スカラー(int), もしくは [batch_size, 1] のテンソル
        """
        device = logits.device
        batch_size, vocab_size = logits.size()

        # (1) temperature をテンソル化して準備
        # --------------------------------
        if isinstance(temperature, (float, int)):
            # スカラー → バッチサイズぶん展開
            temperature = torch.full((batch_size, 1), fill_value=temperature, 
                                    device=device, dtype=logits.dtype)
        else:
            # すでに [batch_size, 1] と仮定
            temperature = temperature.view(batch_size, 1).to(device=device, dtype=logits.dtype)
        # 0.0 はゼロ除算回避のため 1.0 に置き換える
        temperature[temperature == 0.0] = 1.0

        # (2) top_p をテンソル化
        # --------------------------------
        if isinstance(top_p, (float, int)):
            # スカラー → バッチサイズぶん展開
            p = torch.full((batch_size, 1), fill_value=top_p, 
                        device=device, dtype=logits.dtype)
        else:
            # すでに [batch_size, 1] と仮定
            p = top_p.view(batch_size, 1).to(device=device, dtype=logits.dtype)

        # (3) top_k をバッチ単位で処理するための準備
        # --------------------------------
        # None や 0 以下の場合はTop-Kなしとみなす
        top_k_tensor = None
        if top_k is not None and isinstance(top_k, int) and top_k > 0:
            # スカラー (整数) の場合
            # → 全バッチ同じK
            top_k_tensor = torch.full((batch_size,), fill_value=top_k, device=device, dtype=torch.long)
        elif torch.is_tensor(top_k):
            # すでにバッチサイズと同じかどうかを仮定
            # [batch_size, 1] または [batch_size] など
            top_k = top_k.view(-1).to(device=device, dtype=torch.long)
            # k <= 0 のものは使えないので min(k, 1) のチェックなど適宜する
            top_k = torch.clamp(top_k, min=0)
            if top_k.size(0) == batch_size:
                top_k_tensor = top_k
            else:
                # サイズが合わない場合はエラーにするか、適当に処理する
                raise ValueError(f"top_k がバッチサイズ({batch_size})と合いません: {top_k.size()}")
        # top_k_tensor が None のままなら「Top-Kなし」として扱う

        # (4) softmax で確率に変換
        # --------------------------------
        # float16/bfloat16 環境の場合も考慮し、float32 で一度計算する例
        probs = F.softmax(logits.to(dtype=torch.float32), dim=-1)

        # (5) バッチごとに Top-K を適用（top_k_tensor があれば）
        # --------------------------------
        if top_k_tensor is not None:
            new_probs = torch.zeros_like(probs)
            for i in range(batch_size):
                k_i = top_k_tensor[i].item()
                if k_i > 0:
                    # バッチ i の確率分布のみ取り出し
                    row_probs = probs[i]  # shape: [vocab_size]
                    # 上位 k_i を取得
                    topk_probs, topk_indices = torch.topk(row_probs, k_i, dim=-1)
                    # 一旦全部0にして、該当indexにだけ値を割り当てる
                    temp = torch.zeros_like(row_probs)
                    temp.scatter_(0, topk_indices, topk_probs)
                    new_probs[i] = temp
                else:
                    # k_i <= 0 はマスク扱い(Top-K無効と同じ)
                    new_probs[i] = probs[i]
            probs = new_probs

        # (6) Top-P (Nucleus) をバッチ単位で適用
        # --------------------------------
        # 降順にソート
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 累積確率が p を超える部分をマスク
        # 先頭要素を除いて True をずらすことでしきい値を超えた直後からマスク
        sorted_indices_to_remove = (cumulative_probs > p)
        shifted = torch.zeros_like(sorted_indices_to_remove)
        shifted[:, 1:] = sorted_indices_to_remove[:, :-1]
        sorted_indices_to_remove = shifted
        sorted_indices_to_remove[:, 0] = False

        # 元のインデックスにマスクを適用
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)

        # (7) 温度スケーリング
        # --------------------------------
        # （全成分 1.0 のバッチもあるので、==1.0 全部 が False のバッチもあるかもしれませんが、
        #   一応一括で計算）
        # temperature はバッチ単位 [batch_size, 1]
        # probs は [batch_size, vocab_size]
        if not torch.all(temperature == 1.0):
            # (1.0 / temperature): shape [batch_size, 1]
            inv_temp = 1.0 / temperature
            # broadcasting
            probs = probs ** inv_temp
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # (8) multinomial サンプリング
        # --------------------------------
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples

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

    def __init__(self, shift_states, wkv_states,kv_cache=None,pos_cache=None):
        self.wkv_states = wkv_states
        self.shift_states = shift_states
        if kv_cache is not None:
            self.kv_cache = kv_cache
            self.pos_cache = pos_cache

    @staticmethod
    def create(N, B, C, H, device, dtype):
        result = BlockStateList.empty(N, B, C, H, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        return result
    
    @staticmethod
    def x070_create(N, B, n_embd, head_size, device, dtype):
        result = BlockStateList.x070_empty(N, B, n_embd, head_size, device, dtype)
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        return result
    
    def hx078_create(N,GQA_N, B, n_embd,n_kv,head_size,max_len,device, dtype):
        result = BlockStateList.hx078_empty(N,GQA_N, B, n_embd,n_kv,head_size,max_len,device, dtype)
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        result.kv_cache[:] = 0
        result.pos_cache[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        wkv_states = torch.zeros((N, B, H, C//H, C//H),
                                 device=device,
                                 dtype=torch.bfloat16)
        shift_states = torch.zeros((N*2,B,1, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    @staticmethod
    def x070_empty(N, B, n_embd,head_size, device, dtype):
        wkv_states = torch.zeros((N, B, n_embd // head_size, head_size, head_size),
                                 device=device,
                                 dtype=dtype) 
        shift_states = torch.zeros((N*2,B,1,n_embd), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)
    
    @staticmethod
    def hx078_empty(N,GQA_N, B, n_embd,n_kv,head_size,max_len,device, dtype):
        wkv_states = torch.zeros((N, B, n_embd // head_size, head_size, head_size),
                                 device=device,
                                 dtype=dtype) 
        shift_states = torch.zeros((N*2,B,1,n_embd), device=device, dtype=dtype)

        kv_cache = torch.zeros((GQA_N,B,2,max_len,head_size*n_kv), device=device, dtype=dtype)
        #kv_cache = torch.zeros((GQA_N,B,max_len,2,head_size*n_kv), device=device, dtype=dtype)
        pos_cache = torch.zeros((B,1), device=device, dtype=torch.int64)
        return BlockStateList(shift_states, wkv_states,kv_cache=kv_cache,pos_cache=pos_cache)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state