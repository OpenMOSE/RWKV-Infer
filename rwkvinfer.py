#RWKV x060 Multibatch Inference Engine
#2024 OpenMOSE Apache2.0
from abc import ABC, abstractmethod
from enum import Enum, auto
from concurrent.futures import ProcessPoolExecutor
import os, sys
import pathlib
import copy
import re
import time
import torch
from torch.nn import functional as F
import numpy as np
import asyncio
import time
import gc
import multiprocessing
from rwkv.model2 import RWKV
#from rwkv.utils import PIPELINE

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

class PIPELINE():
    def __init__(self, model):
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
    def sample_logits_mose3(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        if temperature == 0:
            temperature = 1.0
        top_p = 0   
        probs = F.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff_value = sorted_probs[cutoff_index]

        # Use boolean indexing instead of torch.where for masking
        mask = probs >= cutoff_value
        probs = probs * mask

        if 0 < top_k < len(probs):
            # Use boolean indexing for masking top_k values
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[sorted_indices[:top_k]] = True
            probs = probs * mask

        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)

        # Normalize probabilities in-place
        probs /= probs.sum()

        # Use Gumbel-max trick for faster sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs)))
        out = torch.argmax(probs + gumbel_noise)

        return int(out)

 


model = None
pipeline = PIPELINE(model)

class RWKVWrapper:
    def __init__(self,debug=False):
        self.model_tokens = []
        self.model_state = None
        self.CHUNK_LEN = 64  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)
        self.model_current_statetuned_filename = ""
        self.model_current_statetuned = None

        self.GEN_TEMP = 1.0
        self.GEN_TOP_P = 0.3
        self.GEN_alpha_presence = 0.0
        self.GEN_alpha_frequency = 1.0
        self.GEN_penalty_decay = 0.996
        self.GEN_MAX_COUNT = 1000
        self.busy = False
        self.debug = debug
        self.Stop = True
    def is_busy(self):
        return self.busy
    def set_busy(self, busy):
        self.busy = busy
    def load_model(self,model_filename,strategy): #this is shared model
        global model
        model = None
        model = RWKV(model=model_filename, strategy=strategy)
    def unload_model(self):                       #this is shared
        global model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
    def load_state(self, state_filename):
        global model
        if self.model_current_statetuned_filename != state_filename and state_filename != "":
            try:
                state_raw = torch.load(state_filename, map_location="cpu")
            except Exception as e:
                print(e)
                return "state file failed to load"
            state_raw_shape = next(iter(state_raw.values())).shape

            args = model.args
            if self.debug:
                print(f"{len(state_raw)} != {args.n_layer}")
                print(f"{state_raw_shape[0] * state_raw_shape[1]} != {args.n_embd}")

            if (
                len(state_raw) != args.n_layer
                or state_raw_shape[0] * state_raw_shape[1] != args.n_embd
            ):
                print("state failed to load")
                return "state shape mismatch"

            strategy = model.strategy

            self.model_current_statetuned = [None] * args.n_layer * 3

            for i in range(args.n_layer):
                dd = strategy[i]
                dev = dd.device
                atype = dd.atype
                self.model_current_statetuned[i * 3 + 0] = torch.zeros(
                    args.n_embd, dtype=atype, requires_grad=False, device=dev
                ).contiguous()
                self.model_current_statetuned[i * 3 + 1] = (
                    state_raw[f"blocks.{i}.att.time_state"]
                    .transpose(1, 2)
                    .to(dtype=torch.float, device=dev)
                    .requires_grad_(False)
                    .contiguous()
                )
                self.model_current_statetuned[i * 3 + 2] = torch.zeros(
                    args.n_embd, dtype=atype, requires_grad=False, device=dev
                ).contiguous()

            self.model_state = None
            self.model_current_statetuned_filename = state_filename
            if self.debug:
                print(f"State-tune model loaded:{state_filename}")
        elif state_filename == "":
            print('state reset')
            self.model_state = None
            self.model_current_statetuned = None
            self.model_current_statetuned_filename = ""
            gc.collect()
            torch.cuda.empty_cache()


    
 

    async def Generate(self, input_prompt, temperature=1.0, top_p=0.5,alpha_presence=0.0,alpha_frequency=1.0, penalty_decay=0.996, MAX_COUNT=1000,STOP=['\n\n']):
        if self.debug:
            print("Generate Command Start")
        self.Stop = False

        self.GEN_TEMP = temperature
        self.GEN_TOP_P = top_p
        self.GEN_alpha_presence = alpha_presence
        self.GEN_alpha_frequency = alpha_frequency
        self.GEN_penalty_decay = penalty_decay
        self.GEN_MAX_COUNT = MAX_COUNT

        occurrence = {}
        out_tokens = []
        out_last = 0

        if self.model_current_statetuned is not None and self.model_state is None:
            self.model_state = copy.deepcopy(self.model_current_statetuned)
            if self.debug:
                print("State-tuned deepcopy")

        output_text = ''

        ctx = input_prompt
        ctx = ctx.replace("\r\n", "\n")

        tokens = pipeline.encode(ctx)
        tokens = [int(x) for x in tokens]
        self.model_tokens += tokens

        while len(tokens) > 0:
            out, self.model_state = await asyncio.to_thread(model.forward, tokens[:self.CHUNK_LEN], self.model_state)
            tokens = tokens[self.CHUNK_LEN:]
            yield ""
            
 
        if self.debug:
            print(input_prompt)
        
        for i in range(self.GEN_MAX_COUNT):
            if self.Stop:
                yield ''
                break
            for n in occurrence:
                out[n] -= self.GEN_alpha_presence + occurrence[n] * self.GEN_alpha_frequency
            out[0] -= 1e10
            token = pipeline.sample_logits_mose2(out, temperature=self.GEN_TEMP, top_p=self.GEN_TOP_P)
            out, self.model_state = await asyncio.to_thread(model.forward, [token], self.model_state)
            self.model_tokens += [token]
            out_tokens += [token]

            for xxx in occurrence:
                occurrence[xxx] *= self.GEN_penalty_decay
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            tmp = pipeline.decode(out_tokens[out_last:])
            #print(tmp,end="")
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                yield tmp
                output_text = output_text + tmp
                out_last = i + 1

            if type(STOP) == str:
                if STOP in tmp:
                    yield tmp
                    output_text = output_text + tmp
                    break
            elif type(STOP) == list:
                exit_flag = False
                for stop in STOP:
                    if stop in tmp:
                        yield tmp
                        output_text = output_text + tmp
                        exit_flag = True
                        break
                if exit_flag:
                    break
