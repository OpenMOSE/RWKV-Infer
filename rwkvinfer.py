#RWKV x060 Multibatch Inference Engine
#2024 OpenMOSE Apache2.0
from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
import time
import torch
import asyncio
import time
import gc
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


model = None
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

class RWKVWrapper:
    def __init__(self,debug=False):
        self.model_tokens = []
        self.model_state = None
        self.CHUNK_LEN = 1024  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)
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
            if self.debug:
                print(f"State-tune model loaded:{state_filename}")
        elif state_filename == "":
            print('state reset')
            self.model_state = None
            self.model_current_statetuned = None
            self.model_current_statetuned_filename = ""
            gc.collect()
            torch.cuda.empty_cache()

    def run_rnn(self, ctx):
        global model

        ctx = ctx.replace("\r\n", "\n")

        tokens = pipeline.encode(ctx)
        tokens = [int(x) for x in tokens]
        self.model_tokens += tokens

        while len(tokens) > 0:
            out, self.model_state = model.forward(tokens[:self.CHUNK_LEN], self.model_state)
            tokens = tokens[self.CHUNK_LEN:]

        return out

    async def Generate(self, input_prompt, temperature=1.0, top_p=0.5,alpha_presence=0.0,alpha_frequency=1.0, penalty_decay=0.996, MAX_COUNT=1000,STOP=['\n\n']):
        if self.debug:
            print("Generate Command Start")

        self.GEN_TEMP = temperature
        self.GEN_TOP_P = top_p
        self.GEN_alpha_presence = alpha_presence
        self.GEN_alpha_frequency = alpha_frequency
        self.GEN_penalty_decay = penalty_decay
        self.GEN_MAX_COUNT = MAX_COUNT

        occurrence = {}
        out_tokens = []
        out_last = 0

        if self.model_current_statetuned is not None:
            self.model_state = copy.deepcopy(self.model_current_statetuned)
            if self.debug:
                print("State-tuned deepcopy")
        else:
            self.model_state = None #ToDo Implement State-Cache for Faster Inference

        output_text = ''

        out = await asyncio.to_thread(self.run_rnn, input_prompt)
        if self.debug:
            print(input_prompt)
        
        for i in range(self.GEN_MAX_COUNT):
            for n in occurrence:
                out[n] -= self.GEN_alpha_presence + occurrence[n] * self.GEN_alpha_frequency
            out[0] -= 1e10
            token = pipeline.sample_logits(out, temperature=self.GEN_TEMP, top_p=self.GEN_TOP_P)
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
