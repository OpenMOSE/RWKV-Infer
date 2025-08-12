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