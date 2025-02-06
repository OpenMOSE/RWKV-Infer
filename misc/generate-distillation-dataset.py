#RWKV x060 Distillation Dataset Generator
#2024 OpenMOSE
import os, copy, types, gc, sys, re
import numpy as np
#from prompt_toolkit import prompt
import torch
from argparse import ArgumentParser
import csv
import random
#import pyarrow as pa
#import pyarrow.parquet as pq
import numpy as np
import random
from typing import List, Tuple
import os
import concurrent.futures
import h5py
from torch.utils.data import Dataset, DataLoader
import glob
import json
import copy

from rwkvengine.rwkvcore import RWKV_x, PIPELINE

pipeline = PIPELINE(mode='world')




parser = ArgumentParser()

parser.add_argument("--load_model", default="myfolder/14B-CoT-merged.pth", type=str)
parser.add_argument("--input_folder", default="myfolder/datasets/General", type=str)
parser.add_argument("--output_parquet", default="myfolder/datasets/14B-General-Topk50.h5", type=str)
parser.add_argument("--strategy", default="cuda fp16", type=str)
#\x17
args2 = parser.parse_args()


########################################################################################################

args = types.SimpleNamespace()

args.strategy = args2.strategy#"cuda fp16"  # use CUDA, fp16

args.MODEL_NAME = args2.load_model

CHUNK_LEN = 1024  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")

model = RWKV_x(args.MODEL_NAME,'bf16',target_device='cuda:1')

model_tokens = []
model_state = None

trainset = []

model_current_statetuned = None






def get_top_k_logits(logits, k=50):
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    return top_k_values, top_k_indices

 

def run_rnn_logits(intoken):
    tokens = copy.deepcopy(intoken)#tokenizer.encode(ctx)
    tokens = [int(x) for x in tokens]

    #accumulated_logits = []

    accumulated_k = []
    accumulated_indice = []

    model_state = None
    i = 0

    States = model.new_state(1)
    shift_states = States.shift_states
    wkv_states = States.wkv_states

    InitialTokenLength  = len(tokens)
    LogitsSum = 0
 
    

    while len(tokens) > 0:
        prompts = []
        print('append tensor')
        prompts.append(torch.tensor(tokens[:CHUNK_LEN]).unsqueeze(0).to(model.target_device))

        idx = torch.cat(prompts, dim=0)

        print(f'idx shape = {idx.shape}')
        print(idx.view(-1))
        i+=1
        print(f'Running RNN Logits = {i}')
        out, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)
 
        out = out.view(-1,65536)

        LogitsSum+=torch.sum(out)

        top_k_values, top_k_indices = get_top_k_logits(out)
  
        print(f'out shape = {out.shape}')
 
        print('accumulate')
 
        accumulated_k.append(top_k_values.cpu())
        accumulated_indice.append(top_k_indices.cpu())
        print('cut tokens')
        tokens = tokens[CHUNK_LEN:]
        print('ok next loop')
        print(len(idx.view(-1)))

 
    print('try cat')
 
    final_k = torch.cat(accumulated_k, dim=0)
    final_indice = torch.cat(accumulated_indice, dim=0)
    print('ok')

    return final_k, final_indice#final_logits



class HDF5TopKTensorDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_length = len(f['tokens'])
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            tokens = torch.from_numpy(f['tokens'][idx][:]).long()
            top_k_values = torch.from_numpy(f['top_k_values'][idx][:]).float()
            top_k_indices = torch.from_numpy(f['top_k_indices'][idx][:]).long()
        
        if self.transform:
            tokens = self.transform(tokens)
            top_k_values = self.transform(top_k_values)
            top_k_indices = self.transform(top_k_indices)
        
        return tokens, top_k_values, top_k_indices
    
# Inputフォルダ内のJSONLファイル一覧を取得
input_folder = args2.input_folder
jsonl_files = glob.glob(os.path.join(input_folder, '*.jsonl'))



top_k = 50
i = 0


with h5py.File(args2.output_parquet, 'w') as f:
        tokens_dataset = f.create_dataset('tokens', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64),
                                          #compression="gzip", compression_opts=9
                                          )
        top_k_values_dataset = f.create_dataset('top_k_values', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.float32),
                                                #compression="gzip", compression_opts=9
                                                )
        top_k_indices_dataset = f.create_dataset('top_k_indices', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int64),
                                                 #compression="gzip", compression_opts=9
                                                 )
        NowProcessing = 0
        totaldatasetpairs = 0
        # 各JSONLファイルを処理
        for jsonl_file in jsonl_files:
            print(f"Processing file: {jsonl_file}")
            NowProcessing = NowProcessing + 1
            
            # JSONLファイルを開いて各行を処理
            with open(jsonl_file, 'r', encoding='utf-8') as file:
                for line in file:
                    # 各行をJSONとしてパース
                    try:
                        json_data = json.loads(line.strip())
                        # 'text'キーの値を取得
                        if 'text' in json_data:
                            text_value = json_data['text']

                            print(f'filename = {jsonl_file} text = {text_value}')

                            tokens = torch.tensor(pipeline.encode(text_value))
                            model_state = None
                            #if model_current_statetuned is not None:
                            #    model_state = copy.deepcopy(model_current_statetuned)
                            #    print('initial state deepcopy')
                                
                            print('Now RNN Processing')
                            totaldatasetpairs += 1
                            print(f'Now TotalProceedPairs {totaldatasetpairs}')

                            #logits = run_rnn_logits(tokens)
                            top_k_values, top_k_indices = run_rnn_logits(tokens)

                            print('finished RNN Processing')
                            

                            
                            # Top-K Logitsの計算
                            #top_k_values, top_k_indices = get_top_k_logits(logits, k=top_k)

                            # データセットのサイズを拡張
                            tokens_dataset.resize((i+1,))
                            top_k_values_dataset.resize((i+1,))
                            top_k_indices_dataset.resize((i+1,))

                            print('logits compute finished')

                            # PyTorch TensorをNumPy配列に変換してから保存
                            tokens_dataset[i] = tokens.numpy().astype(np.int64)
                            
                            # Top-K値とインデックスを1次元に変換して保存
                            top_k_values_dataset[i] = top_k_values.to(dtype=torch.float32).flatten().numpy().astype(np.float32)
                            top_k_indices_dataset[i] = top_k_indices.flatten().numpy().astype(np.int64)
                            


                            i=i+1


                            print(f"Text value: {text_value}")

                            print(f'NowProcessing {NowProcessing}/{len(jsonl_files)}')
                        else:
                            print("No 'text' key found in this line")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {jsonl_file}")
            

            print("-------------------------")














exit()

