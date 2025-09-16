import time
import sys
import os
# 1階層上のディレクトリのパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

#Original Repo is https://github.com/jannalulu/NIAH
#Modified for support RWKV-Infer
#2025 OpenMOSE

import os
import math
#import fla
from transformers import GenerationConfig
import torch
import json
import argparse
import random
import re
import numpy as np
from numpy import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


#RWKV Infer
from rwkvengine.rwkvcore import RWKV_x, PIPELINE

pattern = "sliding_1024"
model_path = "/home/client/Projects/llm/RWKV-Reka-Flash-Gen2/"
max_kv_size = 1024
model_adapter_path = "" # can realtime merge LoRA,Bone,DoRA
model_adapter_mode = "" # set lora,bone,dora
quant_mode = "int8" # int8, OpenMOSE Silly 8bit matmul kernel(triton)
template = "rekaflash3"
rope_theta=8000000.0
rms_norm_eps=1e-5    

def get_gpu_memory():
    """Returns the current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

def parse_config():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments including:
            - Standard evaluation parameters
            - HF model path and cache directory
            - Optional HF model arguments as JSON string
    """
    parser = argparse.ArgumentParser(description='arg parser')
    #parser.add_argument('hf_model', type=str)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--min_tokens', type=int, default=1024, help='minimum token length to start evaluation')
    parser.add_argument('--max_tokens', type=int, default=8192, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1024, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=2, help='number of repeat testing for each length')
    parser.add_argument('--max_depth', type=float, default=1.0, help='max depth ratio to test')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for computation')
    # #parser.add_argument('--hf_model_args', type=str, default='{}',
    #                   help='Additional HuggingFace model arguments as JSON string')
    args = parser.parse_args()
    return args


def generate_prompt_landmark(tokenizer, pass_key, context_length, depth, final_context_length_buffer=250):
    # needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
    # task_description = "<|im_start|>system\nThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. "
    # garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    # question = "<|im_end|>\n<|im_start|>user\nWhat is the pass key?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe pass key is"
    needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. "
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    question = "What is the pass key? The pass key number is"
    
    tokens_in_garbage = len(tokenizer.encode(garbage))
    multiplier = math.ceil((context_length - len(tokenizer.encode(task_description)) - 25) / tokens_in_garbage)
    context = garbage * multiplier
    
    tokens_task = tokenizer.encode(task_description)
    tokens_needle = tokenizer.encode(needle)
    tokens_context = tokenizer.encode(context)
    tokens_question = tokenizer.encode(question)
    tokens_newline = tokenizer.encode("\n")
    
    # Reduce context length by buffer
    context_length = context_length - final_context_length_buffer - len(tokens_task) - len(tokens_question)
    
    # Truncate context if needed
    if len(tokens_context) + len(tokens_task) + len(tokens_needle) + len(tokens_question) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]
    
    if depth >= 1:
        tokens_new_context = tokens_task + tokens_context + tokens_newline + tokens_needle + tokens_newline + tokens_question

    elif depth == 0:
        tokens_new_context = tokens_task + tokens_needle + tokens_newline + tokens_context + tokens_newline + tokens_question

    else:
        insertion_point = int(len(tokens_context) * depth)
        tokens_new_context = tokens_context[:insertion_point]
        
        # Find sentence break
        period_tokens = tokenizer.encode('.')
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        
        tokens_new_context = tokens_task + tokens_new_context + tokens_newline + tokens_needle + tokens_newline + tokens_context[insertion_point:] + tokens_question
    
    print("Total Tokens in Context: ", len(tokens_new_context))
    new_context = tokenizer.decode(tokens_new_context)
    return new_context

def passkey_retrieval_test(model, tokenizer, device, context_length, depth, seed=666):
    # Generate random pass key
    rnd_state = random.get_state()
    random.seed(seed)
    pass_key = random.randint(1, 50000)
    random.set_state(rnd_state)

    prompt = generate_prompt_landmark(tokenizer, pass_key, context_length=context_length, depth=depth)
    answer = str(pass_key)

    #input_token_ids = tokenizer(prompt, return_tensors=None).input_ids
    input_token_ids = tokenizer.encode(prompt)#.input_ids

    input_ids = torch.tensor([input_token_ids], device=device)
    len_token = input_ids.shape[-1]

    #answer_ids = tokenizer(answer).input_ids # Get token IDs for answer length
    answer_ids = tokenizer.encode(answer)#.input_ids
    max_new_tokens = len(answer_ids) + 16

    past_key_values = None
    processed_len = 0
    prefill_ids = input_ids[:, :-1]
    prefill_len = prefill_ids.shape[1]
    chunk_size = 8192

    States = model.new_state(1,max_kv_size)
    shift_states = States.shift_states
    wkv_states = States.wkv_states
    kv_caches = States.kv_cache
    pos_caches = States.pos_cache

    # Process the prompt in chunks (for long context)
    with torch.no_grad():
        # Chunked prefill stage
        for i in range(0, prefill_len, chunk_size):
            chunk = prefill_ids[:, i : min(i + chunk_size, prefill_len)]
            chunk_len = chunk.shape[1]

            outputs, _, wkv_states, kv_caches, pos_caches  = model.forward(chunk, shift_states, wkv_states,kv_caches,pos_caches,full_output=True)

            #if full RNN No Need LoL
            #current_position_ids = torch.arange(processed_len, processed_len + chunk_len, dtype=torch.long, device=device).unsqueeze(0)

            # outputs = model(
            #     input_ids=chunk,
            #     past_key_values=past_key_values,
            #     position_ids=current_position_ids,
            #     use_cache=True,
            # )

            #past_key_values = outputs.past_key_values
            processed_len += chunk_len

        # Final forward pass for the last token
        last_token_input_ids = input_ids[:, -1:]
        last_token_pos_id = torch.tensor([[processed_len]], dtype=torch.long, device=device)
        final_prompt_mask = torch.ones(1, processed_len + 1, dtype=torch.long, device=device)

        # outputs = model(
        #     input_ids=last_token_input_ids,
        #     past_key_values=past_key_values,
        #     position_ids=last_token_pos_id,
        #     use_cache=True,
        # )
        #outputs, shift_states, wkv_states = model.forward(last_token_input_ids, shift_states, wkv_states,KernelMode=1,full_output=True)
        outputs, _, wkv_states, kv_caches, pos_caches  = model.forward(last_token_input_ids, shift_states, wkv_states,kv_caches,pos_caches,full_output=True)


        logits = outputs#.logits  # Logits for the *next* token
        #past_key_values = outputs.past_key_values  # Final cache after full prompt
        processed_len += 1

        # Generation stage
        generated_ids_list = []

        # Get the first generated token ID
        print(f'logits dim = {logits.shape}')
        next_token_logits = logits[:, -1, :]  # Shape [batch_size, vocab_size]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # Shape [batch_size, 1]

        for step in range(max_new_tokens):
            generated_ids_list.append(next_token_id.item())

            # Prepare inputs for the next step model call
            step_position_ids = torch.tensor([[processed_len]], dtype=torch.long, device=device)

            # outputs = model(
            #     input_ids=next_token_id,
            #     past_key_values=past_key_values,
            #     position_ids=step_position_ids,
            #     use_cache=True,
            # )
            #outputs, shift_states, wkv_states = model.forward(next_token_id, shift_states, wkv_states,KernelMode=1,full_output=True)
            outputs, _, wkv_states, kv_caches, pos_caches  = model.forward(next_token_id, shift_states, wkv_states,kv_caches,pos_caches,full_output=True)



            logits = outputs
            #past_key_values = outputs.past_key_values  # Update cache
            processed_len += 1  # Sequence length grows

            # Get the ID for the *next* token
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    # Decode and evaluate the model's output
    generated_ids_list = [i for i in generated_ids_list if i != 0]
    print(generated_ids_list)

    try:
        model_output = tokenizer.decode(generated_ids_list)
    except:
        model_output = "none"

    matches = re.findall(r"[\D]*(\d+)", model_output)
    model_answer = matches[0] if matches else ""
    is_correct = (model_answer == answer)
    
    print(f"Model's output: {model_output}")
    print(f"Found answer: {model_answer}")
    print(f"Correct answer: {answer}")
    print(f"Is correct: {is_correct}\n")

    # Clean up
    del past_key_values, input_ids, logits, generated_ids_list, model_output, next_token_id
    torch.cuda.empty_cache()

    return is_correct, len_token

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    print("RWKV-Infer Model", model_path)

    # Parse additional HF model arguments
    #hf_model_args = json.loads(args.hf_model_args)
    
    # Load model and tokenizer
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.hf_model,
    #     trust_remote_code=True,
    #     **hf_model_args
    # ).bfloat16().to(device)
    # tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    # model.eval()

    #RWKV-Infer with flash-linear-attention
    model = RWKV_x(model_path,quant_mode,
                   adapter_model=model_adapter_path,
                   adapter_mode=model_adapter_mode,
                   fully_fusedrecurrent=True,
                   rope_theta=rope_theta,
                   rms_norm_eps=rms_norm_eps
                   ) #use Fused Kernel(much faster)
    tokenizer = PIPELINE(template)

    # Calculate number of test points starting from min_tokens
    total_test_points = (args.max_tokens - args.min_tokens) // args.interval + 1
    all_accuracies = []
    
    for i in range(total_test_points):
        # Calculate context length starting from min_tokens
        current_tokens = args.min_tokens + (i * args.interval)
        
        # Calculate depth steps to max_depth
        depth_steps = np.linspace(0, args.max_depth, 10) # 10 steps from 0 to max_depth
        
        for depth in depth_steps:
            passed_tests = 0
            total_tokens = 0
            
            for k in range(args.num_tests):
                is_correct, len_tokens = passkey_retrieval_test(
                    model, tokenizer, device, 
                    context_length=current_tokens,
                    depth=depth,
                    seed=k
                )
                passed_tests += is_correct
                total_tokens += len_tokens
                
            avg_tokens = total_tokens // args.num_tests
            accuracy = float(passed_tests) / args.num_tests
            print(f"accuracy on the token length {avg_tokens}, depth {depth:.2f}, is {accuracy:.2f}")
            
            result = {
                "Context Length": avg_tokens,
                "Document Depth": round(depth * 100, -1),
                "Score": passed_tests
            }
            all_accuracies.append(result)

    total_tests = len(all_accuracies)
    total_passed = sum(result['Score'] for result in all_accuracies)
    total_score = (total_passed / (total_tests * args.num_tests)) * 100

    print("\nFinal Results Summary:")
    print(f"Total Tests Run: {total_tests * args.num_tests}")
    print(f"Total Tests Passed: {total_passed}")
    print(f"Overall Score: {total_score:.2f}%")

    # Print detailed breakdown
    df_summary = pd.DataFrame(all_accuracies)
    print("\nDetailed Results by Context Length and Depth:")
    print(df_summary.groupby(['Context Length', 'Document Depth'])['Score'].mean().to_string())

    # Create visualization
    df = pd.DataFrame(all_accuracies)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    pivot_table = pd.pivot_table(
        df, values='Score', index=['Document Depth', 'Context Length'], 
        aggfunc='mean'
    ).reset_index()
    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )
    
    plt.figure(figsize=(17.5, 8))
    sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'}
    )

    plt.xlabel('Token Limit')
    plt.ylabel('Depth Percent')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Extract last 2 path components and create sanitized filename
    model_path_parts = model_path.split('/')
    sanitized_model_name = '_'.join(model_path_parts[-2:] if len(model_path_parts) > 1 else model_path_parts[-1:])
   
    plt.savefig(f"misc/data/heatmap_tokenized_{args.max_tokens}_{sanitized_model_name}{pattern}.png")
    df_summary.to_csv(f"misc/data/results_tokenized_{args.max_tokens}_{sanitized_model_name}{pattern}.csv", index=False)

if __name__ == "__main__":
    args = parse_config()
    main(args)