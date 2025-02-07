import torch
from rwkvengine.rwkvcore import RWKV_x, PIPELINE
import time
import copy
import torch.nn.functional as F
import numpy as np

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tb", default="1", type=int) #batch size 
    parser.add_argument("--fully_fused", default="1", type=int) #batch size 
    args = parser.parse_args()
    print('RWKV x070Core with FLA Test')

    pipeline = PIPELINE()
    model = RWKV_x('/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth','fp8',adapter_mode='lora',adapter_model='/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/x070-0B4-moe-cjev4/rwkv-4.pth',fully_fusedrecurrent=args.fully_fused)
    #model = RWKV_x('/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/rwkv-x070-2b9-cje-instruct-1.pth','fp8')
    Target_batch = args.tb

    States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    States2 = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    context =  'User: 君の心は何色？\n\n\x17Assistant:'
    

    shift_states = States.shift_states
    wkv_states = States.wkv_states

    shift_states2 = States2.shift_states
    wkv_states2 = States2.wkv_states

    def print_tensor_shapes(tensor_list):
        for i, tensor in enumerate(tensor_list):
            if isinstance(tensor, torch.Tensor):
                print(f"Tensor {i}: Shape = {tensor.shape}")
            else:
                print(f"Item {i} is not a Tensor")

    #print_tensor_shapes(model.model_current_statetuned )
    #print(f'state-tune-file = {model.model_current_statetuned    }')

    print('////////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    print(f'wkv_states = {wkv_states.shape    }')
    print(f'shift_states = {shift_states.shape    }')


    tokens0 = pipeline.encode(context)
    #tokens = pipeline.encode(context2)
    #tokens2 = pipeline.encode(context3)

    #print(len(tokens))



    prompts = []
    for i in range(Target_batch):
            prompts.append(torch.tensor(tokens0).unsqueeze(0).to('cuda'))

    idx = torch.cat(prompts, dim=0)

    #print(idx.shape)
    # this is warmup for triton kernels
    x1, shift_states1, wkv_states1 = model.forward(copy.deepcopy(idx), copy.deepcopy(shift_states), copy.deepcopy(wkv_states),KernelMode=1) #FLA
    del x1
    del shift_states1
    del wkv_states1

    t_prefill_0 = time.perf_counter()
    x, shift_states, wkv_states = model.forward(copy.deepcopy(idx), shift_states, wkv_states,KernelMode=0) #FLA
    t_prefill_1 = time.perf_counter()


    prefilltoken_total = len(idx.view(-1))

    prefill_time = (t_prefill_1 - t_prefill_0)
    totaltime = prefill_time
    prefill_time = (float(prefilltoken_total)) / prefill_time

    print(f'totaltime = {totaltime} totaltoken = {prefilltoken_total} Prefill {prefill_time}t/s')




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #exit()

    print(context)
    out_tokens = [[] for _ in range(Target_batch)]
    out_last = [0 for _ in range(Target_batch)]
    output_text = ['' for _ in range(Target_batch)]
    

    FirstTime = 1

    t000 = time.perf_counter()
    min_time = 1e10
    min_time_all = 1e10
    min_time_all_single = 1e10

    maxtoken= 500

    temperature = torch.full((Target_batch,), 1.0)
    top_p = torch.full((Target_batch,), 0.5)


    SamplingSum = 0
    ForwardSum = 0
    DecodeSum = 0

    for i in range(maxtoken):
        
        t0 = time.perf_counter()

        #x = x.view(-1,1)
        if x.dim() == 2:
            x = x.view(x.shape[0],1,x.shape[1])
        x[:, -1, 0] -= 1e10

        otokens = pipeline.improved_nucleus_sampling_multi_static(x[:, -1], temperature=temperature, top_p=top_p).tolist()

        tokens = []
        for j in range(Target_batch):
            tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0).to('cuda'))

        idx = torch.cat(tokens, dim=0)
        t1 = time.perf_counter()
        for j in range(Target_batch):
            out_tokens[j] += [otokens[j]]
            try:
                tmp = pipeline.decode(out_tokens[j][out_last[j]:])
                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                        #if j == Target_batch - 1:
                        print(tmp,end="", flush=True)
                        output_text[j] = output_text[j] + tmp
                        out_last[j] = i + 1
            except:
                pass
        t2 = time.perf_counter()

        x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states,one_mode=True)
        if x.dim() == 2:
            x = x.view(x.shape[0],1,x.shape[1])
        #print(x)
        t3 = time.perf_counter()
        ForwardSum += (t3 - t2)
        DecodeSum += (t2 - t1)
        SamplingSum += (t1 - t0)

    ForwardSum = ForwardSum / (float(maxtoken)) * 1000
    DecodeSum = DecodeSum / (float(maxtoken)) * 1000
    SamplingSum = SamplingSum / (float(maxtoken)) * 1000

    print('performance')
    print(f'ForwardAverage= {round(ForwardSum,4)} ms')
    print(f'DecodeSum= {round(DecodeSum,4)} ms')
    print(f'SamplingSum= {round(SamplingSum,4)} ms')



    t001 = time.perf_counter()

    print(output_text)
    print('RWKV-Infer FLA Refactor')

    tokensec = maxtoken / (t001-t000)
    print(f'totaltime = {totaltime} totaltoken = {prefilltoken_total} Prefill {prefill_time}t/s')
    print(f'TargetBatch = {Target_batch} Total token/s = {round(tokensec*Target_batch,2)} Single token/s = {round(tokensec,2)}')
