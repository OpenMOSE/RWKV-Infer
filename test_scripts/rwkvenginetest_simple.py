import torch
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(parent_dir)
import torch.profiler
import torch
from rwkvengine.rwkvcore import RWKV_x, PIPELINE
import sys
 
import time

sys.path.append(parent_dir)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tb", default="1", type=int) #batch size 
    args = parser.parse_args()
    print('RWKV-Infer Test')

    pipeline = PIPELINE()
    #This is RWKV "World" model only :)
    model = RWKV_x('models/rwkv7-g0-7.2b-20250722-ctx4096.pth','nf4') #bf16, fp8, fp6, fp5
    Target_batch = 1#args.tb#16

    States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    #Input Context 
    context =  'User: あなたの趣味をおしえて\n\nAssistant:'

    

    shift_states = States.shift_states
    wkv_states = States.wkv_states


    print('////////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    print(f'wkv_states = {wkv_states.shape    }')
    print(f'shift_states = {shift_states.shape    }')


    #Apply State-tuned wkv to InitialState
    #if no state-tuned commentout
    # state_tuned_wkv = model.load_state('states/ojousama2.pth')
    # wkv_states = state_tuned_wkv.view(state_tuned_wkv.shape[0],1,state_tuned_wkv.shape[1],state_tuned_wkv.shape[2],state_tuned_wkv.shape[3]).to('cuda')


    tokens = pipeline.encode(context)
    prompts = []
    for i in range(Target_batch):
            prompts.append(torch.tensor(tokens).unsqueeze(0).to('cuda'))

    idx = torch.cat(prompts, dim=0)

    #prefill
    x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)

    if x.dim() == 2:
        x = x.view(x.shape[0],1,x.shape[1])

  
    out_tokens = [[] for _ in range(Target_batch)]
    out_last = [0 for _ in range(Target_batch)]
    output_text = ['' for _ in range(Target_batch)]
    

    FirstTime = 1

    t000 = time.perf_counter()
    min_time = 1e10
    min_time_all = 1e10
    min_time_all_single = 1e10

    maxtoken= 100

    temperature = torch.full((Target_batch,), 1.0)
    top_p = torch.full((Target_batch,), 0.3)


    SamplingSum = 0
    ForwardSum = 0
    DecodeSum = 0

    for i in range(maxtoken):
        
        t0 = time.perf_counter()
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

        x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)
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
    print(f'TargetBatch = {Target_batch} Total token/s = {round(tokensec*Target_batch,2)} Single token/s = {round(tokensec,2)}')