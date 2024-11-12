import torch
from rwkvengine.rwkvcore import RWKV_6, PIPELINE
import time


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tb", default="1", type=int) #batch size 
    args = parser.parse_args()
    print('RWKV x060Core with FLA Test')

    pipeline = PIPELINE()
    model = RWKV_6('models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth','fp5')
    Target_batch = args.tb#16

    States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    context =  'User: Tell me advantage of C++.\n\nAssistant:'
    context2 = 'User: Tell me advantage of C++.\n\nAssistant:'

    #model.load_state('states/ojousama2.pth')

    shift_states = States.shift_states
    wkv_states = States.wkv_states

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

    #wkv_states[0] = model.model_current_statetuned
    #for i in range(model.n_layer):
    #    wkv_states[i][0] = model.model_current_statetuned[i*3 + 1]
    #exit()

    tokens = pipeline.encode(context)
    tokens2 = pipeline.encode(context2)
    prompts = []
    for i in range(Target_batch):
        if i%2 == 0:
            prompts.append(torch.tensor(tokens).unsqueeze(0).to('cuda'))
        else:
            prompts.append(torch.tensor(tokens2).unsqueeze(0).to('cuda'))


    idx = torch.cat(prompts, dim=0)

    print(f'{idx.shape}')

    x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)

    print(f'x = {x}')

    print(context)
    out_tokens = [[] for _ in range(Target_batch)]
    out_last = [0 for _ in range(Target_batch)]
    output_text = ['' for _ in range(Target_batch)]
    

    FirstTime = 1

    t000 = time.perf_counter()
    min_time = 1e10
    min_time_all = 1e10
    min_time_all_single = 1e10

    maxtoken= 1000

    temperature = torch.full((Target_batch,), 1.0)
    top_p = torch.full((Target_batch,), 0.7)


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
                        #print(tmp,end="", flush=True)
                        output_text[j] = output_text[j] + tmp
                        out_last[j] = i + 1
            except:
                pass
        t2 = time.perf_counter()

        x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)
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