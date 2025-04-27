import torch
from rwkvengine.rwkvcore import RWKV_x, PIPELINE
import time
from torch.nn import functional as F


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tb", default="2", type=int) #batch size 
    args = parser.parse_args()
    print('RWKV x070Core Test')

    pipeline = PIPELINE(mode='world')
    model = RWKV_x('models/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth','fp16')
    Target_batch = args.tb#16

    

    States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)
    #print(States)
    #exit()

    context =  'User: What is Artificial Intelligence?\n\nAssistant:'
    context2 = 'User: What is Artificial Intelligence?\n\nAssistant:'

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

    #print(f'wkv_states = {wkv_states.shape    }')
    #print(f'shift_states = {shift_states.shape    }')

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
            #prompts.append(tokens)
        else:
            prompts.append(torch.tensor(tokens2).unsqueeze(0).to('cuda'))
            #prompts.append(tokens2)


    idx = torch.cat(prompts, dim=0)
    #idx = torch.tensor(prompts,dtype=torch.int64)

    print(prompts)
   

    print(f'{idx.shape}')

    x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)
    print(x)

    # init_out = x
    # probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
    # print(init_out.shape)
    # #print(f'\n{prompt}')

    # print(f'probs shape = {probs.shape}')

    # _, indices = torch.topk(probs, 10,dim=-1) # print top-10 possibilities

    # # バッチごとのインデックス表示
    # for batch_idx in range(indices.size(0)):
    #     print(f"Batch {batch_idx}:")
    #     indice = indices[batch_idx]

    #     # 各シーケンス内の上位トークン表示
    #     for topk_idx in range(indice.size(0)):  # 上位10件を処理
    #         token_id = indice[topk_idx].item()
    #         token = pipeline.decode([token_id])
    #         token_prob = probs[batch_idx,token_id].item()
    #         print(f"{token} [probability {token_prob:.2%}]")

    #print(f'x = {x}')

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

    maxtoken= 1000

    temperature = torch.full((Target_batch,), 1.0)
    top_p = torch.full((Target_batch,), 0.3)


    SamplingSum = 0
    ForwardSum = 0
    DecodeSum = 0

    for i in range(maxtoken):
        
        t0 = time.perf_counter()
        #print(x.shape)
        if x.dim() == 2:
            x[:,0] -= 1e10
        else:
            x[:, -1, 0] -= 1e10
        if x.dim() == 2:
            otokens = pipeline.improved_nucleus_sampling_multi_static(x, temperature=temperature, top_p=top_p).tolist()
        else:
            otokens = pipeline.improved_nucleus_sampling_multi_static(x[:, -1], temperature=temperature, top_p=top_p).tolist()

        tokens = []
        for j in range(Target_batch):
            tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0).to('cuda'))
            #tokens.append(otokens[j])

        idx = torch.cat(tokens, dim=0).reshape(Target_batch)
        
        #print(f'idx shape = {idx.shape}')
        #idx = torch.tensor(tokens,dtype=torch.int64).view(Target_batch)

        #print(idx)

        #print(f'idx = {idx.shape}')

        t1 = time.perf_counter()
        for j in range(Target_batch):
            out_tokens[j] += [otokens[j]]
            try:
                tmp = pipeline.decode(out_tokens[j][out_last[j]:])
                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                        if j == Target_batch - 1:
                            print(tmp,end="", flush=True)
                        output_text[j] = output_text[j] + tmp
                        out_last[j] = i + 1
            except:
                pass
        t2 = time.perf_counter()

        x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states ,one_mode=True)
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
    print('RWKV-Infer x070 Test')

    tokensec = maxtoken / (t001-t000)
    print(f'TargetBatch = {Target_batch} Total token/s = {round(tokensec*Target_batch,2)} Single token/s = {round(tokensec,2)}')