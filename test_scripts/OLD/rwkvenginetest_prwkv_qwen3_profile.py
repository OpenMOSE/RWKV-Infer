import time
import sys
import os
# 1階層上のディレクトリのパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch.profiler
import torch
from rwkvengine.rwkvcore import RWKV_x, PIPELINE
import time
import copy
import torch.nn.functional as F
import numpy as np
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tb", default="1", type=int) #batch size 
    parser.add_argument("--fully_fused", default="1", type=int) #batch size 
    args = parser.parse_args()
    print('RWKV x070Core with FLA Test')

    pipeline = PIPELINE("qwen3")

    text = "The Large Language Model is "
    # エンコード：通常は input_ids というキーでID列が得られます
    encoded = pipeline.encode(text)
    #print("Encoded IDs:", encoded)

    # デコード
   # decoded = pipeline.decode(text)
    #print("Decoded text:", decoded)

    messages = [
        #{'role':'system', 'content':"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.Your knowledge base was last updated on 2023-10-01. The current date is 2025-01-30.When you're not sure about some information, you say that you don't have the information and don't make up anything."},
        {'role':'system', 'content':"You are Mistral Small 3"},
        {'role':'assistant', 'content':"Gooday! how can i help you?"},
        {'role':'user', 'content':"Hi! Tell me what is Large Language Model?"},
    ]

    messages = [
        {'role':'system', 'content':"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.Your knowledge base was last updated on 2023-10-01. The current date is 2025-01-30.When you're not sure about some information, you say that you don't have the information and don't make up anything."},
        #{'role':'system', 'content':"You are Mistral Small 3"},
        {'role':'assistant', 'content':"hi! how can i help you?"},
        {'role':'user', 'content':"What is Large Language Model?"},
    ]


    context = pipeline.generate_prompt_from_config(pipeline.modeltemplate,messages,True)

    #context = " "


    #exit()

    model = RWKV_x('models/PRWKV-7-Qwen3-14B-Preview-stage2final-ctx3072.pth','fp8',
                   adapter_model='/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/prwkvtest/rwkv-0.pth',
                   adapter_mode='',
                   fully_fusedrecurrent=args.fully_fused)

    Target_batch = args.tb
    target_temp = 1.0
    target_topp = 0.3
    while True:

        while True:
            textinput = input(f'Input temp={target_temp} topp={target_topp}:')
            if textinput == "temperature":
                target_temp = input('temperature:')
            elif textinput == "topp":
                target_topp = input('topp:') 
            else:
                break

        messages = [
        #{'role':'system', 'content':"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.Your knowledge base was last updated on 2023-10-01. The current date is 2025-01-30.When you're not sure about some information, you say that you don't have the information and don't make up anything."},
        {'role':'system', 'content':"You are helpful assistant."},
        {'role':'user', 'content':textinput},
        ]


        context = pipeline.generate_prompt_from_config(pipeline.modeltemplate,messages,True)

        States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)
        States2 = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

        

        print(context)
        #exit()
    

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
        print(f'token = {tokens0}')
        decodedtest = pipeline.decode(tokens0)

        print(f'decoded = {decodedtest}')
        #exit()

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
        x, shift_states, wkv_states = model.forward(copy.deepcopy(idx), shift_states, wkv_states,KernelMode=1) #FLA
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

        maxtoken= 120

        temperature = torch.full((Target_batch,), float(target_temp))
        top_p = torch.full((Target_batch,), float(target_topp))


        SamplingSum = 0
        ForwardSum = 0
        DecodeSum = 0

        occurrence = [{},{},{},{}]

        ProfileMode = 0

        

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir="./log")

        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA  # GPUも記録
        #     ],
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),  # GPUにはwarmup推奨
        #     on_trace_ready=tensorboard_trace_handler("./log"),  # ★ TensorBoard向け
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:

        for i in range(maxtoken):
            
            t0 = time.perf_counter()

            #x = x.view(-1,1)
            if x.dim() == 2:
                x = x.view(x.shape[0],1,x.shape[1])
            x[:, -1, 0] -= 1e10

            for j in range(Target_batch):
                for n in occurrence[j]:
                    x[j][-1][n] -= 0.2 + occurrence[j][n] * 0.3

            otokens = pipeline.improved_nucleus_sampling_multi_static_topk(x[:, -1], temperature=temperature, top_p=top_p,top_k=50).tolist()

            for j in range(Target_batch):
                for xxx in occurrence[j]:
                        occurrence[j][xxx] *= 0.996
                tk = otokens[j]
                occurrence[j][tk] = 1 + (occurrence[j][tk] if tk in occurrence[j] else 0)

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
                # try:
                #     #print(out_tokens[j])
                #     # すべてのトークンをデコード
                #     new_tmp = pipeline.decode(out_tokens[j])

                #     # 差分を求めて新しい部分のみ取得
                #     diff_text = new_tmp[len(output_text[j]):]

                #     # 新しくデコードされた部分があれば出力
                #     if diff_text and ("\ufffd" not in diff_text) and (not diff_text.endswith("\n")):
                #         print(diff_text, end="", flush=True)
                #         output_text[j] =new_tmp#+= diff_text  # 出力したテキストを保存
                #         out_last[j] = i + 1  # 出力済みのインデックス更新
                # except Exception as e:
                #     print("Decode error:", e)
            t2 = time.perf_counter()

            x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states,one_mode=True)
            #prof.step()
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
        #writer.add_scalar("profiling/test", 1, 0)  # これで .tfevents を強制出力
        #writer.close()


        t001 = time.perf_counter()

        print(output_text)
        print('RWKV-Infer FLA Refactor')

        tokensec = maxtoken / (t001-t000)
        print(f'totaltime = {totaltime} totaltoken = {prefilltoken_total} Prefill {prefill_time}t/s')
        print(f'TargetBatch = {Target_batch} Total token/s = {round(tokensec*Target_batch,2)} Single token/s = {round(tokensec,2)}')

        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
