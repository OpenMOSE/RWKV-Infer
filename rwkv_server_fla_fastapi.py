#OpenAI API Compatible RWKV Inference Engine with Flash-Linear-Attention
#Fast API Version
#2024 OpenMOSE
#Modified Dynamic State Cache (wkv-state and shift-state)
import os
from rwkvinfer_fla import prompt_queue,LLMWorker,Prompt,PromptStatus
import pandas as pd
import asyncio
import json
import threading
import copy
import re
import torch
from functools import wraps
from argparse import ArgumentParser
import uvicorn
from starlette.concurrency import run_in_threadpool
from starlette.background import BackgroundTask

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

parser = ArgumentParser()
parser.add_argument("--localhost", default="0.0.0.0", type=str) 
parser.add_argument("--port", default=9000, type=int) 
parser.add_argument("--debug", default=False, type=bool) 
parser.add_argument("--workers", default=8, type=int) 
parser.add_argument("--dynamic_state_cache_size", default=512, type=int)  # for 14B need 16GB of PC RAM
parser.add_argument("--dynamic_state_cache_store", default='cpu', type=str) #if gpu need more vram for storing state

parser.add_argument("--admin_key1", default='123874139713915425423541', type=str) 
parser.add_argument("--admin_key2", default='d46871245412541544408014', type=str) 

args = parser.parse_args()
engine1 = LLMWorker(max_batch_size = args.workers)



#for debug 
AUTHORIZED_TOKEN = 'your_secure_token'
ModelList = [{"object":"models","id":"RWKV-14B x060 'finch'"}]
StateList = []
#model = None

DynamicStateList = []
DynamicStateList_lock = threading.Lock()

model_filename = "Anarchy-RWKV-2B-154.pth"
model_viewname = ""


def move_tensors_to_cpu(state):
    return [tensor.to('cpu') if isinstance(tensor, torch.Tensor) else tensor for tensor in state]

def move_tensors_to_gpu(state):
    return [tensor.to('cuda') if isinstance(tensor, torch.Tensor) else tensor for tensor in state]


def add_to_dynamic_state_list(text_prompt,target_state_filename,raw_state,raw_state2):
    global DynamicStateList
    global DynamicStateList_lock
    with DynamicStateList_lock:
        if len(DynamicStateList) >= args.dynamic_state_cache_size:
            DynamicStateList.pop(0)  # 先頭の要素を削除
        text_prompt = re.sub(r'\n{3,}', '\n\n', text_prompt)
        if args.debug == True:
            print(f'Added DynamicStateList a = {len(text_prompt)} ')
        if args.dynamic_state_cache_store == 'cpu':
            DynamicStateList.append({'text_prompt':text_prompt,'target_state_filename':target_state_filename,'raw_state': copy.deepcopy(move_tensors_to_cpu(raw_state)),'raw_state2': copy.deepcopy(move_tensors_to_cpu(raw_state2))})
        else:
            DynamicStateList.append({'text_prompt':text_prompt,'target_state_filename':target_state_filename,'raw_state': copy.deepcopy(raw_state),'raw_state2': copy.deepcopy(raw_state2)})


def search_dynamic_state_list(inputprompt,state_filename):
    if args.debug == True:
        print('Search Dynamic State List')
        print(f'statefile={state_filename}')
    inputprompt = re.sub(r'\n{3,}', '\n\n', inputprompt)
    raw_state = None
    raw_state2 = None
    target_state_filename = None
    text_prompt = None
    global DynamicStateList
    global DynamicStateList_lock
    with DynamicStateList_lock:
        for DynamicState in DynamicStateList:
            text_prompt = DynamicState['text_prompt']
            target_state_filename = DynamicState['target_state_filename']
            if args.debug:
                print('--------------------------------------------------------------')
                print(f'text_prompt {text_prompt[-100:]}')
                print('--------------------------------------------------------------')
                print(f'inputprompt {inputprompt[-100:]}')
                print('--------------------------------------------------------------')
            print(f'a = {len(text_prompt)} b = {len(inputprompt)} state_filename = {state_filename} target = {target_state_filename}')
            if text_prompt == inputprompt and state_filename == target_state_filename:
                raw_state = DynamicState['raw_state']
                raw_state2 = DynamicState['raw_state2']
                if args.dynamic_state_cache_store == 'cpu':
                    raw_state = move_tensors_to_gpu(raw_state)
                    raw_state2 = move_tensors_to_gpu(raw_state2)
                if args.debug == True:
                    print(f'Dynamic State Cache Found!')
                break
    if raw_state is not None:
        #print(raw_state)
        return copy.deepcopy(raw_state), copy.deepcopy(raw_state2)
    else:
        return None,None


app = FastAPI()

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

params_base = {
            "system_name": "System",
            "user_name": "User", 
            "assistant_name": "Assistant",
            "model": 'model',
            "max_tokens": 1024,
            "top_p": 0.3,
            "temperature": 1,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "penalty_decay": 0.996,
            "half_life": 400,
            "stop": ['\x00','\n\n']
        }
@app.post("/removemodel")
async def removemodel():
    #global wrappers
    global ModelList
    global engine1
    try:
        #wrappers[0].unload_model()
        engine1.UnloadModel()
        ModelList = []
        #return jsonify({"status": "success"}), 200
        return {"status": "success"}
    except Exception as e:
        print(f'error {str(e)}')
        #return jsonify({"error": str(e)}), 500
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/loadmodel")
async def loadmodel(request: Request):
    #global wrappers
    global engine1
    global ModelList
    try:
        data = await request.json()
        model_filename = data.get('model_filename')
        model_viewname = data.get('model_viewname','default model')
        model_strategy = data.get('model_strategy','None')
        #wrappers[0].load_model(model_filename,model_strategy)
        Quant = False
        if model_strategy == 'quant':
            Quant = True
        engine1.LoadModel(model_filename,Quant)
        ModelList = [{"object":"models","id":f"{model_viewname}"}]
        #return jsonify({"status": "success"}), 200
        return {"status": "success"}
    except Exception as e:
        print(f'error {str(e)}')
        #return jsonify({"error": str(e)}), 500
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/loadstatemodel")
async def loadstatemodel(request: Request):
     global StateList
     global engine1
     try:
         data = await request.json()
         state_filename = data.get('state_filename')
         state_viewname = data.get('state_viewname')
         state_tensor_wkv = engine1.model.load_state(state_filename) # if correct, have tensors :)
         if type(state_tensor_wkv) == str:
             print('State Loading Error')
             raise HTTPException(status_code=500, detail='State file is incorrect. check filename or tensor size.')

         StateList.append({"state_filename":state_filename,"state_viewname":state_viewname,'state_tensor':state_tensor_wkv})
         return {"status": "success"}
     except Exception as e:
         print(f'error {str(e)}')
         raise HTTPException(status_code=500, detail=str(e))
     
@app.post("/removestatemodel")
async def removestatemodel(request: Request):
    global StateList
    
    try:
        data = await request.json()
        StateList = None
        StateList = []
        #return jsonify({"status": "success"}), 200
        return {"status": "success"}
    except Exception as e:
         print(f'error {str(e)}')
         #return jsonify({"error": str(e)}), 500
         raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/v1/models")
@app.get("/models")
async def models():
    global StateList
    try:
        models2 = copy.deepcopy([ModelList[0]])
        i = 0
        for State in StateList:
            i = i + 1
            models2.append({"object":"models","id":f"{models2[0]['id']} {State['state_viewname']}"})
        return models2
    except Exception as e:
        print(f'error {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    
async def collect_chunks(async_gen):
    return [chunk async for chunk in async_gen]
    
def generate_response(wrapper, input_prompt, params):
    print(f"Start Processing prompt {input_prompt}")
    return wrapper.Generate(input_prompt)

async def shutdown():
    # 少し待ってからシャットダウン（レスポンスを返す時間を確保）
    global engine1
    await asyncio.sleep(1)
    uvicorn.Server.should_exit = True
    engine1.ForceExit = True
    exit()

@app.post("/fjdkgmzak9sd/sksf_appkill")
async def verify_keys(request: Request):
    data = await request.json()
    key1 = data.get("key1")
    key2 = data.get("key2")

    if key1 == args.admin_key1 and key2 == args.admin_key2:
        # キーが正しい場合、シャットダウンをスケジュール
        asyncio.create_task(shutdown())
        return {"message": "Keys verified successfully", "status_code": 200}
    else:
        raise HTTPException(status_code=400, detail="Invalid keys")



@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def rwkv_completions(request: Request):
    data = await request.json()

    if args.debug:
        print(data)

    model = data.get('model')
    state = data.get('state','')
    stream = data.get('stream', False)

    delete_ragprompt = data.get('delete_ragprompt',False)
    minimum_gen_count = data.get('minimum_gen_count',1)


    messages = data.get('messages')
    params = data.get('params', params_base)
    system_name = params.get('system_name', 'system')
    rag_name = params.get('rag_name', 'rag')

    user_name = params.get('user_name', 'user')
    assistant_name = params.get('assistant_name', 'assistant')


    max_tokens = params.get('max_tokens', 1000)  
    top_p = params.get('top_p', 0.3)
    temperature = params.get('temperature', 1.0)
    presence_penalty = params.get('presence_penalty', 0.3)
    frequency_penalty = params.get('frequency_penalty', 0.3)
    penalty_decay = params.get('penalty_decay', 0.996)
    stop = params.get('stop', ['\x00','\n\n'])
    #stop = params.get('stop', ['\n\n'])
    #stop = params.get('stop', ['\x00'])

    max_tokens = data.get('max_tokens',max_tokens)
    top_p = data.get('top_p',top_p)
    temperature = data.get('temperature',temperature)
    presence_penalty = data.get('presence_penalty',presence_penalty)
    frequency_penalty = data.get('frequency_penalty',frequency_penalty)
    penalty_decay = data.get('penalty_decay',penalty_decay)
    stop = data.get('stop',stop)

    input_prompt = ""
    input_prompt_stm = ""
    for element in messages[:-minimum_gen_count]:
        if element['role'] == 'user':
            input_prompt = input_prompt + f'{user_name}:{element["content"]}'
            input_prompt = re.sub(r'\n{3,}', '\n\n', input_prompt)
            input_prompt_stm = input_prompt_stm + f'{user_name}:{element["content"]}'
            input_prompt_stm = re.sub(r'\n{3,}', '\n\n', input_prompt_stm)
            if not input_prompt_stm.endswith('\n\n'):
                input_prompt_stm += '\n\n'
            if not input_prompt.endswith('\n\n'):
                input_prompt += '\n\n'
        elif element['role'] == 'assistant':
            input_prompt = input_prompt + f'{assistant_name}:{element["content"]}'
            input_prompt = re.sub(r'\n{3,}', '\n\n', input_prompt)
            input_prompt_stm = input_prompt_stm + f'{assistant_name}:{element["content"]}'
            input_prompt_stm = re.sub(r'\n{3,}', '\n\n', input_prompt_stm)
            if not input_prompt_stm.endswith('\n\n'):
                input_prompt_stm += '\n\n'
            if not input_prompt.endswith('\n\n'):
                input_prompt += '\n\n'
        elif element['role'] == 'system':
            input_prompt = input_prompt + f'{system_name}:{element["content"]}'
            input_prompt = re.sub(r'\n{3,}', '\n\n', input_prompt)
            input_prompt_stm = input_prompt_stm + f'{system_name}:{element["content"]}'
            input_prompt_stm = re.sub(r'\n{3,}', '\n\n', input_prompt_stm)
            if not input_prompt_stm.endswith('\n\n'):
                input_prompt_stm += '\n\n'
            if not input_prompt.endswith('\n\n'):
                input_prompt += '\n\n'
        elif element['role'] == 'rag':
            input_prompt = input_prompt + f'{system_name}:{element["content"]}'
            input_prompt = re.sub(r'\n{3,}', '\n\n', input_prompt)

            if delete_ragprompt == False:
                input_prompt_stm = input_prompt_stm + f'{system_name}:{element["content"]}'
                input_prompt_stm = re.sub(r'\n{3,}', '\n\n', input_prompt_stm)
                if not input_prompt_stm.endswith('\n\n'):
                    input_prompt_stm += '\n\n'

            if not input_prompt.endswith('\n\n'):
                input_prompt += '\n\n'

    input_prompt_b = input_prompt
    input_prompt_stm_b = input_prompt_stm

    if args.debug:
        print(f'minimum_gen_count = {minimum_gen_count} delete_ragprompt = {delete_ragprompt}')

    last_element = messages[-minimum_gen_count]
    input_prompt = ""
    input_prompt_stm = ""

    last_two_elements = messages[-minimum_gen_count:]
    for element in last_two_elements:
        if element['role'] == 'user':
            input_prompt = input_prompt + f'{user_name}:{element["content"]}'
            input_prompt = re.sub(r'\n{3,}', '\n\n', input_prompt)
            if not input_prompt.endswith('\n\n'):
                    input_prompt += '\n\n'
            input_prompt_stm = input_prompt_stm + f'{user_name}:{element["content"]}'
            input_prompt_stm = re.sub(r'\n{3,}', '\n\n', input_prompt_stm)
            if not input_prompt_stm.endswith('\n\n'):
                input_prompt_stm += '\n\n'
        elif element['role'] == 'assistant':
            input_prompt = input_prompt + f'{assistant_name}:{element["content"]}'
            input_prompt = re.sub(r'\n{3,}', '\n\n', input_prompt)
            if not input_prompt.endswith('\n\n'):
                    input_prompt += '\n\n'
            input_prompt_stm = input_prompt_stm + f'{assistant_name}:{element["content"]}'
            input_prompt_stm = re.sub(r'\n{3,}', '\n\n', input_prompt_stm)
            if not input_prompt_stm.endswith('\n\n'):
                input_prompt_stm += '\n\n'
        elif element['role'] == 'system':
            input_prompt = input_prompt + f'{system_name}:{element["content"]}'
            input_prompt = re.sub(r'\n{3,}', '\n\n', input_prompt)
            if not input_prompt.endswith('\n\n'):
                    input_prompt += '\n\n'
            input_prompt_stm = input_prompt_stm + f'{system_name}:{element["content"]}'
            input_prompt_stm = re.sub(r'\n{3,}', '\n\n', input_prompt_stm)
            if not input_prompt_stm.endswith('\n\n'):
                input_prompt_stm += '\n\n'

        elif element['role'] == 'rag':
            input_prompt = input_prompt + f'{system_name}:{element["content"]}'
            input_prompt = re.sub(r'\n{3,}', '\n\n', input_prompt)

            if delete_ragprompt == False:
                input_prompt_stm = input_prompt_stm + f'{system_name}:{element["content"]}'
                input_prompt_stm = re.sub(r'\n{3,}', '\n\n', input_prompt_stm)
                if not input_prompt_stm.endswith('\n\n'):
                    input_prompt_stm += '\n\n'

            if not input_prompt.endswith('\n\n'):
                input_prompt += '\n\n'

    input_prompt = input_prompt + f'{assistant_name}:'
    input_prompt_stm = input_prompt_stm + f'{assistant_name}:'

    models2 = [ModelList[0]]
    models2[0]['filename'] = ""
    models2[0]['state_tensor'] = None
    for State in StateList:
        models2.append({"object":"models","id":f"{ModelList[0]['id']} {State['state_viewname']}","filename":State['state_filename'],"state_tensor":State['state_tensor']})

    target_state_filename = ''
    target_state_tensor_wkv = None

    for modelname in models2:
        if modelname['id'] == model:
            target_state_filename = modelname['filename']
            target_state_tensor_wkv = modelname['state_tensor']

    QueryDatas = Prompt()

    wkv_state,shift_state = search_dynamic_state_list(input_prompt_stm_b,target_state_filename)
    StateCacheMode = False
    if wkv_state is not None and shift_state is not None:
        if args.debug:
            print('resume state detected.')
        QueryDatas.base_state_tuned = target_state_filename
        QueryDatas.use_exist_state_wkv = copy.deepcopy(wkv_state)
        QueryDatas.use_exist_state_shift = copy.deepcopy(shift_state)
        StateCacheMode = True
    else:
        if args.debug:
            print('plane state')
        if target_state_tensor_wkv is not None:
            QueryDatas.use_exist_state_wkv = copy.deepcopy(target_state_tensor_wkv)

        input_prompt = input_prompt_b + input_prompt
        input_prompt_stm = input_prompt_stm_b + input_prompt_stm 

    QueryDatas.prompts = input_prompt
    QueryDatas.maxtokens = max_tokens
    QueryDatas.temperature = temperature
    QueryDatas.top_p = top_p
    QueryDatas.endtoken = stop

    def handle_stream_disconnection(f):
        @wraps(f)
        def decorated_function(*argss, **kwargss):
            global args
            try:
                if args.debug:
                    print('handle_stream_disconnection start')
                return f(*argss, **kwargss)
            except GeneratorExit:
                if args.debug:
                    print("Stream disconnected")
            except Exception as e:
                if args.debug:
                    print("Stream disconnected")

        return decorated_function

    @handle_stream_disconnection
    async def sync_stream():
        totaltext = ''
        wkv_state = None
        shift_state = None
        try:
            async for response_chunk, d1, d2 in engine1.FLAGenerate(QueryDatas):
                    if d1 is not None:
                        wkv_state = d1
                    if d2 is not None:
                        shift_state = d2
                    totaltext = totaltext+response_chunk
                    response_data = [
                        {
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": response_chunk
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                    ]
                    yield f'data: {json.dumps(response_data[0])}\n\n'
        except StopAsyncIteration:
            if args.debug:
                print('Stop Async Iteration detected.')
            if StateCacheMode:
                output_prompt = input_prompt_stm_b + input_prompt_stm + totaltext
            else:
                output_prompt = input_prompt_stm + totaltext
            add_to_dynamic_state_list(output_prompt,target_state_filename,wkv_state,shift_state)
            response_data = [
                "[DONE]"
            ]
            yield f'data: {json.dumps(response_data)}\n\n'
            pass
        except Exception as e:
            if args.debug:
                print(f"An error occurred during generation: {str(e)}")
        finally:
            if args.debug:
                print('Stop Async Iteration detected.')
            if StateCacheMode:
                output_prompt = input_prompt_stm_b + input_prompt_stm + totaltext
            else:
                output_prompt = input_prompt_stm + totaltext
            if wkv_state is not None and shift_state is not None:
                add_to_dynamic_state_list(output_prompt,target_state_filename,wkv_state,shift_state)
            response_data = [
                "[DONE]"
            ]
            yield f'data: {json.dumps(response_data)}\n\n'
            pass

    if stream:
        generator = sync_stream()
        return StreamingResponse(generator, media_type='text/event-stream',background=BackgroundTask(sync_stream))
    else:
        response_chunk = []
        if args.debug:
            print('Non Stream Start Generate')
        wkv_state = None
        shift_state = None
        async for item, d1, d2 in engine1.FLAGenerate(QueryDatas):
            response_chunk.append(item)
            if d1 is not None:
                wkv_state = d1
            if d2 is not None:
                shift_state = d2
        response = ''
        for chunk in response_chunk:
            response = response + chunk

        OutputText = response

        if args.debug:
            print(f'Non Stream: {OutputText}')

        if StateCacheMode:
            output_prompt = input_prompt_stm_b + input_prompt_stm + OutputText
        else:
            output_prompt = input_prompt_stm + OutputText

        if wkv_state is not None and shift_state is not None:
                add_to_dynamic_state_list(output_prompt,target_state_filename,wkv_state,shift_state)

        jsonResponse = {
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        (
                            {
                                "message": {
                                    "role": assistant_name,
                                    "content": OutputText,
                                },
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        )
                    ],
                }
        return (jsonResponse)

async def run_uvicorn(host, port):
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    await asyncio.gather(
        engine1.RunLLM(),
        engine1.QueueManage(),
        run_uvicorn(host=args.localhost, port=args.port)
    )

if __name__ == '__main__':
    print('RWKV Infer v0.0.1 with Flash-Linear-Attention')
    print('wrapper code by OpenMOSE')
    asyncio.run(main())
    
