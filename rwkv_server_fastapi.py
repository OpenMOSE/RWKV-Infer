#OpenAI API Compatible RWKV Inference Engine
#Fast API Version
#2024 OpenMOSE
import os
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
from rwkvinfer import RWKVWrapper
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
parser.add_argument("--port", default=8000, type=int) 
parser.add_argument("--debug", default=True, type=bool) 
parser.add_argument("--workers", default=16, type=int) 
parser.add_argument("--dynamic_state_cache_size", default=512, type=int)  # for 14B need 16GB of PC RAM
parser.add_argument("--dynamic_state_cache_store", default='cpu', type=str) #if gpu need more vram for storing state

args = parser.parse_args()

#for debug 
AUTHORIZED_TOKEN = 'your_secure_token'
ModelList = [{"object":"models","id":"RWKV-14B x060 'finch'"}]
StateList = []
model = None
#pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

DynamicStateList = []
DynamicStateList_lock = threading.Lock()

worker_lock = threading.Lock()

wrappers = [RWKVWrapper(debug=args.debug) for _ in range(args.workers)]

model_filename = "Anarchy-RWKV-2B-154.pth"
model_viewname = ""


def move_tensors_to_cpu(state):
    return [tensor.to('cpu') if isinstance(tensor, torch.Tensor) else tensor for tensor in state]

def move_tensors_to_gpu(state):
    return [tensor.to('cuda') if isinstance(tensor, torch.Tensor) else tensor for tensor in state]


def add_to_dynamic_state_list(text_prompt,target_state_filename,raw_state):
    global DynamicStateList
    global DynamicStateList_lock
    with DynamicStateList_lock:
        if len(DynamicStateList) >= args.dynamic_state_cache_size:
            DynamicStateList.pop(0)  # 先頭の要素を削除
        text_prompt = re.sub(r'\n{3,}', '\n\n', text_prompt)
        if args.debug == True:
            print(f'Added DynamicStateList a = {len(text_prompt)} ')
        if args.dynamic_state_cache_store == 'cpu':
            DynamicStateList.append({'text_prompt':text_prompt,'target_state_filename':target_state_filename,'raw_state': copy.deepcopy(move_tensors_to_cpu(raw_state))})
        else:
            DynamicStateList.append({'text_prompt':text_prompt,'target_state_filename':target_state_filename,'raw_state': copy.deepcopy(raw_state)})


def search_dynamic_state_list(inputprompt,state_filename):
    if args.debug == True:
        print('Search Dynamic State List')
        print(f'statefile={state_filename}')
    inputprompt = re.sub(r'\n{3,}', '\n\n', inputprompt)
    raw_state = None
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
                if args.dynamic_state_cache_store == 'cpu':
                    raw_state = move_tensors_to_gpu(raw_state)
                if args.debug == True:
                    print(f'Dynamic State Cache Found!')
                break
    if raw_state is not None:
        #print(raw_state)
        return copy.deepcopy(raw_state)
    else:
        return None


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
#@app.route('/removemodel', methods=['POST'])
@app.post("/removemodel")
async def removemodel():
    global wrappers
    global ModelList
    try:
        wrappers[0].unload_model()
        ModelList = []
        #return jsonify({"status": "success"}), 200
        return {"status": "success"}
    except Exception as e:
        print(f'error {str(e)}')
        #return jsonify({"error": str(e)}), 500
        raise HTTPException(status_code=500, detail=str(e))


#@app.route('/loadmodel', methods=['POST'])
@app.post("/loadmodel")
async def loadmodel(request: Request):
    global wrappers
    global ModelList
    #raw_body = await request.body()
    ## JSONとしてパース
    #try:
    #    data = json.loads(raw_body)
    #except json.JSONDecodeError:
    #    return {"error": "Invalid JSON"}
    try:
        data = await request.json()
        model_filename = data.get('model_filename')
        model_viewname = data.get('model_viewname','default model')
        model_strategy = data.get('model_strategy','cuda fp16')
        wrappers[0].load_model(model_filename,model_strategy)
        ModelList = [{"object":"models","id":f"{model_viewname}"}]
        #return jsonify({"status": "success"}), 200
        return {"status": "success"}
    except Exception as e:
        print(f'error {str(e)}')
        #return jsonify({"error": str(e)}), 500
        raise HTTPException(status_code=500, detail=str(e))
    
    
#@app.route('/loadstatemodel', methods=['POST'])
@app.post("/loadstatemodel")
async def loadstatemodel(request: Request):
     global StateList
     try:
         data = await request.json()
         state_filename = data.get('state_filename')
         state_viewname = data.get('state_viewname')
         #wrappers[0].load_model(model_filename,model_viewname)
         StateList.append({"state_filename":state_filename,"state_viewname":state_viewname})
         #return jsonify({"status": "success"}), 200
         return {"status": "success"}
     except Exception as e:
         print(f'error {str(e)}')
         #return jsonify({"error": str(e)}), 500
         raise HTTPException(status_code=500, detail=str(e))
     
#@app.route('/removestatemodel', methods=['POST'])
@app.post("/removestatemodel")
async def removestatemodel(request: Request):
    global StateList
    
    try:
        data = await request.json()
        StateList = []
        #return jsonify({"status": "success"}), 200
        return {"status": "success"}
    except Exception as e:
         print(f'error {str(e)}')
         #return jsonify({"error": str(e)}), 500
         raise HTTPException(status_code=500, detail=str(e))
    

#@app.route('/v1/models', methods=['GET'])
#@app.route('/models', methods=['GET'])
@app.get("/v1/models")
@app.get("/models")
def models():
    try:
        models2 = [ModelList[0]]
        i = 0
        for State in StateList:
            i = i + 1
            print('before')
            print(models2)
            models2.append({"object":"models","id":f"{ModelList[0]['id']} {State['state_viewname']}"})
            print(i)
            print('after')
            print(models2)
        #print(models2)
        #return jsonify(models2)
        return models2
    except Exception as e:
        print(f'error {str(e)}')
        #return jsonify({"error": str(e)}), 500
        raise HTTPException(status_code=500, detail=str(e))
    
async def collect_chunks(async_gen):
    return [chunk async for chunk in async_gen]
    
def generate_response(wrapper, input_prompt, params):
    print(f"Start Processing prompt {input_prompt}")
    return wrapper.Generate(input_prompt)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def rwkv_completions(request: Request):
    data = await request.json()

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

    # Choose boring worker
    selected_wrapper = None
    with worker_lock:
        for wrapper in wrappers:
            if not wrapper.is_busy():  # is_busy() means have work
                selected_wrapper = wrapper
                break

    if selected_wrapper is None:
        #return jsonify({"error": "All workers are busy"}), 503
        raise HTTPException(status_code=503, detail=str('All workers are busy'))

    selected_wrapper.set_busy(True)

    models2 = [ModelList[0]]
    models2[0]['filename'] = ""
    for State in StateList:
        models2.append({"object":"models","id":f"{ModelList[0]['id']} {State['state_viewname']}","filename":State['state_filename']})

    #selected_wrapper.load_state("")

    target_state_filename = ''

    for modelname in models2:
        if modelname['id'] == model:
            target_state_filename = modelname['filename']
            #selected_wrapper.load_state(modelname['filename'])

    if state != '':
        selected_wrapper.load_state(state)
        target_state_filename = selected_wrapper.model_current_statetuned_filename

    #statecache = search_dynamic_state_list(input_prompt_b,selected_wrapper.model_current_statetuned_filename)
    statecache = search_dynamic_state_list(input_prompt_stm_b,target_state_filename)
    StateCacheMode = False
    if statecache is not None:
        print('resume state detected.')
        selected_wrapper.model_state = copy.deepcopy(statecache)
        StateCacheMode = True
    else:
        print('plane state')
        selected_wrapper.load_state(target_state_filename)
        input_prompt = input_prompt_b + input_prompt
        input_prompt_stm = input_prompt_stm_b + input_prompt_stm

    #async def generate_and_stream():
    #    #global selected_wrapper
    #    i=0
    #    try:
    #        async for response_chunk in selected_wrapper.Generate(input_prompt, temperature=temperature, top_p=top_p,alpha_presence=presence_penalty,alpha_frequency=frequency_penalty, penalty_decay=penalty_decay, MAX_COUNT=max_tokens,STOP=stop):
    #            yield response_chunk
    #    finally:
    #        i=1
    #        #selected_wrapper.set_busy(False)

    #async def generate_and_stream(selected_wrapper, input_prompt, temperature, top_p, presence_penalty, frequency_penalty, penalty_decay, max_tokens, stop):
    #async def generate_and_stream():
    #    try:
    #        async for response_chunk in selected_wrapper.Generate(
    #            input_prompt, 
    #            temperature=temperature, 
    #            top_p=top_p,
    #            alpha_presence=presence_penalty,
    #            alpha_frequency=frequency_penalty, 
    #            penalty_decay=penalty_decay, 
    #            MAX_COUNT=max_tokens,
    #            STOP=stop
    #        ):
    #            yield response_chunk
    #    finally:
    #        # クリーンアップ処理がある場合はここに記述
    #        pass

    

    def handle_stream_disconnection(f):
        nonlocal selected_wrapper
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                print('handle_stream_disconnection start')
                return f(*args, **kwargs)
            except GeneratorExit:
                print("Stream disconnected")
                selected_wrapper.Stop = True
                selected_wrapper.set_busy(False)
            except Exception as e:
                print("Stream disconnected")
                selected_wrapper.Stop = True
                selected_wrapper.set_busy(False)
    #            print(f"An error occurred: {str(e)}")
        return decorated_function
    #def handle_stream_disconnection(f):
    #    @wraps(f)
    #    async def decorated_function(*args, **kwargs):
    #        try:
    #            return await f(*args, **kwargs)
    #        except GeneratorExit:
    #            print("Stream disconnected")
    #            selected_wrapper = kwargs.get('selected_wrapper')
    #            if selected_wrapper:
    #                await run_in_threadpool(lambda: setattr(selected_wrapper, 'Stop', True))
    #                await run_in_threadpool(selected_wrapper.set_busy, False)
    #        except Exception as e:
    #            print(f"An error occurred: {str(e)}")
    #            # 適切なエラーハンドリングを行う
    #    return decorated_function
    #async def cleanup(generator):
    #    nonlocal selected_wrapper
    #    try:
    #        await generator.close()
    #    except Exception as e:
    #        print(f"Error during cleanup: {e}")
    #        print("Stream disconnected")
    #        selected_wrapper.Stop = True
    #        selected_wrapper.set_busy(False)
    @handle_stream_disconnection
    async def sync_stream():
        totaltext = ''
        nonlocal selected_wrapper
        try:
            async for response_chunk in selected_wrapper.Generate(
                    input_prompt, 
                    temperature=temperature, 
                    top_p=top_p,
                    alpha_presence=presence_penalty,
                    alpha_frequency=frequency_penalty, 
                    penalty_decay=penalty_decay, 
                    MAX_COUNT=max_tokens,
                    STOP=stop
                ):
                    #yield response_chunk
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
            print('Stop Async Iteration detected.')
            if StateCacheMode:
                output_prompt = input_prompt_stm_b + input_prompt_stm + totaltext
            else:
                output_prompt = input_prompt_stm + totaltext
            add_to_dynamic_state_list(output_prompt,selected_wrapper.model_current_statetuned_filename,selected_wrapper.model_state)
            #print('OutputPrompt-----------------------------------------------------------------------')
            #print(output_prompt[-100:])
            #print('-----------------------------------------------------------------------------------')
            selected_wrapper.Stop = True
            selected_wrapper.set_busy(False)
            response_data = [
                "[DONE]"
            ]
            yield f'data: {json.dumps(response_data)}\n\n'
            pass
        except Exception as e:
            print(f"An error occurred during generation: {str(e)}")
        finally:
            print('Stop Async Iteration detected.')
            if StateCacheMode:
                output_prompt = input_prompt_stm_b + input_prompt_stm + totaltext
            else:
                output_prompt = input_prompt_stm + totaltext
            add_to_dynamic_state_list(output_prompt,selected_wrapper.model_current_statetuned_filename,selected_wrapper.model_state)
            #print('OutputPrompt-----------------------------------------------------------------------')
            #print(output_prompt[-100:])
            #print('-----------------------------------------------------------------------------------')
            selected_wrapper.Stop = True
            selected_wrapper.set_busy(False)
            response_data = [
                "[DONE]"
            ]
            yield f'data: {json.dumps(response_data)}\n\n'
            pass

    if stream:
        #return Response(stream_with_context(sync_stream()), content_type='text/event-stream')
        generator = sync_stream()
        return StreamingResponse(generator, media_type='text/event-stream',background=BackgroundTask(sync_stream))
    else:
        response_chunk = []
        print('Non Stream Start Generate')
        async for item in selected_wrapper.Generate(input_prompt, temperature=temperature, top_p=top_p,alpha_presence=presence_penalty,alpha_frequency=frequency_penalty, penalty_decay=penalty_decay, MAX_COUNT=max_tokens,STOP=stop):
            response_chunk.append(item)
        response = ''
        for chunk in response_chunk:
            response = response + chunk
        #response = asyncio.run(collect_chunks(selected_wrapper.Generate(input_prompt, temperature=temperature, top_p=top_p,alpha_presence=presence_penalty,alpha_frequency=frequency_penalty, penalty_decay=penalty_decay, MAX_COUNT=max_tokens,STOP=stop)))
        #response = (collect_chunks(selected_wrapper.Generate(input_prompt, temperature=temperature, top_p=top_p,alpha_presence=presence_penalty,alpha_frequency=frequency_penalty, penalty_decay=penalty_decay, MAX_COUNT=max_tokens,STOP=stop)))
        
        #OutputText = ''
        #for chunk in response:
        #    OutputText = OutputText+chunk
        OutputText = response

        print(f'Non Stream: {OutputText}')

        if StateCacheMode:
            output_prompt = input_prompt_stm_b + input_prompt_stm + OutputText
        else:
            output_prompt = input_prompt_stm + OutputText
        #print('-----------------------------------------------------------------------------------')
        #print(output_prompt)
        #print('-----------------------------------------------------------------------------------')
        add_to_dynamic_state_list(output_prompt,selected_wrapper.model_current_statetuned_filename,selected_wrapper.model_state)
        selected_wrapper.set_busy(False)
        jsonResponse = {
                    "object": "chat.completion",
                    # "response": response,
                    "model": model,
                    # "usage": {
                    #     "prompt_tokens": prompt_tokens,
                    #     "completion_tokens": completion_tokens,
                    #     "total_tokens": prompt_tokens + completion_tokens,
                    # },
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
        #print(f'Not Stream Output:{response}')
        return (jsonResponse)
        #return jsonify({"response": ''.join(response)})



if __name__ == '__main__':
    #app.run(debug=False, host=args.localhost, port=args.port)
    uvicorn.run(app, host=args.localhost, port=args.port)
