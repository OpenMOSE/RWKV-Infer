#OpenAI API Compatible RWKV Inference Engine
#2024 OpenMOSE
import os
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
from rwkvinfer import RWKVWrapper

from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS

import pandas as pd
from flask import Flask, Response, request
from rwkv.utils import PIPELINE
import asyncio
import json
import threading
import copy
import re

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--localhost", default="0.0.0.0", type=str) 
parser.add_argument("--port", default=8000, type=int) 
parser.add_argument("--debug", default=True, type=bool) 
parser.add_argument("--workers", default=16, type=int) 
parser.add_argument("--dynamic_state_cache_size", default=64, type=int) 

args = parser.parse_args()

#for debug 
AUTHORIZED_TOKEN = 'your_secure_token'
ModelList = [{"object":"models","id":"RWKV-14B x060 'finch'"}]
StateList = []
model = None
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

DynamicStateList = []
DynamicStateList_lock = threading.Lock()

worker_lock = threading.Lock()

wrappers = [RWKVWrapper(debug=args.debug) for _ in range(args.workers)]

model_filename = "Anarchy-RWKV-2B-154.pth"
model_viewname = ""

def add_to_dynamic_state_list(text_prompt,target_state_filename,raw_state):
    global DynamicStateList
    global DynamicStateList_lock
    with DynamicStateList_lock:
        if len(DynamicStateList) >= args.dynamic_state_cache_size:
            DynamicStateList.pop(0)  # 先頭の要素を削除
        text_prompt = re.sub(r'\n{3,}', '\n\n', text_prompt)
        if args.debug == True:
            print(f'Added DynamicStateList a = {len(text_prompt)} ')
        DynamicStateList.append({'text_prompt':text_prompt,'target_state_filename':target_state_filename,'raw_state': copy.deepcopy(raw_state)})  # 新しい要素を追加
        #print({'text_prompt':text_prompt,'target_state_filename':target_state_filename})#,'raw_state': copy.deepcopy(raw_state)})

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
                if args.debug == True:
                    print(f'Dynamic State Cache Found!')
                break
    if raw_state is not None:
        #print(raw_state)
        return copy.deepcopy(raw_state)
    else:
        return None


app = Flask(__name__)
CORS(app)

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
@app.route('/removemodel', methods=['POST'])
def removemodel():
    global wrappers
    global ModelList
    try:
        wrappers[0].unload_model()
        ModelList = []
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f'error {str(e)}')
        return jsonify({"error": str(e)}), 500


@app.route('/loadmodel', methods=['POST'])
def loadmodel():
    global wrappers
    global ModelList
    try:
        data = request.json
        model_filename = data.get('model_filename')
        model_viewname = data.get('model_viewname','default model')
        model_strategy = data.get('model_strategy','cuda fp16')
        wrappers[0].load_model(model_filename,model_strategy)
        ModelList = [{"object":"models","id":f"{model_viewname}"}]
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f'error {str(e)}')
        return jsonify({"error": str(e)}), 500
    
@app.route('/loadstatemodel', methods=['POST'])
def loadstatemodel():
     global StateList
     try:
         data = request.json
         state_filename = data.get('state_filename')
         state_viewname = data.get('state_viewname')
         #wrappers[0].load_model(model_filename,model_viewname)
         StateList.append({"state_filename":state_filename,"state_viewname":state_viewname})
         return jsonify({"status": "success"}), 200
     except Exception as e:
         print(f'error {str(e)}')
         return jsonify({"error": str(e)}), 500
     
@app.route('/removestatemodel', methods=['POST'])
def removestatemodel():
    global StateList
    
    try:
        data = request.json
        StateList = []
        return jsonify({"status": "success"}), 200
    except Exception as e:
         print(f'error {str(e)}')
         return jsonify({"error": str(e)}), 500
    





@app.route('/v1/models', methods=['GET'])
@app.route('/models', methods=['GET'])
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
        return jsonify(models2)
    except Exception as e:
        print(f'error {str(e)}')
        return jsonify({"error": str(e)}), 500
    
async def collect_chunks(async_gen):
    return [chunk async for chunk in async_gen]
    
def generate_response(wrapper, input_prompt, params):
    print(f"Start Processing prompt {input_prompt}")
    return wrapper.Generate(input_prompt)
    
@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def rwkv_completions():
    data = request.json

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
        return jsonify({"error": "All workers are busy"}), 503

    selected_wrapper.set_busy(True)

    models2 = [ModelList[0]]
    models2[0]['filename'] = ""
    for State in StateList:
        models2.append({"object":"models","id":f"{ModelList[0]['id']} {State['state_viewname']}","filename":State['state_filename']})

    #selected_wrapper.load_state("")

    for modelname in models2:
        if modelname['id'] == model:
            selected_wrapper.load_state(modelname['filename'])

    if state != '':
        selected_wrapper.load_state(state)

    #statecache = search_dynamic_state_list(input_prompt_b,selected_wrapper.model_current_statetuned_filename)
    statecache = search_dynamic_state_list(input_prompt_stm_b,selected_wrapper.model_current_statetuned_filename)
    StateCacheMode = False
    if statecache is not None:
        print('resume state detected.')
        selected_wrapper.model_state = copy.deepcopy(statecache)
        StateCacheMode = True
    else:
        print('plane state')
        input_prompt = input_prompt_b + input_prompt
        input_prompt_stm = input_prompt_stm_b + input_prompt_stm

    async def generate_and_stream():
        #global selected_wrapper
        i=0
        try:
            async for response_chunk in selected_wrapper.Generate(input_prompt, temperature=temperature, top_p=top_p,alpha_presence=presence_penalty,alpha_frequency=frequency_penalty, penalty_decay=penalty_decay, MAX_COUNT=max_tokens,STOP=stop):
                yield response_chunk
        finally:
            i=1
            #selected_wrapper.set_busy(False)

    

    def sync_stream():
        #global selected_wrapper
        totaltext = ''
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = generate_and_stream()

        try:
            while True:
                chunk = loop.run_until_complete(async_gen.__anext__())
                totaltext = totaltext+chunk
                response_data = [
                    {
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                ]
                #print(response_data[0])
                yield f'data: {json.dumps(response_data[0])}\n\n'

            
        except StopAsyncIteration:
            if StateCacheMode:
                output_prompt = input_prompt_stm_b + input_prompt_stm + totaltext
            else:
                output_prompt = input_prompt_stm + totaltext
            add_to_dynamic_state_list(output_prompt,selected_wrapper.model_current_statetuned_filename,selected_wrapper.model_state)
            #print('OutputPrompt-----------------------------------------------------------------------')
            #print(output_prompt[-100:])
            #print('-----------------------------------------------------------------------------------')
            selected_wrapper.set_busy(False)
            response_data = [
                "[DONE]"
            ]
            yield f'data: {json.dumps(response_data)}\n\n'
            pass

    if stream:
        return Response(stream_with_context(sync_stream()), content_type='text/event-stream')
    else:
        response = asyncio.run(collect_chunks(selected_wrapper.Generate(input_prompt, temperature=temperature, top_p=top_p,alpha_presence=presence_penalty,alpha_frequency=frequency_penalty, penalty_decay=penalty_decay, MAX_COUNT=max_tokens,STOP=stop)))
        
        OutputText = ''
        for chunk in response:
            OutputText = OutputText+chunk
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
        return jsonify(jsonResponse)
        #return jsonify({"response": ''.join(response)})


if __name__ == '__main__':
    app.run(debug=False, host=args.localhost, port=args.port)