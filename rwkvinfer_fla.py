#RWKVInfer with Flash-Linear-Attention
#2024 OpenMOSE

import asyncio
import queue
from dataclasses import dataclass,field
from typing import Dict, List, Optional
from enum import Enum
import torch
import copy

from rwkv6fla import PIPELINE, RWKV6



# ロックオブジェクトの作成
lock = asyncio.Lock()


class PromptStatus(Enum):
    QUEUED = 1
    PROCESSING = 2
    COMPLETED = 3
    FAILED = 4

@dataclass
class Prompt:
    id: int
    prompts: str
    status: PromptStatus = 3
    result: Optional[str] = None
    maxtokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.3
    endtoken: List[str] = field(default_factory=lambda: ['\n\n'])
    base_state_tuned: str = "None"
    use_exist_state_wkv: Optional[torch.Tensor] = None
    use_exist_state_shift: Optional[torch.Tensor] = None


class PromptQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.prompts: Dict[int, Prompt] = {}
        self.current_id = 0

    async def add_prompt(self, promptdata:Prompt) -> int:
        self.current_id += 1
        prompt = promptdata#Prompt(id=self.current_id, prompts=content, status=PromptStatus.QUEUED,temperature=)
        self.prompts[prompt.id] = prompt
        await self.queue.put(prompt)
        return prompt.id

    async def get_prompt(self) -> Prompt:
        return await self.queue.get()

    def update_prompt(self, prompt_id: int, status: PromptStatus, result: Optional[str] = None):
        if prompt_id in self.prompts:
            self.prompts[prompt_id].status = status
            if result is not None:
                self.prompts[prompt_id].result = result

prompt_queue = PromptQueue()


class LLMWorker:
    def __init__(self):
        print('Initializing LLM Worker')
        
        self.llm_batch_chunk = 512 #FLA Preprocess Prompt chunks
        self.llm_batch_cycle = 4 #Preprocess cycle if 4, Pre,single,single,single,Pre,....
        self.llm_work_cycle = 0

        self.llm_max_batch_count = 16

        self.llm_last_batch_info = []
        self.llm_last_batch_count = 0

        self.llM_current_batch_info = []
        self.llm_current_batch_count = 0

        self.llm_dynamic_state_cache = []

        self.pipeline = PIPELINE()

        self.proceed_total_batches = 0


        #async with lock:
        for i in range(self.llm_max_batch_count):
            data = {'slotstatus':'idle', # idle,preprocess,processing,finished
                    'prompt':[], #tokenized tokens
                    'prompt_id':-1, #queue prompt_id
                    'proceedtokens':0, 
                    'max_tokens': 200,
                    'temperature': 1.0,
                    'top_p' : 0.5,
                    'end_token': ['default'],
                    'remarks':'',
                    'use_state_tuned':'None',
                    'wkv_states' : None,
                    'shift_states' : None,
                    'output':'',
                    'currenttoken':None,
                    'currenttokencount': 0,
                    }
            self.llM_current_batch_info.append(data)
            self.llm_last_batch_info.append(data)





        

    def LoadModel(self,modelpath,quantize=False):
        self.model = RWKV6(modelpath,quantize=quantize)




    async def QueueManage(self):
        global prompt_queue
        print('Start Queue Management')
        while True:
            IdleSlot = 0
            async with lock:
                for i in range(self.llm_max_batch_count):
                    if self.llM_current_batch_info[i]['slotstatus'] == 'idle':
                        IdleSlot = IdleSlot + 1
                        prompt = await prompt_queue.get_prompt()
                        print(prompt)
                        prompt_queue.update_prompt(prompt.id, PromptStatus.PROCESSING)
                        self.llM_current_batch_info[i]['slotstatus'] = 'processing'

                        data = {'slotstatus':'processing', # idle,preprocess,processing,finished
                                'prompt':self.pipeline.encode(prompt.prompts), #tokenized tokens
                                'prompt_id':prompt.id, #queue prompt_id
                                'proceedtokens':0, 
                                'max_tokens': prompt.maxtokens,
                                'currenttokencount': 0,
                                'temperature': prompt.temperature,
                                'top_p' : prompt.top_p,
                                'end_token': prompt.endtoken,
                                'remarks':f'{str(self.proceed_total_batches)}',
                                'use_state-tuned':prompt.base_state_tuned,
                                'wkv_states' : prompt.use_exist_state_wkv,
                                'shift_states' : prompt.use_exist_state_shift,
                                'current_prob' : None,
                                'output':'',
                                'out_tokens':[],
                                'out_last':0,
                                'currenttoken':self.pipeline.encode(''),
                                'occurrence':{},
                                'count':0,
                                }
                        
                        self.llM_current_batch_info[i] = copy.deepcopy(data)
                        
                        self.proceed_total_batches = self.proceed_total_batches + 1
                        if self.proceed_total_batches > 2000000000:
                            self.proceed_total_batches = 0
                    

                    
            
            await asyncio.sleep(0.01) # Everyone 10ms loop


    async def RunLLM(self):
        print('Start LLM Engine')
        global prompt_queue
        while True:
            NoProcessing = 1
            for i in range(self.llm_max_batch_count):
                if self.llM_current_batch_info[i]['slotstatus'] == 'processing':
                    NoProcessing = 0
            if NoProcessing == 0:
                #print('have work :)')
                self.llm_work_cycle = self.llm_work_cycle + 1
                if self.llm_work_cycle > 1000:
                    self.llm_work_cycle = 0

                if self.llm_work_cycle % self.llm_batch_cycle == 0:
                    #batch Pre Process Mode:
                    #print('pre processing mode')
                    #check
                    prompts = []
                    prompts_ids = [] 
                    b_wkv_states = []
                    b_shift_states = []
                    token_max = self.llm_batch_chunk
                    #for work in self.llM_current_batch_info:
                    #    if work['proceedtokens'] < len(work['prompt']):
                    #        if len(work['prompt']) - work['proceedtokens'] < token_max:
                    #            token_max = len(work['prompt']) - work['proceedtokens']
                    for i in range(self.llm_max_batch_count):
                        work = self.llM_current_batch_info[i]
                        
                        if work['proceedtokens'] < len(work['prompt']) and work['slotstatus'] == 'processing':
                            print(f"batch {i} input tokens = {len(work['prompt'])}")
                            if len(work['prompt']) - work['proceedtokens'] < token_max:
                                token_max = len(work['prompt']) - work['proceedtokens']
                                
                    for i in range(self.llm_max_batch_count):
                        work = self.llM_current_batch_info[i]
                        if work['proceedtokens'] < len(work['prompt']) and work['slotstatus'] == 'processing':
                                prompts.append(work['prompt'][work['proceedtokens']:work['proceedtokens']+token_max])
                                prompts_ids.append(work['prompt_id'])
                                b_wkv_states.append(work['wkv_states'])
                                b_shift_states.append(work['shift_states'])

                    #print(f'prompts = {prompts}')
                    
                    if len(prompts)>0: # if have pre process work
                        prompts_tensor = []
                        for tx in prompts:
                            prompts_tensor.append(torch.tensor(tx).unsqueeze(0).to('cuda'))

                        idx = torch.cat(prompts_tensor, dim=0)
        
                        self.States = self.model.new_state(len(prompts))

                        shift_states = self.States.shift_states.permute(1, 0, 2, 3)
                        wkv_states = self.States.wkv_states.permute(1, 0, 2, 3, 4)

                        for i in range(len(prompts)):
                            if b_wkv_states[i] is not None:
                                wkv_states[i] = b_wkv_states[i]#prompts['wkv_states']
                            else:
                                print('wkv is none')
                            if b_shift_states[i] is not None:
                                shift_states[i] = b_shift_states[i]#prompts['shift_states']
                            else:
                                print('shift is none')

                        shift_states = shift_states.permute(1,0,2,3)
                        wkv_states = wkv_states.permute(1, 0, 2, 3, 4)
                        print(f'{token_max} forwarded')

                        x, shift_states, wkv_states = self.model.forward(idx, shift_states, wkv_states)

                        shift_states = shift_states.permute(1,0,2,3)
                        wkv_states = wkv_states.permute(1, 0, 2, 3, 4)

                        j = -1
                        for id in prompts_ids:
                            j = j + 1
                            for i in range(self.llm_max_batch_count):
                                if self.llM_current_batch_info[i]['prompt_id'] == id:
                                    self.llM_current_batch_info[i]['wkv_states'] = wkv_states[j]
                                    self.llM_current_batch_info[i]['shift_states'] = shift_states[j]
                                    self.llM_current_batch_info[i]['current_prob'] = x[j]
                                    self.llM_current_batch_info[i]['proceedtokens'] = self.llM_current_batch_info[i]['proceedtokens'] + token_max

                        #print(self.llM_current_batch_info)

                else:
                    #print('recurrent infer mode')
                    #prompts = []
                    token = []
                    token_ids = [] 
                    b_wkv_states = []
                    b_shift_states = []
                    current_prob = []
                    temperature = []
                    top_p = []
                    outputs = []
                    out_tokens = []
                    out_last = []
                    max_tokens = []
                    statuss = []
                    end_token = []
                    occurrence = []
                    counts = []

                    for i in range(self.llm_max_batch_count):
                        work = self.llM_current_batch_info[i]
                        if work['proceedtokens'] >= len(work['prompt']) and work['slotstatus'] == 'processing':
                                #prompts.append(work['prompt'][work['proceedtokens']:work['proceedtokens']+token_max])
                                token.append(work['currenttoken'])
                                token_ids.append(work['prompt_id'])
                                b_wkv_states.append(work['wkv_states'])
                                b_shift_states.append(work['shift_states'])
                                current_prob.append(work['current_prob'])
                                temperature.append(work['temperature'])
                                top_p.append(work['top_p'])
                                outputs.append(work['output'])
                                out_tokens.append(work['out_tokens'])
                                out_last.append(work['out_last'])
                                max_tokens.append(work['max_tokens'])
                                statuss.append(work['slotstatus']) 
                                end_token.append(work['end_token']) 
                                occurrence.append(work['occurrence']) 
                                counts.append(work['count']) 
                                #outputs.append(work['output'])
                    if len(token) > 0:
                        #tokens_tensor = []
                        #for i in range(len(token)):
                        #    tokens_tensor.append(torch.tensor(token[i]).unsqueeze(0).to('cuda'))

                        otokens = []
                        for j in range(len(token)):
                            #x[j][0][0] -= 1e10
                            for n in occurrence[j]:
                                current_prob[j][0][n] -= 0 + occurrence[j][n] * 1.0
                            
                            current_prob[j][0][0] -= 1e10
                            tk = self.pipeline.sample_logits(current_prob[j][0], temperature=temperature[j], top_p=top_p[j])

                            for xxx in occurrence[j]:
                                occurrence[j][xxx] *= 0.996
                            occurrence[j][tk] = 1 + (occurrence[j][tk] if tk in occurrence[j] else 0)



                            otokens.append(tk)

                        tokens = []
                        for j in range(len(token)):
                            tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0).to('cuda'))

                        for j in range(len(token)):
                            out_tokens[j] += [otokens[j]]
                            try:
                                tmp = self.pipeline.decode(out_tokens[j][out_last[j]:])
                                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                                        #yield tmp
                                        #if j == Target_batch - 1:
                                        #print(tmp,end="", flush=True)
                                        if j == 0:
                                            print(tmp,end="", flush=True)
                                        #print(tmp,end="", flush=True)
                                        outputs[j] = outputs[j] + tmp
                                        out_last[j] = counts[j] + 1
                                if out_last[j] > max_tokens[j]:
                                    #Reached Max Token
                                    statuss[j] = 'idle'
                                    print(f'batch {j} is finished')
                                    print(outputs[j])
                                exit_flag = False
                                #print(end_token[j])
                                for stop in end_token[j]:
                                    if stop in tmp:
                                        #yield tmp
                                        #output_text = output_text + tmp
                                        outputs[j] = outputs[j] + tmp
                                        exit_flag = True
                                if exit_flag:
                                    statuss[j] = 'idle'
                                    print(f'batch {j} is finished cause got endtoken')
                                    print(outputs[j])

                            except:
                                pass

                        idx = torch.cat(tokens, dim=0)

                        self.States = self.model.new_state(len(token))

                        shift_states = self.States.shift_states.permute(1, 0, 2, 3)
                        wkv_states = self.States.wkv_states.permute(1, 0, 2, 3, 4)

                        for i in range(len(token)):
                            if b_wkv_states[i] is not None:
                                wkv_states[i] = b_wkv_states[i]#prompts['wkv_states']
                            else:
                                print('wkv is none')
                            if b_shift_states[i] is not None:
                                shift_states[i] = b_shift_states[i]#prompts['shift_states']
                            else:
                                print('shift is none')

                        shift_states = shift_states.permute(1,0,2,3)
                        wkv_states = wkv_states.permute(1, 0, 2, 3, 4)

                        x, shift_states, wkv_states = self.model.forward(idx, shift_states, wkv_states)

                        shift_states = shift_states.permute(1,0,2,3)
                        wkv_states = wkv_states.permute(1, 0, 2, 3, 4)

                        j = -1
                        for id in token_ids:
                            j = j + 1
                            for i in range(self.llm_max_batch_count):
                                if self.llM_current_batch_info[i]['prompt_id'] == id:

                                    #print('waiting lock')

                                    
                                        #print('locked')
                                    if statuss[j] == 'processing':
                                        prompt_queue.update_prompt(id,PromptStatus.PROCESSING,outputs[j])
                                    else:
                                        prompt_queue.update_prompt(id,PromptStatus.COMPLETED,outputs[j])
                                        statuss[j] == 'idle'

                                    self.llM_current_batch_info[i]['wkv_states'] = wkv_states[j]
                                    self.llM_current_batch_info[i]['shift_states'] = shift_states[j]
                                    self.llM_current_batch_info[i]['current_prob'] = x[j]
                                    self.llM_current_batch_info[i]['currenttoken'] = otokens[j]
                                    self.llM_current_batch_info[i]['slotstatus'] = statuss[j]

                                    self.llM_current_batch_info[i]['output'] = outputs[j]
                                    self.llM_current_batch_info[i]['out_tokens'] = out_tokens[j]
                                    self.llM_current_batch_info[i]['out_last'] = out_last[j]
                                    self.llM_current_batch_info[i]['occurrence'] = occurrence[j]
                                    self.llM_current_batch_info[i]['count'] = counts[j] + 1
                                    break


                    

                    




            await asyncio.sleep(0.001) # Every 1ms

texts = '''User: Who is doraemon ?\n\nAssistant:'''



texts2 = '''User: 日本の三権分立について詳しく教えてください\n\nAssistant:'''
class Test:
    async def test(self):
        print('Test thread')
        while True:
            print('test prompt send')
            i = await prompt_queue.add_prompt(texts)
            i = await prompt_queue.add_prompt(texts2)
            #i = await prompt_queue.add_prompt(texts)
            #i = await prompt_queue.add_prompt(texts2)
            #i = await prompt_queue.add_prompt(texts)
            #i = await prompt_queue.add_prompt(texts2)
            #i = await prompt_queue.add_prompt(texts)
            #i = await prompt_queue.add_prompt(texts2)

            await asyncio.sleep(3)

async def main():
    engine1 = LLMWorker()
    engine2 = Test()
    engine1.LoadModel('models/7b-rwkv-114-merged.pth',quantize=False)
    await asyncio.gather(
        engine1.RunLLM(),
        engine1.QueueManage(),
        engine2.test()
    )

if __name__ == '__main__':
    print('RWKV x060 FLA Inference Test')
    asyncio.run(main())
    