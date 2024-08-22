import requests
import json
import time


class LLMBackend():
    def __init__(self,RWKVServerURL='http://0.0.0.0:9000/',RWKVServerAPIKey='rwkv'):
        
        self.initial_prompt = [ {"role": "system", "content": "あなたは優しい女性です。すべての質問に対して、優しく簡潔に答えてくれます。特技はピアノです。"},
                                {"role": "user", "content": "こんにちは"},
                                {"role": "assistant", "content": "ご主人様！なにか御用でしょうか？"},
                                {"role": "user", "content": "あなたは誰ですか？"},
                                {"role": "assistant", "content": "はい！ご主人様。私はRWKV言語モデルで動作するメイドです。なんでも聞いてくださいね。"},
                                {"role": "user", "content": "最近は何をしているの？"},
                                {"role": "assistant", "content": "はい！ご主人様。私はいつもあなたのお世話をしています。"},
                              ]
        
        self.chat_history = []
        self.OpenAICompatibleURL = RWKVServerURL
        self.OpenAIAPIKey = RWKVServerAPIKey
        self.generate_context_count = 50

        self.data = {
            "model": "RWKV",
            "messages": "",#self.initial_prompt + self.chat_history,
            "max_tokens": 500,
            "temperature": 1,
            "stream": True,
            "names": {'system': 'system', 'user': 'user', 'assistant': 'assistant'}
        }

    def chat(self,prompt):
        self.chat_history.append( {"role": "user", "content": prompt} )
        # リクエストボディの設定
        

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.OpenAIAPIKey}'
        }

        self.chat_history.append({"role": "assistant", "content": ""} )

        self.chat_history[-1]['content'] = ""

        self.data['messages'] = self.initial_prompt + self.chat_history

        response = requests.post(self.OpenAICompatibleURL + 'chat/completions', headers=headers, json=self.data, stream=True)

        for line in response.iter_lines():
            if line:
                json_response = json.loads(line.decode('utf-8').split('data: ')[1])
                if 'choices' in json_response and len(json_response['choices']) > 0:
                    delta = json_response['choices'][0]['delta']
                    if 'content' in delta:
                        #print(delta['content'], end='', flush=True)
                        self.chat_history[-1]['content'] = self.chat_history[-1]['content'] + delta['content']
                        yield delta['content']


BaseEngine = LLMBackend()

if __name__ == '__main__':
    print('RWKV Infer API Test')
    BaseEngine.initial_prompt = [ {"role": "system", "content": "あなたは優しい女性です。すべての質問に対して、優しく簡潔に答えてくれます。特技はピアノです。"},
                                {"role": "user", "content": "こんにちは"},
                                {"role": "assistant", "content": "ご主人様！なにか御用でしょうか？"},
                                {"role": "user", "content": "あなたは誰ですか？"},
                                {"role": "assistant", "content": "はい！ご主人様。私はRWKV言語モデルで動作するメイドです。なんでも聞いてくださいね。"},
                                {"role": "user", "content": "最近は何をしているの？"},
                                {"role": "assistant", "content": "はい！ご主人様。私はいつもあなたのお世話をしています。"},
                              ]
    BaseEngine.data = {
            "model": "RWKV x060 7B JPN MRSS Test",
            "messages": "", #set model viewname. if model+state(set '{ModelViewname} {StateViewname}')
            "max_tokens": 500,
            #"temperature": 1,
            "stream": True,
            "names": {'system': 'system', 'user': 'user', 'assistant': 'assistant'},
            #"mrss_gatingweight":['0.01','0.01','0.9','0.1']
        }
    
    for message in BaseEngine.initial_prompt:
        if message['role'] == 'system':
            print(f"System : {message['content']}")
        if message['role'] == 'user':
            print(f"User : {message['content']}")
        if message['role'] == 'assistant':
            print(f"Assistant : {message['content']}")
    
    while True:
        user_input = input("User: ")
        print('Assistant :', end='',flush=True)
        for output in BaseEngine.chat(user_input):
            print(output, end='',flush=True)

    
