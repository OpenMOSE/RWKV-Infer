def RWKV_World(promptcount,role,content,split_token='\n\n'):
    Role = 'User'
    if role == 'system':
        Role = 'System'
    if role == 'user':
        Role = 'User'
    if role == 'assistant':
        Role = 'Assistant'
    Template = f'{Role}: {content}{split_token}'

    if content == None:
        Template = f'{Role}:'
    return Template

def Qwen2(promptcount,role,content,split_token='<|im_end|>'):
    Role = 'user'
    if role == 'system':
        Role = 'system'
    if role == 'user':
        Role = 'user'
    if role == 'assistant':
        Role = 'assistant'
    Template = f'<|im_start|>{Role}\n{content}\n{split_token}\n'

    if content == None:
        Template = f'<|im_start|>{Role}\n'
    return Template

def Llama(promptcount,role,content,split_token='<|eot_id|>'):
    Role = 'user'
    if role == 'system':
        Role = 'system'
    if role == 'user':
        Role = 'user'
    if role == 'assistant':
        Role = 'assistant'
    Template = f'<|start_header_id|>{Role}<|end_header_id|>{content}{split_token}'
    if promptcount == 0:
        Template = '<|start_of_text|>' + Template
    
    if content == None:
        Template = f'<|start_header_id|>{Role}<|end_header_id|>'
    return Template


from jinja2 import Template

class LLMJPChatFormatter:
    def __init__(self):
        self.config = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "cls_token": "<CLS|LLM-jp>",
            "eod_token": "</s>",
            "clean_up_tokenization_spaces": False
        }
        
        self.template = Template("""{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '\n\n### 指示:\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。' }}{% elif message['role'] == 'assistant' %}{{ '\n\n### 応答:\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\n### 応答:\n' }}{% endif %}{% endfor %}""")

    def format_chat(self, messages, add_generation_prompt=True):
        return self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **self.config
        )
    
class PHI3ChatFormatter:
    def __init__(self):
        self.config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "added_tokens_decoder": {
                "0": {
                    "content": "<unk>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "1": {
                    "content": "<s>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "2": {
                    "content": "</s>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": False
                },
                "32000": {
                    "content": "<|endoftext|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "32001": {
                    "content": "<|assistant|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32002": {
                    "content": "<|placeholder1|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32003": {
                    "content": "<|placeholder2|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32004": {
                    "content": "<|placeholder3|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32005": {
                    "content": "<|placeholder4|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32006": {
                    "content": "<|system|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32007": {
                    "content": "<|end|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32008": {
                    "content": "<|placeholder5|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32009": {
                    "content": "<|placeholder6|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                },
                "32010": {
                    "content": "<|user|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": True,
                    "single_word": False,
                    "special": True
                }
            },
            "bos_token": "<s>",
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' and message['content'] %}"
                "{{'<|system|>' + message['content'] + '<|end|>'}}"
                "{% elif message['role'] == 'user' %}"
                "{{'<|user|>' + message['content'] + '<|end|>'}}"
                "{% elif message['role'] == 'assistant' %}"
                "{{'<|assistant|>' + message['content'] + '<|end|>'}}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|assistant|>' }}"
                "{% else %}"
                "{{ eos_token }}"
                "{% endif %}"
            ),
            "clean_up_tokenization_spaces": False,
            "eos_token": "<|endoftext|>",
            "legacy": False,
            "model_max_length": 131072,
            "pad_token": "<|endoftext|>",
            "padding_side": "left",
            "sp_model_kwargs": {},
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "<unk>",
            "use_default_system_prompt": False
        }

        # chat_template を Jinja2 テンプレートとして読み込み
        self.template = Template(self.config["chat_template"])
    
    def format_chat(self, messages, add_generation_prompt=True):
        """
        与えられたmessages（リスト）を、TokenizerConfigのchat_templateに基づいて
        フォーマットし文字列として返す。
        """
        return self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **self.config
        )
    
llmjpformatter = LLMJPChatFormatter()
phi3formatter = PHI3ChatFormatter()


def GetTemplate(lastprompt,role,content,split_token,mode = 'world'):
    
    if mode == 'qwen':
        if split_token is None:
            return Qwen2(lastprompt,role,content)
        else:
            return Qwen2(lastprompt,role,content,split_token)
    if mode == 'llmjp':
        if split_token is None:
            return Llama(lastprompt,role,content)
        else:
            return Llama(lastprompt,role,content,split_token)
    else:
        if split_token is None:
            return RWKV_World(lastprompt,role,content)
        else:
            return RWKV_World(lastprompt,role,content,split_token)