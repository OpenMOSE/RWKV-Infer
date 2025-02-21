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
        Template = '<|begin_of_text|>' + Template
    
    if content == None:
        Template = f'<|start_header_id|>{Role}<|end_header_id|>'
    return Template


def GetTemplate(lastprompt,role,content,split_token,mode = 'world'):
    
    if mode == 'qwen':
        if split_token is None:
            return Qwen2(lastprompt,role,content)
        else:
            return Qwen2(lastprompt,role,content,split_token)
    if mode == 'llama':
        if split_token is None:
            return Llama(lastprompt,role,content)
        else:
            return Llama(lastprompt,role,content,split_token)
    else:
        if split_token is None:
            return RWKV_World(lastprompt,role,content)
        else:
            return RWKV_World(lastprompt,role,content,split_token)