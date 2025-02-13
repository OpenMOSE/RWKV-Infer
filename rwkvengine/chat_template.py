def RWKV_World(role,content,split_token='\n\n'):
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

def Qwen2(role,content,split_token='<|im_end|>'):
    Role = 'user'
    if role == 'system':
        Role = 'system'
    if role == 'user':
        Role = 'user'
    if role == 'assistant':
        Role = 'assistant'
    Template = f'<|im_start|>{Role}\n{content}\n{split_token}\n'

    if content == None:
        Template = f'<|im_start|>{Role}'
    return Template


def GetTemplate(role,content,split_token,mode = 'world'):
    
    if mode == 'qwen':
        if split_token is None:
            return Qwen2(role,content)
        else:
            return Qwen2(role,content,split_token)
    else:
        if split_token is None:
            return RWKV_World(role,content)
        else:
            return RWKV_World(role,content,split_token)