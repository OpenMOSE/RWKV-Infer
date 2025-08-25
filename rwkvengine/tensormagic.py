from safetensors import safe_open
from safetensors.torch import load_file
import torch
import os
from einops import rearrange, reduce, repeat
def load_split_safetensors(model_dir, device="cpu"):
                """
                分割されたSafeTensorファイルを読み込む関数
                
                Args:
                    model_dir: 分割されたSafeTensorファイルが格納されているディレクトリパス
                    device: ロード先のデバイス（デフォルトは"cpu"）
                
                Returns:
                    dict: 統合されたモデルの状態辞書
                """
                # ディレクトリから全てのsafetensorファイルを取得
                files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
                
                if not files:
                    raise ValueError(f"No safetensors files found in {model_dir}")
                
                # 状態辞書を初期化
                state_dict = {}
                
                # 各ファイルを読み込んで統合
                for file in files:
                    file_path = os.path.join(model_dir, file)
                    # load_fileはtorch形式で読み込む
                    file_state_dict = load_file(file_path, device=device)
                    state_dict.update(file_state_dict)
                
                return state_dict



def pad_weight_tensor_to_256(weight_tensor):
    """重みテンソルを256の倍数にゼロパディング"""
    # 現在のサイズを取得
    original_shape = weight_tensor.shape
    print(f'original shape = {original_shape}')
    
    # 各次元を256の倍数に切り上げ
    padded_shape = []
    for dim in original_shape:
        padded_dim = ((dim + 255) // 256) * 256
        padded_shape.append(padded_dim)
    
    # すでに256の倍数の場合はそのまま返す
    if list(original_shape) == padded_shape:
        return weight_tensor
    
    # 新しいゼロテンソルを作成
    padded_tensor = torch.zeros(
        padded_shape,
        dtype=weight_tensor.dtype,
        device=weight_tensor.device,
        requires_grad=weight_tensor.requires_grad
    )
    
    # 元のデータをコピー
    if len(original_shape) == 2:  # Linear層の重み
        padded_tensor[:original_shape[0], :original_shape[1]] = weight_tensor
    elif len(original_shape) == 4:  # Conv2d層の重み
        padded_tensor[:original_shape[0], :original_shape[1], 
                     :original_shape[2], :original_shape[3]] = weight_tensor
    else:  # その他の次元
        slices = tuple(slice(0, dim) for dim in original_shape)
        padded_tensor[slices] = weight_tensor

    print(f'padded_tensor shape = {padded_tensor.shape}')
    
    return padded_tensor

def RenameToHFStyle(z,inputkey,gqa=0):
    if gqa == 1:
        name_mapping = {
            #'model.': '',
            'blocks.':'model.layers.',
            'att.':'self_attn.',
            'ffn.': 'mlp.',
            'gate_up.':'gate_up_proj.',
            'down.':'down_proj.',
            'gate.':'gate_proj.',
            'up.':'up_proj.',
            'ln1.':'input_layernorm.',
            'ln2.':'post_attention_layernorm.',
            'head.':'lm_head.',
            'ln_r.':'q_norm.',
            'ln_k.':'k_norm.',
            'ln_out.':'model.norm.',
            'emb.':'model.embed_tokens.'
        }
    else:
        name_mapping = {
            #'model.': '',
            'blocks.':'model.layers.',
            'att.':'self_attn.',
            'ffn.': 'mlp.',
            'gate_up.':'gate_up_proj.',
            'down.':'down_proj.',
            'gate.':'gate_proj.',
            'up.':'up_proj.',
            'ln1.':'input_layernorm.',
            'ln2.':'post_attention_layernorm.',
            'head.':'lm_head.',
            'ln_r.':'r_norm.',
            'ln_k.':'k_norm.',
            'ln_out.':'model.norm.',
            'emb.':'model.embed_tokens.'
        }
    
    #new_state_dict = OrderedDict()

    old_key = inputkey
    new_key = old_key
    for old_pattern, new_pattern in name_mapping.items():
        new_key = new_key.replace(old_pattern, new_pattern)

    if new_key != old_key:
        z[new_key] = z[old_key].detach().clone().to(z[old_key].dtype)
        print(f'old = {old_key} new = {new_key}')
        z[old_key] = None
        del z[old_key]

    return z

import torch

def Attach_Adapter(keyname,weight,adapter,mode,scaling=1.0,device='cuda'): #from JL-er lora merge inspired
            
            print(f'AttachAdapter = {keyname}')
            k = keyname
            if keyname.endswith('.weight') or keyname.endswith('head'):
                adapterkeys = list(adapter.keys())
                #print(adapterkeys)
                #exit()

                if mode != '':
                    #print(f'scaling = {scaling}')
                    prefix = keyname[:-len('.weight')]
                    lora_A = prefix + '.lora_A'
                    lora_B = prefix + '.lora_B'
                    lora_M = prefix + '.lora_M'
                    gbmm = prefix + '.bone'
                    if lora_A in adapterkeys:
                        w=adapter
                        assert lora_B in adapterkeys

                        if lora_M in adapterkeys:
                            print('dora merging {lora_A} and {lora_B} and {lora_M} into {k}')
                            assert w[lora_B].shape[1] == w[lora_A].shape[0]

                            w[lora_A] = w[lora_A].to(device=device)
                            w[lora_B] = w[lora_B].to(device=device)
                            w[lora_M] = w[lora_M].to(device=device)
                            weight = weight + w[lora_B] @ w[lora_A] * scaling
                            norm = weight.norm(dim=0, keepdim=True) + 1e-6
                            weight = (w[lora_M] * weight) / norm  

                            del w[lora_A]
                            del w[lora_B]
                            del w[lora_M]
                            return weight
                        
                        else:
                            print(f'lora merging {lora_A} and {lora_B} into {k}')
                            
                            assert w[lora_B].shape[1] == w[lora_A].shape[0]

                            w[lora_A] = w[lora_A].to(device=device)
                            w[lora_B] = w[lora_B].to(device=device)
                            weight = weight + w[lora_B] @ w[lora_A] * scaling
                            del w[lora_A]
                            del w[lora_B]
                            return weight

                    if gbmm in adapterkeys :
                        w=adapter
                        print(f'bone merging {gbmm} into {k}')
                        w[gbmm] = w[gbmm].to(device=device)
                        b,r,_ = w[gbmm].shape
                        bone = rearrange(weight, '(a r1) (b r2) -> a b r1 r2', r1 = r, r2 = r)@w[gbmm]+w[gbmm]
                        weight += rearrange(bone, 'a b r1 r2 ->(a r1) (b r2) ')
                        print(weight)
                        del w[gbmm]
                        return weight
                    for key in adapterkeys:
                        if key == keyname:
                            weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                            print(f'key = {key} is swapped from Adapter')
                    return weight
                else:
                    return weight
            else:
                adapterkeys = list(adapter.keys())
                for key in adapterkeys:
                    if key == keyname:
                        weight = adapter[key].to(dtype=torch.bfloat16,device=device)
                        print(f'key = {key} is swapped from Adapter')

                return weight