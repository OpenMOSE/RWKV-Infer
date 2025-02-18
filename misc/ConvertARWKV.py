import os
import torch
from safetensors.torch import load_file
from collections import OrderedDict

def convert_weight_names(state_dict):
    """
    重みの名前を変換する関数
    """
    name_mapping = {
        'model.': '',
        'layers.': 'blocks.',
        'self_attn.time_mixer.': 'att.',
        'mlp.': 'ffn.',
        'down_proj': 'down',
        'gate_proj': 'gate',
        'up_proj': 'up',
        'input_layernorm': 'ln1',
        'post_attention_layernorm': 'ln2',
        'lm_head': 'head',
        'norm.': 'ln_out.',
        'embed_tokens.': 'emb.'
    }
    
    new_state_dict = OrderedDict()
    
    with torch.no_grad():
        # 重み名の変換とBF16への変換
        for old_key, tensor in state_dict.items():
            new_key = old_key
            for old_pattern, new_pattern in name_mapping.items():
                new_key = new_key.replace(old_pattern, new_pattern)
            # テンソルをBF16に変換し、grad情報なしでコピー
            new_state_dict[new_key] = tensor.detach().clone().to(torch.bfloat16)
    
    return new_state_dict

def merge_safetensors(input_dir):
    """
    指定されたディレクトリ内のすべてのsafetensorファイルをマージする
    """
    safetensor_files = [f for f in os.listdir(input_dir) if f.endswith('.safetensors')]
    
    if not safetensor_files:
        raise FileNotFoundError("Inputフォルダにsafetensorファイルが見つかりません。")
    
    print(f"処理するファイル数: {len(safetensor_files)}")
    
    merged_weights = OrderedDict()
    
    for file_name in safetensor_files:
        safetensor_path = os.path.join(input_dir, file_name)
        print(f"\n処理中のファイル: {file_name}")
        
        try:
            with torch.no_grad():
                # ファイルを読み込み
                current_weights = load_file(safetensor_path)
                
                # 重み名を変換
                converted_weights = convert_weight_names(current_weights)
                
                # 重複をチェック
                overlap = set(merged_weights.keys()) & set(converted_weights.keys())
                if overlap:
                    print(f"\n警告: {file_name}で重複する重みが見つかりました:")
                    for key in overlap:
                        print(f"  {key}")
                    choice = input("重複する重みを上書きしますか？ (y/n): ").strip().lower()
                    if choice != 'y':
                        print("処理を中断します。")
                        return None
                
                # 重みをマージ（BF16形式、grad情報なし）
                for key, tensor in converted_weights.items():
                    merged_weights[key] = tensor.clone()
                    
                print(f"{file_name}の処理が完了しました。")
                print(f"現在の総重み数: {len(merged_weights)}")
            
        except Exception as e:
            print(f"ファイル{file_name}の処理中にエラーが発生しました: {str(e)}")
            return None
    
    return merged_weights

def main():
    input_dir = "/home/client/Projects/ARWKV/ARWKV_7B_R1_16K"
    output_file = "ARWKV_7B_R1_16K.pth"
    
    try:
        print("safetensorファイルのマージを開始します...")
        with torch.no_grad():
            merged_weights = merge_safetensors(input_dir)
            
            if merged_weights is not None:
                # BF16形式でマージ結果を保存
                torch.save(merged_weights, output_file, _use_new_zipfile_serialization=True)
                print(f"\nマージが完了しました。")
                print(f"マージされた重みの総数: {len(merged_weights)}")
                print(f"保存先: {output_file}")
                
                # 最終的な重み名一覧を表示
                print("\n最終的な重み名一覧:")
                for key in merged_weights.keys():
                    print(f"  {key}")
            else:
                print("\nマージ処理が中断されました。")
            
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()