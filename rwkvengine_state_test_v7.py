import torch
from rwkvengine.rwkvcore import RWKV_x, PIPELINE
import time
import copy
import cv2
import numpy as np


def visualize_tensor_colored(layers,tensor, normalization_method='standard', gamma=1.0, contrast_limit=None, scale_factor=2.0):
    """
    テンソルの可視化（カラーマップ付き、拡大表示、黒枠付き）
    
    Parameters:
    - scale_factor: 拡大倍率（デフォルト2倍）
    """
    # テンソルをNumPy配列に変換
    tensor = tensor.detach().cpu().numpy()
    
    if contrast_limit is not None:
        lower = np.percentile(tensor, contrast_limit * 100)
        upper = np.percentile(tensor, (1 - contrast_limit) * 100)
        tensor = np.clip(tensor, lower, upper)

    # 正規化
    if normalization_method == 'standard':
        tensor = (tensor - np.mean(tensor)) / np.std(tensor)
        tensor = (tensor * 64) + 128
    elif normalization_method == 'minmax':
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    elif normalization_method == 'adaptive':
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    if gamma != 1.0:
        tensor = np.power(tensor, gamma)

    tensor = (tensor * 255).astype(np.uint8)

    # 拡大後のセルサイズを計算
    cell_height = int(64 * scale_factor)
    cell_width = int(64 * scale_factor)
    
    # グリッドの作成 - 黒枠用に余白を追加
    border_size = 2  # 黒枠の太さ
    rows, cols = 4, 8
    grid_height = rows * (cell_height + border_size) + border_size
    grid_width = cols * (cell_width + border_size) + border_size
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # 各チャネルをグリッドに配置
    for idx in range(layers):
        i, j = idx // cols, idx % cols
        cell = tensor[idx, layers, :, :]
        
        if normalization_method == 'adaptive':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cell = clahe.apply(cell)

        # セルの拡大
        cell_resized = cv2.resize(cell, (cell_width, cell_height), interpolation=cv2.INTER_LINEAR)
        
        # カラーマップの適用
        colored_cell = cv2.applyColorMap(cell_resized, cv2.COLORMAP_JET)
        
        # インデックス番号を追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = scale_factor * 0.4
        cv2.putText(colored_cell, f'{idx}', 
                    (5, 20), font, font_scale, (255, 0, 0), 1)
        
        # グリッドに配置（黒枠を考慮した位置）
        y_start = i * (cell_height + border_size) + border_size
        x_start = j * (cell_width + border_size) + border_size
        grid[y_start:y_start + cell_height, 
             x_start:x_start + cell_width] = colored_cell

    return grid

def add_colorbar(image, height=None):
    """
    可視化画像にカラーバーを追加（拡大に対応）
    """
    if height is None:
        height = image.shape[0]
    
    colorbar_width = int(30 * (image.shape[0] / 256))  # スケールに応じて幅を調整
    colorbar = np.zeros((height, colorbar_width, 3), dtype=np.uint8)
    
    for i in range(height):
        value = 255 - int((i / height) * 255)
        color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        colorbar[i, :] = color
    
    # ラベルのフォントサイズを調整
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = image.shape[0] / 512  # 画像サイズに応じてスケール調整
    cv2.putText(colorbar, 'High', (2, int(20 * font_scale * 2)), 
                font, font_scale, (255, 255, 255), 1)
    cv2.putText(colorbar, 'Low', (2, height-int(10 * font_scale * 2)), 
                font, font_scale, (255, 255, 255), 1)
    
    return np.hstack([image, colorbar])


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tb", default="1", type=int) #batch size 
    args = parser.parse_args()
    print('RWKV x060Core with FLA Test')

    pipeline = PIPELINE()
    model = RWKV_x('/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/rwkv-x070-1b5-world-v3-60%trained-20250113-ctx4k.pth','fp16')
    Target_batch = 1 #args.tb#16

    States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    context =  'User: おつかれさま。元気してる？\n\nAssistant:'
    #context2 = 'User: Tell me advantage of C++.\n\nAssistant:'

    tuned_state = model.load_state('/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/x070-1b5may-state/rwkv-0-state.pth')

    

    shift_states = States.shift_states
    wkv_states = States.wkv_states

    layers = wkv_states.shape[0]

    def print_tensor_shapes(tensor_list):
        for i, tensor in enumerate(tensor_list):
            if isinstance(tensor, torch.Tensor):
                print(f"Tensor {i}: Shape = {tensor.shape}")
            else:
                print(f"Item {i} is not a Tensor")

    print_tensor_shapes(tuned_state)
    print(f'state-tune-file = {tuned_state.shape   }')

    print('////////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    print(f'wkv_states = {wkv_states.shape    }')
    print(f'shift_states = {shift_states.shape    }')

    #wkv_states[0] = model.model_current_statetuned




    
    for i in range(model.n_layer):
        wkv_states[i][0] = copy.deepcopy(tuned_state[i])

    tuned_state = copy.deepcopy(wkv_states)




    #print('base wkv_states = {}')
    #exit()

    tokens = pipeline.encode(context)
    #tokens2 = pipeline.encode(context2)
    prompts = []
    prompts.append(torch.tensor(tokens).unsqueeze(0).to('cuda'))
    # for i in range(Target_batch):
    #     if i%2 == 0:
    #         prompts.append(torch.tensor(tokens).unsqueeze(0).to('cuda'))
    #     else:
    #         prompts.append(torch.tensor(tokens2).unsqueeze(0).to('cuda'))

    def check_wkv_states(reference_state, current_state):
        shape = current_state.shape

        diff = reference_state.to(device='cuda',dtype=torch.float32) - current_state.to(dtype=torch.float32)
        sums = torch.sum(diff)
        print(f'RefState - CurrentState Sum = {sums}')
        tensor = current_state.to(dtype=torch.float32).view(-1,shape[2],shape[2],shape[2]).to(device='cpu')

        # scale_factorを2.0に設定して2倍に拡大
        scale_factor = 3
        # vis_standard = visualize_tensor_colored(tensor, normalization_method='standard', 
        #                                     scale_factor=scale_factor)
        vis_adaptive = visualize_tensor_colored(layers,tensor, normalization_method='adaptive', 
                                            scale_factor=scale_factor)
        # vis_gamma = visualize_tensor_colored(tensor, normalization_method='standard', 
        #                                 gamma=0.7, scale_factor=scale_factor)
        # vis_contrast = visualize_tensor_colored(tensor, normalization_method='standard', 
        #                                     contrast_limit=0.02, scale_factor=scale_factor)

        # カラーバーの追加
        #vis_standard = add_colorbar(vis_standard)
        vis_adaptive = add_colorbar(vis_adaptive)
        #vis_gamma = add_colorbar(vis_gamma)
        #vis_contrast = add_colorbar(vis_contrast)

        # 結果の表示
        #cv2.imshow('Standard Normalization (with colormap)', vis_standard)
        cv2.imshow('Adaptive Histogram Equalization (with colormap)', vis_adaptive)
        #cv2.imshow('Gamma Correction (with colormap)', vis_gamma)
        #cv2.imshow('Contrast Limited (with colormap)', vis_contrast)
        cv2.waitKey(1)
        


    idx = torch.cat(prompts, dim=0)

    print(f'{idx.shape}')

    check_wkv_states(tuned_state,wkv_states)
    check_wkv_states(tuned_state,wkv_states)

    time.sleep(5)

    x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)
    if x.dim() == 2:
        x = x.view(x.shape[0],1,x.shape[1])
    check_wkv_states(tuned_state,wkv_states)
    check_wkv_states(tuned_state,wkv_states)
    time.sleep(5)
    #exit()

    print(f'x = {x}')

    print(context)
    out_tokens = [[] for _ in range(Target_batch)]
    out_last = [0 for _ in range(Target_batch)]
    output_text = ['' for _ in range(Target_batch)]
    

    FirstTime = 1

    t000 = time.perf_counter()
    min_time = 1e10
    min_time_all = 1e10
    min_time_all_single = 1e10

    maxtoken= 100

    temperature = torch.full((Target_batch,), 1.0)
    top_p = torch.full((Target_batch,), 0.7)


    SamplingSum = 0
    ForwardSum = 0
    DecodeSum = 0

    for i in range(maxtoken):
        time.sleep(0.1)
        check_wkv_states(tuned_state,wkv_states)
        
        t0 = time.perf_counter()
        x[:, -1, 0] -= 1e10

        otokens = pipeline.improved_nucleus_sampling_multi_static(x[:, -1], temperature=temperature, top_p=top_p).tolist()

        tokens = []
        for j in range(Target_batch):
            tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0).to('cuda'))

        idx = torch.cat(tokens, dim=0)
        t1 = time.perf_counter()
        for j in range(Target_batch):
            out_tokens[j] += [otokens[j]]
            try:
                tmp = pipeline.decode(out_tokens[j][out_last[j]:])
                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                        #if j == Target_batch - 1:
                        #print(tmp,end="", flush=True)
                        print(tmp)
                        output_text[j] = output_text[j] + tmp
                        out_last[j] = i + 1
            except:
                pass
        t2 = time.perf_counter()

        x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states)
        if x.dim() == 2:
            x = x.view(x.shape[0],1,x.shape[1])
        t3 = time.perf_counter()
        ForwardSum += (t3 - t2)
        DecodeSum += (t2 - t1)
        SamplingSum += (t1 - t0)

    ForwardSum = ForwardSum / (float(maxtoken)) * 1000
    DecodeSum = DecodeSum / (float(maxtoken)) * 1000
    SamplingSum = SamplingSum / (float(maxtoken)) * 1000

    print('performance')
    print(f'ForwardAverage= {round(ForwardSum,4)} ms')
    print(f'DecodeSum= {round(DecodeSum,4)} ms')
    print(f'SamplingSum= {round(SamplingSum,4)} ms')



    t001 = time.perf_counter()

    print(output_text)
    print('RWKV-Infer FLA Refactor')

    tokensec = maxtoken / (t001-t000)
    print(f'TargetBatch = {Target_batch} Total token/s = {round(tokensec*Target_batch,2)} Single token/s = {round(tokensec,2)}')