import torch
from rwkvengine.rwkvcore import RWKV_x, PIPELINE
import time
import copy
import torch.nn.functional as F
import numpy as np

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tb", default="1", type=int) #batch size 
    parser.add_argument("--fully_fused", default="1", type=int) #batch size 
    args = parser.parse_args()
    print('RWKV x070Core with FLA Test')

    pipeline = PIPELINE()
    #model = RWKV_x('/home/client/Projects/RWKV-LM-RLHF/main/myfolder/converted/x060-upgraded.pth','fp8',adapter_mode='bone',adapter_model='/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/x070-7b-cje/rwkv-14.pth',fully_fusedrecurrent=args.fully_fused)
    model = RWKV_x('/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/rwkv-x070-2b9-cje-instruct-1.pth','fp8')
    Target_batch = args.tb

    States = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    States2 = model.new_state(Target_batch)#state_empty(32, 1, 2560, 2560 // 32)

    context =  'User: Tell me advantage of C++. also write examples\n\nAssistant:'

    context2 = """User: Translate this japanese text to english\n\n元モー娘。道重さゆみ、夏のツアーで芸能界引退へ 理由と「心から、感謝の気持ち」長文表明【全文】
ニュース｜ 元モーニング娘。で歌手の道重さゆみ（35）が19日、自身の公式ブログを更新し、「私、道重さゆみは、今年の夏に開催予定のコンサートツアー...
.7 時間前

Yahoo!ニュース
【バレー】大友愛さん630字の長文投稿 娘・秋本美空の日本一に「心の底からありがとう」（日刊スポーツ）
バレーボール全日本高校選手権（春高バレー）女子決勝は12日、東京体育館で行われ、共栄学園（東京）が、05年度大会以来19年ぶり3度目の優勝を飾った。
.6日前

毎日新聞
毎小ニュース：教育 衝撃の読書離れ 「長文」読めない？
6割（わり）の人（ひと）が1か月（げつ）に1冊（さつ）も本（ほん）を読（よ）まない――。そんなデータが、国（くに）の役所（やくしょ）・文化庁（ぶんかちょう）が...
.2024/09/26

Yahoo!ニュース
【春高バレー】大友愛さん「毎日頑張りに驚かされた」 娘・美空が主将、共栄学園Vに感謝の長文（THE ANSWER）
全日本バレーボール高等学校選手権大会（春高バレー）は12日、東京体育館で女子決勝が行われ、共栄学園が下北沢成徳（ともに東京）に...
.6日前

BIGLOBEニュース
写真ニュース(3/3): 大学入学共通テスト「情報I」対策！オリジナルの長文予想問題を使った短期集中講座を豊川高校にて開催
写真ニュース(3/3): 大学入学共通テスト「情報I」対策！オリジナルの長文予想問題を使った短期集中講座を豊川高校にて開催.
.1週間前

Yahoo!ニュース
バスケ高田真希が長文投稿 「大好きな井上先生」「先生のバスケットが世界に通用する事を証明できて嬉しかった」 亡き恩師への思いつづる（デイリースポーツ）
バスケットボール女子日本代表の高田真希が４日、自身のＳＮＳを更新。高校女子の名門・桜花学園監督で、１２月３１日に７８歳で亡くなった恩師の井上真一さんへ...
.2週間前

Yahoo!ニュース
SixTONES、念願の6人ディズニー裏側語る 高地優吾は帰宅後にメンバーへ長文LINE「激キモエモLINE」 (モデルプレス)
SixTONES、念願の6人ディズニー裏側語る 高地優吾は帰宅後にメンバーへ長文LINE「激キモエモLINE」. 写真の記事を読む...
.1週間前

Yahoo!ニュース
横浜FM退団のDF畠中槙之輔が長文で感謝をつづる「セレッソ大阪に移籍することになりました」（GOAL）
横浜F・マリノスは28日、DF畠中槙之輔がセレッソ大阪へ完全移籍することを発表した。 東京ヴェルディの下部組織で育ち、同クラブのトップチームへ昇格...
.3週間前

Yahoo!ニュース
バスケ高田真希が長文投稿 「大好きな井上先生」「先生のバスケットが世界に通用する事を証明できて嬉しかった」 亡き恩師への思いつづる (デイリースポーツ)
バスケ高田真希が長文投稿 「大好きな井上先生」「先生のバスケットが世界に通用する事を証明できて嬉しかった」 亡き恩師への思いつづる.
.2週間前

Yahoo!ニュース
NewJeansダニエルが金髪ロング、韓服や制服姿など様々な姿で2024年を振り返り！ファンへの想いを綴る超長文メッセージも
NewJeansダニエルが金髪ロング、韓服や制服姿など様々な姿で2024年を振り返り！ファンへの想いを綴る超長文メッセージも.
.2週間前
Assistant:"""

    context3 = """User: Translate this japanese text to english\n\n田真希が４日、自身のＳＮＳを更新。高校女子の名門・桜花学園監督で、１２月３１日に７８歳で亡くなった恩師の井上真一さんへ...
    .2週間前

    Yahoo!ニュース
    SixTONES、念願の6人ディズニー裏側語る 高地優吾は帰宅後にメンバーへ長文LINE「激キモエモLINE」 (モデルプレス)
    SixTONES、念願の6人ディズニー裏側語る 高地優吾は帰宅後にメンバーへ長文LINE「激キモエモLINE」. 写真の記事を読む...
    .1週間前

    Yahoo!ニュース
    横浜FM退団のDF畠中槙之輔が長文で感謝をつづる「セレッソ大阪に移籍することになりました」（GOAL）
    横浜F・マリノスは28日、DF畠中槙之輔がセレッソ大阪へ完全移籍することを発表した。 東京ヴェルディの下部組織で育ち、同クラブのトップチームへ昇格...
    .3週間前

    Yahoo!ニュース
    バスケ高田真希が長文投稿 「大好きな井上先生」「先生のバスケットが世界に通用する事を証明できて嬉しかった」 亡き恩師への思いつづる (デイリースポーツ)
    バスケ高田真希が長文投稿 「大好きな井上先生」「先生のバスケットが世界に通用する事を証明できて嬉しかった」 亡き恩師への思いつづる.
    .2週間前

    Yahoo!ニュース
    NewJeansダニエルが金髪ロング、韓服や制服姿など様々な姿で2024年を振り返り！ファンへの想いを綴る超長文メッセージも
    NewJeansダニエルが金髪ロング、韓服や制服姿など様々な姿で2024年を振り返り！ファンへの想いを綴る超長文メッセージも.
    .2週間前
    Assistant:"""


    shift_states = States.shift_states
    wkv_states = States.wkv_states

    shift_states2 = States2.shift_states
    wkv_states2 = States2.wkv_states

    def print_tensor_shapes(tensor_list):
        for i, tensor in enumerate(tensor_list):
            if isinstance(tensor, torch.Tensor):
                print(f"Tensor {i}: Shape = {tensor.shape}")
            else:
                print(f"Item {i} is not a Tensor")

    #print_tensor_shapes(model.model_current_statetuned )
    #print(f'state-tune-file = {model.model_current_statetuned    }')

    print('////////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    print(f'wkv_states = {wkv_states.shape    }')
    print(f'shift_states = {shift_states.shape    }')

    #wkv_states[0] = model.model_current_statetuned
    #for i in range(model.n_layer):
    #    wkv_states[i][0] = model.model_current_statetuned[i*3 + 1]
    #exit()
    tokens0 = pipeline.encode(context)
    tokens = pipeline.encode(context2)
    tokens2 = pipeline.encode(context3)

    print(len(tokens))

    # prompts = []
    # for i in range(Target_batch):
    #         prompts.append(torch.tensor(tokens).unsqueeze(0).to('cuda'))

    # def analyze_vectors(v1, v2):
    #     # 基本統計量
    #     print("Pytorch State Vector 1 stats:",
    #         "\n Mean:", v1.mean().item(),
    #         "\n Std:", v1.std().item(),
    #         "\n Min:", v1.min().item(),
    #         "\n Max:", v1.max().item())
        
    #     print("\nFLA Vector 2 stats:",
    #         "\n Mean:", v2.mean().item(),
    #         "\n Std:", v2.std().item(),
    #         "\n Min:", v2.min().item(),
    #         "\n Max:", v2.max().item())
        
    #     # 要素ごとの相関係数
    #     corr = torch.corrcoef(torch.stack([v1, v2]))
    #     print("\nCorrelation:", corr[0,1].item())
        
    #     # 分散の比較
    #     print("\nVariance ratio:", v1.var() / v2.var())





    # for i in range(4):
         
    #     ctxlen = int((i+1)*4)

    #     prompts = []
    #     prompts2 = []
    #     for i in range(Target_batch):
    #             prompts.append(torch.tensor(tokens[:ctxlen]).unsqueeze(0).to('cuda'))
    #             prompts2.append(torch.tensor(tokens[:ctxlen]).unsqueeze(0).to('cuda'))

    #     idx = torch.cat(prompts, dim=0)
    #     idx2 = torch.cat(prompts2, dim=0)

    #     #print(idx.shape)
    #     x, shift_states0, wkv_states0 = model.forward(copy.deepcopy(idx2), copy.deepcopy(shift_states), copy.deepcopy(wkv_states),KernelMode=0) #Pytorch

    #     x1, shift_states1, wkv_states1 = model.forward(copy.deepcopy(idx), copy.deepcopy(shift_states2), copy.deepcopy(wkv_states2),KernelMode=1) #FLA

    #     #if x.dim() == 2:
    #     #    x = x.view(x.shape[0],1,x.shape[1])

    #     print(x.shape)
    #     print(x1.shape)

    #     print(f'state dim = {wkv_states1.shape}')

    #     #wkv_states1 = wkv_states1.permute(0,1,2,4,3).contiguous()
    #     analyze_vectors(wkv_states0.view(-1), wkv_states1.view(-1))




    #     x_similarity = F.cosine_similarity(x.view(-1), x1.view(-1), dim=-1)
    #     s_similarity = F.cosine_similarity(wkv_states0.view(-1), wkv_states1.view(-1), dim=-1)



    #     x_sum = torch.sum(x.float())
    #     #del x
    #     s_sum = torch.sum(wkv_states0.float())
    #     #del wkv_states0

    #     x1_sum = torch.sum(x1.float())
    #     #del x1
    #     s1_sum = torch.sum(wkv_states1.float())
    #     #del wkv_states1

    #     diff_x = x1_sum - x_sum
    #     diff_s = s1_sum - s_sum

        


    #     print(f'ctxlen = {idx.shape} x_similarity = {x_similarity} s_similarity = {s_similarity} x_sum = {x_sum} x1_sum = {x1_sum} diff_x = {diff_x} diff_s = {diff_s}')


    prompts = []
    for i in range(Target_batch):
            prompts.append(torch.tensor(tokens0).unsqueeze(0).to('cuda'))

    idx = torch.cat(prompts, dim=0)

    #print(idx.shape)
    # this is warmup for triton kernels
    x1, shift_states1, wkv_states1 = model.forward(copy.deepcopy(idx), copy.deepcopy(shift_states), copy.deepcopy(wkv_states),KernelMode=1) #FLA
    del x1
    del shift_states1
    del wkv_states1

    t_prefill_0 = time.perf_counter()
    x, shift_states, wkv_states = model.forward(copy.deepcopy(idx), shift_states, wkv_states,KernelMode=0) #FLA
    t_prefill_1 = time.perf_counter()


    prefilltoken_total = len(idx.view(-1))

    prefill_time = (t_prefill_1 - t_prefill_0)
    totaltime = prefill_time
    prefill_time = (float(prefilltoken_total)) / prefill_time

    print(f'totaltime = {totaltime} totaltoken = {prefilltoken_total} Prefill {prefill_time}t/s')




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #exit()

    print(context)
    out_tokens = [[] for _ in range(Target_batch)]
    out_last = [0 for _ in range(Target_batch)]
    output_text = ['' for _ in range(Target_batch)]
    

    FirstTime = 1

    t000 = time.perf_counter()
    min_time = 1e10
    min_time_all = 1e10
    min_time_all_single = 1e10

    maxtoken= 1000

    temperature = torch.full((Target_batch,), 1.0)
    top_p = torch.full((Target_batch,), 0.3)


    SamplingSum = 0
    ForwardSum = 0
    DecodeSum = 0

    for i in range(maxtoken):
        
        t0 = time.perf_counter()

        #x = x.view(-1,1)
        if x.dim() == 2:
            x = x.view(x.shape[0],1,x.shape[1])
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
                        #    print(tmp,end="", flush=True)
                        output_text[j] = output_text[j] + tmp
                        out_last[j] = i + 1
            except:
                pass
        t2 = time.perf_counter()

        x, shift_states, wkv_states = model.forward(idx, shift_states, wkv_states,one_mode=True)
        if x.dim() == 2:
            x = x.view(x.shape[0],1,x.shape[1])
        #print(x)
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
    print(f'totaltime = {totaltime} totaltoken = {prefilltoken_total} Prefill {prefill_time}t/s')
    print(f'TargetBatch = {Target_batch} Total token/s = {round(tokensec*Target_batch,2)} Single token/s = {round(tokensec,2)}')
