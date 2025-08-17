import torchao
from torchao.dtypes.floatx import to_scaled_tc_floatx
from torchao.ops import quant_llm_linear


import torch
import torch.nn as nn
from typing import Optional,List
import types, gc, os, time, re
from torch.nn import functional as F
import numpy as np
import os, sys
import time
#import bitsandbytes as bnb
import functools
from einops import rearrange
from torch.nn import functional as F
import math

#from rwkvengine.misc import PIPELINE
from rwkvengine.misc import PIPELINE, TimeMixState, ChannelMixState,BlockState,BlockStateList
from rwkvengine.matmularena import hybrid_matmul
from rwkvengine.fla.ops.rwkv6.chunk import chunk_rwkv6,ChunkRWKV6Function
from rwkvengine.fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
#from rwkvengine.fla.ops.rwkv7 import chunk_rwkv7,fused_recurrent_rwkv7
from rwkvengine.cuda.wkv7triton import rwkv7_attn_triton

from fla.ops.rwkv7 import chunk_rwkv7,fused_recurrent_rwkv7,fused_mul_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)


MyStatic = torch.jit.script

@MyStatic
def x070_ChannelMix_Experts_LoRA(x,K_ref,V_ref,K_lora_a,K_lora_b,V_lora_a,V_lora_b,scaling:float=2.0):
    #print('Channel Mix MoE')
    k = torch.relu(hybrid_matmul(x,K_ref) + scaling * F.linear(F.linear(x, K_lora_a), K_lora_b) ) ** 2
    v = hybrid_matmul(k,V_ref) + scaling * F.linear(F.linear(k, V_lora_a), V_lora_b)
    return v

def x070_ChannelMix_Experts_Bone2(x,K_ref,V_ref,K_bone,V_bone):
    #print('Channel Mix MoE')
    temporalweight = K_ref.to(dtype=x.dtype)#.t()
    # print(f'kbone shape = {K_bone.shape}')
    # print(f'vbone shape = {V_bone.shape}')
    # print(f'temporalweight shape = {temporalweight.shape}')
    w = rearrange(temporalweight, '(a r1) (b r2) -> a b r1 r2', r1 = K_bone.shape[1], r2 = K_bone.shape[1]) @ K_bone + K_bone
    w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
    k = torch.relu(x @ (w + temporalweight).t()) ** 2

    #del temporalweight
    #del w

    temporalweight = V_ref.to(dtype=x.dtype)
    w = rearrange(temporalweight, '(a r1) (b r2) -> a b r1 r2', r1 = V_bone.shape[1], r2 = V_bone.shape[1]) @ V_bone + V_bone
    w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')

    v = k @ (w + temporalweight).t()

    return v
@MyStatic
def x070_ChannelMix_Experts_Bone(x,K_ref,V_ref,K_bone,V_bone):
    temporalweight = K_ref.to(dtype=x.dtype)
    # w = rearrange(temporalweight, '(a r1) (b r2) -> a b r1 r2', r1 = K_bone.shape[1], r2 = K_bone.shape[1]) @ K_bone + K_bone
    # w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
    # k = torch.relu(x @ (w + temporalweight).t()) ** 2




    # weight の形状は (a*r, b*r) を想定
    r = K_bone.shape[1]
    a = temporalweight.size(0) // r
    b = temporalweight.size(1) // r

    # 1. (a*r, b*r) -> (a, r, b, r)
    # 2. (a, r, b, r) -> (a, b, r, r) に permute で変形（einops でいう 'a b r1 r2' と同等）
    w_4d = temporalweight.view(a, r, b, r).permute(0, 2, 1, 3)
    # w_4d の形状: [a, b, r, r]

    # 3. バッチ行列積 @ bone + bone （最後の2次元が (r, r) のためブロードキャスト加算が可能）
    w_4d = torch.matmul(w_4d, K_bone) + K_bone  # shape: [a, b, r, r]

    # 4. 再び元の形状 (a*r, b*r) に戻す (permuteを元に戻してから view)
    w_delta = w_4d.permute(0, 2, 1, 3).reshape(a*r, b*r)

    # 5. 最後に (weight + 変形した w_delta) を使って F.linear
    k = torch.relu(x @ (w_delta + temporalweight).t()) ** 2


    # temporalweight = V_ref.to(dtype=x.dtype)
    # w = rearrange(temporalweight, '(a r1) (b r2) -> a b r1 r2', r1 = V_bone.shape[1], r2 = V_bone.shape[1]) @ V_bone + V_bone
    # w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')

    # v = k @ (w + temporalweight).t()
    temporalweight = V_ref.to(dtype=x.dtype)
    r = V_bone.shape[1]
    a = temporalweight.size(0) // r
    b = temporalweight.size(1) // r

    # 1. (a*r, b*r) -> (a, r, b, r)
    # 2. (a, r, b, r) -> (a, b, r, r) に permute で変形（einops でいう 'a b r1 r2' と同等）
    w_4d = temporalweight.view(a, r, b, r).permute(0, 2, 1, 3)
    # w_4d の形状: [a, b, r, r]

    # 3. バッチ行列積 @ bone + bone （最後の2次元が (r, r) のためブロードキャスト加算が可能）
    w_4d = torch.matmul(w_4d, V_bone) + V_bone  # shape: [a, b, r, r]

    # 4. 再び元の形状 (a*r, b*r) に戻す (permuteを元に戻してから view)
    w_delta = w_4d.permute(0, 2, 1, 3).reshape(a*r, b*r)

    v = k @ (w_delta + temporalweight).t()

    return v


class RWKV_7(nn.Module):
    # x070 Multi batch Implementation
    # modified from RWKV-LM v7 demo_fast code @ BlinkDL
    # Now fully supported flash-linear-attention
    # Unofficial MoE Support 

    
    #@MyStatic
    def x070_TimeMix_fla_Step1(layer_id: int, H: int, N: int,
                        x, x_prev, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2,
                        v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_,
                        ln_w, ln_b):
        dtype = x.dtype
        B, T, _ = x.shape  # B, T, H*N
        
        #xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        
        xx = torch.cat([x_prev, x[:, :-1]], dim=1) - x  # (B,T,H*N) 
        #print(x_prev)
        #print(xx)
        #print(f'xx shape = {xx.shape}')
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        #r = xr @ R_
        r = hybrid_matmul(xr,R_)

        #w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        w_lora_result = w0 + (torch.tanh(xw @ w1) @ w2).float()
        log_w = -math.exp(-0.5) * torch.sigmoid(w_lora_result.float())   
        w = log_w#.exp()

        #k = xk @ K_
        k = hybrid_matmul(xk,K_)

        #v = xv @ V_
        v = hybrid_matmul(xv,V_)

        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)


        #k = k * (1 + (a-1) * k_a)
        k = fused_k_rwkv7(k, a, k_a)

        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
     

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)
        

        aa=-kk
        bb=kk*a

        return r,w,k,v,g,aa,bb,xx,v_first
    #@torch.compile
    def x070_TimeMix_fla_Step2(r, w, k, v, aa, bb,state,FullyFusedMode = True,offset_tensor:torch.Tensor=None):

        B,T,HC = w.shape
        C = state.shape[3]#64
        H = int(HC//C)

        #w=-torch.exp(w)
        #w = -w.to(dtype=torch.float32).exp()
        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,C) for i in [r,w,k,v,aa,bb]]
        B,T,_,_ = r_.shape
        if T>128 and FullyFusedMode == False:
            xx, state = chunk_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state,cu_seqlens=None, output_final_state=True, head_first=False)
        else:
            xx, state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=state, output_final_state=True, head_first=False)

        return xx, state
    @MyStatic
    def x070_TimeMix_fla_Step3(B:int,T:int,H:int,N:int,r,k,r_k,v,g,O_,x,xx,state,v_first,ln_w,ln_b):

        xx = xx.view(B,T,-1).to(dtype=r.dtype)
        # xx = xx.permute(0, 2, 1)  # (B,H*N,T)
        # xx = torch.nn.functional.group_norm(xx, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)#.view(B*T,-1)
        # xx = xx.permute(0, 2, 1)

        xx = xx.view(B * T, -1)
        xx = torch.nn.functional.group_norm(xx.to(dtype=ln_w.dtype),num_groups = H, weight=ln_w,bias=ln_b, eps= 64e-5).view(B, T, -1)
        





        #xx = xx.view(B,T,-1)
        xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
        xx=xx.to(dtype=g.dtype)
        #return (xx * g) @ O_, x[:,-1], state.float(), v_first
        return hybrid_matmul((xx * g) , O_), x[:,-1:], state.float(), v_first
        #return hybrid_matmul((xx * g) , O_), x[:,-1], state, v_first
        #hybrid_matmul


    


    @MyStatic
    def x070_ChannelMix_seq(x, x_prev, x_k, K_, V_):
        #xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1, :]], dim=1) - x  # (B,T,H*N)
        xx = torch.cat([x_prev, x[:, :-1]], dim=1) - x  # (B,T,H*N)
        k = x + xx * x_k
        #k = torch.relu(k @ K_) ** 2
        k = torch.relu(hybrid_matmul(k , K_)) ** 2

        #hybrid_matmul
        #return k @ V_, x[:,-1,:]
        return hybrid_matmul(k , V_), x[:,-1:]
    




    @torch.compile
    def x070_ChannelMix_MoE_2(
        x, x_prev, x_k,
        Router_ref,
        K_ref, V_ref,
        Experts_K: List[List[torch.Tensor]],
        Experts_V: List[List[torch.Tensor]],
        MoETopk: int = 2,
        MoECount: int = 4,
        MoEMode: int = 0
    ):

        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1, :]], dim=1) - x
        hidden_with_tokenshift = x + xx * x_k

        # (B, S, n_embd) → (B*S, n_embd)
        flat_hidden = hidden_with_tokenshift.reshape(-1, hidden_with_tokenshift.size(-1))
        B = flat_hidden.size(0)


        flat_value = torch.zeros_like(flat_hidden)


        if MoEMode == 0:
            out_0 = x070_ChannelMix_Experts_LoRA(
                flat_hidden,
                K_ref, V_ref,
                Experts_K[0][0], Experts_K[0][1],
                Experts_V[0][0], Experts_V[0][1]
            )
        else:
            # 別の MoEMode 実装？
            raise NotImplementedError()

        flat_value += out_0

        router_scores = F.linear(flat_hidden, Router_ref)  # weight = Router_ref

        # topk
        AdaptiveActiveExperts = MoETopk - 1  # 例: 2 なら 1
        topk_values, topk_experts = torch.topk(router_scores, k=AdaptiveActiveExperts, dim=-1)
        gating = F.softmax(topk_values, dim=-1)  # shape [B, AdaptiveActiveExperts]


        topk_experts_flat = topk_experts.reshape(-1)
        gating_flat = gating.reshape(-1)

        source_indices = torch.arange(B, device=flat_hidden.device).unsqueeze(1)
        source_indices = source_indices.expand(B, AdaptiveActiveExperts).reshape(-1)

        for e in range(1, MoECount):
            mask_e = (topk_experts_flat == (e - 0))  # 例: 1番目expertを topk_experts_flat==1

            if not mask_e.any():
                continue

            indices_e = mask_e.nonzero(as_tuple=True)[0]
            input_e = flat_hidden[source_indices[indices_e]]
            # forward
            if MoEMode == 0:
                out_e = x070_ChannelMix_Experts_LoRA(
                    input_e,
                    K_ref, V_ref,
                    Experts_K[e][0], Experts_K[e][1],
                    Experts_V[e][0], Experts_V[e][1]
                )
            else:
                raise NotImplementedError()

            out_e = out_e * gating_flat[indices_e].unsqueeze(-1)
            flat_value.index_add_(0, source_indices[indices_e], out_e)

        # (B*S, n_embd) → (B, S, n_embd) に戻す
        kv = flat_value.view(x.size(0), x.size(1), x.size(2))
        return kv,  x[:,-1,:]
    

    @torch.compile
    def x070_ChannelMix_MoE(
        x, x_prev, x_k,
        Router_ref,
        K_ref, V_ref,
        Experts_K: List[List[torch.Tensor]],
        Experts_V: List[List[torch.Tensor]],
        MoETopk: int = 2,
        MoECount: int = 4,
        MoEMode: int = 0
    ):
        """
        推論用 x070_ChannelMix_MoE 関数（学習時と同じ処理を再現）
        
        Args:
            x: Tensor of shape (B, S, n_embd)
            x_prev: Tensor of shape (B, S, n_embd)
            x_k: Tensor of shape (1, 1, n_embd)（token shift 用のパラメータ）
            Router_ref: ルーター用の重みテンソル（F.linear によるスコア計算に使用）
            K_ref, V_ref: 各 Expert のベースとなる重み（LoRA での変換に使用）
            Experts_K: 各 Expert 用の K 関連パラメータのリスト。Experts_K[e] は Expert e 用の [W_A, W_B] など。
            Experts_V: 各 Expert 用の V 関連パラメータのリスト。
            MoETopk: トークンごとに選択する Expert の総数（例: 2 なら、expert0 ともう1つを選択）
            MoECount: Expert の総数（例: 4）
            MoEMode: モード選択（ここでは 0 のみ実装）
            
        Returns:
            kv: 出力テンソル (B, S, n_embd)
            x[:, -1, :]: x の最終トークン（下位層への情報伝達などに利用）
        """
        # 1. Token Shift 処理
        # x_prev の先頭に1次元追加し、x のシフト版を作成（x_prev と x のずれ分を計算）
        xx = torch.cat([x_prev.unsqueeze(1), x[:, :-1, :]], dim=1) - x
        hidden_with_tokenshift = x + xx * x_k  # (B, S, n_embd)
        
        # (B, S, n_embd) -> (B*S, n_embd)
        flat_hidden = hidden_with_tokenshift.reshape(-1, hidden_with_tokenshift.size(-1))
        B_tokens = flat_hidden.size(0)  # 総トークン数

        # 2. ルーターによる各 Expert のスコア計算
        # shared_expert の場合、expert0 は常に計算するので、expert0 のスコアは 0 として付加する
        if MoEMode == 0:
            # experts 1～MoECount-1 のルーター出力を計算（shape: [B_tokens, MoECount-1]）
            router_scores_shared = F.linear(flat_hidden, Router_ref)
            # expert0 のスコアは 0（shape: [B_tokens, 1]）
            expert0_score = torch.zeros(B_tokens, 1, device=flat_hidden.device, dtype=router_scores_shared.dtype)
            # 全 Expert のスコアを連結（shape: [B_tokens, MoECount]）
            router_scores_all = torch.cat([expert0_score, router_scores_shared], dim=-1)
        else:
            raise NotImplementedError("MoEMode==0 のみ実装しています。")
        
        # 3. トークンごとに top-k の Expert を選び、正規化した gating 重みを算出
        if MoETopk < MoECount:
            # 各トークンについて、上位 MoETopk のスコアとそのインデックスを取得
            topk_values, topk_indices = torch.topk(router_scores_all, k=MoETopk, dim=-1)  # shape: (B_tokens, MoETopk)
            gating_active = F.softmax(topk_values, dim=-1)  # 各トークンにおける選択された Expert の重み（和は1）
            # 全 Expert 用の gating テンソルを作成（初期値0）
            gating_full = torch.zeros_like(router_scores_all)
            # topk_indices に対応する位置に gating_active の値を散らばせる
            gating_full.scatter_(1, topk_indices, gating_active)
        else:
            # MoETopk == MoECount の場合、全 Expert に対して softmax を適用
            gating_full = F.softmax(router_scores_all, dim=-1)
            topk_indices = torch.arange(MoECount, device=flat_hidden.device).unsqueeze(0).expand(B_tokens, MoECount)
        
        # 4. 各 Expert の出力を、対応する gating 重みで加重平均する
        flat_value = torch.zeros_like(flat_hidden)
        for e in range(MoECount):
            # 各トークンについて、top-k 選択された Expert インデックスが e である箇所を抽出
            mask = (topk_indices == e)  # shape: (B_tokens, MoETopk)
            if mask.sum() == 0:
                continue
            # mask==True となる各要素について、flat_hidden 内のトークン番号と topk 内の位置を取得
            token_info = torch.nonzero(mask, as_tuple=False)  # shape: (N, 2)；各行 [token_index, topk_position]
            tokens = token_info[:, 0]  # flat_hidden のインデックス
            # gating 重みは gating_full から取得（各トークンについて該当 Expert の重み）
            gating_weights = gating_full[tokens, e]  # shape: (N,)
            
            # 対象トークンに対し、Expert e の forward を実行
            if MoEMode == 0:
                out_e = x070_ChannelMix_Experts_LoRA(
                    flat_hidden[tokens],
                    K_ref, V_ref,
                    Experts_K[e][0], Experts_K[e][1],
                    Experts_V[e][0], Experts_V[e][1]
                )
            else:
                raise NotImplementedError("MoEMode==0 のみ実装しています。")
            # 各出力に gating 重みを乗算
            out_e = out_e * gating_weights.unsqueeze(-1)
            # 結果を flat_value に index_add_ で加算
            flat_value.index_add_(0, tokens, out_e)
        
        # 5. (B*S, n_embd) を (B, S, n_embd) に変形して出力
        kv = flat_value.view(x.size(0), x.size(1), x.size(2))
        
        return kv, x[:, -1, :]
    


    def x070_forward_one(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor] ):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, time_mix_shift, time_mix_state, v_first = self.x070_TimeMix_one(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, channel_mix_state = self.x070_ChannelMix_one(xx, channel_mix_state, z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

                last_shift_states[i*2] = time_mix_shift.view(time_mix_shift.shape[0],-1)
                last_shift_states[i*2+1] = channel_mix_state.view(channel_mix_state.shape[0],-1)
                
                last_wkv_states[i] = time_mix_state
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            #exit()
            return x, last_shift_states, last_wkv_states
        
    def x070_forward_seq(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor],  full_output:bool=False, KernelMode:int=0,time_offset_state:torch.Tensor=None):
        with torch.no_grad(): 
            z = self.z


            #x = z['emb.weight'][idx]
            if self.emboncpu:
                x = z['emb.weight'][idx.cpu()].to(device=self.device,dtype=self.dtype)
            else:
                x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)

            StrategyMode = 0 # 0 is Fully BF16 or FP16 or FP8
            if self.bit4quant == True:
                StrategyMode = 2
            elif self.bitfp6quant == True:
                StrategyMode = 3



            if time_offset_state is None:
                time_offset_state = None#torch.zeros((self.n_layer,idx.shape[0],self.n_head,self.head_size,self.head_size),dtype=x.dtype, device=x.device)


            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                time_mix_shift = last_shift_states[i*2]
                channel_mix_state = last_shift_states[i*2+1]
                time_mix_state = last_wkv_states[i]


                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                B, T, X = xx.shape

                if StrategyMode == 0:
        
                    r,w,k,v,g,aa,bb,xx_step1,v_first = RWKV_7.x070_TimeMix_fla_Step1(i, self.n_head, self.head_size, xx, time_mix_shift, v_first, time_mix_state,
                                                                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                                                                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                                                                        z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                                                                        z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                                                                        z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                    #print(f'time_offset_state total shape = {time_offset_state.shape}')
                    xx_step2, time_mix_state = RWKV_7.x070_TimeMix_fla_Step2(r,w,k,v,aa,bb,time_mix_state,self.fully_fusedrecurrent)

                    xx, time_mix_shift, time_mix_state, v_first = RWKV_7.x070_TimeMix_fla_Step3(B,T,self.n_head,self.head_size,r,k,z[att+'r_k'],v,g,z[att+'output.weight'],
                                                                                                xx,xx_step2,time_mix_state,v_first,z[att+'ln_x.weight'], z[att+'ln_x.bias'])



                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])


                if self.MoE == 1:
                    ExpertsCount = self.MoEExperts
                    Experts_K = []
                    Experts_V = []
                    keys = list(z.keys())

                    bonetext = ffn + "expert_0.key.bone_expert_0"
                    #loratext = ffn + "expert_0.key.lora_A_expert_0"
                    expertmode = self.MoELayerMode[i]
                    # for key in keys:
                    #     if bonetext in key:
                    #         #print(f'found bone mode in {key}' )
                    #         expertmode = 1
                    #         break

                    #print('MoE TestMode')
                    for a in range(ExpertsCount):
                        Parts = []
                        if expertmode ==1:
                            Parts.append(z[ffn+f'expert_{a}.key.bone_expert_0'])
                            Parts.append(torch.tensor(0))
                        else:
                            Parts.append(z[ffn+f'expert_{a}.key.lora_A_expert_0'])
                            Parts.append(z[ffn+f'expert_{a}.key.lora_B_expert_0'])
                        Experts_K.append(Parts)

                        Parts2 = []
                        if expertmode == 1:
                            Parts2.append(z[ffn+f'expert_{a}.value.bone_expert_0'])
                            Parts2.append(torch.tensor(0))
                        else:
                            Parts2.append(z[ffn+f'expert_{a}.value.lora_A_expert_0'])
                            Parts2.append(z[ffn+f'expert_{a}.value.lora_B_expert_0'])
                        Experts_V.append(Parts2)
                        

                    
                    xx, channel_mix_state = RWKV_7.x070_ChannelMix_MoE(xx,channel_mix_state,z[ffn+'x_k'],z[ffn+'router.linear.weight'],z[ffn+'key.weight'],z[ffn+'value.weight'],
                                                                           Experts_K,Experts_V,MoETopk=self.ActiveMoEs,MoEMode=expertmode,MoECount=self.MoEExperts                                                                           
                                                                           )
                else:
                    xx, channel_mix_state = RWKV_7.x070_ChannelMix_seq(xx, channel_mix_state, z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])

                x = x + xx

                last_shift_states[i*2] = time_mix_shift
                last_shift_states[i*2+1] = channel_mix_state
                last_wkv_states[i] = time_mix_state

            
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = hybrid_matmul(x , z['head.weight'])
            if not full_output: x = x[:, -1, :]  # 最後のタイムステップだけを選択し、バッチ次元を保持

            return x, last_shift_states, last_wkv_states
        
    def x070_forward(self, idx, last_shift_states: List[torch.Tensor],
                last_wkv_states: List[torch.Tensor], full_output=False,one_mode=False,KernelMode = 0,time_offset_state:torch.Tensor=None):
        #if one_mode:
        #    return self.x070_forward_one(idx, last_shift_states, last_wkv_states)
        return RWKV_7.x070_forward_seq(self,idx, last_shift_states,last_wkv_states, full_output,KernelMode,time_offset_state=time_offset_state)

 
    
 