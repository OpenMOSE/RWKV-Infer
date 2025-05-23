# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch

from rwkvengine.fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule


def chunk_rwkv7(
    r: torch.Tensor,
    log_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
):
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        log_w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (torch.Tensor):
            initial state of shape `[B, H, K, V]` if cu_seqlens is None else `[N, H, K, V]` where N = len(cu_seqlens) - 1.
        output_final_state (bool):
            whether to output the final state.
        cu_seqlens (torch.LongTensor):
            cu_seqlens of shape `[B + 1]`: the cumulative sequence lengths, used to index into hidden_states as in Flash Attention.
            If None, it is used for variable length training.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """
    return chunk_dplr_delta_rule(
        q=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=log_w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        head_first=head_first,
        offsets=cu_seqlens
    )
