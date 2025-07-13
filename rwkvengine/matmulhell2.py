import torch
import triton
import triton.language as tl

# --------------------------------
# Triton fused dequant + vecmat kernel for T==1
# --------------------------------
@triton.jit
def fused_dequant_vecmat_dot_kernel(
    A_ptr, W_ptr, scale_ptr, C_ptr,
    B, K, N,
    sAB, sAK, sWK, sWN, sS, sCB, sCN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    nb = tl.program_id(1)

    # タイル内の行・列インデックス
    rm = tl.arange(0, BLOCK_M)   # M=16
    cn = tl.arange(0, BLOCK_N)   # N タイル
    rb = b * sAB
    cb = nb * BLOCK_N * sCN

    # M×K の入力タイル。最初の行だけ A[b,*]、残り行は 0。
    offs_k = tl.arange(0, BLOCK_K)
    off_A = rb + offs_k * sAK  # [BLOCK_K]
    mask_k = offs_k < K
    A_line = tl.load(A_ptr + off_A, mask=mask_k, other=0.0).to(tl.float16)  # [BLOCK_K]
    # [M,K]: 第一行に入れて、他は0に
    A_sub = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float16)
    A_sub = tl.where(rm[:, None] == 0, A_line[None, :], A_sub)

    # dequant + load W_sub
    off_W = offs_k[:, None] * sWK + (nb * BLOCK_N + cn)[None, :] * sWN  # [BLOCK_K, BLOCK_N]
    mask_W = (offs_k[:, None] < K) & ((nb * BLOCK_N + cn)[None, :] < N)
    W_i8 = tl.load(W_ptr + off_W, mask=mask_W, other=0).to(tl.int8)
    off_s = offs_k * sS
    mask_s = offs_k < K
    scale_f = tl.load(scale_ptr + off_s, mask=mask_s, other=1.0).to(tl.float16)
    W_sub = W_i8.to(tl.float16) * scale_f[:, None]  # [BLOCK_K, BLOCK_N]

    # M×N タイルとして dot 実行
    C_block = tl.dot(A_sub, W_sub)  # [BLOCK_M, BLOCK_N] float32 accumulation
    # FP16 に戻して store。rm==0 の行だけ残す
    C_f16 = C_block.to(tl.float16)
    # 出力先オフセット
    base_C = b * sCB + (nb * BLOCK_N) * sCN
    offs_M = tl.arange(0, BLOCK_M)[:, None] * sCB
    offs_N = cn[None, :] * sCN
    off_C  = base_C + offs_M + offs_N
    mask_out = (rm[:, None] == 0) & (cn[None, :] < N)
    tl.store(C_ptr + off_C, C_f16, mask=mask_out)


# --------------------------------
# Wrapper dispatching vector-mat for T=1
# --------------------------------
def fused_dequant_gemm(A: torch.Tensor,
                       W_int8: torch.Tensor,
                       scale: torch.Tensor,
                       BLOCK_N=16, BLOCK_K=32):
    """
    A:      (B, 1, K) float16
    W_int8: (K, N)     int8
    scale:   (K,)      float16
    ->      (B, 1, N)  float16
    """
    B, T, K = A.shape
    _, N    = W_int8.shape
    assert T == 1, "This wrapper is for T=1 only"
    # flatten A to (B,K)
    A2 = A.view(B, K)

    C_out = torch.zeros((B, N), device=A.device, dtype=torch.float16)
    grid  = (B, (N + BLOCK_N - 1) // BLOCK_N)
    fused_dequant_vecmat_dot_kernel[grid](
        A2, W_int8, scale, C_out,
        B, K, N,
        A2.stride(0), A2.stride(1),
        W_int8.stride(0), W_int8.stride(1),
        scale.stride(0),
        C_out.stride(0), C_out.stride(1),BLOCK_M=16,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C_out.unsqueeze(1)


# --------------------------------
# Test & benchmark
# --------------------------------
def quantize_weight(W_fp16, qmax=127):
    max_abs = W_fp16.abs().amax(dim=1)
    scale   = (max_abs / qmax).clamp(min=1e-8)
    W_scaled= W_fp16 / scale.unsqueeze(1)
    W_int8  = W_scaled.round().clamp(-qmax, qmax).to(torch.int8)
    return W_int8, scale.to(torch.float16)

def reference(A, W_int8, scale):
    W_fp16 = W_int8.to(torch.float16) * scale.view(-1,1)
    return torch.matmul(A.view(-1, K:=W_int8.shape[0]), W_fp16).view(A.shape[0],1,-1)

def test():
    torch.manual_seed(0)
    B,K,N = 4,512,1024
    A      = torch.randn(B,1,K, device='cuda', dtype=torch.float16)
    W_fp16 = torch.randn(K,N, device='cuda', dtype=torch.float16)
    W_int8, scale = quantize_weight(W_fp16)
    C1 = fused_dequant_gemm(A, W_int8, scale)
    C2 = reference(A, W_int8, scale)
    assert torch.allclose(C1, C2, atol=1e-2)
    print("✅ T=1 fused vecmat test passed!")

def benchmark():
    import time
    B,K,N,iters = 64,4096,4096,200
    A      = torch.randn(B,1,K, device='cuda', dtype=torch.float16)
    W_fp16 = torch.randn(K,N, device='cuda', dtype=torch.float16)
    W_int8, scale = quantize_weight(W_fp16)
    # warmup
    for _ in range(10):
        fused_dequant_gemm(A, W_int8, scale)
    torch.cuda.synchronize()
    start,end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fused_dequant_gemm(A, W_int8, scale)
    end.record(); torch.cuda.synchronize()
    print(f"T=1 fused vecmat: {start.elapsed_time(end)/iters:.3f} ms/iter")

if __name__ == "__main__":
    test()
    benchmark()

