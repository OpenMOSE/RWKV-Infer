import torch
import triton
import triton.language as tl
# Help me if you have good idea:)
# This is int8 fused matmul triton kernel


# --------------------------------
# Triton fused dequant→FP16 GEMM kernel (3D grid: B, T_blocks, N_blocks)
# --------------------------------
@triton.jit
def fused_dequant_gemm_T1_kernel(
    A_ptr, W_ptr, scale_ptr, C_ptr,
    M, K, N,                      # ここで M = B*T
    stride_A_M, stride_A_K,
    stride_W_K, stride_W_N,
    stride_scale,
    stride_C_M, stride_C_N,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # --- 行 (m) と 列タイル (n_block) を決定 ---
    m_idx   = tl.program_id(0)                # [0 .. M)
    n_block = tl.program_id(1)                # [0 .. ceil(N/BLOCK_N))

    offs_n  = n_block * BLOCK_N + tl.arange(0, BLOCK_N)   # 列インデックス

    # 出力バッファ（float32 Accum）
    acc = tl.zeros((BLOCK_N,), tl.float32)

    # K を BLOCK_K ごとに回す
    num_kb = tl.cdiv(K, BLOCK_K)
    for kb in range(num_kb):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        # ---- A_vec : [BLOCK_K] ----
        a_ptr = A_ptr + m_idx * stride_A_M + offs_k * stride_A_K
        a_vec = tl.load(a_ptr, mask=offs_k < K, other=0.).to(tl.float16)

        # ---- W_sub (int8 → fp16) : [BLOCK_K, BLOCK_N] ----
        w_ptr = (
            W_ptr
            + offs_k[:, None] * stride_W_K
            + offs_n[None, :] * stride_W_N
        )
        w_int8 = tl.load(w_ptr, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                         other=0).to(tl.int8)

        # ---- スケール ----
        s_ptr   = scale_ptr + offs_k * stride_scale
        scale_k = tl.load(s_ptr, mask=offs_k < K, other=1.).to(tl.float16)

        w_sub = w_int8.to(tl.float16) * scale_k[:, None]   # de‑quant

        # ---- FMA ----  (1 × K) · (K × BLOCK_N)
        acc += tl.sum(a_vec[:, None] * w_sub, 0)

    # ---- store ----
    c_ptr = C_ptr + m_idx * stride_C_M + offs_n * stride_C_N
    tl.store(c_ptr, acc.to(tl.float16), mask=offs_n < N)

@triton.jit
def fused_dequant_gemm_kernel(
    A_ptr,               # ptr to FP16 A [B, T, K]
    W_ptr,               # ptr to int8 W [K, N]
    scale_ptr,           # ptr to FP16 scale [K]
    C_ptr,               # ptr to FP16 output [B, T, N]
    B, T, K, N,
    stride_A_B, stride_A_T, stride_A_K,
    stride_W_K, stride_W_N,
    stride_scale,
    stride_C_B, stride_C_T, stride_C_N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # グリッドインデックス
    b_idx   = tl.program_id(0)   # [0..B)
    t_block = tl.program_id(1)   # [0..ceil(T/BLOCK_M))
    n_block = tl.program_id(2)   # [0..ceil(N/BLOCK_N))

    # タイル内オフセット
    offs_m = tl.arange(0, BLOCK_M)    # M
    offs_n = tl.arange(0, BLOCK_N)    # N
    t_idx  = t_block * BLOCK_M + offs_m  # [M]
    n_idx  = n_block * BLOCK_N + offs_n  # [N]

    # FP32 accumulation buffer
    C_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K を BLOCK_K ずつ回すセグメントループ
    num_kb = tl.cdiv(K, BLOCK_K)
    for kb in range(num_kb):
        k_start = kb * BLOCK_K
        offs_k  = k_start + tl.arange(0, BLOCK_K)  # [K_seg]

        # --- A_sub: [M, K_seg] のロード ---
        off_A  = (
            b_idx * stride_A_B +
            t_idx[:, None] * stride_A_T +
            offs_k[None, :] * stride_A_K
        )
        mask_A = (t_idx[:, None] < T) & (offs_k[None, :] < K)
        A_sub  = tl.load(A_ptr + off_A, mask=mask_A, other=0.0).to(tl.float16)

        # --- W_sub(int8) のロード＆デクォント → FP16 ---
        off_W   = (
            offs_k[:, None] * stride_W_K +
            n_idx[None, :]  * stride_W_N
        )
        mask_W  = (offs_k[:, None] < K) & (n_idx[None, :] < N)
        W_int8  = tl.load(W_ptr + off_W, mask=mask_W, other=0).to(tl.int8)
        off_s   = offs_k * stride_scale
        mask_s  = offs_k < K
        scale_s = tl.load(scale_ptr + off_s, mask=mask_s, other=1.0).to(tl.float16)
        W_sub   = W_int8.to(tl.float16) * scale_s[:, None]  # [K_seg, N]

        # 部分 GEMM → float32 累積
        C_block += tl.dot(A_sub, W_sub)

    # FP16 にキャストして出力へストア
    off_C  = (
        b_idx * stride_C_B +
        t_idx[:, None] * stride_C_T +
        n_idx[None, :]  * stride_C_N
    )
    mask_C = (t_idx[:, None] < T) & (n_idx[None, :] < N)
    tl.store(C_ptr + off_C, C_block.to(tl.float16), mask=mask_C)


# --------------------------------
# Python wrapper
# --------------------------------
def fused_dequant_gemm(A: torch.Tensor,
                       W_int8: torch.Tensor,
                       scale: torch.Tensor,
                       BLOCK_M=16, BLOCK_N=16, BLOCK_K=1024):
    """
    A:       (B, T, K) float16
    W_int8:  (K, N)     int8
    scale:   (K,)       float16
    returns: (B, T, N)  float16
    """
    B, T, K = A.shape
    K2, N   = W_int8.shape
    assert K2 == K, "A の last dim と W の first dim が一致しません"
    assert scale.shape[0] == K

    C_out = torch.zeros((B, T, N), device=A.device, dtype=torch.float16)

    if T <= 4 and B <= 16:
        A_flat = A.view(B * T, K).to(dtype=torch.float16)
        C_flat = C_out.view(B * T, N)

        grid = (B * T,
                (N + BLOCK_N - 1) // BLOCK_N)

        fused_dequant_gemm_T1_kernel[grid](
            A_flat, W_int8, scale, C_flat,
            B * T, K, N,
            # strides
            A_flat.stride(0), A_flat.stride(1),
            W_int8.stride(0), W_int8.stride(1),
            scale.stride(0),
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=16, 
        )
        return C_out.to(dtype=A.dtype)
    
    else:
        return reference_dequant_gemm(A.to(dtype=torch.float16),W_int8,scale).to(dtype=A.dtype)
   

    
    grid  = (B,
             (T + BLOCK_M - 1) // BLOCK_M,
             (N + BLOCK_N - 1) // BLOCK_N)
    fused_dequant_gemm_kernel[grid](
        A.to(dtype=torch.float16), W_int8, scale, C_out,
        B, T, K, N,
        # strides for A
        A.stride(0), A.stride(1), A.stride(2),
        # strides for W
        W_int8.stride(0), W_int8.stride(1),
        # stride for scale
        scale.stride(0),
        # strides for C_out
        C_out.stride(0), C_out.stride(1), C_out.stride(2),
        # tile sizes
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C_out.to(dtype=A.dtype)


# --------------------------------
# Quantization helper (pre-process)
# --------------------------------
def quantize_weight(W_fp16: torch.Tensor, qmax: int = 127):
    """
    W_fp16: (K, N) float16
    returns:
      W_int8: (K, N) int8
      scale:  (K,)   float16
    """
    max_abs = W_fp16.abs().amax(dim=1)            # [K]
    scale   = (max_abs / qmax).clamp(min=1e-8)     # [K]
    W_scaled= W_fp16 / scale.unsqueeze(1)         # [K, N]
    W_int8  = W_scaled.round().clamp(-qmax, qmax).to(torch.int8)
    return W_int8, scale.to(torch.float16)


# --------------------------------
# Reference implementation & test
# --------------------------------
@torch.compile
def reference_dequant_gemm(A, W_int8, scale):
    """
    A:      (B, T, K)
    returns: (B, T, N)
    """
    W_fp16 = W_int8.to(torch.float16) * scale.view(-1, 1)
    return torch.matmul(A, W_fp16)


def test_quant_fused():
    torch.manual_seed(0)
    B, T, K, N = 2, 37, 128, 64  # 任意の非正方行列サイズも OK
    A      = torch.randn(B, T, K, device='cuda', dtype=torch.float16)
    W_fp16 = torch.randn(K, N, device='cuda', dtype=torch.float16)
    W_int8, scale = quantize_weight(W_fp16)

    C_pred = fused_dequant_gemm(A, W_int8, scale)
    C_ref  = reference_dequant_gemm(A, W_int8, scale)
    max_err= (C_pred - C_ref).abs().max().item()
    print(f"Max abs error = {max_err:.4f}")
    assert max_err < 1e-2, f"誤差 {max_err} が許容範囲を超えました"
    print("✅ quant→fused GEMM test passed!")


# --------------------------------
# Benchmark
# --------------------------------
def benchmark(B=1, T=1, K=4096, N=1024, iters=100):
    A      = torch.randn(B, T, K, device='cuda', dtype=torch.float16)
    W_fp16 = torch.randn(K, N, device='cuda', dtype=torch.float16)
    W_int8, scale = quantize_weight(W_fp16)

    # warm-up
    for _ in range(10):
        fused_dequant_gemm(A, W_int8, scale)
    torch.cuda.synchronize()

    # Triton kernel
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fused_dequant_gemm(A, W_int8, scale)
    end.record(); torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / iters

    # PyTorch matmul
    start2 = torch.cuda.Event(True); end2 = torch.cuda.Event(True)
    start2.record()
    for _ in range(iters):
        reference_dequant_gemm(A, W_int8, scale)
    end2.record(); torch.cuda.synchronize()
    torch_ms = start2.elapsed_time(end2) / iters

    print(f"Triton kernel:  {triton_ms:.3f} ms/iter")
    print(f"PyTorch matmul: {torch_ms:.3f} ms/iter")
    print(f"Speedup:        {torch_ms / triton_ms:.2f}x")


# --------------------------------
# Main
# --------------------------------
if __name__ == "__main__":
    test_quant_fused()
    print()
    print("Benchmark (B=16, T=1, K=4096, N=1024, 100 iters):")
    benchmark()
