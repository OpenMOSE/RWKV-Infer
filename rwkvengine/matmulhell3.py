import triton
import triton.language as tl
import torch
from triton.runtime import driver
try:
    import pandas as pd
except ImportError:
    print("pandas not found, some features will be limited")
    pd = None


def fused_dequant_gemm(A: torch.Tensor,
                       W_int8: torch.Tensor,
                       scale: torch.Tensor,
                       BLOCK_N=16, BLOCK_K=256):
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
# Auto-tuning configuration
# --------------------------------
def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=2, num_warps=8),
    ]

# --------------------------------
# Optimized kernel with auto-tuning
# --------------------------------
@triton.autotune(
    configs=get_autotune_config(),
    key=['B', 'K', 'N'],
)
@triton.jit
def fused_dequant_vecmat_optimized_kernel(
    A_ptr, W_ptr, scale_ptr, C_ptr,
    B, K, N,
    sAB, sAK, sWK, sWN, sS, sCB, sCN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    nb = tl.program_id(1)
    
    # タイル内の行・列インデックス
    rm = tl.arange(0, BLOCK_M)
    cn = tl.arange(0, BLOCK_N)
    rb = b * sAB
    cb = nb * BLOCK_N * sCN
    
    # アキュムレータの初期化（最初の行のみ必要）
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # K次元をブロック単位でループ
    K_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(K_tiles):
        k = k_tile * BLOCK_K
        
        # A行列のロード（最初の行のみ）
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        off_A = rb + offs_k * sAK
        A_vec = tl.load(A_ptr + off_A, mask=mask_k, other=0.0).to(tl.float16)
        
        # W行列のロード（int8）
        off_W = offs_k[:, None] * sWK + (nb * BLOCK_N + cn)[None, :] * sWN
        mask_W = (offs_k[:, None] < K) & ((nb * BLOCK_N + cn)[None, :] < N)
        W_i8 = tl.load(W_ptr + off_W, mask=mask_W, other=0).to(tl.int8)
        
        # スケールのロード
        off_s = offs_k * sS
        mask_s = offs_k < K
        scale_vec = tl.load(scale_ptr + off_s, mask=mask_s, other=1.0).to(tl.float16)
        
        # デクオンタイズ
        W_f16 = W_i8.to(tl.float16) * scale_vec[:, None]
        
        # ドット積の計算（ベクトル×行列）
        acc += tl.sum(A_vec[:, None] * W_f16, axis=0)
    
    # 結果の格納（最初の行のみ）
    C_f16 = acc.to(tl.float16)
    base_C = b * sCB + (nb * BLOCK_N) * sCN
    offs_N = cn * sCN
    off_C = base_C + offs_N
    mask_out = (nb * BLOCK_N + cn) < N
    tl.store(C_ptr + off_C, C_f16, mask=mask_out)

# --------------------------------
# 追加の最適化：ベクトル専用カーネル
# --------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_stages=2, num_warps=8),
    ],
    key=['K', 'N'],
)
@triton.jit
def vector_dequant_matmul_kernel(
    A_ptr, W_ptr, scale_ptr, C_ptr,
    K, N,
    sAK, sWK, sWN, sS, sCN,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """T=1専用の最適化されたベクトル×行列カーネル"""
    b = tl.program_id(0)
    nb = tl.program_id(1)
    
    # アキュムレータ
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Nタイルのオフセット
    cn = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cn < N
    
    # K次元をループ
    A_base = b * sAK
    for k in range(0, K, BLOCK_K):
        # Aベクトルのロード
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        A_vec = tl.load(A_ptr + A_base + offs_k * sAK, mask=mask_k, other=0.0).to(tl.float16)
        
        # W行列のロード
        off_W = offs_k[:, None] * sWK + cn[None, :] * sWN
        mask_W = mask_k[:, None] & mask_n[None, :]
        W_i8 = tl.load(W_ptr + off_W, mask=mask_W, other=0).to(tl.int8)
        
        # スケールのロード
        scale_vec = tl.load(scale_ptr + offs_k * sS, mask=mask_k, other=1.0).to(tl.float16)
        
        # デクオンタイズ
        W_f16 = W_i8.to(tl.float16) * scale_vec[:, None]
        
        # ドット積の計算
        acc += tl.sum(A_vec[:, None] * W_f16, axis=0)
    
    # 結果の格納
    C_base = b * N + cn
    tl.store(C_ptr + C_base * sCN, acc.to(tl.float16), mask=mask_n)

# --------------------------------
# 改良版ラッパー関数
# --------------------------------
def fused_dequant_gemm_optimized(
    A: torch.Tensor,
    W_int8: torch.Tensor,
    scale: torch.Tensor,
    use_vector_kernel: bool = True
):
    """
    最適化された8ビット量子化行列積
    
    Args:
        A: (B, 1, K) float16
        W_int8: (K, N) int8
        scale: (K,) float16
        use_vector_kernel: T=1用の専用カーネルを使うか
    
    Returns:
        (B, 1, N) float16
    """
    B, T, K = A.shape
    _, N = W_int8.shape
    assert T == 1, "This wrapper is for T=1 only"
    
    # デバイスとデータ型の確認
    assert A.device == W_int8.device == scale.device
    assert A.dtype == torch.float16
    assert W_int8.dtype == torch.int8
    assert scale.dtype == torch.float16
    
    # メモリアライメントの確認
    def is_aligned(tensor, alignment=16):
        return tensor.data_ptr() % alignment == 0
    
    # 必要に応じてcontiguousにする
    if not A.is_contiguous():
        A = A.contiguous()
    if not W_int8.is_contiguous():
        W_int8 = W_int8.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    
    A2 = A.view(B, K)
    
    if use_vector_kernel:
        # ベクトル専用カーネルを使用
        C_out = torch.zeros((B, N), device=A.device, dtype=torch.float16)
        grid = (B, triton.cdiv(N, 128))  # BLOCK_Nのデフォルト値
        
        vector_dequant_matmul_kernel[grid](
            A2, W_int8, scale, C_out,
            K, N,
            A2.stride(1),
            W_int8.stride(0), W_int8.stride(1),
            scale.stride(0),
            C_out.stride(1),
        )
    else:
        # 元のカーネル（改良版）を使用
        C_out = torch.zeros((B, N), device=A.device, dtype=torch.float16)
        grid = (B, triton.cdiv(N, 16))  # BLOCK_Nのデフォルト値
        
        fused_dequant_vecmat_optimized_kernel[grid](
            A2, W_int8, scale, C_out,
            B, K, N,
            A2.stride(0), A2.stride(1),
            W_int8.stride(0), W_int8.stride(1),
            scale.stride(0),
            C_out.stride(0), C_out.stride(1),
        )
    
    return C_out.unsqueeze(1)

# --------------------------------
# PyTorch参照実装
# --------------------------------
def pytorch_dequant_matmul(A, W_int8, scale):
    """PyTorchによる参照実装"""
    # int8をfloat16にデクオンタイズ
    W_fp16 = W_int8.to(torch.float16) * scale.unsqueeze(1)
    # 行列積の計算
    return torch.matmul(A, W_fp16)

# --------------------------------
# 精度検証関数
# --------------------------------
def verify_accuracy(B=32, K=1024, N=1024, tolerance=1e-2):
    """Triton実装とPyTorch実装の精度を比較"""
    print(f"\n=== Accuracy Verification (B={B}, K={K}, N={N}) ===")
    
    # テストデータの生成（小さめの値で精度を確認しやすくする）
    torch.manual_seed(42)
    A = torch.randn(B, 1, K, device='cuda', dtype=torch.float16) * 0.1
    W_int8 = torch.randint(-64, 64, (K, N), device='cuda', dtype=torch.int8)
    scale = torch.rand(K, device='cuda', dtype=torch.float16) * 0.01 + 0.001
    
    # PyTorch実装
    pytorch_out = pytorch_dequant_matmul(A, W_int8, scale)
    
    # Triton実装（ベクトルカーネル）
    triton_vec_out = fused_dequant_gemm_optimized(A, W_int8, scale, use_vector_kernel=True)
    
    # Triton実装（元のカーネル改良版）
    triton_orig_out = fused_dequant_gemm_optimized(A, W_int8, scale, use_vector_kernel=False)
    
    # 元の実装（比較用）
    original_out = fused_dequant_gemm(A, W_int8, scale)
    
    # 精度メトリクスの計算
    def compute_metrics(ref, test, name):
        abs_diff = torch.abs(ref - test)
        rel_diff = abs_diff / (torch.abs(ref) + 1e-6)
        
        metrics = {
            'name': name,
            'max_abs_error': abs_diff.max().item(),
            'mean_abs_error': abs_diff.mean().item(),
            'max_rel_error': rel_diff.max().item(),
            'mean_rel_error': rel_diff.mean().item(),
            'allclose': torch.allclose(ref, test, rtol=tolerance, atol=tolerance)
        }
        return metrics
    
    # 各実装の精度を検証
    results = [
        compute_metrics(pytorch_out, triton_vec_out, "Triton Vector Kernel"),
        compute_metrics(pytorch_out, triton_orig_out, "Triton Optimized Kernel"),
        compute_metrics(pytorch_out, original_out, "Original Triton Kernel")
    ]
    
    # 結果の表示
    print(f"\n{'Implementation':<25} {'Max Abs Err':<12} {'Mean Abs Err':<12} {'Max Rel Err':<12} {'Mean Rel Err':<12} {'All Close':<10}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<25} {r['max_abs_error']:<12.6f} {r['mean_abs_error']:<12.6f} {r['max_rel_error']:<12.6f} {r['mean_rel_error']:<12.6f} {str(r['allclose']):<10}")
    
    return results

# --------------------------------
# 包括的なベンチマーク関数
# --------------------------------
def comprehensive_benchmark(sizes=[(32, 2048, 2048), (64, 4096, 4096), (128, 8192, 8192)]):
    """異なるサイズでの包括的なベンチマーク"""
    import time
    
    results = []
    
    for B, K, N in sizes:
        print(f"\n=== Benchmarking B={B}, K={K}, N={N} ===")
        
        # テストデータの生成
        A = torch.randn(B, 1, K, device='cuda', dtype=torch.float16)
        W_int8 = torch.randint(-127, 127, (K, N), device='cuda', dtype=torch.int8)
        scale = torch.rand(K, device='cuda', dtype=torch.float16) * 0.1
        
        # 各実装のベンチマーク
        implementations = [
            ("PyTorch", lambda: pytorch_dequant_matmul(A, W_int8, scale)),
            ("Triton Vector", lambda: fused_dequant_gemm_optimized(A, W_int8, scale, use_vector_kernel=True)),
            ("Triton Optimized", lambda: fused_dequant_gemm_optimized(A, W_int8, scale, use_vector_kernel=False)),
            ("Triton Original", lambda: fused_dequant_gemm(A, W_int8, scale))
        ]
        
        for name, func in implementations:
            # ウォームアップ
            for _ in range(5):
                _ = func()
            torch.cuda.synchronize()
            
            # タイミング測定
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            num_iterations = 50
            start_event.record()
            for _ in range(num_iterations):
                _ = func()
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event) / num_iterations  # ms
            
            # FLOPS計算（8ビット演算として）
            flops = 2 * B * K * N  # multiply-add
            tflops = flops / (elapsed_time * 1e9)  # TFLOPS
            
            result = {
                'Implementation': name,
                'B': B, 'K': K, 'N': N,
                'Time (ms)': elapsed_time,
                'TFLOPS': tflops,
                'Speedup': 1.0  # 後で計算
            }
            results.append(result)
            
            print(f"{name:<20} Time: {elapsed_time:>8.3f} ms, TFLOPS: {tflops:>6.2f}")
    
    # スピードアップの計算
    for size in sizes:
        pytorch_time = None
        for result in results:
            if (result['B'] == size[0] and result['K'] == size[1] and 
                result['N'] == size[2] and result['Implementation'] == 'PyTorch'):
                pytorch_time = result['Time (ms)']
                break
        
        if pytorch_time:
            for result in results:
                if (result['B'] == size[0] and result['K'] == size[1] and 
                    result['N'] == size[2]):
                    result['Speedup'] = pytorch_time / result['Time (ms)']
    
    return results

# --------------------------------
# メモリ帯域幅の分析
# --------------------------------
def analyze_memory_bandwidth(B=64, K=4096, N=4096):
    """メモリ帯域幅の使用率を分析"""
    print(f"\n=== Memory Bandwidth Analysis (B={B}, K={K}, N={N}) ===")
    
    # データサイズの計算
    input_size = B * K * 2  # float16
    weight_size = K * N * 1  # int8
    scale_size = K * 2  # float16
    output_size = B * N * 2  # float16
    total_bytes = input_size + weight_size + scale_size + output_size
    
    print(f"Input size: {input_size / 1e6:.2f} MB")
    print(f"Weight size: {weight_size / 1e6:.2f} MB")
    print(f"Scale size: {scale_size / 1e6:.2f} MB")
    print(f"Output size: {output_size / 1e6:.2f} MB")
    print(f"Total data: {total_bytes / 1e6:.2f} MB")
    
    # 実行時間の測定
    A = torch.randn(B, 1, K, device='cuda', dtype=torch.float16)
    W_int8 = torch.randint(-127, 127, (K, N), device='cuda', dtype=torch.int8)
    scale = torch.rand(K, device='cuda', dtype=torch.float16) * 0.1
    
    # ベクトルカーネルの測定
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    num_iterations = 100
    start_event.record()
    for _ in range(num_iterations):
        _ = fused_dequant_gemm_optimized(A, W_int8, scale, use_vector_kernel=True)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / num_iterations  # ms
    bandwidth_achieved = total_bytes / (elapsed_time * 1e6)  # GB/s
    
    # GPU理論帯域幅との比較（例：V100は900GB/s、A100は1555GB/s）
    print(f"\nAchieved bandwidth: {bandwidth_achieved:.2f} GB/s")
    print(f"Kernel execution time: {elapsed_time:.3f} ms")
    
    return bandwidth_achieved

# 使用例
if __name__ == "__main__":
    # 精度検証
    print("=" * 100)
    print("ACCURACY VERIFICATION")
    print("=" * 100)
    verify_accuracy(B=32, K=1024, N=1024)
    verify_accuracy(B=64, K=2048, N=2048)
    
    # 包括的なベンチマーク
    print("\n" + "=" * 100)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 100)
    df = comprehensive_benchmark(sizes=[
        (32, 2048, 2048),
        (64, 4096, 4096),
        (128, 8192, 8192)
    ])
    
    # 結果のサマリー表示
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    if pd is not None:
        print(df.to_string(index=False))
    else:
        for result in df:
            print(f"{result['Implementation']:<20} B={result['B']:>3} K={result['K']:>4} N={result['N']:>4} "
                  f"Time={result['Time (ms)']:>8.3f}ms TFLOPS={result['TFLOPS']:>6.2f} Speedup={result['Speedup']:>5.2f}x")
    
    # メモリ帯域幅の分析
    print("\n" + "=" * 100)
    print("MEMORY BANDWIDTH ANALYSIS")
    print("=" * 100)
    analyze_memory_bandwidth(B=64, K=4096, N=4096)
    
    # 最速実装の確認
    print("\n" + "=" * 100)
    print("BEST PERFORMING IMPLEMENTATION")
    print("=" * 100)
    if pd is not None:
        best_impl = df.groupby('Implementation')['Speedup'].mean().idxmax()
        avg_speedup = df.groupby('Implementation')['Speedup'].mean()[best_impl]
    else:
        # pandasなしでの処理
        impl_speedups = {}
        for result in df:
            impl = result['Implementation']
            if impl not in impl_speedups:
                impl_speedups[impl] = []
            impl_speedups[impl].append(result['Speedup'])
        
        best_impl = max(impl_speedups, key=lambda x: sum(impl_speedups[x])/len(impl_speedups[x]))
        avg_speedup = sum(impl_speedups[best_impl]) / len(impl_speedups[best_impl])
    
    print(f"Best implementation: {best_impl}")
    print(f"Average speedup over PyTorch: {avg_speedup:.2f}x")