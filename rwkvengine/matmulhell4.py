import torch
import triton
import triton.language as tl
import numpy as np
import time
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")
print(f"Triton version: {triton.__version__}")

# --------------------------------
# 最適化されたTritonカーネル（AMD向け）
# --------------------------------

@triton.jit
def fused_dequant_gemv_amd(
    A_ptr, W_ptr, scale_ptr, C_ptr,
    K, N,
    stride_ak, stride_wk, stride_wn, stride_s, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """AMD GPU向けに最適化されたGEMVカーネル"""
    # プログラムID（Nの次元でのタイル番号）
    pid = tl.program_id(0)
    
    # 各スレッドが処理するN次元のオフセット
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    
    # 累積器の初期化
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # K次元をループ
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        
        # Aベクトルのロード（ブロードキャスト）
        a_vals = tl.load(A_ptr + offs_k * stride_ak, mask=mask_k, other=0.0)
        
        # スケールのロード
        scale_vals = tl.load(scale_ptr + offs_k * stride_s, mask=mask_k, other=1.0)
        
        # Wのロード（INT8）
        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        mask_w = mask_k[:, None] & mask_n[None, :]
        w_int8 = tl.load(w_ptrs, mask=mask_w, other=0)
        
        # デクオントと積和演算を融合
        # AMD GPUでは小さい中間結果の方が効率的
        for i in range(BLOCK_K):
            if k + i < K:
                w_row = w_int8[i, :].to(tl.float32)
                acc += a_vals[i].to(tl.float32) * scale_vals[i].to(tl.float32) * w_row
    
    # 結果の保存
    c_ptrs = C_ptr + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_n)

@triton.jit
def fused_dequant_gemm_amd(
    A_ptr, W_ptr, scale_ptr, C_ptr,
    B, K, N,
    stride_ab, stride_ak,
    stride_wk, stride_wn,
    stride_s,
    stride_cb, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """バッチ処理対応のAMD最適化カーネル"""
    bid = tl.program_id(0)  # バッチインデックス
    nid = tl.program_id(1)  # N次元のタイルインデックス
    
    # オフセット計算
    offs_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    
    # バッチごとのポインタ
    A_batch = A_ptr + bid * stride_ab
    C_batch = C_ptr + bid * stride_cb
    
    # 累積器
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # K次元をタイル処理
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K
        
        # データロード
        a_vals = tl.load(A_batch + k_offs * stride_ak, mask=mask_k, other=0.0)
        scale_vals = tl.load(scale_ptr + k_offs * stride_s, mask=mask_k, other=1.0)
        
        # INT8ウェイトのロード
        w_ptrs = W_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_int8 = tl.load(w_ptrs, mask=w_mask, other=0)
        
        # 融合演算
        w_fp32 = w_int8.to(tl.float32) * scale_vals[:, None].to(tl.float32)
        acc += tl.sum(a_vals.to(tl.float32)[:, None] * w_fp32, axis=0)
    
    # 結果を保存
    c_ptrs = C_batch + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_n)

# --------------------------------
# PyTorch実装のバリエーション
# --------------------------------

def pytorch_fp16_baseline(A, W):
    """FP16ベースライン"""
    return torch.matmul(A, W.T)

def pytorch_int8_dequant_matmul(A, W_int8, scale):
    """標準的なINT8実装"""
    W_fp16 = W_int8.to(torch.float16) * scale.view(-1, 1)
    return torch.matmul(A, W_fp16.T)

def pytorch_int8_mixed_precision(A, W_int8, scale):
    """混合精度実装"""
    # FP32で計算してFP16に戻す
    A_fp32 = A.to(torch.float32)
    W_fp32 = W_int8.to(torch.float32) * scale.view(-1, 1).to(torch.float32)
    result_fp32 = torch.matmul(A_fp32, W_fp32.T)
    return result_fp32.to(torch.float16)

# JITコンパイル版
@torch.jit.script
def pytorch_int8_jit_optimized(A: torch.Tensor, W_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # 効率的なメモリアクセスパターン
    B, T, K = A.shape
    K_w, N = W_int8.shape
    
    # ビューを使って効率化
    A_2d = A.view(B * T, K)
    
    # ベクトル化されたデクオント
    scale_expanded = scale.unsqueeze(1)
    W_fp16 = (W_int8.to(torch.float32) * scale_expanded.to(torch.float32)).to(torch.float16)
    
    # 効率的な転置とmatmul
    result = torch.mm(A_2d, W_fp16.t())
    
    return result.view(B, T, N)

# --------------------------------
# Triton実装のラッパー
# --------------------------------

def triton_int8_inference_amd(A, W_int8, scale, block_n=64, block_k=256):
    """AMD最適化Triton実装"""
    B, T, K = A.shape
    K_w, N = W_int8.shape
    
    assert T == 1, "T=1 only"
    
    # 出力テンソル
    C = torch.zeros(B, N, device=A.device, dtype=torch.float16)
    
    # グリッドサイズ
    grid = (B, (N + block_n - 1) // block_n)
    
    # カーネル実行
    fused_dequant_gemm_amd[grid](
        A.view(B, K), W_int8, scale, C,
        B, K, N,
        A.stride(0), A.stride(2),
        W_int8.stride(0), W_int8.stride(1),
        scale.stride(0),
        C.stride(0), C.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k
    )
    
    return C.unsqueeze(1)

# --------------------------------
# ベンチマーク
# --------------------------------

def benchmark_with_profiling(func, *args, iterations=100, warmup=20):
    """詳細なプロファイリング付きベンチマーク"""
    # ウォームアップ
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()
    
    # 時間測定
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = func(*args)
        end.record()
        
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    
    # 統計計算
    timings = np.array(timings)
    return {
        "mean_ms": np.mean(timings),
        "std_ms": np.std(timings),
        "min_ms": np.min(timings),
        "max_ms": np.max(timings),
        "median_ms": np.median(timings)
    }

def run_comprehensive_benchmark():
    """包括的なベンチマーク（大規模サイズ含む）"""
    # より大きなサイズでテスト
    test_configs = [
        # (B, K, N, name)
        (1, 4096, 4096, "Medium"),
        (8, 4096, 4096, "Medium Batch"),
        (1, 8192, 8192, "Large"),
        (1, 16384, 16384, "XLarge"),
        (64, 2048, 2048, "Large Batch"),
    ]
    
    results = {}
    
    for B, K, N, name in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing {name}: B={B}, K={K}, N={N}")
        print(f"Memory requirement: ~{2 * (B*K + K*N + B*N) * 2 / 1e9:.2f} GB")
        
        try:
            # データ準備
            torch.manual_seed(42)
            A = torch.randn(B, 1, K, device='cuda', dtype=torch.float16)
            W = torch.randn(K, N, device='cuda', dtype=torch.float16) * (2.0 / np.sqrt(K))
            
            # 量子化
            W_abs_max = W.abs().amax(dim=1, keepdim=True)
            scale = (W_abs_max / 127).clamp(min=1e-8)
            W_int8 = (W / scale).round().clamp(-127, 127).to(torch.int8)
            scale = scale.squeeze(1)
            
            config_results = {}
            
            # 1. PyTorch FP16
            print("\nPyTorch FP16:")
            stats = benchmark_with_profiling(pytorch_fp16_baseline, A, W.T)
            print(f"  Mean: {stats['mean_ms']:.3f} ms (±{stats['std_ms']:.3f})")
            config_results["pytorch_fp16"] = stats
            
            # 2. PyTorch INT8 Standard
            print("\nPyTorch INT8 Standard:")
            stats = benchmark_with_profiling(pytorch_int8_dequant_matmul, A, W_int8, scale)
            print(f"  Mean: {stats['mean_ms']:.3f} ms (±{stats['std_ms']:.3f})")
            config_results["pytorch_int8_standard"] = stats
            
            # 3. PyTorch INT8 JIT
            print("\nPyTorch INT8 JIT:")
            stats = benchmark_with_profiling(pytorch_int8_jit_optimized, A, W_int8, scale)
            print(f"  Mean: {stats['mean_ms']:.3f} ms (±{stats['std_ms']:.3f})")
            config_results["pytorch_int8_jit"] = stats
            
            # 4. Triton実装（複数設定）
            triton_configs = [(32, 256), (64, 256), (64, 512), (128, 512)]
            
            for bn, bk in triton_configs:
                print(f"\nTriton INT8 (BN={bn}, BK={bk}):")
                try:
                    stats = benchmark_with_profiling(
                        triton_int8_inference_amd, A, W_int8, scale, bn, bk
                    )
                    print(f"  Mean: {stats['mean_ms']:.3f} ms (±{stats['std_ms']:.3f})")
                    config_results[f"triton_bn{bn}_bk{bk}"] = stats
                except Exception as e:
                    print(f"  Failed: {e}")
            
            results[f"{name}_B{B}_K{K}_N{N}"] = config_results
            
            # メモリクリーンアップ
            del A, W, W_int8, scale
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to test {name}: {e}")
            continue
    
    return results

def analyze_results(results):
    """結果の分析と可視化"""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    # 各設定での勝者を特定
    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        
        # 最速の手法を見つける
        best_method = min(config_results.items(), 
                         key=lambda x: x[1]['mean_ms'])
        baseline = config_results.get("pytorch_fp16", {}).get("mean_ms", float('inf'))
        
        print(f"  Best: {best_method[0]} ({best_method[1]['mean_ms']:.3f} ms)")
        
        # スピードアップ分析
        print("  Speedup vs FP16:")
        for method, stats in sorted(config_results.items()):
            if stats['mean_ms'] > 0:
                speedup = baseline / stats['mean_ms']
                print(f"    {method}: {speedup:.2f}x")
    
    # プロット作成
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # パフォーマンス比較
        configs = list(results.keys())
        methods = set()
        for r in results.values():
            methods.update(r.keys())
        
        for method in sorted(methods):
            times = []
            labels = []
            for config, config_results in results.items():
                if method in config_results:
                    times.append(config_results[method]['mean_ms'])
                    labels.append(config.split('_')[0])
            
            if times:
                ax1.plot(labels, times, 'o-', label=method, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Performance Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # INT8 vs FP16 効率
        int8_speedups = []
        config_names = []
        
        for config, results_dict in results.items():
            if "pytorch_fp16" in results_dict and "pytorch_int8_jit" in results_dict:
                fp16_time = results_dict["pytorch_fp16"]["mean_ms"]
                int8_time = results_dict["pytorch_int8_jit"]["mean_ms"]
                speedup = fp16_time / int8_time
                int8_speedups.append(speedup)
                config_names.append(config.split('_')[0])
        
        ax2.bar(range(len(int8_speedups)), int8_speedups)
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('INT8 Speedup vs FP16')
        ax2.set_title('INT8 Efficiency')
        ax2.set_xticks(range(len(config_names)))
        ax2.set_xticklabels(config_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Plotting failed: {e}")

# --------------------------------
# メイン実行
# --------------------------------

def main():
    print("AMD GPU Optimized INT8 Inference Benchmark")
    print("="*60)
    
    # ベンチマーク実行
    results = run_comprehensive_benchmark()
    
    # 結果分析
    analyze_results(results)
    
    # 推奨事項
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR AMD RADEON PRO W7900")
    print("="*80)
    print("1. For small matrices (<4K): Use PyTorch FP16 (hipBLAS is highly optimized)")
    print("2. For large matrices (>8K): INT8 can provide memory bandwidth benefits")
    print("3. JIT compilation provides the best INT8 performance on AMD")
    print("4. Triton benefits appear mainly in memory-bound scenarios")
    print("5. Consider mixed precision (FP32 accumulation) for better accuracy")

if __name__ == "__main__":
    main()