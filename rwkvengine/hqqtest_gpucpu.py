import torch
import time
import numpy as np
from torch.utils.cpp_extension import load
from dataclasses import dataclass
from typing import List, Dict

# カスタムカーネルをロード（CPU専用）
def load_custom_kernel():
    """カスタム量子化カーネルをロード（CPU専用）"""
    return load(
        name='quantized_linear_cpu',
        sources=['cpu_kernel/quantized_linear.cpp'],
        extra_cflags=['-O3', '-march=native', '-fopenmp'],
        extra_ldflags=['-fopenmp'],
        verbose=True,
        with_cuda=False  # 明示的にCPUのみ
    )

class HQQCompatibleLinear(torch.nn.Module):
    """HQQと互換性のあるカスタム量子化線形層（CPU専用）"""
    
    def __init__(self, hqq_layer):
        super().__init__()
        
        # カスタムカーネルをロード
        self.kernel = load_custom_kernel()
        
        # HQQレイヤーから情報を抽出
        self.in_features = hqq_layer.in_features
        self.out_features = hqq_layer.out_features
        self.nbits = hqq_layer.meta['nbits']
        self.group_size = hqq_layer.meta['group_size']
        self.compute_dtype = hqq_layer.compute_dtype
        
        # HQQのメタデータを抽出（CPUに転送）
        self.scale = hqq_layer.meta['scale'].cpu()
        self.zero = hqq_layer.meta['zero'].cpu() if 'zero' in hqq_layer.meta else None
        self.shape = hqq_layer.meta['shape'] if 'shape' in hqq_layer.meta else None
        
        # 量子化された重みを抽出（CPUに転送）
        self.W_q = hqq_layer.W_q.cpu()
        
        # バイアス（もしあれば、CPUに転送）
        self.bias = hqq_layer.bias.cpu() if hasattr(hqq_layer, 'bias') and hqq_layer.bias is not None else None
        
        # パッキング形式を決定
        self.packing = self._determine_packing()
        
        print(f"Custom kernel initialized for CPU operation")
        print(f"  Input dtype: {self.compute_dtype}")
        print(f"  Quantization: {self.nbits}-bit")
        print(f"  W_q device: {self.W_q.device}")
        print(f"  Scale device: {self.scale.device}")
        
    def _determine_packing(self):
        """ビット数に基づいてパッキング形式を決定"""
        if self.nbits == 8:
            return "8bit_u8"
        elif self.nbits == 4:
            return "4bit_u8"
        elif self.nbits == 2:
            return "2bit_u8"
        elif self.nbits == 1:
            return "1bit_u8"
        else:
            raise ValueError(f"Unsupported nbits: {self.nbits}")
    
    def forward(self, x):
        """順伝播（CPU専用）"""
        # 入力が必ずCPU上にあることを確認
        original_device = x.device
        x_cpu = x.cpu().to(self.compute_dtype)
        
        # バイアスの準備
        bias = self.bias if self.bias is not None else torch.empty(0)
        
        # ゼロポイントの準備
        if self.zero is None:
            zero = torch.zeros_like(self.scale)
        else:
            zero = self.zero
        
        # W_shapeの準備
        W_shape = [self.out_features, self.in_features]
        
        # 空のテンソル（スケールとゼロポイントの量子化なし）
        empty = torch.empty(0)
        empty_shape = [0]
        
        # カスタムカーネルを呼び出し（CPU上で実行）
        output_cpu = self.kernel.forward_with_quant_fast(
            x_cpu, bias, 
            self.W_q, self.scale, zero,
            W_shape, self.group_size, self.nbits, 1, self.packing
        )
        
        # 元のデバイスに戻す
        return output_cpu.to(original_device)

@dataclass
class BenchmarkResult:
    """ベンチマーク結果を保持するデータクラス"""
    method: str
    times: List[float]
    errors: List[float]
    device: str
    
    @property
    def mean_time(self) -> float:
        return np.mean(self.times)
    
    @property
    def std_time(self) -> float:
        return np.std(self.times)
    
    @property
    def mean_error(self) -> float:
        return np.mean(self.errors)
    
    @property
    def std_error(self) -> float:
        return np.std(self.errors)
    
    def print_summary(self):
        print(f"\n{self.method} ({self.device}):")
        print(f"  Time: {self.mean_time:.3f} ± {self.std_time:.3f} ms")
        print(f"  Error: {self.mean_error:.6f} ± {self.std_error:.6f}")

def benchmark_cpu_vs_gpu(hqq_layer_gpu, custom_layer_cpu, x_gpu, W_gpu, num_trials=10):
    """CPU（カスタムカーネル）とGPU（HQQ）のベンチマーク"""
    
    print(f"\n=== CPU vs GPU Benchmark ({num_trials} trials) ===")
    
    # ウォームアップ
    print("Warming up...")
    for _ in range(5):
        _ = torch.matmul(x_gpu, W_gpu.T)
        _ = custom_layer_cpu(x_gpu)
        _ = hqq_layer_gpu(x_gpu)
    
    torch.cuda.synchronize()
    
    results = {
        'torch_gpu': BenchmarkResult('PyTorch matmul', [], [], 'GPU'),
        'custom_cpu': BenchmarkResult('Custom kernel', [], [], 'CPU'),
        'hqq_gpu': BenchmarkResult('HQQ', [], [], 'GPU'),
    }
    
    # リファレンス結果
    y_ref = torch.matmul(x_gpu, W_gpu.T)
    
    print(f"Running {num_trials} trials...")
    for trial in range(num_trials):
        # PyTorch GPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_torch = torch.matmul(x_gpu, W_gpu.T)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # Custom CPU kernel
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        y_custom = custom_layer_cpu(x_gpu)  # 内部でCPU転送
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        # HQQ GPU
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        y_hqq = hqq_layer_gpu(x_gpu)
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        
        # 記録
        results['torch_gpu'].times.append((t1 - t0) * 1000)
        results['custom_cpu'].times.append((t3 - t2) * 1000)
        results['hqq_gpu'].times.append((t5 - t4) * 1000)
        
        results['torch_gpu'].errors.append(0.0)
        results['custom_cpu'].errors.append((y_ref - y_custom).abs().mean().item())
        results['hqq_gpu'].errors.append((y_ref - y_hqq).abs().mean().item())
    
    return results

def test_cpu_only_performance(hqq_layer_cpu, custom_layer_cpu, x_cpu, W_cpu, num_trials=10):
    """CPU同士の公平な比較"""
    
    print(f"\n=== CPU-Only Performance Comparison ({num_trials} trials) ===")
    
    # ウォームアップ
    print("Warming up...")
    for _ in range(5):
        _ = torch.matmul(x_cpu, W_cpu.T)
        _ = custom_layer_cpu(x_cpu)
        _ = hqq_layer_cpu(x_cpu)
    
    results = {
        'torch_cpu': BenchmarkResult('PyTorch matmul', [], [], 'CPU'),
        'custom_cpu': BenchmarkResult('Custom kernel', [], [], 'CPU'),
        'hqq_cpu': BenchmarkResult('HQQ', [], [], 'CPU'),
    }
    
    # リファレンス結果
    y_ref = torch.matmul(x_cpu, W_cpu.T)
    
    print(f"Running {num_trials} trials...")
    for trial in range(num_trials):
        # PyTorch CPU
        t0 = time.perf_counter()
        y_torch = torch.matmul(x_cpu, W_cpu.T)
        t1 = time.perf_counter()
        
        # Custom CPU kernel
        t2 = time.perf_counter()
        y_custom = custom_layer_cpu(x_cpu)
        t3 = time.perf_counter()
        
        # HQQ CPU
        t4 = time.perf_counter()
        y_hqq = hqq_layer_cpu(x_cpu)
        t5 = time.perf_counter()
        
        # 記録
        results['torch_cpu'].times.append((t1 - t0) * 1000)
        results['custom_cpu'].times.append((t3 - t2) * 1000)
        results['hqq_cpu'].times.append((t5 - t4) * 1000)
        
        results['torch_cpu'].errors.append(0.0)
        results['custom_cpu'].errors.append((y_ref - y_custom).abs().mean().item())
        results['hqq_cpu'].errors.append((y_ref - y_hqq).abs().mean().item())
    
    return results

# メインのテストコード
if __name__ == "__main__":
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
    
    # パラメータ設定
    in_features, out_features = 4096, 4096
    W_nbits, group_size = 4, 64
    compute_dtype = torch.float16
    
    print(f"Configuration: {in_features}x{out_features}, {W_nbits}-bit, group_size={group_size}")
    print(f"Available devices: CPU" + (", CUDA" if torch.cuda.is_available() else ""))
    
    # 線形層を作成
    linear = torch.nn.Linear(
        in_features=in_features, 
        out_features=out_features, 
        bias=True,
        device='cpu'
    )
    
    # HQQ量子化設定
    quant_config = BaseQuantizeConfig(
        nbits=W_nbits, 
        group_size=group_size, 
        quant_zero=False, 
        quant_scale=False, 
        axis=1
    )
    
    # テスト入力
    x_cpu = torch.randn((8, in_features), dtype=compute_dtype) / 10.
    
    if torch.cuda.is_available():
        print("\n=== Testing CPU Custom Kernel vs GPU HQQ ===")
        
        # GPU用HQQレイヤー
        hqq_layer_gpu = HQQLinear(
            linear, 
            quant_config=quant_config, 
            compute_dtype=compute_dtype, 
            device='cuda:0', 
            del_orig=False
        )
        
        # CPU用カスタムレイヤー（HQQ GPUレイヤーから作成）
        custom_layer_cpu = HQQCompatibleLinear(hqq_layer_gpu)
        
        # デクォンタイズされた重み
        W_gpu = hqq_layer_gpu.dequantize().reshape((out_features, in_features))
        x_gpu = x_cpu.cuda()
        
        # CPU vs GPU ベンチマーク
        results_mixed = benchmark_cpu_vs_gpu(
            hqq_layer_gpu, custom_layer_cpu, x_gpu, W_gpu, num_trials=10
        )
        
        for result in results_mixed.values():
            result.print_summary()
        
        print("\n=== Performance Analysis ===")
        cpu_time = results_mixed['custom_cpu'].mean_time
        gpu_time = results_mixed['hqq_gpu'].mean_time
        transfer_overhead = cpu_time - gpu_time
        
        print(f"CPU kernel time (with transfers): {cpu_time:.3f} ms")
        print(f"GPU HQQ time: {gpu_time:.3f} ms")
        print(f"Estimated CPU↔GPU transfer overhead: {transfer_overhead:.3f} ms")
    
    # CPU同士の公平な比較
    print("\n" + "="*60)
    print("=== CPU-Only Fair Comparison ===")
    
    # CPU用HQQレイヤー
    hqq_layer_cpu = HQQLinear(
        linear, 
        quant_config=quant_config, 
        compute_dtype=torch.float32,  # CPUではFP32を使用
        device='cpu', 
        del_orig=False
    )
    
    # CPU用カスタムレイヤー
    custom_layer_cpu = HQQCompatibleLinear(hqq_layer_cpu)
    
    # デクォンタイズされた重み（CPU）
    W_cpu = hqq_layer_cpu.dequantize().reshape((out_features, in_features))
    x_cpu_fp32 = x_cpu.to(torch.float32)  # CPUではFP32
    
    # CPU同士のベンチマーク
    results_cpu = test_cpu_only_performance(
        hqq_layer_cpu, custom_layer_cpu, x_cpu_fp32, W_cpu, num_trials=10
    )
    
    for result in results_cpu.values():
        result.print_summary()
    
    # スピードアップ計算
    torch_time = results_cpu['torch_cpu'].mean_time
    custom_time = results_cpu['custom_cpu'].mean_time
    hqq_time = results_cpu['hqq_cpu'].mean_time
    
    print(f"\n=== CPU Performance Summary ===")
    print(f"Custom kernel vs PyTorch: {torch_time/custom_time:.2f}x speedup")
    print(f"HQQ vs PyTorch: {torch_time/hqq_time:.2f}x speedup")
    print(f"Custom kernel vs HQQ: {hqq_time/custom_time:.2f}x faster")
    
    # メモリ使用量
    print("\n=== Memory Usage ===")
    def get_memory_usage(layer):
        total = 0
        for name, param in layer.named_parameters():
            total += param.numel() * param.element_size()
        for name, buffer in layer.named_buffers():
            total += buffer.numel() * buffer.element_size()
        return total / (1024 * 1024)  # MB
    
    orig_memory = get_memory_usage(linear)
    hqq_memory = get_memory_usage(hqq_layer_cpu)
    custom_memory = get_memory_usage(custom_layer_cpu)
    
    print(f"Original linear: {orig_memory:.2f} MB")
    print(f"HQQ layer: {hqq_memory:.2f} MB")
    print(f"Custom layer: {custom_memory:.2f} MB")
    print(f"Compression ratio: {orig_memory / hqq_memory:.2f}x")