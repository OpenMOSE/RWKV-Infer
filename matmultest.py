# matmul_bench_llm.py
import math, time, csv, os
import torch

# ====== 設定 ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE=="cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16  if (DEVICE=="cuda") else torch.float32)
ALLOW_TF32 = True  # TF32を許可（Ampere+のFP32 matmul高速化）
B = 4              # バッチ
T = 1            # シーケンス長（B*T がMになる）
REPEAT = 30        # 実測イテレーション
WARMUP = 10
STEP = 8           # Cの刻み幅（8から32768までSTEP刻み）
C_MAX = 5120
CSV_PATH = "matmul_bench.csv"
TORCH_COMPILE = False  # Trueにするとモデルラッパをtorch.compileで計測（Dynamo/Inductor）

# ====== ユーティリティ ======
def pad_to_multiple(n, m):
    r = (-n) % m
    return n + r, r

def flop_gemm(m, n, k):
    # GEMM FLOPs ≈ 2 * M * N * K
    return 2.0 * m * n * k

def timer_start_end():
    if DEVICE == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        return start, end
    else:
        return None, None

def elapsed_ms(start, end, t0=None):
    if DEVICE == "cuda":
        return start.elapsed_time(end)  # ms
    else:
        return (time.perf_counter() - t0) * 1000.0

def maybe_contiguous(x):
    # 連続ストライドを保証
    return x.contiguous()

# ====== 計測コア ======
class FFNLike(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.up = torch.nn.Linear(C, 4*C, bias=False, device=DEVICE, dtype=DTYPE)
        self.down = torch.nn.Linear(4*C, C, bias=False, device=DEVICE, dtype=DTYPE)

    def forward(self, x):
        # x: [B*T, C]
        y = self.up(x)
        z = self.down(y)
        return z

def bench_one(C, use_padding=False):
    C_eff = C
    pad_added = 0
    if use_padding:
        C_eff, pad_added = pad_to_multiple(C, 256)

    M = B * T
    K = C_eff
    N1 = 4 * C_eff  # up-proj
    N2 = C_eff      # down-proj

    # 入力と重み作成（連続・アライン重視）
    x = torch.randn(M, C_eff, device=DEVICE, dtype=DTYPE)
    w1 = torch.randn(C_eff, 4*C_eff, device=DEVICE, dtype=DTYPE)
    w2 = torch.randn(4*C_eff, C_eff, device=DEVICE, dtype=DTYPE)

    if ALLOW_TF32 and DEVICE=="cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        if DEVICE=="cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    # オプション：torch.compile で linear 合成をモデル化
    if TORCH_COMPILE and DEVICE=="cuda":
        mod = torch.nn.Sequential(
            torch.nn.Linear(C_eff, 4*C_eff, bias=False, device=DEVICE, dtype=DTYPE),
            torch.nn.Linear(4*C_eff, C_eff, bias=False, device=DEVICE, dtype=DTYPE),
        )
        # 既存の重みを移植
        with torch.no_grad():
            mod[0].weight.copy_(w1.T)
            mod[1].weight.copy_(w2.T)
        mod = torch.compile(mod, dynamic=False)

    # Warmup
    for _ in range(WARMUP):
        if DEVICE=="cuda":
            torch.cuda.synchronize()
        if TORCH_COMPILE and DEVICE=="cuda":
            y = mod(x)
        else:
            y = x @ w1
            y = y @ w2
        if DEVICE=="cuda":
            torch.cuda.synchronize()

    # 実測（up+down を1セットとみなす）
    start, end = timer_start_end()
    times = []
    for _ in range(REPEAT):
        if DEVICE=="cuda":
            torch.cuda.synchronize()
            start.record()
            if TORCH_COMPILE and DEVICE=="cuda":
                y = mod(x)
            else:
                y = x @ w1
                y = y @ w2
            end.record()
            torch.cuda.synchronize()
            t_ms = elapsed_ms(start, end)
        else:
            t0 = time.perf_counter()
            y = x @ w1
            y = y @ w2
            t_ms = elapsed_ms(None, None, t0)
        times.append(t_ms)

    avg_ms = sum(times) / len(times)

    # FLOPs（up + down の合計）
    flops = flop_gemm(M, N1, K) + flop_gemm(M, N2, 4*K)
    tflops = (flops / (avg_ms * 1e-3)) / 1e12

    return {
        "C": C,
        "C_eff": C_eff,
        "padded": int(use_padding),
        "pad_added": pad_added,
        "divisible_256": int((C % 256) == 0),
        "B": B,
        "T": T,
        "dtype": str(DTYPE).replace("torch.", ""),
        "device": DEVICE,
        "avg_ms": avg_ms,
        "est_TFLOPS": tflops
    }

def main():
    sizes = list(range(4096, C_MAX+1, STEP))
    results = []

    # そのまま / パディング有り の両方を計測
    for C in sizes:
        try:
            results.append(bench_one(C, use_padding=False))
            results.append(bench_one(C, use_padding=True))
        except RuntimeError as e:
            # でかすぎた等はスキップ
            results.append({
                "C": C, "C_eff": None, "padded": -1, "pad_added": None,
                "divisible_256": int((C % 256)==0),
                "B": B, "T": T, "dtype": str(DTYPE).replace("torch.",""),
                "device": DEVICE, "avg_ms": float("nan"), "est_TFLOPS": float("nan"),
                "error": str(e).split("\n")[0]
            })

    # CSV保存
    fieldnames = ["C","C_eff","padded","pad_added","divisible_256","B","T","dtype","device","avg_ms","est_TFLOPS","error"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            if "error" not in r:
                r["error"] = ""
            w.writerow(r)

    # 簡易サマリ出力
    best_nonpad = max([r for r in results if r["padded"]==0 and not math.isnan(r["est_TFLOPS"])], key=lambda x: x["est_TFLOPS"], default=None)
    best_pad    = max([r for r in results if r["padded"]==1 and not math.isnan(r["est_TFLOPS"])], key=lambda x: x["est_TFLOPS"], default=None)
    print(f"Saved: {CSV_PATH}")
    print(f"Best(non-padded): {best_nonpad}")
    print(f"Best(padded)    : {best_pad}")

if __name__ == "__main__":
    main()
