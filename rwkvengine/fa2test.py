# Lets define a helpful benchmarking function:
import torch
import torch.utils.benchmark as benchmark
from torch.nn import functional as F
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# Lets define the hyper-parameters of our input
batch_size = 32
max_sequence_len = 1024
num_heads = 32
embed_dimension = 32
device='cuda'

dtype = torch.float16

query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# Lets explore the speed of each of the 3 implementations
from torch.nn.attention import SDPBackend, sdpa_kernel


with sdpa_kernel(SDPBackend.MATH):
    math_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
    print(f"The math implementation runs in {math_time:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")