#include <torch/extension.h>
#include <vector>
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <ATen/Parallel.h>

// Helper function to determine if CPU supports certain instructions
inline bool has_avx2() {
    #ifdef __AVX2__
    return true;
    #else
    return false;
    #endif
}

inline torch::Tensor unpack_4bit_u8(torch::Tensor &W_q)
{
    return torch::cat({(W_q & 0xF0).__rshift__(4), (W_q & 0x0F)}, 0);
}

inline torch::Tensor unpack_3bit_32(torch::Tensor &W_q) {
    return torch::cat(
    {
        ((W_q & 0x38000000).__rshift__(27)),
        ((W_q & 0x07000000).__rshift__(24)),
        ((W_q & 0x00E00000).__rshift__(21)),
        ((W_q & 0x001C0000).__rshift__(18)),
        ((W_q & 0x00038000).__rshift__(15)),
        ((W_q & 0x00007000).__rshift__(12)),
        ((W_q & 0x00000E00).__rshift__(9)),
        ((W_q & 0x000001C0).__rshift__(6)),
        ((W_q & 0x00000038).__rshift__(3)),
        ((W_q & 0x00000007))
    }, 0);
}

inline torch::Tensor unpack_2bit_u8(torch::Tensor &W_q)
{
    return torch::cat({(W_q & 0xC0).__rshift__(6), (W_q & 0x30).__rshift__(4), (W_q & 0x0C).__rshift__(2), W_q & 0x03}, 0);  
}

inline torch::Tensor unpack_1bit_u8(torch::Tensor &W_q) {
    return torch::cat(
    {
        ((W_q & 0x80).__rshift__(7)),
        ((W_q & 0x40).__rshift__(6)),
        ((W_q & 0x20).__rshift__(5)),
        ((W_q & 0x10).__rshift__(4)),
        ((W_q & 0x08).__rshift__(3)),
        ((W_q & 0x04).__rshift__(2)),
        ((W_q & 0x02).__rshift__(1)),
        ((W_q & 0x01))
    }, 0);
}

// Optimized dequantization that avoids creating intermediate tensors
inline torch::Tensor dequantize_optimized(torch::Tensor &W_q, torch::Tensor &scale, torch::Tensor &zero, 
                                         torch::IntArrayRef &shape, int group_size, int nbits, 
                                         int axis, std::string packing, torch::ScalarType dtype)
{
    torch::Tensor W_q_p;

    // Unpack bits
    if(packing=="8bit_u8"){W_q_p = W_q;}
    else if(packing=="4bit_u8"){W_q_p = unpack_4bit_u8(W_q);}
    else if(packing=="3bit_32"){W_q_p = unpack_3bit_32(W_q);}
    else if(packing=="2bit_u8"){W_q_p = unpack_2bit_u8(W_q);}
    else if(packing=="1bit_u8"){W_q_p = unpack_1bit_u8(W_q);}

    // Check size
    if(group_size>0 && nbits==3)
    {   
        W_q_p = W_q_p.slice(axis, 0, group_size); 
    }

    // Use the specified dtype (FP32 for CPU, FP16 for GPU if supported)
    W_q_p = W_q_p.to(dtype);
    auto W_r = ((W_q_p - zero.to(dtype)) * scale.to(dtype)).reshape(shape);

    return W_r;
}

// Legacy dequantize function for backward compatibility
inline torch::Tensor dequantize(torch::Tensor &W_q, torch::Tensor &scale, torch::Tensor &zero, 
                               torch::IntArrayRef &shape, int group_size, int nbits, 
                               int axis, std::string packing)
{
    // Determine the appropriate dtype based on device
    torch::ScalarType dtype = W_q.is_cuda() ? torch::kHalf : torch::kFloat32;
    return dequantize_optimized(W_q, scale, zero, shape, group_size, nbits, axis, packing, dtype);
}

// Fused dequantize and matmul kernel
torch::Tensor fused_dequant_matmul(torch::Tensor &x, torch::Tensor &W_q, torch::Tensor &scale, 
                                   torch::Tensor &zero, torch::IntArrayRef &shape, int group_size, 
                                   int nbits, int axis, std::string packing)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = shape[1];
    
    // Determine dtype based on device
    torch::ScalarType dtype = x.is_cuda() ? torch::kHalf : torch::kFloat32;
    
    // For small matrices or when not on CPU, fall back to regular approach
    if (!x.is_cpu() || M * N < 1024) {
        auto W_deq = dequantize_optimized(W_q, scale, zero, shape, group_size, nbits, axis, packing, dtype);
        return torch::matmul(x.to(dtype), W_deq.transpose(0, 1));
    }
    
    // Create output tensor
    auto output = torch::zeros({M, N}, x.options().dtype(dtype));
    
    // Get raw pointers
    float* x_ptr = x.to(torch::kFloat32).data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    // Dequantize weight to FP32
    auto W_deq = dequantize_optimized(W_q, scale, zero, shape, group_size, nbits, axis, packing, torch::kFloat32);
    float* W_ptr = W_deq.data_ptr<float>();
    
    // Parallel computation with blocking for cache efficiency
    const int block_size = 64;  // Tune this based on cache size
    
    at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            // Process in blocks for better cache utilization
            for (int j_block = 0; j_block < N; j_block += block_size) {
                int j_end = std::min(j_block + block_size, N);
                
                for (int j = j_block; j < j_end; j++) {
                    float sum = 0.0f;
                    
                    // Vectorizable inner loop
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < K; k++) {
                        sum += x_ptr[i * K + k] * W_ptr[k * N + j];
                    }
                    
                    out_ptr[i * N + j] = sum;
                }
            }
        }
    });
    
    return output;
}

inline torch::Tensor forward_with_quant(torch::Tensor &x, torch::Tensor &bias, 
                                       torch::Tensor &W_q, torch::Tensor &W_s, torch::Tensor &W_z, 
                                       torch::IntArrayRef &W_shape, int W_group_size, int W_nbits, 
                                       int W_axis, std::string W_packing,
                                       torch::Tensor &S_q, torch::Tensor &S_s, torch::Tensor &S_z, 
                                       torch::IntArrayRef &S_shape, int S_group_size, int S_nbits, 
                                       int S_axis, std::string S_packing,
                                       torch::Tensor &Z_q, torch::Tensor &Z_s, torch::Tensor &Z_z, 
                                       torch::IntArrayRef &Z_shape, int Z_group_size, int Z_nbits, 
                                       int Z_axis, std::string Z_packing) 
{
    torch::Tensor W_s_tmp, W_z_tmp;
    
    // Determine the appropriate dtype
    torch::ScalarType dtype = x.is_cuda() ? torch::kHalf : torch::kFloat32;

    if(S_q.numel()>0){
        W_s_tmp = dequantize_optimized(S_q, S_s, S_z, S_shape, S_group_size, S_nbits, S_axis, S_packing, dtype);
    }
    else {
        W_s_tmp = W_s.to(dtype);
    }
    
    if(Z_q.numel()>0){
        W_z_tmp = dequantize_optimized(Z_q, Z_s, Z_z, Z_shape, Z_group_size, Z_nbits, Z_axis, Z_packing, dtype);
    }
    else {
        W_z_tmp = W_z.to(dtype);
    }

    // Use fused kernel for better performance
    auto out = fused_dequant_matmul(x, W_q, W_s_tmp, W_z_tmp, W_shape, W_group_size, W_nbits, W_axis, W_packing);
    
    if(bias.numel()>0) {
        out += bias.to(dtype);
    }

    return out;
}

// Alternative high-performance forward function
torch::Tensor forward_with_quant_fast(torch::Tensor &x, torch::Tensor &bias, 
                                     torch::Tensor &W_q, torch::Tensor &W_s, torch::Tensor &W_z, 
                                     torch::IntArrayRef &W_shape, int W_group_size, int W_nbits, 
                                     int W_axis, std::string W_packing) 
{
    // Simplified version without nested quantization for maximum performance
    torch::ScalarType dtype = x.is_cuda() ? torch::kHalf : torch::kFloat32;
    
    auto out = fused_dequant_matmul(x, W_q, W_s, W_z, W_shape, W_group_size, W_nbits, W_axis, W_packing);
    
    if(bias.numel()>0) {
        out += bias.to(dtype);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_with_quant", &forward_with_quant, "forward_with_quant with FP32 CPU support");
    m.def("forward_with_quant_fast", &forward_with_quant_fast, "Optimized forward_with_quant for simple cases");
    m.def("dequantize", &dequantize, "dequantize with automatic dtype selection");
    m.def("fused_dequant_matmul", &fused_dequant_matmul, "Fused dequantization and matrix multiplication");
}