#include <torch/extension.h>
#include <vector>
#include <string>
#include <iostream>
#include <ATen/Parallel.h>

// Simple and safe HQQ-compatible kernel for group_size=64
torch::Tensor simple_hqq_matmul(torch::Tensor &x, torch::Tensor &W_q, 
                                torch::Tensor &scale, torch::Tensor &zero, 
                                torch::IntArrayRef &shape, int group_size) {
    
    // Force group_size to 64
    if (group_size != 64) {
        std::cout << "Warning: group_size must be 64, got " << group_size << std::endl;
        return torch::empty(0);
    }
    
    const int M = x.size(0);  // batch_size
    const int K = x.size(1);  // in_features
    const int N = shape[1];   // out_features
    
    // Verify input dimensions
    if (K != shape[0]) {
        std::cout << "Error: K (" << K << ") != shape[0] (" << shape[0] << ")" << std::endl;
        return torch::empty(0);
    }
    
    // Create output tensor
    auto output = torch::zeros({M, N}, x.options().dtype(torch::kFloat32));
    
    // Convert inputs to CPU float32
    auto x_f32 = x.to(torch::kFloat32);
    auto scale_f32 = scale.to(torch::kFloat32).squeeze();  // Remove extra dimensions
    auto zero_f32 = zero.to(torch::kFloat32).squeeze();
    
    // Get raw pointers for faster access
    float* x_ptr = x_f32.data_ptr<float>();
    uint8_t* W_q_ptr = W_q.data_ptr<uint8_t>();
    float* scale_ptr = scale_f32.data_ptr<float>();
    float* zero_ptr = zero_f32.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    // Simple parallel computation - no complex tiling to avoid segfaults
    at::parallel_for(0, M, 1, [&](int64_t batch_start, int64_t batch_end) {
        for (int64_t m = batch_start; m < batch_end; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                
                for (int k = 0; k < K; k++) {
                    // Calculate the linear index in the original weight matrix
                    int weight_idx = k * N + n;  // Row-major indexing
                    
                    // Map to HQQ's packed format
                    // HQQ uses shape [total_weights / (group_size * 2), group_size]
                    int elements_per_row = group_size * 2;  // 2 4-bit values per byte
                    int packed_row = weight_idx / elements_per_row;
                    int packed_col_group = (weight_idx % elements_per_row) / 2;  // Which byte in the row
                    int bit_pos = weight_idx % 2;  // Which 4-bit value in the byte
                    
                    // Bounds check
                    if (packed_row >= W_q.size(0) || packed_col_group >= W_q.size(1)) {
                        std::cout << "Index out of bounds: " << packed_row << ", " << packed_col_group << std::endl;
                        continue;
                    }
                    
                    // Get the packed byte
                    uint8_t packed_byte = W_q_ptr[packed_row * W_q.size(1) + packed_col_group];
                    
                    // Extract the 4-bit value
                    uint8_t val_4bit;
                    if (bit_pos == 0) {
                        val_4bit = packed_byte & 0x0F;  // Lower 4 bits
                    } else {
                        val_4bit = (packed_byte & 0xF0) >> 4;  // Upper 4 bits
                    }
                    
                    // Dequantize
                    int scale_idx = weight_idx / group_size;
                    if (scale_idx >= scale_f32.numel() || scale_idx >= zero_f32.numel()) {
                        std::cout << "Scale index out of bounds: " << scale_idx << std::endl;
                        continue;
                    }
                    
                    float dequant_weight = (static_cast<float>(val_4bit) - zero_ptr[scale_idx]) * scale_ptr[scale_idx];
                    
                    // Accumulate
                    sum += x_ptr[m * K + k] * dequant_weight;
                }
                
                out_ptr[m * N + n] = sum;
            }
        }
    });
    
    return output;
}

// Forward function
torch::Tensor forward_simple(torch::Tensor &x, torch::Tensor &bias, 
                            torch::Tensor &W_q, torch::Tensor &W_s, torch::Tensor &W_z, 
                            torch::IntArrayRef &W_shape, int W_group_size, int W_nbits, 
                            int W_axis, std::string W_packing) {
    
    // Only support 4-bit, group_size=64
    if (W_nbits != 4 || W_packing != "4bit_u8" || W_group_size != 64 || !x.is_cpu()) {
        return torch::empty(0);
    }
    
    auto out = simple_hqq_matmul(x, W_q, W_s, W_z, W_shape, W_group_size);
    
    if (out.numel() == 0) {
        return torch::empty(0);
    }
    
    if (bias.numel() > 0) {
        out += bias.to(out.dtype());
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_simple", &forward_simple, "Simple safe quantized forward pass");
    m.def("simple_hqq_matmul", &simple_hqq_matmul, "Simple HQQ matrix multiplication");
}