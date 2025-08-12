#include <torch/extension.h>
#include <vector>
#include <string>
#include <iostream>
#include <ATen/Parallel.h>
#include <immintrin.h>
#include <cstring>
#include <algorithm>

// Turbo HQQ kernel with aggressive optimizations
torch::Tensor turbo_hqq_matmul(torch::Tensor &x, torch::Tensor &W_q, 
                               torch::Tensor &scale, torch::Tensor &zero, 
                               torch::IntArrayRef &shape, int group_size) {
    
    if (group_size != 64) {
        return torch::empty(0);
    }
    
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = shape[1];
    
    auto output = torch::zeros({M, N}, x.options().dtype(torch::kFloat32));
    
    auto x_f32 = x.to(torch::kFloat32);
    auto scale_f32 = scale.to(torch::kFloat32).squeeze();
    auto zero_f32 = zero.to(torch::kFloat32).squeeze();
    
    float* x_ptr = x_f32.data_ptr<float>();
    uint8_t* W_q_ptr = W_q.data_ptr<uint8_t>();
    float* scale_ptr = scale_f32.data_ptr<float>();
    float* zero_ptr = zero_f32.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    // Pre-compute dequantization lookup table for 4-bit values
    std::vector<std::vector<float>> dequant_lut(scale_f32.numel());
    for (int scale_idx = 0; scale_idx < scale_f32.numel(); scale_idx++) {
        dequant_lut[scale_idx].resize(16);  // 4-bit = 16 possible values
        float scale_val = scale_ptr[scale_idx];
        float zero_val = zero_ptr[scale_idx];
        for (int val = 0; val < 16; val++) {
            dequant_lut[scale_idx][val] = (static_cast<float>(val) - zero_val) * scale_val;
        }
    }
    
    // Aggressive tiling for maximum cache efficiency
    const int tile_M = 8;   // Small for better parallelization
    const int tile_N = 64;  // Moderate for SIMD efficiency
    const int tile_K = 32;  // Small for cache efficiency
    
    at::parallel_for(0, M, 1, [&](int64_t m_start, int64_t m_end) {
        // Process each batch element independently
        for (int64_t m = m_start; m < m_end; m++) {
            
            // Process output in chunks
            for (int n_start = 0; n_start < N; n_start += tile_N) {
                int n_end = std::min(n_start + tile_N, N);
                int n_count = n_end - n_start;
                
                // Initialize output chunk
                std::fill(&out_ptr[m * N + n_start], &out_ptr[m * N + n_end], 0.0f);
                
                // Process K dimension in tiles
                for (int k_start = 0; k_start < K; k_start += tile_K) {
                    int k_end = std::min(k_start + tile_K, K);
                    int k_count = k_end - k_start;
                    
                    // Optimized inner loop with reduced overhead
                    for (int k = k_start; k < k_end; k++) {
                        float x_val = x_ptr[m * K + k];
                        
                        // Vectorized processing where possible
                        #ifdef __AVX2__
                        if (n_count >= 8) {
                            for (int n = n_start; n < n_end - 7; n += 8) {
                                // Load 8 output values
                                __m256 out_vec = _mm256_loadu_ps(&out_ptr[m * N + n]);
                                
                                // Process 8 weights
                                alignas(32) float weights[8];
                                for (int i = 0; i < 8; i++) {
                                    int n_global = n + i;
                                    int weight_idx = k * N + n_global;
                                    
                                    int elements_per_row = group_size * 2;
                                    int packed_row = weight_idx / elements_per_row;
                                    int packed_col_group = (weight_idx % elements_per_row) / 2;
                                    int bit_pos = weight_idx % 2;
                                    
                                    if (packed_row < W_q.size(0) && packed_col_group < W_q.size(1)) {
                                        uint8_t packed_byte = W_q_ptr[packed_row * W_q.size(1) + packed_col_group];
                                        uint8_t val_4bit = (bit_pos == 0) ? (packed_byte & 0x0F) : ((packed_byte & 0xF0) >> 4);
                                        
                                        int lut_idx = weight_idx / group_size;
                                        if (lut_idx < dequant_lut.size()) {
                                            weights[i] = dequant_lut[lut_idx][val_4bit];
                                        } else {
                                            weights[i] = 0.0f;
                                        }
                                    } else {
                                        weights[i] = 0.0f;
                                    }
                                }
                                
                                // Broadcast x_val and multiply with weights
                                __m256 x_broadcast = _mm256_set1_ps(x_val);
                                __m256 w_vec = _mm256_load_ps(weights);
                                __m256 result = _mm256_fmadd_ps(x_broadcast, w_vec, out_vec);
                                
                                // Store result
                                _mm256_storeu_ps(&out_ptr[m * N + n], result);
                            }
                            
                            // Handle remaining elements
                            for (int n = ((n_end - n_start) / 8) * 8 + n_start; n < n_end; n++) {
                                int weight_idx = k * N + n;
                                
                                int elements_per_row = group_size * 2;
                                int packed_row = weight_idx / elements_per_row;
                                int packed_col_group = (weight_idx % elements_per_row) / 2;
                                int bit_pos = weight_idx % 2;
                                
                                if (packed_row < W_q.size(0) && packed_col_group < W_q.size(1)) {
                                    uint8_t packed_byte = W_q_ptr[packed_row * W_q.size(1) + packed_col_group];
                                    uint8_t val_4bit = (bit_pos == 0) ? (packed_byte & 0x0F) : ((packed_byte & 0xF0) >> 4);
                                    
                                    int lut_idx = weight_idx / group_size;
                                    if (lut_idx < dequant_lut.size()) {
                                        float weight = dequant_lut[lut_idx][val_4bit];
                                        out_ptr[m * N + n] += x_val * weight;
                                    }
                                }
                            }
                        } else
                        #endif
                        {
                            // Scalar processing with lookup table
                            for (int n = n_start; n < n_end; n++) {
                                int weight_idx = k * N + n;
                                
                                int elements_per_row = group_size * 2;
                                int packed_row = weight_idx / elements_per_row;
                                int packed_col_group = (weight_idx % elements_per_row) / 2;
                                int bit_pos = weight_idx % 2;
                                
                                if (packed_row < W_q.size(0) && packed_col_group < W_q.size(1)) {
                                    uint8_t packed_byte = W_q_ptr[packed_row * W_q.size(1) + packed_col_group];
                                    uint8_t val_4bit = (bit_pos == 0) ? (packed_byte & 0x0F) : ((packed_byte & 0xF0) >> 4);
                                    
                                    int lut_idx = weight_idx / group_size;
                                    if (lut_idx < dequant_lut.size()) {
                                        float weight = dequant_lut[lut_idx][val_4bit];
                                        out_ptr[m * N + n] += x_val * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    
    return output;
}

// Ultra-turbo kernel for 256-aligned matrices
torch::Tensor ultra_turbo_256_aligned_matmul(torch::Tensor &x, torch::Tensor &W_q, 
                                             torch::Tensor &scale, torch::Tensor &zero, 
                                             torch::IntArrayRef &shape, int group_size) {
    
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = shape[1];
    
    // Verify alignment
    if (K % 256 != 0 || N % 256 != 0) {
        return turbo_hqq_matmul(x, W_q, scale, zero, shape, group_size);
    }
    
    auto output = torch::zeros({M, N}, x.options().dtype(torch::kFloat32));
    
    auto x_f32 = x.to(torch::kFloat32);
    auto scale_f32 = scale.to(torch::kFloat32).squeeze();
    auto zero_f32 = zero.to(torch::kFloat32).squeeze();
    
    float* x_ptr = x_f32.data_ptr<float>();
    uint8_t* W_q_ptr = W_q.data_ptr<uint8_t>();
    float* scale_ptr = scale_f32.data_ptr<float>();
    float* zero_ptr = zero_f32.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    // Pre-compute all dequantized weights for 256-aligned case
    std::vector<float> W_dequant(K * N);
    
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            int weight_idx = k * N + n;
            
            int elements_per_row = group_size * 2;
            int packed_row = weight_idx / elements_per_row;
            int packed_col_group = (weight_idx % elements_per_row) / 2;
            int bit_pos = weight_idx % 2;
            
            if (packed_row < W_q.size(0) && packed_col_group < W_q.size(1)) {
                uint8_t packed_byte = W_q_ptr[packed_row * W_q.size(1) + packed_col_group];
                uint8_t val_4bit = (bit_pos == 0) ? (packed_byte & 0x0F) : ((packed_byte & 0xF0) >> 4);
                
                int scale_idx = weight_idx / group_size;
                if (scale_idx < scale_f32.numel()) {
                    W_dequant[k * N + n] = (static_cast<float>(val_4bit) - zero_ptr[scale_idx]) * scale_ptr[scale_idx];
                } else {
                    W_dequant[k * N + n] = 0.0f;
                }
            } else {
                W_dequant[k * N + n] = 0.0f;
            }
        }
    }
    
    // High-performance GEMM with pre-dequantized weights
    const int tile_M = 16;
    const int tile_N = 256;  // Match alignment
    const int tile_K = 128;  // Larger for better amortization
    
    at::parallel_for(0, (M + tile_M - 1) / tile_M, 1, [&](int64_t m_tile_start, int64_t m_tile_end) {
        for (int64_t m_tile = m_tile_start; m_tile < m_tile_end; m_tile++) {
            int m_start = m_tile * tile_M;
            int m_end = std::min(m_start + tile_M, static_cast<int>(M));
            
            for (int n_start = 0; n_start < N; n_start += tile_N) {
                int n_end = std::min(n_start + tile_N, N);
                
                for (int k_start = 0; k_start < K; k_start += tile_K) {
                    int k_end = std::min(k_start + tile_K, K);
                    
                    // Optimized GEMM kernel
                    for (int m = m_start; m < m_end; m++) {
                        #ifdef __AVX2__
                        for (int n = n_start; n < n_end; n += 8) {
                            __m256 sum_vec = (k_start == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(&out_ptr[m * N + n]);
                            
                            for (int k = k_start; k < k_end; k++) {
                                __m256 x_broadcast = _mm256_broadcast_ss(&x_ptr[m * K + k]);
                                __m256 w_vec = _mm256_loadu_ps(&W_dequant[k * N + n]);
                                sum_vec = _mm256_fmadd_ps(x_broadcast, w_vec, sum_vec);
                            }
                            
                            _mm256_storeu_ps(&out_ptr[m * N + n], sum_vec);
                        }
                        #else
                        for (int n = n_start; n < n_end; n++) {
                            float sum = (k_start == 0) ? 0.0f : out_ptr[m * N + n];
                            
                            for (int k = k_start; k < k_end; k++) {
                                sum += x_ptr[m * K + k] * W_dequant[k * N + n];
                            }
                            
                            out_ptr[m * N + n] = sum;
                        }
                        #endif
                    }
                }
            }
        }
    });
    
    return output;
}

// Forward function with turbo optimization
torch::Tensor forward_turbo(torch::Tensor &x, torch::Tensor &bias, 
                           torch::Tensor &W_q, torch::Tensor &W_s, torch::Tensor &W_z, 
                           torch::IntArrayRef &W_shape, int W_group_size, int W_nbits, 
                           int W_axis, std::string W_packing) {
    
    if (W_nbits != 4 || W_packing != "4bit_u8" || W_group_size != 64 || !x.is_cpu()) {
        return torch::empty(0);
    }
    
    const int K = x.size(1);
    const int N = W_shape[1];
    
    torch::Tensor out;
    if (K % 256 == 0 && N % 256 == 0) {
        out = ultra_turbo_256_aligned_matmul(x, W_q, W_s, W_z, W_shape, W_group_size);
    } else {
        out = turbo_hqq_matmul(x, W_q, W_s, W_z, W_shape, W_group_size);
    }
    
    if (out.numel() == 0) {
        return torch::empty(0);
    }
    
    if (bias.numel() > 0) {
        out += bias.to(out.dtype());
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_turbo", &forward_turbo, "Turbo optimized quantized forward pass");
    m.def("turbo_hqq_matmul", &turbo_hqq_matmul, "Turbo HQQ matrix multiplication");
    m.def("ultra_turbo_256_aligned_matmul", &ultra_turbo_256_aligned_matmul, "Ultra-turbo 256-aligned matrix multiplication");
}