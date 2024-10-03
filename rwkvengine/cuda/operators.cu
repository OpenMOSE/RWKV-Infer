#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

//#include <mma.h>

//using namespace nvcuda;
#define MIN_VALUE (-1e38)
typedef at::Half fp16;
__half *cast(fp16 *ptr) {
    return reinterpret_cast<__half *>(ptr);
}

#define MM8_MONE_JSPLIT 24
#define MM8_MONE_TILE 512

template <typename F>
void cuda_mm8_mone(int B, int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);

__global__ void kernel_mm_mone_fp16i8(
    const int B, const int N, const int M,
    const __half *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    float *__restrict__ const y) {

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j_start = blockIdx.x * MM8_MONE_JSPLIT;
    const int j_end = min(N, (blockIdx.x + 1) * MM8_MONE_JSPLIT);

    if (k < M) {

        half unpacked_weights[MM8_MONE_JSPLIT];
        //#pragma unroll
        for (int j = j_start; j < j_end; ++j) {

            half w_f16 = __hadd(__uint2half_rn(w[j * w_stride + k]), __float2half(0.5f));

            half rx_f16 = rx[k];  // rx is already half
            half ry_f16 = ry[j];  // ry is already half
            half mx_f16 = mx[k];  // mx is already half
            half my_f16 = my[j];  // my is already half
            
            unpacked_weights[j - j_start] = __hfma(w_f16, __hmul(rx_f16, ry_f16), __hadd(mx_f16, my_f16));

        }

        //#pragma unroll
        for (int b = 0; b < B; ++b) {
            float y_local = 0;
            //#pragma unroll
            for (int j = j_start; j < j_end; ++j) {
                half xbnj = x[b * N + j];
                //y_local += __half2float(xbnj) * __half2float(unpacked_weights[j - j_start]);
                y_local +=__half2float(__hmul(xbnj,unpacked_weights[j - j_start]));
            }
            atomicAdd(&y[b * M + k], y_local);
        }
    }
}

__global__ void kernel_mm_mone_fp16i8_shared(
    const int B, const int N, const int M,
    const __half *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    float *__restrict__ const y) {

    extern __shared__ __half shared_mem[];
    __half* shared_x = shared_mem;
    __half* shared_w = shared_mem + blockDim.x * MM8_MONE_JSPLIT;

    const int tid = threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j_start = blockIdx.x * MM8_MONE_JSPLIT;
    const int j_end = min(N, (blockIdx.x + 1) * MM8_MONE_JSPLIT);

    if (k < M) {
        // 重みを共有メモリにロード
        for (int j = tid; j < (j_end - j_start); j += blockDim.x) {
            __half w_f16 = __hadd(__uint2half_rn(w[(j + j_start) * w_stride + k]), __float2half(0.5f));
            __half rx_f16 = rx[k];
            __half ry_f16 = ry[j + j_start];
            __half mx_f16 = mx[k];
            __half my_f16 = my[j + j_start];
            shared_w[j * blockDim.y + threadIdx.y] = __hfma(w_f16, __hmul(rx_f16, ry_f16), __hadd(mx_f16, my_f16));
        }
        __syncthreads();

        for (int b = 0; b < B; ++b) {
            float y_local = 0.0f;
            // xを共有メモリにロード
            for (int j = tid; j < (j_end - j_start); j += blockDim.x) {
                shared_x[j] = x[b * N + j + j_start];
            }
            __syncthreads();

            // 内積計算
            for (int j = 0; j < (j_end - j_start); ++j) {
                y_local += __half2float(__hmul(shared_x[j], shared_w[j * blockDim.y + threadIdx.y]));
            }
            atomicAdd(&y[b * M + k], y_local);
            __syncthreads();
        }
    }
}


template <>
void cuda_mm8_mone<fp16>(int B, int N, int M,
                        fp16 *x,
                        uint8_t *w, int w_stride,
                        fp16 *mx, fp16 *rx,
                        fp16 *my, fp16 *ry,
                        float *y) {
     dim3 blockSize(1, min(MM8_MONE_TILE, M));
    dim3 gridSize(
        (N + MM8_MONE_JSPLIT - 1) / MM8_MONE_JSPLIT,
        (M + blockSize.y - 1) / blockSize.y
        //1//B
    );
    
    // ゼロで初期化
    cudaMemset(y, 0, B * M * sizeof(float));

    kernel_mm_mone_fp16i8<<<gridSize, blockSize>>>(
        B, N, M, cast(x), w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), y); 
    // cudaMemset(y, 0, B * M * sizeof(float));
    // dim3 blockSize(32, 8);  // 共有メモリの容量に応じて調整
    // dim3 gridSize(
    //     (N + MM8_MONE_JSPLIT - 1) / MM8_MONE_JSPLIT,
    //     (M + blockSize.y - 1) / blockSize.y
    // );
    // size_t sharedMemSize = (blockSize.x * MM8_MONE_JSPLIT + MM8_MONE_JSPLIT * blockSize.y) * sizeof(__half);

    // kernel_mm_mone_fp16i8_shared<<<gridSize, blockSize, sharedMemSize>>>(
    //     B, N, M, cast(x), w, w_stride,
    //     cast(mx), cast(rx), cast(my), cast(ry), y);

}
 
 



/*

#define MM8_MONE_JSPLIT 64
#define MM8_MONE_TILE 1024 

template <typename F>
void cuda_mm8_mone(int B, int N, int M,
                   F *x,
                   uint8_t *w, int w_stride,
                   F *mx, F *rx,
                   F *my, F *ry,
                   float *y);

__global__ void kernel_mm_mone_fp16i8_optimized(
    const int B, const int N, const int M,
    const __half *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    float *__restrict__ const y) {

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j_start = blockIdx.x * MM8_MONE_JSPLIT;
    const int j_end = min(N, (blockIdx.x + 1) * MM8_MONE_JSPLIT);

    if (k < M && b < B) {

        // Preload rx and mx for the current 'k'
        __half rx_f16 = rx[k];
        __half mx_f16 = mx[k];

        float y_local = 0.0f;

        #pragma unroll
        for (int j = j_start; j < j_end; ++j) {

            // Load and compute unpacked weight
            __half w_f16 = __hadd(__uint2half_rn(w[j * w_stride + k]), __float2half(0.5f));
            __half ry_f16 = ry[j];
            __half my_f16 = my[j];

            __half unpacked_weight = __hfma(w_f16, __hmul(rx_f16, ry_f16), __hadd(mx_f16, my_f16));

            // Load 'x' and compute partial sum
            __half xbnj = x[b * N + j];
            y_local += __half2float(__hmul(xbnj, unpacked_weight));
        }

        // Write the result to 'y' without atomic operation
        y[b * M + k] = y_local;
    }
}

template <>
void cuda_mm8_mone<fp16>(int B, int N, int M,
                           fp16 *x,
                           uint8_t *w, int w_stride,
                           fp16 *mx, fp16 *rx,
                           fp16 *my, fp16 *ry,
                           float *y) {
    // Define block and grid dimensions
    const int block_k = 32; // Number of threads processing 'k'
    const int block_b = 8;  // Number of threads processing 'b'
    dim3 blockSize(1, block_k, block_b);
    dim3 gridSize(
        (N + MM8_MONE_JSPLIT - 1) / MM8_MONE_JSPLIT,
        (M + block_k - 1) / block_k,
        (B + block_b - 1) / block_b
    );

    // Launch the optimized kernel
    kernel_mm_mone_fp16i8_optimized<<<gridSize, blockSize>>>(
        B, N, M, cast(x), w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), y);
}


*/













// template <typename T>
// __global__ void kernel_decompress_weights(
//     const int N, const int M,
//     const uint8_t* __restrict__ const w, const int w_stride,
//     const T* __restrict__ const mx,
//     const T* __restrict__ const rx,
//     const T* __restrict__ const my,
//     const T* __restrict__ const ry,
//     T* __restrict__ const y) {

//     const int j = blockIdx.x * blockDim.x + threadIdx.x;
//     const int k = blockIdx.y * blockDim.y + threadIdx.y;

//     if (j < N && k < M) {
//         y[j * M + k] = (
//             (float(w[j * w_stride + k]) + 0.5f)
//             * rx[k] * ry[j]
//             + mx[k] + my[j]
//         );
//     }
// }

// template <typename T>
// void cuda_decompress_weights(int N, int M,
//                              uint8_t* w, int w_stride,
//                              T* mx, T* rx,
//                              T* my, T* ry,
//                              T* y) {
//     dim3 blockSize(4, 256);
//     dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
//     kernel_decompress_weights<<<gridSize, blockSize>>>(
//         N, M, w, w_stride, mx, rx, my, ry, y);
// }

// // fp16 (half) に対する特殊化
// template <>
// void cuda_decompress_weights<fp16>(int N, int M,
//                                    uint8_t* w, int w_stride,
//                                    fp16* mx, fp16* rx,
//                                    fp16* my, fp16* ry,
//                                    fp16* y) {
//     dim3 blockSize(4, 256);
//     dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
//     kernel_decompress_weights<<<gridSize, blockSize>>>(
//         N, M, w, w_stride, mx, rx, my, ry, y);
// }

