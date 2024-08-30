#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <mma.h>

using namespace nvcuda;
#define MIN_VALUE (-1e38)
typedef at::Half fp16;
__half *cast(fp16 *ptr) {
    return reinterpret_cast<__half *>(ptr);
}




template <typename F>
void cuda_mm8_seq(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);

__global__ void kernel_mm_seq_fp16i8(
    const int B, const int N, const int M,
    const __half *__restrict__ const x, const int x_stride,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    __half *__restrict__ const y, const int y_stride) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && k < M) {
        float y_local = 0;
        for (int j = 0; j < N; ++j) {
            y_local += __half2float(x[i * x_stride + j]) * (
                (float(w[j * w_stride + k]) + 0.5f)
                * __half2float(rx[k]) * __half2float(ry[j])
                + __half2float(mx[k]) + __half2float(my[j])
            );
        }
        y[i * y_stride + k] = __float2half(y_local);
    }
}

#define TILE_SIZE 32


template <>
void cuda_mm8_seq<fp16>(int B, int N, int M,
                        fp16 *x, int x_stride,
                        uint8_t *w, int w_stride,
                        fp16 *mx, fp16 *rx,
                        fp16 *my, fp16 *ry,
                        fp16 *y, int y_stride) {
    dim3 blockSize(1, 256);
    dim3 gridSize((B + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm_seq_fp16i8<<<gridSize, blockSize>>>(
        B, N, M, cast(x), x_stride, w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), cast(y), y_stride);
}



#define MM8_ONE_JSPLIT 48
#define MM8_ONE_TILE 256
//#define MM8_ONE_JSPLIT 24
//#define MM8_ONE_TILE 1024


template <typename F>
void cuda_mm8_one(int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);

__global__ void kernel_mm_one_fp16i8(
    const int N, const int M,
    const __half *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    float *__restrict__ const y) {

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M) {
        float y_local = 0;
        for (int j = j0; j < j1; ++j) {
            y_local += __half2float(x[j]) * (
                (float(w[j * w_stride + k]) + 0.5f)
                * __half2float(rx[k]) * __half2float(ry[j])
                + __half2float(mx[k]) + __half2float(my[j])
            );
        }
        atomicAdd(&y[k], y_local);
    }
}

template <>
void cuda_mm8_one<fp16>(int N, int M,
                        fp16 *x,
                        uint8_t *w, int w_stride,
                        fp16 *mx, fp16 *rx,
                        fp16 *my, fp16 *ry,
                        float *y) {
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm_one_fp16i8<<<gridSize, blockSize>>>(
        N, M, cast(x), w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), y);
}






// #define MM8_MONE_JSPLIT 48
// #define MM8_MONE_TILE 256

// template <typename F>
// void cuda_mm8_mone(int B, int N, int M,
//                   F *x,
//                   uint8_t *w, int w_stride,
//                   F *mx, F *rx,
//                   F *my, F *ry,
//                   float *y);

// __global__ void kernel_mm_mone_fp16i8(
//     const int B, const int N, const int M,
//     const __half *__restrict__ const x,
//     const uint8_t *__restrict__ const w, const int w_stride,
//     const __half *__restrict__ const mx,
//     const __half *__restrict__ const rx,
//     const __half *__restrict__ const my,
//     const __half *__restrict__ const ry,
//     float *__restrict__ const y) {

//     const int b = blockIdx.z;
//     const int k = blockIdx.y * blockDim.y + threadIdx.y;
//     const int j_start = blockIdx.x * MM8_MONE_JSPLIT;
//     const int j_end = min(N, (blockIdx.x + 1) * MM8_MONE_JSPLIT);

//     if (k < M) {
//         float y_local = 0;
//         for (int j = j_start; j < j_end; ++j) {
//             y_local += __half2float(x[b * N + j]) * (
//                 (float(w[j * w_stride + k]) + 0.5f)
//                 * __half2float(rx[k]) * __half2float(ry[j])
//                 + __half2float(mx[k]) + __half2float(my[j])
//             );
//         }
//         atomicAdd(&y[b * M + k], y_local);
//     }
// }


// template <>
// void cuda_mm8_mone<fp16>(int B, int N, int M,
//                         fp16 *x,
//                         uint8_t *w, int w_stride,
//                         fp16 *mx, fp16 *rx,
//                         fp16 *my, fp16 *ry,
//                         float *y) {
//     dim3 blockSize(1, min(MM8_MONE_TILE, M));
//     dim3 gridSize(
//         (N + MM8_MONE_JSPLIT - 1) / MM8_MONE_JSPLIT,
//         (M + blockSize.y - 1) / blockSize.y,
//         B
//     );
    
//     // ゼロで初期化
//     cudaMemset(y, 0, B * M * sizeof(float));

//     kernel_mm_mone_fp16i8<<<gridSize, blockSize>>>(
//         B, N, M, cast(x), w, w_stride,
//         cast(mx), cast(rx), cast(my), cast(ry), y);
// }


#define MM8_MONE_JSPLIT 24
#define MM8_MONE_TILE 1024

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
        #pragma unroll
        for (int j = j_start; j < j_end; ++j) {

            half w_f16 = __hadd(__uint2half_rn(w[j * w_stride + k]), __float2half(0.5f));

            half rx_f16 = rx[k];  // rx is already half
            half ry_f16 = ry[j];  // ry is already half
            half mx_f16 = mx[k];  // mx is already half
            half my_f16 = my[j];  // my is already half
            
            unpacked_weights[j - j_start] = __hfma(w_f16, __hmul(rx_f16, ry_f16), __hadd(mx_f16, my_f16));

        }

        #pragma unroll
        for (int b = 0; b < B; ++b) {
            float y_local = 0;
            #pragma unroll
            for (int j = j_start; j < j_end; ++j) {
                half xbnj = x[b * N + j];
                //y_local += __half2float(xbnj) * __half2float(unpacked_weights[j - j_start]);
                y_local +=__half2float(__hmul(xbnj,unpacked_weights[j - j_start]));
            }
            atomicAdd(&y[b * M + k], y_local);
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
}




















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


#define WEIGHT_DECOMPRESS_BLOCK_SIZE 256

template <typename F>
void cuda_decompress_weights(int N, int M,
                             uint8_t *w, int w_stride,
                             F *mx, F *rx,
                             F *my, F *ry,
                             F *y);

__global__ void kernel_decompress_weights_fp16(
    const int N, const int M,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    __half *__restrict__ const y) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx / M;
    const int k = idx % M;

    if (j < N && k < M) {
        y[j * M + k] = __float2half(
            (float(w[j * w_stride + k]) + 0.5f)
            * __half2float(rx[k]) * __half2float(ry[j])
            + __half2float(mx[k]) + __half2float(my[j])
        );
    }
}

template <>
void cuda_decompress_weights<fp16>(int N, int M,
                                   uint8_t *w, int w_stride,
                                   fp16 *mx, fp16 *rx,
                                   fp16 *my, fp16 *ry,
                                   fp16 *y) {
    dim3 blockSize(WEIGHT_DECOMPRESS_BLOCK_SIZE);
    dim3 gridSize((N * M + blockSize.x - 1) / blockSize.x);

    kernel_decompress_weights_fp16<<<gridSize, blockSize>>>(
        N, M, w, w_stride,
        reinterpret_cast<__half*>(mx),
        reinterpret_cast<__half*>(rx),
        reinterpret_cast<__half*>(my),
        reinterpret_cast<__half*>(ry),
        reinterpret_cast<__half*>(y));
}