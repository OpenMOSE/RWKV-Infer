#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

typedef at::Half fp16;

/*
template <typename F>
void cuda_mm8_seq(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);
template <typename F>
void cuda_mm8_seq_optimized(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);
template <typename F>
void cuda_mm8_one(int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);*/
template <typename F>
void cuda_mm8_mone(int B, int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);
/*template <typename T>
void cuda_decompress_weights(int N, int M,
                             uint8_t* w, int w_stride,
                             T* mx, T* rx,
                             T* my, T* ry,
                             T* y);*/



void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(1) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(1) == 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
        //cuda_mm8_seq(
        cuda_mm8_mone(
            B, N, M,
            x.data_ptr<fp16>(), //x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<float>()//, y.stride(0)
            );
        


        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}
/*void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(0) == 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_mm8_one(
            N, M,
            x.data_ptr<fp16>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<float>());
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

void dequantize_8bit(int64_t N, int64_t M,
             torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y){

    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(0) == 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));

    switch (y.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_decompress_weights(
            N, M,
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<fp16>());
        break;
    default:
        assert(false && "Only FP16 is currently supported");
    }       
        
}*/

using torch::Tensor;

#ifndef DISABLE_CUBLAS_GEMM
void gemm_fp16_cublas(Tensor a, Tensor b, Tensor c);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //m.def("mm8_seq", &mm8_seq, "mm8 seq");
    m.def("mm8_seq", &mm8_seq, "mm8 seq");
    //m.def("mm8_one", &mm8_one, "mm8 one");
    //m.def("dequantize_8bit", &dequantize_8bit, "dequantize 8bit");
#ifndef DISABLE_CUBLAS_GEMM
    m.def("gemm_fp16_cublas", &gemm_fp16_cublas, "gemv fp16 cublas");
#endif
}

TORCH_LIBRARY(rwkv, m) {
    m.def("mm8_seq", mm8_seq);
    //m.def("mm8_one", mm8_one);
    //m.def("dequantize_8bit", dequantize_8bit);
#ifndef DISABLE_CUBLAS_GEMM
    m.def("gemm_fp16_cublas", gemm_fp16_cublas);
#endif
}
