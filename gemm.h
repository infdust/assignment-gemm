#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas.h>
#ifdef MACRO_CUTLASS
#define private public
#include <cutlass/gemm/device/gemm.h>
#undef private
#endif
//#include "ptx_gemm.h"
#include "ptx_gemm.h"
using namespace nvcuda::wmma;
template <size_t M, size_t K, size_t N>
class MatMul
{
    cublasHandle_t handle;
    half *A;
    half *B;
    float *C;
    half *dA;
    half *dB;
    float *dC;
    void *buffer;
    float alpha;
    float beta;
#ifdef MACRO_CUTLASS
    using ElementAccumulator = float;                  // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
    using ElementOutput = float;                       // <- data type of elements in output matrix D
    // The code section below describes matrix layout of input and output matrices. Column Major for
    // Matrix A, Row Major for Matrix B and Row Major for Matrix C
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;
    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm80;
    // This code section describes the tile size a thread block will compute
    using ShapeMMAThreadBlock =
        cutlass::gemm::GemmShape<128, 256, 32>; // <- threadblock tile M = 128, N = 256, K = 32
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; // <- warp tile M = 64, N = 64, K = 32
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>; // <- MMA Op tile M = 16, N = 8, K = 8
    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??
    // This code section describes the epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                    // <- data type of output matrix
        128 / cutlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
                                                          // memory access. For a byte, it's 16
                                                          // elements. This becomes the vector width of
                                                          // math instructions in the epilogue too
        ElementAccumulator,                               // <- data type of accumulator
        ElementComputeEpilogue,                           // <- data type for alpha/beta in linear combination function
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
    // Number of pipelines you want to use
    static constexpr int NumStages = 4;
    using CutlassGemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                    LayoutInputA,
                                                    ElementInputB,
                                                    LayoutInputB,
                                                    ElementOutput,
                                                    LayoutOutput,
                                                    ElementAccumulator,
                                                    MMAOp,
                                                    SmArch,
                                                    ShapeMMAThreadBlock,
                                                    ShapeMMAWarp,
                                                    ShapeMMAOp,
                                                    EpilogueOp,
                                                    SwizzleThreadBlock,
                                                    NumStages>;
    mutable CutlassGemm cutlass_gemm;
    CutlassGemm::Arguments *args;
    dim3 grid;
    dim3 block;
#endif
    size_t smem_size;

    static constexpr size_t ptxGemmMode = 0;

public:
    MatMul(void *_buffer = nullptr) : buffer(_buffer), alpha(1.0f), beta(0.0f)
    {
        cudaSetDevice(1);
        cublasCreate_v2(&handle);
        const size_t absize = (M * K + K * N) * sizeof(half);
        const size_t csize = (M * N) * sizeof(float);
        bool flag = false;
        if (!buffer)
            buffer = malloc(absize > csize ? absize : csize);
        else
            flag = true;
        A = (half *)buffer;
        B = A + M * K;
        C = (float *)buffer;
        if (flag)
            buffer = nullptr;
        cudaMalloc(&dA, M * K * sizeof(half));
        cudaMalloc(&dB, K * N * sizeof(half));
        cudaMalloc(&dC, M * N * sizeof(float));
#ifdef MACRO_CUTLASS
        args = new CutlassGemm::Arguments(cutlass::gemm::GemmCoord(M, N, K),
                                          cutlass::TensorRef<const cutlass::half_t, cutlass::layout::RowMajor>((cutlass::half_t *)dA, K),
                                          cutlass::TensorRef<const cutlass::half_t, cutlass::layout::RowMajor>((cutlass::half_t *)dB, N),
                                          cutlass::TensorRef<const float, cutlass::layout::RowMajor>(dC, N),
                                          cutlass::TensorRef<float, cutlass::layout::RowMajor>(dC, N));
        cutlass_gemm.initialize(*args);
        SwizzleThreadBlock swizzle{};
        grid = swizzle.get_grid_shape(cutlass_gemm.params_.grid_tiled_shape);
        block = dim3(CutlassGemm::GemmKernel::kThreadCount, 1, 1);
        smem_size = size_t(sizeof(typename CutlassGemm::GemmKernel::SharedStorage));
        if (smem_size >= (0xC000))
            cudaFuncSetAttribute(cutlass::Kernel<CutlassGemm::GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
#endif
        if (sizeof(typename PTX_GEMM<M, K, N>::Smem) >= (0xC000))
            cudaFuncSetAttribute(ptx_gemm<M, K, N>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(typename PTX_GEMM<M, K, N>::Smem));
    }
    void run(int method = 0) const
    {
        switch (method)
        {
        case 0:
            ptx_gemm<M, K, N><<<dim3(PTX_GEMM<M, K, N>::nBlocks, PTX_GEMM<M, K, N>::mBlocks), (PTX_GEMM<M, K, N>::mBlockWarps * PTX_GEMM<M, K, N>::nBlockWarps) * PTX_GEMM<M, K, N>::WarpSize, sizeof(typename PTX_GEMM<M, K, N>::Smem)>>>(dA, dB, dC);
            break;
        case 1:
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dB, CUDA_R_16F, K, dA, CUDA_R_16F, N, &beta, dC, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            break;
#ifdef MACRO_CUTLASS
        case 2:
            cutlass::Kernel<CutlassGemm::GemmKernel><<<grid, block, smem_size>>>(cutlass_gemm.params_);
            break;
#endif
        }
    }
    void print(double sec, int method = 0) const
    {
        static const char *_name[] = {"mygemm", "cublas", "cutlass"};
        static const char **name = _name + 0;
        double tflops = 2.0 * M * N * K / (1024.0 * 1024.0 * 1024.0 * 1024.0) / sec;
        printf("kern = %s, M = %lu, K = %lu, N = %lu, perf = %g Tflops\n", name[method], M, K, N, tflops);
    }
    ~MatMul()
    {
#ifdef MACRO_CUTLASS
        delete args;
#endif
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cublasDestroy_v2(handle);
        free(buffer);
    }
    half *getA() { return A; }
    half *getB() { return B; }
    void set()
    {
        cudaMemcpy(dA, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    };
    float *get()
    {
        cudaMemcpy(A, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        return C;
    };
};
