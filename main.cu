#include <stdio.h>
#include "gemm.h"
#include "perf_test.h"
class GPUTimer
{
    cudaEvent_t __begin;
    cudaEvent_t __end;

public:
    GPUTimer()
    {
        cudaEventCreate(&__begin);
        cudaEventCreate(&__end);
    }
    void start()
    {
        cudaDeviceSynchronize();
        cudaEventRecord(__begin);
        cudaEventQuery(__begin);
    }
    double stop()
    {
        auto err = cudaDeviceSynchronize();
        cudaEventRecord(__end);
        cudaEventSynchronize(__end);
        if (err != cudaSuccess)
        {
            printf("ERROR! %s:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
            return 0.0;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("ERROR! %s:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
            return 0.0;
        }
        float time = 0.0f;
        cudaEventElapsedTime(&time, __begin, __end);
        printf("%gms\n", time);
        return time / 1000.0;
    }
    ~GPUTimer()
    {
        cudaEventDestroy(__begin);
        cudaEventDestroy(__end);
    }
};
template <size_t MN, size_t K>
using gemmtest = PerfTest<MatMul<MN, K, MN>, GPUTimer>;
#ifndef MACRO_MN
#define MACRO_MN 8192
#endif
#ifndef MACRO_K
#define MACRO_K 8192
#endif
int main()
{
#ifdef MACRO_CUTLASS
    gemmtest<MACRO_MN, MACRO_K>().run<200, 1000>(0).run<200, 1000>(1).run<200, 1000>(2);
#else 
    gemmtest<MACRO_MN, MACRO_K>().run<200, 1000>(0).run<200, 1000>(1);
#endif
    return 0;
}