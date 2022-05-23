#include <cuda_runtime.h>
#include <mma.h> 
#include <stdio.h>
using namespace nvcuda::wmma;
//#define USE_CUBLAS
static constexpr size_t mBlockWarps = 2;
static constexpr size_t nBlockWarps = 8;
static constexpr size_t kBlockWarps = 2;
static constexpr size_t mWarpCores = 4;
static constexpr size_t nWarpCores = 2;
static constexpr size_t kWarpCores = 1;
static constexpr size_t smemPad = 8;
using copy_t = int4;

inline void _CUassert(cudaError_t err) {
	//return;
	if (err == cudaSuccess || err == cudaErrorNotReady) return;
	printf("ERROR! %s:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
}

static constexpr size_t M = 8192;
static constexpr size_t N = 8192;
static constexpr size_t K = 8192;
static constexpr size_t mCoreWidth = 16;
static constexpr size_t nCoreWidth = 16;
static constexpr size_t kCoreWidth = 16;
static constexpr size_t mWarpWidth = mWarpCores * mCoreWidth;
static constexpr size_t nWarpWidth = nWarpCores * nCoreWidth;
static constexpr size_t mBlockCores = mBlockWarps * mWarpCores;
static constexpr size_t nBlockCores = nBlockWarps * nWarpCores;
static constexpr size_t kBlockCores = kBlockWarps * kWarpCores;
static constexpr size_t mBlockWidth = mBlockCores * mCoreWidth;
static constexpr size_t nBlockWidth = nBlockCores * nCoreWidth;
static constexpr size_t kBlockWidth = kBlockCores * kCoreWidth;
static constexpr size_t mBlocks = M / mBlockWidth;
static constexpr size_t nBlocks = N / nBlockWidth;
static constexpr size_t kBlocks = K / kBlockWidth;
static constexpr size_t blockWarps = mBlockWarps * nBlockWarps;
static constexpr size_t WarpSize = 32;
static constexpr size_t kBlockI4a = kBlockWidth / (sizeof(copy_t) / sizeof(half));
static constexpr size_t nBlockI4b = nBlockWidth / (sizeof(copy_t) / sizeof(half));
static constexpr size_t mBlockI4a = blockWarps * WarpSize / kBlockI4a;
static constexpr size_t kBlockI4b = blockWarps * WarpSize / nBlockI4b;

__global__ void gemm(const half* a, const half* b, float* c) {
#define nWarpIndex ((threadIdx.x / WarpSize) % nBlockWarps)
#define mWarpIndex ((threadIdx.x / WarpSize) / nBlockWarps)
#define threadRank (threadIdx.x % (blockWarps * WarpSize))
#define nBlockIndex blockIdx.x
#define mBlockIndex blockIdx.y
	__shared__ half smemA[mBlockWidth][kBlockWidth + smemPad];
	__shared__ half smemB[kBlockWidth][nBlockWidth + smemPad];
	fragment<matrix_a, mCoreWidth, kCoreWidth, nCoreWidth, half, row_major> a_frag[mWarpCores];
	fragment<matrix_b, mCoreWidth, kCoreWidth, nCoreWidth, half, row_major> b_frag;
	fragment<accumulator, mCoreWidth, kCoreWidth, nCoreWidth, float> c_frag[mWarpCores][nWarpCores];
	copy_t Areg[mBlockWidth / mBlockI4a];
	copy_t Breg[kBlockWidth / kBlockI4b];
#pragma unroll
	for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex) {
#pragma unroll
		for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex) {
			fill_fragment(c_frag[mCoreIndex][nCoreIndex], half(0));
		}
	}
#pragma unroll
	for (size_t m = 0; m < mBlockWidth; m += mBlockI4a)
		*(copy_t*)&(smemA[m + threadRank / kBlockI4a][(threadRank % kBlockI4a) * (sizeof(copy_t) / sizeof(half))]) = *(copy_t*)&(a[(threadRank % kBlockI4a) * (sizeof(copy_t) / sizeof(half)) + (mBlockIndex * mBlockWidth + m + threadRank / kBlockI4a) * K]);
#pragma unroll
	for (size_t k = 0; k < kBlockWidth; k += kBlockI4b)
		*(copy_t*)&(smemB[k + threadRank / nBlockI4b][(threadRank % nBlockI4b) * (sizeof(copy_t) / sizeof(half))]) = *(copy_t*)&(b[nBlockIndex * nBlockWidth + (threadRank % nBlockI4b) * (sizeof(copy_t) / sizeof(half)) + (k + threadRank / nBlockI4b) * N]);
	__syncthreads();
	for (size_t kBlockIndex = 1; kBlockIndex < kBlocks; ++kBlockIndex) {
#pragma unroll
		for (size_t m = 0; m < mBlockWidth; m += mBlockI4a)
			Areg[m / mBlockI4a] = *(copy_t*)&(a[kBlockIndex * kBlockWidth + (threadRank % kBlockI4a) * (sizeof(copy_t) / sizeof(half)) + (mBlockIndex * mBlockWidth + m + threadRank / kBlockI4a) * K]);
#pragma unroll
		for (size_t k = 0; k < kBlockWidth; k += kBlockI4b)
			Breg[k / kBlockI4b] = *(copy_t*)&(b[nBlockIndex * nBlockWidth + (threadRank % nBlockI4b) * (sizeof(copy_t) / sizeof(half)) + (kBlockIndex * kBlockWidth + k + threadRank / nBlockI4b) * N]);
#pragma unroll
		for (size_t kCoreIndex = 0; kCoreIndex < kBlockWarps * kWarpCores; ++kCoreIndex) {
#pragma unroll
			for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
				load_matrix_sync(a_frag[mCoreIndex], &(smemA[mWarpIndex * mWarpWidth + mCoreIndex * mCoreWidth][kCoreIndex * kCoreWidth]), kBlockWidth + smemPad);
#pragma unroll
			for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
				load_matrix_sync(b_frag, &(smemB[kCoreIndex * kCoreWidth][nWarpIndex * nWarpWidth + nCoreIndex * nCoreWidth]), nBlockWidth + smemPad);
#pragma unroll
			for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
#pragma unroll
				for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
					mma_sync(c_frag[mCoreIndex][nCoreIndex], a_frag[mCoreIndex], b_frag, c_frag[mCoreIndex][nCoreIndex]);
		}
		__syncthreads();
#pragma unroll
		for (size_t m = 0; m < mBlockWidth; m += mBlockI4a)
			*(copy_t*)&(smemA[m + threadRank / kBlockI4a][(threadRank % kBlockI4a) * (sizeof(copy_t) / sizeof(half))]) = Areg[m / mBlockI4a];
#pragma unroll
		for (size_t k = 0; k < kBlockWidth; k += kBlockI4b)
			*(copy_t*)&(smemB[k + threadRank / nBlockI4b][(threadRank % nBlockI4b) * (sizeof(copy_t) / sizeof(half))]) = Breg[k / kBlockI4b];
		__syncthreads();
	}
#pragma unroll
	for (size_t kCoreIndex = 0; kCoreIndex < kBlockWarps * kWarpCores; ++kCoreIndex) {
#pragma unroll
		for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
			load_matrix_sync(a_frag[mCoreIndex], &(smemA[mWarpIndex * mWarpWidth + mCoreIndex * mCoreWidth][kCoreIndex * kCoreWidth]), kBlockWidth + smemPad);
#pragma unroll
		for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
			load_matrix_sync(b_frag, &(smemB[kCoreIndex * kCoreWidth][nWarpIndex * nWarpWidth + nCoreIndex * nCoreWidth]), nBlockWidth + smemPad);
#pragma unroll
		for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
#pragma unroll
			for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
				mma_sync(c_frag[mCoreIndex][nCoreIndex], a_frag[mCoreIndex], b_frag, c_frag[mCoreIndex][nCoreIndex]);
	}
#pragma unroll
	for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
#pragma unroll
		for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
			store_matrix_sync(&c[nBlockIndex * nBlockWidth + nWarpIndex * nWarpWidth + nCoreIndex * nCoreWidth + (mBlockIndex * mBlockWidth + mWarpIndex * mWarpWidth + mCoreIndex * mCoreWidth) * N], c_frag[mCoreIndex][nCoreIndex], M, mem_row_major);
}
#include <cublas.h>
int main() {
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	float alpha = 1, beta = 0;
	cudaEvent_t __begin;
	cudaEvent_t __end;
	cudaEventCreate(&__begin);
	cudaEventCreate(&__end);
	void* buffer = malloc((M * K + K * N) * sizeof(half));
	auto A = (half*)buffer;
	auto B = A + M * K;
	auto C = (float*)buffer;
	for (size_t i = 0; i < M * K + K * N; ++i)A[i] = half(1);
	half* dA, * dB;
	float* dC;
	cudaMalloc(&dA, M * K * sizeof(half));
	cudaMalloc(&dB, K * N * sizeof(half));
	cudaMalloc(&dC, M * N * sizeof(float));
	cudaMemcpy(dA, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, K * N * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemset(dC, 0, M * N * sizeof(float));
	cudaDeviceSynchronize();
	cudaEventRecord(__begin);
	cudaEventQuery(__begin);
#ifdef USE_CUBLAS
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 8192, 8192, 8192, &alpha, dA, CUDA_R_16F, 8192, dB, CUDA_R_16F, 8192, &beta, dC, CUDA_R_32F, 8192, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
	gemm << <dim3(nBlocks, mBlocks), (mBlockWarps * nBlockWarps)* WarpSize >> > (dA, dB, dC);
#endif
	_CUassert(cudaDeviceSynchronize());
	cudaEventRecord(__end);
	cudaEventSynchronize(__end);
	float time = 0.0f;
	cudaEventElapsedTime(&time, __begin, __end);
	cudaMemcpy(A, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	size_t ans = 0;
	for (size_t i = 0; i < M * N; ++i)if (C[i] == K)++ans;
	printf("ans=%lu:%lu, time=%gms, perform=%gTflops\n", ans, M * N, time, 1000.0f / time);
	free(buffer);
	cudaEventDestroy(__begin);
	cudaEventDestroy(__end);
	return 0;
}
