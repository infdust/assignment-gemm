#pragma once
#include <stdio.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
template <size_t M, size_t K, size_t N>
struct MMA_GEMM
{
    static constexpr size_t mBlockWarps = 4;
    static constexpr size_t nBlockWarps = 4;
    static constexpr size_t kBlockWarps = 1;
    static constexpr size_t mWarpCores = 4;
    static constexpr size_t nWarpCores = 8;
    static constexpr size_t kWarpCores = 4;
    static constexpr size_t mCoreWidth = 16;
    static constexpr size_t nCoreWidth = 8;
    static constexpr size_t kCoreWidth = 16;
    static constexpr size_t mWarpWidth = mWarpCores * mCoreWidth;
    static constexpr size_t nWarpWidth = nWarpCores * nCoreWidth;
    static constexpr size_t kWarpWidth = kWarpCores * kCoreWidth;
    static constexpr size_t mBlockCores = mBlockWarps * mWarpCores;
    static constexpr size_t nBlockCores = nBlockWarps * nWarpCores;
    static constexpr size_t kBlockCores = kBlockWarps * kWarpCores;
    static constexpr size_t mBlockWidth = mBlockCores * mCoreWidth;
    static constexpr size_t nBlockWidth = nBlockCores * nCoreWidth;
    static constexpr size_t kBlockWidth = kBlockCores * kCoreWidth;
    static constexpr size_t mBlocks = M / mBlockWidth;
    static constexpr size_t nBlocks = N / nBlockWidth;
    static constexpr size_t kBlocks = K / kBlockWidth;
    static constexpr size_t ldA = sizeof(half) * K;
    static constexpr size_t ldB = sizeof(half) * N;
    static constexpr size_t ldC = sizeof(float) * N;
    static constexpr size_t sldA = sizeof(half) * (kBlockWidth + 8);
    static constexpr size_t sldB = sizeof(half) * (nBlockWidth + 16);
    static constexpr size_t sldC = sizeof(float) * (nBlockWidth + 8);
    using copy_t = int4;
    static constexpr size_t WarpSize = 32;
    template <typename T, size_t _m, size_t _n, size_t _ld>
    struct Matrix
    {
        static constexpr size_t m = _m;
        static constexpr size_t n = _n;
        static constexpr size_t ld = _ld;
        static constexpr size_t mn = m * n;
        using Type = T;
        struct Row
        {
            static_assert(n * sizeof(T) <= ld);
            union
            {
                T buffer[n];
                char padding[ld];
            };
            __device__ T &operator[](size_t i) { return buffer[i]; }
            __device__ const T &operator[](size_t i) const { return buffer[i]; }
        };
        Row buffer[m];
        __device__ Row &operator[](size_t i) { return buffer[i]; }
        template <size_t mWidth, typename = std::enable_if_t<m % mWidth == 0>>
        __device__ Matrix<T, mWidth, n, ld> &sliceM(size_t i) { return *(Matrix<T, mWidth, n, ld> *)&(buffer[mWidth * i][0]); }
        template <size_t nWidth, typename = std::enable_if_t<n % nWidth == 0>>
        __device__ Matrix<T, m, nWidth, ld> &sliceN(size_t i) { return *(Matrix<T, m, nWidth, ld> *)&(buffer[0][nWidth * i]); }
        template <size_t mWidth, size_t nWidth, typename = std::enable_if_t<m % mWidth == 0 && n % nWidth == 0>>
        __device__ Matrix<T, mWidth, nWidth, ld> &slice(size_t im, size_t in) { return *(Matrix<T, mWidth, nWidth, ld> *)&(buffer[mWidth * im][nWidth * in]); }
        __device__ const Row &operator[](size_t i) const { return buffer[i]; }
        template <size_t mWidth, typename = std::enable_if_t<m % mWidth == 0>>
        __device__ const Matrix<T, mWidth, n, ld> &sliceM(size_t i) const { return *(Matrix<T, mWidth, n, ld> *)&(buffer[mWidth * i][0]); }
        template <size_t nWidth, typename = std::enable_if_t<n % nWidth == 0>>
        __device__ const Matrix<T, m, nWidth, ld> &sliceN(size_t i) const { return *(Matrix<T, m, nWidth, ld> *)&(buffer[0][nWidth * i]); }
        template <size_t mWidth, size_t nWidth, typename = std::enable_if_t<m % mWidth == 0 && n % nWidth == 0>>
        __device__ const Matrix<T, mWidth, nWidth, ld> &slice(size_t im, size_t in) const { return *(Matrix<T, mWidth, nWidth, ld> *)&(buffer[mWidth * im][nWidth * in]); }
        using CopyT = Matrix<copy_t, m, n * sizeof(T) / sizeof(copy_t), ld>;
        __device__ CopyT &copyT() { return *(CopyT *)this; }
        __device__ const CopyT &copyT() const { return *(CopyT *)this; }
    };
    struct MMA_A : public Matrix<half, 16, 16, sldA>
    {
        using Matrix<half, 16, 16, sldA>::operator[];
        using Matrix<half, 16, 16, sldA>::Matrix;
        struct Piece
        {
            struct Layout
            {
                half a;
                half b;
            };
            union
            {
                Layout layout;
                u_int32_t reg;
            };
        };
        __device__ Piece get0() const { return {(*this)[(threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4)], (*this)[(threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4) + 1]}; }
        __device__ Piece get1() const { return {(*this)[8 + (threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4)], (*this)[8 + (threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4) + 1]}; }
        __device__ Piece get2() const { return {(*this)[(threadIdx.x % WarpSize) / 4][8 + 2 * ((threadIdx.x % WarpSize) % 4)], (*this)[(threadIdx.x % WarpSize) / 4][8 + 2 * ((threadIdx.x % WarpSize) % 4) + 1]}; }
        __device__ Piece get3() const { return {(*this)[8 + (threadIdx.x % WarpSize) / 4][8 + 2 * ((threadIdx.x % WarpSize) % 4)], (*this)[8 + (threadIdx.x % WarpSize) / 4][8 + 2 * ((threadIdx.x % WarpSize) % 4) + 1]}; }
        __device__ const void *getRow() const { return &((*this)[(threadIdx.x % WarpSize) % 16][8 * ((threadIdx.x % WarpSize) / 16)]); }
    };
    struct MMA_B : public Matrix<half, 16, 8, sldB>
    {
        using Matrix<half, 16, 8, sldB>::operator[];
        using Matrix<half, 16, 8, sldB>::Matrix;
        struct Piece
        {
            struct Layout
            {
                half a;
                half b;
            };
            union
            {
                Layout layout;
                u_int32_t reg;
            };
        };
        __device__ Piece get0() const { return {(*this)[2 * ((threadIdx.x % WarpSize) % 4)][(threadIdx.x % WarpSize) / 4], (*this)[2 * ((threadIdx.x % WarpSize) % 4) + 1][(threadIdx.x % WarpSize) / 4]}; }
        __device__ Piece get1() const { return {(*this)[8 + 2 * ((threadIdx.x % WarpSize) % 4)][(threadIdx.x % WarpSize) / 4], (*this)[8 + 2 * ((threadIdx.x % WarpSize) % 4) + 1][(threadIdx.x % WarpSize) / 4]}; }
        __device__ const void *getRow() const { return &((*this)[(threadIdx.x % WarpSize) % 16][0]); }
    };
    struct MMA_B2 : public Matrix<half, 16, 16, sldB>
    {
        using Matrix<half, 16, 16, sldB>::operator[];
        using Matrix<half, 16, 16, sldB>::Matrix;
        struct Piece
        {
            struct Layout
            {
                half a;
                half b;
            };
            union
            {
                Layout layout;
                u_int32_t reg;
            };
        };
        __device__ Piece get0() const { return {(*this)[2 * ((threadIdx.x % WarpSize) % 4)][(threadIdx.x % WarpSize) / 4], (*this)[2 * ((threadIdx.x % WarpSize) % 4) + 1][(threadIdx.x % WarpSize) / 4]}; }
        __device__ Piece get1() const { return {(*this)[8 + 2 * ((threadIdx.x % WarpSize) % 4)][(threadIdx.x % WarpSize) / 4], (*this)[8 + 2 * ((threadIdx.x % WarpSize) % 4) + 1][(threadIdx.x % WarpSize) / 4]}; }
        __device__ Piece get0_() const { return {(*this)[2 * ((threadIdx.x % WarpSize) % 4)][8 + (threadIdx.x % WarpSize) / 4], (*this)[2 * ((threadIdx.x % WarpSize) % 4) + 1][8 + (threadIdx.x % WarpSize) / 4]}; }
        __device__ Piece get1_() const { return {(*this)[8 + 2 * ((threadIdx.x % WarpSize) % 4)][8 + (threadIdx.x % WarpSize) / 4], (*this)[8 + 2 * ((threadIdx.x % WarpSize) % 4) + 1][8 + (threadIdx.x % WarpSize) / 4]}; }
        __device__ const void *getRow() const { return &((*this)[(threadIdx.x % WarpSize) % 16][8 * ((threadIdx.x % WarpSize) / 16)]); }
    };
    struct MMA_C : public Matrix<float, 16, 8, sldC>
    {
        using Matrix<float, 16, 8, sldC>::operator[];
        using Matrix<float, 16, 8, sldC>::Matrix;
        struct Piece
        {
            union
            {
                float layout;
                u_int32_t reg;
            };
        };
        __device__ Piece &ref0() { return *(Piece *)&(*this)[(threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4)]; }
        __device__ Piece &ref1() { return *(Piece *)&(*this)[(threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4) + 1]; }
        __device__ Piece &ref2() { return *(Piece *)&(*this)[8 + (threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4)]; }
        __device__ Piece &ref3() { return *(Piece *)&(*this)[8 + (threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4) + 1]; }
    };
    struct MMA_C_Global : public Matrix<float, 16, 8, ldC>
    {
        using Matrix<float, 16, 8, ldC>::operator[];
        using Matrix<float, 16, 8, ldC>::Matrix;
        struct Piece
        {
            union
            {
                float layout;
                u_int32_t reg;
            };
        };
        __device__ Piece &ref0() { return *(Piece *)&(*this)[(threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4)]; }
        __device__ Piece &ref1() { return *(Piece *)&(*this)[(threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4) + 1]; }
        __device__ Piece &ref2() { return *(Piece *)&(*this)[8 + (threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4)]; }
        __device__ Piece &ref3() { return *(Piece *)&(*this)[8 + (threadIdx.x % WarpSize) / 4][2 * ((threadIdx.x % WarpSize) % 4) + 1]; }
    };
    __device__ static void MMA(typename MMA_A::Piece A0, typename MMA_A::Piece A1, typename MMA_A::Piece A2, typename MMA_A::Piece A3,
                               typename MMA_B::Piece B0, typename MMA_B::Piece B1,
                               typename MMA_C::Piece &C0, typename MMA_C::Piece &C1, typename MMA_C::Piece &C2, typename MMA_C::Piece &C3)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            : "=r"(C0.reg), "=r"(C1.reg), "=r"(C2.reg), "=r"(C3.reg)
            : "r"(A0.reg), "r"(A1.reg), "r"(A2.reg), "r"(A3.reg),
              "r"(B0.reg), "r"(B1.reg),
              "r"(C0.reg), "r"(C1.reg), "r"(C2.reg), "r"(C3.reg));
    }
    __device__ static void __copyt_global2smem_async(copy_t &dst, const copy_t &src)
    {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
            :
            : "r"((__uint32_t)__cvta_generic_to_shared(&dst)), "l"(&src));
    }
    template <size_t m, size_t n, size_t ld1, size_t ld2>
    __device__ static void __global2smem_async(Matrix<copy_t, m, n, ld1> &dst, const Matrix<copy_t, m, n, ld2> &src)
    {
#pragma unroll
        for (size_t i = 0; i < m * n / blockWarps; ++i)
        {
            auto &_dst = dst.sliceM<blockWarps / n>(i);
            const auto &_src = src.sliceM<blockWarps / n>(i);
            __copyt_global2smem_async(_dst[threadIdx.x / n][threadIdx.x % n], _src[threadIdx.x / n][threadIdx.x % n]);
        }
    }
    __device__ static void __async_wait()
    {
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();
    }
    __device__ static void __ldmatrixA(const MMA_A &A, typename MMA_A::Piece &A0, typename MMA_A::Piece &A1, typename MMA_A::Piece &A2, typename MMA_A::Piece &A3)
    {
        asm volatile("    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(A0.reg), "=r"(A1.reg), "=r"(A2.reg), "=r"(A3.reg)
            : "r"((__uint32_t)__cvta_generic_to_shared(A.getRow())));
    }
    __device__ static void __ldmatrixB2(const MMA_B2 &B, typename MMA_B::Piece &B0, typename MMA_B::Piece &B1, typename MMA_B::Piece &B0_, typename MMA_B::Piece &B1_)
    {
        asm volatile("    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(B0.reg), "=r"(B1.reg), "=r"(B0_.reg), "=r"(B1_.reg)
            : "r"((__uint32_t)__cvta_generic_to_shared(B.getRow())));
    }
    using WarpMatrix_A = Matrix<half, mWarpWidth, kWarpWidth, sldA>;
    using WarpMatrix_B = Matrix<half, kWarpWidth, nWarpWidth, sldB>;
    using WarpMatrix_C = Matrix<float, mWarpWidth / mWarpCores, nWarpWidth, sldC>;
    using SmemMatrix_A = Matrix<half, mBlockWidth, kBlockWidth, sldA>;
    using SmemMatrix_B = Matrix<half, kBlockWidth, nBlockWidth, sldB>;
    using SmemMatrix_C = Matrix<float, mBlockWidth / mWarpCores, nBlockWidth, sldC>;
    using GlobalMatrix_A = Matrix<half, M, K, ldA>;
    using GlobalMatrix_B = Matrix<half, K, N, ldB>;
    using GlobalMatrix_C = Matrix<float, M, N, ldC>;
    using BlockMatrix_A = Matrix<half, mBlockWidth, K, ldA>;
    using BlockMatrix_B = Matrix<half, K, nBlockWidth, ldB>;
    using BlockMatrix_C = Matrix<float, mBlockWidth, nBlockWidth, ldC>;
    using tBlockMatrix_A = Matrix<half, mBlockWidth, kBlockWidth, ldA>;
    using tBlockMatrix_B = Matrix<half, kBlockWidth, nBlockWidth, ldB>;
    struct Smem
    {
        struct AB
        {
            SmemMatrix_A a;
            SmemMatrix_B b;
        };
        union
        {
            AB ab[2];
            SmemMatrix_C c;
        };
    };
    using SCA = typename SmemMatrix_A::CopyT;
    using SCB = typename SmemMatrix_B::CopyT;
    using SCC = typename SmemMatrix_C::CopyT;
    static constexpr size_t blockWarps = mBlockWarps * nBlockWarps * WarpSize;
    static constexpr size_t iCopyA = SCA::mn / blockWarps;
    static constexpr size_t iCopyB = SCB::mn / blockWarps;
    static constexpr size_t iCopyC = SCC::mn / blockWarps;
    static constexpr size_t mCopyA = SCA::m / iCopyA;
    static constexpr size_t mCopyB = SCB::m / iCopyB;
    static constexpr size_t mCopyC = SCC::m / iCopyC;
    static constexpr size_t nCopyA = SCA::n;
    static constexpr size_t nCopyB = SCB::n;
    static constexpr size_t nCopyC = SCC::n;
    __device__ static void gemm(Smem *smem, const GlobalMatrix_A *globalA, const GlobalMatrix_B *globalB, GlobalMatrix_C *globalC)
    {
        const size_t nWarpIndex = (threadIdx.x / WarpSize) % nBlockWarps;
        const size_t mWarpIndex = (threadIdx.x / WarpSize) / nBlockWarps;
        const BlockMatrix_A &blockA = globalA->sliceM<mBlockWidth>(blockIdx.y);
        const BlockMatrix_B &blockB = globalB->sliceN<nBlockWidth>(blockIdx.x);
        auto &_blockC = globalC->sliceM<mBlockWidth>(blockIdx.y);
        BlockMatrix_C &blockC = _blockC.sliceN<nBlockWidth>(blockIdx.x);
        SCA &smemA0 = smem->ab[0].a.copyT();
        SCA &smemA1 = smem->ab[1].a.copyT();
        SCB &smemB0 = smem->ab[0].b.copyT();
        SCB &smemB1 = smem->ab[1].b.copyT();
        SCC &smemC = smem->c.copyT();
        WarpMatrix_A &warpmemA0 = smem->ab[0].a.sliceM<mWarpWidth>(mWarpIndex);
        WarpMatrix_A &warpmemA1 = smem->ab[1].a.sliceM<mWarpWidth>(mWarpIndex);
        WarpMatrix_B &warpmemB0 = smem->ab[0].b.sliceN<nWarpWidth>(nWarpIndex);
        WarpMatrix_B &warpmemB1 = smem->ab[1].b.sliceN<nWarpWidth>(nWarpIndex);
        auto &_warpmemC = smem->c.sliceM<mWarpWidth / mWarpCores>(mWarpIndex);
        WarpMatrix_C &warpmemC = _warpmemC.sliceN<nWarpWidth>(nWarpIndex);
        struct MMA_REG
        {
            typename MMA_A::Piece A0[mWarpCores];
            typename MMA_A::Piece A1[mWarpCores];
            typename MMA_A::Piece A2[mWarpCores];
            typename MMA_A::Piece A3[mWarpCores];
            typename MMA_B::Piece B0[nWarpCores];
            typename MMA_B::Piece B1[nWarpCores];
        };
        typename MMA_C::Piece C0[mWarpCores][nWarpCores];
        typename MMA_C::Piece C1[mWarpCores][nWarpCores];
        typename MMA_C::Piece C2[mWarpCores][nWarpCores];
        typename MMA_C::Piece C3[mWarpCores][nWarpCores];
        MMA_REG mmaReg[2];
        for (size_t i = 0; i < mWarpCores; ++i)
            for (size_t j = 0; j < nWarpCores; ++j)
            {
                C0[i][j].reg = 0;
                C1[i][j].reg = 0;
                C2[i][j].reg = 0;
                C3[i][j].reg = 0;
            }
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < mWarpCores; ++j)
            {
                mmaReg[i].A0[j].reg = 0;
                mmaReg[i].A1[j].reg = 0;
                mmaReg[i].A2[j].reg = 0;
                mmaReg[i].A3[j].reg = 0;
            }
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < nWarpCores; ++j)
            {
                mmaReg[i].B0[j].reg = 0;
                mmaReg[i].B1[j].reg = 0;
            }
        {
            const tBlockMatrix_A &tBlockA = blockA.sliceN<kBlockWidth>(0);
            __global2smem_async(smemA0, tBlockA.copyT());
            const tBlockMatrix_B &tBlockB = blockB.sliceM<kBlockWidth>(0);
            __global2smem_async(smemB0, tBlockB.copyT());
            __async_wait();
        }
        for (size_t k = 1; k < kBlocks - 1; k += 2)
        {
            {
                const tBlockMatrix_A &tBlockA = blockA.sliceN<kBlockWidth>(k);
                __global2smem_async(smemA1, tBlockA.copyT());
                const tBlockMatrix_B &tBlockB = blockB.sliceM<kBlockWidth>(k);
                __global2smem_async(smemB1, tBlockB.copyT());
                for (size_t kCoreIndex = 0; kCoreIndex < kWarpCores; kCoreIndex += 2)
                {
                    {
                        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                                MMA(mmaReg[0].A0[mCoreIndex], mmaReg[0].A1[mCoreIndex], mmaReg[0].A2[mCoreIndex], mmaReg[0].A3[mCoreIndex],
                                    mmaReg[0].B0[nCoreIndex], mmaReg[0].B1[nCoreIndex],
                                    C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);
                        const auto &_A = warpmemA0.sliceN<kCoreWidth>(kCoreIndex);
                        const auto &_B = warpmemB0.sliceM<kCoreWidth>(kCoreIndex);
                        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                        {
                            __ldmatrixA(*(const MMA_A *)&(_A.sliceM<mCoreWidth>(mCoreIndex)),
                                        mmaReg[1].A0[mCoreIndex],
                                        mmaReg[1].A1[mCoreIndex],
                                        mmaReg[1].A2[mCoreIndex],
                                        mmaReg[1].A3[mCoreIndex]);
                        }
                        for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores / 2; ++nCoreIndex)
                        {
                            __ldmatrixB2(*(const MMA_B2 *)&(_B.sliceN<nCoreWidth * 2>(nCoreIndex)),
                                        mmaReg[1].B0[nCoreIndex],
                                        mmaReg[1].B1[nCoreIndex],
                                        mmaReg[1].B0[nCoreIndex + 1],
                                        mmaReg[1].B1[nCoreIndex + 1]);
                        }
                    }
                    {
                        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                                MMA(mmaReg[1].A0[mCoreIndex], mmaReg[1].A1[mCoreIndex], mmaReg[1].A2[mCoreIndex], mmaReg[1].A3[mCoreIndex],
                                    mmaReg[1].B0[nCoreIndex], mmaReg[1].B1[nCoreIndex],
                                    C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);
                        const auto &_A = warpmemA0.sliceN<kCoreWidth>(kCoreIndex + 1);
                        const auto &_B = warpmemB0.sliceM<kCoreWidth>(kCoreIndex + 1);
                        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                        {
                            __ldmatrixA(*(const MMA_A *)&(_A.sliceM<mCoreWidth>(mCoreIndex)),
                                        mmaReg[0].A0[mCoreIndex],
                                        mmaReg[0].A1[mCoreIndex],
                                        mmaReg[0].A2[mCoreIndex],
                                        mmaReg[0].A3[mCoreIndex]);
                        }
                        for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores / 2; ++nCoreIndex)
                        {
                            __ldmatrixB2(*(const MMA_B2 *)&(_B.sliceN<nCoreWidth * 2>(nCoreIndex)),
                                        mmaReg[0].B0[nCoreIndex],
                                        mmaReg[0].B1[nCoreIndex],
                                        mmaReg[0].B0[nCoreIndex + 1],
                                        mmaReg[0].B1[nCoreIndex + 1]);
                        }
                    }
                }
                __async_wait();
            }
            {
                const tBlockMatrix_A &tBlockA = blockA.sliceN<kBlockWidth>(k + 1);
                __global2smem_async(smemA0, tBlockA.copyT());
                const tBlockMatrix_B &tBlockB = blockB.sliceM<kBlockWidth>(k + 1);
                __global2smem_async(smemB0, tBlockB.copyT());
                for (size_t kCoreIndex = 0; kCoreIndex < kWarpCores; kCoreIndex += 2)
                {
                    {
                        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                                MMA(mmaReg[0].A0[mCoreIndex], mmaReg[0].A1[mCoreIndex], mmaReg[0].A2[mCoreIndex], mmaReg[0].A3[mCoreIndex],
                                    mmaReg[0].B0[nCoreIndex], mmaReg[0].B1[nCoreIndex],
                                    C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);
                        const auto &_A = warpmemA1.sliceN<kCoreWidth>(kCoreIndex);
                        const auto &_B = warpmemB1.sliceM<kCoreWidth>(kCoreIndex);
                        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                        {
                            __ldmatrixA(*(const MMA_A *)&(_A.sliceM<mCoreWidth>(mCoreIndex)),
                                        mmaReg[1].A0[mCoreIndex],
                                        mmaReg[1].A1[mCoreIndex],
                                        mmaReg[1].A2[mCoreIndex],
                                        mmaReg[1].A3[mCoreIndex]);
                        }
                        for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores / 2; ++nCoreIndex)
                        {
                            __ldmatrixB2(*(const MMA_B2 *)&(_B.sliceN<nCoreWidth * 2>(nCoreIndex)),
                                        mmaReg[1].B0[nCoreIndex],
                                        mmaReg[1].B1[nCoreIndex],
                                        mmaReg[1].B0[nCoreIndex + 1],
                                        mmaReg[1].B1[nCoreIndex + 1]);
                        }
                    }
                    {
                        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                                MMA(mmaReg[1].A0[mCoreIndex], mmaReg[1].A1[mCoreIndex], mmaReg[1].A2[mCoreIndex], mmaReg[1].A3[mCoreIndex],
                                    mmaReg[1].B0[nCoreIndex], mmaReg[1].B1[nCoreIndex],
                                    C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);
                        const auto &_A = warpmemA1.sliceN<kCoreWidth>(kCoreIndex + 1);
                        const auto &_B = warpmemB1.sliceM<kCoreWidth>(kCoreIndex + 1);
                        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                        {
                            __ldmatrixA(*(const MMA_A *)&(_A.sliceM<mCoreWidth>(mCoreIndex)),
                                        mmaReg[0].A0[mCoreIndex],
                                        mmaReg[0].A1[mCoreIndex],
                                        mmaReg[0].A2[mCoreIndex],
                                        mmaReg[0].A3[mCoreIndex]);
                        }
                        for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores / 2; ++nCoreIndex)
                        {
                            __ldmatrixB2(*(const MMA_B2 *)&(_B.sliceN<nCoreWidth * 2>(nCoreIndex)),
                                        mmaReg[0].B0[nCoreIndex],
                                        mmaReg[0].B1[nCoreIndex],
                                        mmaReg[0].B0[nCoreIndex + 1],
                                        mmaReg[0].B1[nCoreIndex + 1]);
                        }
                    }
                }
                __async_wait();
            }
        }
        {
            const tBlockMatrix_A &tBlockA = blockA.sliceN<kBlockWidth>(kBlocks - 1);
            __global2smem_async(smemA1, tBlockA.copyT());
            const tBlockMatrix_B &tBlockB = blockB.sliceM<kBlockWidth>(kBlocks - 1);
            __global2smem_async(smemB1, tBlockB.copyT());
            for (size_t kCoreIndex = 0; kCoreIndex < kWarpCores; kCoreIndex += 2)
            {
                {
                    for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                        for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                            MMA(mmaReg[0].A0[mCoreIndex], mmaReg[0].A1[mCoreIndex], mmaReg[0].A2[mCoreIndex], mmaReg[0].A3[mCoreIndex],
                                mmaReg[0].B0[nCoreIndex], mmaReg[0].B1[nCoreIndex],
                                C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);
                    const auto &_A = warpmemA0.sliceN<kCoreWidth>(kCoreIndex);
                    const auto &_B = warpmemB0.sliceM<kCoreWidth>(kCoreIndex);
                    for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                    {
                        __ldmatrixA(*(const MMA_A *)&(_A.sliceM<mCoreWidth>(mCoreIndex)),
                                    mmaReg[1].A0[mCoreIndex],
                                    mmaReg[1].A1[mCoreIndex],
                                    mmaReg[1].A2[mCoreIndex],
                                    mmaReg[1].A3[mCoreIndex]);
                    }
                    for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores / 2; ++nCoreIndex)
                    {
                        __ldmatrixB2(*(const MMA_B2 *)&(_B.sliceN<nCoreWidth * 2>(nCoreIndex)),
                                    mmaReg[1].B0[nCoreIndex],
                                    mmaReg[1].B1[nCoreIndex],
                                    mmaReg[1].B0[nCoreIndex + 1],
                                    mmaReg[1].B1[nCoreIndex + 1]);
                    }
                }
                {
                    for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                        for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                            MMA(mmaReg[1].A0[mCoreIndex], mmaReg[1].A1[mCoreIndex], mmaReg[1].A2[mCoreIndex], mmaReg[1].A3[mCoreIndex],
                                mmaReg[1].B0[nCoreIndex], mmaReg[1].B1[nCoreIndex],
                                C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);
                    const auto &_A = warpmemA0.sliceN<kCoreWidth>(kCoreIndex + 1);
                    const auto &_B = warpmemB0.sliceM<kCoreWidth>(kCoreIndex + 1);
                    for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                    {
                        __ldmatrixA(*(const MMA_A *)&(_A.sliceM<mCoreWidth>(mCoreIndex)),
                                    mmaReg[0].A0[mCoreIndex],
                                    mmaReg[0].A1[mCoreIndex],
                                    mmaReg[0].A2[mCoreIndex],
                                    mmaReg[0].A3[mCoreIndex]);
                    }
                    for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores / 2; ++nCoreIndex)
                    {
                        __ldmatrixB2(*(const MMA_B2 *)&(_B.sliceN<nCoreWidth * 2>(nCoreIndex)),
                                    mmaReg[0].B0[nCoreIndex],
                                    mmaReg[0].B1[nCoreIndex],
                                    mmaReg[0].B0[nCoreIndex + 1],
                                    mmaReg[0].B1[nCoreIndex + 1]);
                    }
                }
            }
            __async_wait();
        }
        {
            for (size_t kCoreIndex = 0; kCoreIndex < kWarpCores; kCoreIndex += 2)
            {
                {
                    for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                        for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                            MMA(mmaReg[0].A0[mCoreIndex], mmaReg[0].A1[mCoreIndex], mmaReg[0].A2[mCoreIndex], mmaReg[0].A3[mCoreIndex],
                                mmaReg[0].B0[nCoreIndex], mmaReg[0].B1[nCoreIndex],
                                C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);
                    const auto &_A = warpmemA1.sliceN<kCoreWidth>(kCoreIndex);
                    const auto &_B = warpmemB1.sliceM<kCoreWidth>(kCoreIndex);
                    for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                    {
                        __ldmatrixA(*(const MMA_A *)&(_A.sliceM<mCoreWidth>(mCoreIndex)),
                                    mmaReg[1].A0[mCoreIndex],
                                    mmaReg[1].A1[mCoreIndex],
                                    mmaReg[1].A2[mCoreIndex],
                                    mmaReg[1].A3[mCoreIndex]);
                    }
                    for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores / 2; ++nCoreIndex)
                    {
                        __ldmatrixB2(*(const MMA_B2 *)&(_B.sliceN<nCoreWidth * 2>(nCoreIndex)),
                                    mmaReg[1].B0[nCoreIndex],
                                    mmaReg[1].B1[nCoreIndex],
                                    mmaReg[1].B0[nCoreIndex + 1],
                                    mmaReg[1].B1[nCoreIndex + 1]);
                    }
                }
                {
                    for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                        for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                            MMA(mmaReg[1].A0[mCoreIndex], mmaReg[1].A1[mCoreIndex], mmaReg[1].A2[mCoreIndex], mmaReg[1].A3[mCoreIndex],
                                mmaReg[1].B0[nCoreIndex], mmaReg[1].B1[nCoreIndex],
                                C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);
                    const auto &_A = warpmemA1.sliceN<kCoreWidth>(kCoreIndex + 1);
                    const auto &_B = warpmemB1.sliceM<kCoreWidth>(kCoreIndex + 1);
                    for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
                    {
                        const MMA_A &A = *(const MMA_A *)&(_A.sliceM<mCoreWidth>(mCoreIndex));
                        mmaReg[0].A0[mCoreIndex] = A.get0();
                        mmaReg[0].A1[mCoreIndex] = A.get1();
                        mmaReg[0].A2[mCoreIndex] = A.get2();
                        mmaReg[0].A3[mCoreIndex] = A.get3();
                    }
                    for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                    {
                        const MMA_B &B = *(const MMA_B *)&(_B.sliceN<nCoreWidth>(nCoreIndex));
                        mmaReg[0].B0[nCoreIndex] = B.get0();
                        mmaReg[0].B1[nCoreIndex] = B.get1();
                    }
                }
            }
        }
        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                MMA(mmaReg[1].A0[mCoreIndex], mmaReg[1].A1[mCoreIndex], mmaReg[1].A2[mCoreIndex], mmaReg[1].A3[mCoreIndex],
                    mmaReg[1].B0[nCoreIndex], mmaReg[1].B1[nCoreIndex],
                    C0[mCoreIndex][nCoreIndex], C1[mCoreIndex][nCoreIndex], C2[mCoreIndex][nCoreIndex], C3[mCoreIndex][nCoreIndex]);

        auto &___C = blockC.sliceM<mWarpWidth>(mWarpIndex);
        auto &__C = ___C.sliceN<nWarpWidth>(nWarpIndex);
        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
        {
            auto &_C = __C.sliceM<mCoreWidth>(mCoreIndex);
            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; nCoreIndex += 4)
            {
                MMA_C_Global &C = *(MMA_C_Global *)&(_C.sliceN<nCoreWidth>(nCoreIndex));
                C.ref0().layout = C0[mCoreIndex][nCoreIndex].layout;
                C.ref1().layout = C1[mCoreIndex][nCoreIndex].layout;
                C.ref2().layout = C2[mCoreIndex][nCoreIndex].layout;
                C.ref3().layout = C3[mCoreIndex][nCoreIndex].layout;
            }
        }
    }
};
template <size_t M, size_t K, size_t N>
__global__ void mma_gemm(const half *A, const half *B, float *C)
{
    __launch_bounds__(MMA_GEMM<M, K, N>::mBlockWarps * MMA_GEMM<M, K, N>::nBlockWarps * MMA_GEMM<M, K, N>::WarpSize);
    extern __shared__ typename MMA_GEMM<M, K, N>::Smem smem[];
    MMA_GEMM<M, K, N>::gemm(smem, (const typename MMA_GEMM<M, K, N>::GlobalMatrix_A *)A, (const typename MMA_GEMM<M, K, N>::GlobalMatrix_B *)B, (typename MMA_GEMM<M, K, N>::GlobalMatrix_C *)C);
}