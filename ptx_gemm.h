#pragma once
#include <stdio.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef COMPILING
template <size_t M, size_t K, size_t N>
struct PTX_GEMM
{
#else
static constexpr size_t M = 8192;
static constexpr size_t K = 8192;
static constexpr size_t N = 8192;
#endif

    static constexpr size_t mBlockWarps = 2;
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
    static constexpr size_t sldA = sizeof(half) * (kBlockWidth);
    static constexpr size_t sldB = sizeof(half) * (nBlockWidth);
    static constexpr size_t sldC = sizeof(float) * (nBlockWidth + 8);
    using copy_t = int4;
    using reg_t = __uint32_t;
    static constexpr size_t CopyRate = sizeof(copy_t) / sizeof(half);
    static constexpr size_t WarpSize = 32;
    static constexpr size_t BlockThreads = mBlockWarps * kBlockWarps * nBlockWarps * WarpSize;

    __device__ static void cp_async(void *dst, const void *src)
    {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                     :
                     : "r"((__uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
    }
    __device__ static size_t getY() { return (blockIdx.y * 4 % 64) + blockIdx.x / 8; }
    __device__ static size_t getX() { return (blockIdx.x % 8) + 8 * (blockIdx.y / 16); }
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
        __device__ const Row &operator[](size_t i) const { return buffer[i]; }

        template <typename slcT, typename = std::enable_if_t<m % slcT::m == 0 && n % slcT::n == 0 && ld == slcT::ld && std::is_same<Type, typename slcT::Type>::value>>
        __device__ slcT &slice(size_t im, size_t in) { return *(slcT *)&(buffer[slcT::m * im][slcT::n * in]); }
        template <typename slcT, typename = std::enable_if_t<m % slcT::m == 0 && n % slcT::n == 0 && ld == slcT::ld && std::is_same<Type, typename slcT::Type>::value>>
        __device__ const slcT &slice(size_t im, size_t in) const { return *(const slcT *)&(buffer[slcT::m * im][slcT::n * in]); }

        __device__ Matrix<T, m, n, ld> &move(size_t im = 0, size_t in = 0) { return *(Matrix<T, m, n, ld> *)&((*this)[im][in]); }
        __device__ const Matrix<T, m, n, ld> &move(size_t im = 0, size_t in = 0) const { return *(const Matrix<T, m, n, ld> *)&((*this)[im][in]); }

        using CopyT = Matrix<copy_t, m, n * sizeof(T) / sizeof(copy_t), ld>;
        template <typename cvtT = CopyT, typename = std::enable_if_t<cvtT::m * cvtT::ld == m * ld>>
        __device__ cvtT &convert() { return *(cvtT *)this; }
        template <typename cvtT = CopyT, typename = std::enable_if_t<cvtT::m * cvtT::ld == m * ld>>
        __device__ const cvtT &convert() const { return *(const cvtT *)this; }
    };
    template <typename BaseMatrix, size_t m, size_t n>
    using CombineMatrix = Matrix<typename BaseMatrix::Type, BaseMatrix::m * m, BaseMatrix::n * n, BaseMatrix::ld>;

    struct KBlockCopyMatrixA_Global : public Matrix<copy_t, mBlockWidth, kBlockWidth * sizeof(half) / sizeof(copy_t), ldA>
    {
        using Base = Matrix<copy_t, mBlockWidth, kBlockWidth * sizeof(half) / sizeof(copy_t), ldA>;
        using Base::operator[];
        using Piece = Matrix<copy_t, BlockThreads * sizeof(copy_t) / (kBlockWidth * sizeof(half)), kBlockWidth * sizeof(half) / sizeof(copy_t), ldA>;
        static constexpr size_t mRange = Base::m / Piece::m;
        static constexpr size_t nRange = Base::n / Piece::n;
        template <size_t im>
        __device__ const copy_t *get() const { return &(Base::template slice<Piece>(im, 0)[0][0]); }
    };
    struct KBlockCopyMatrixB_Global : public Matrix<copy_t, kBlockWidth, nBlockWidth * sizeof(half) / sizeof(copy_t), ldB>
    {
        using Base = Matrix<copy_t, kBlockWidth, nBlockWidth * sizeof(half) / sizeof(copy_t), ldB>;
        using Base::operator[];
        using Piece = Matrix<copy_t, BlockThreads * sizeof(copy_t) / (nBlockWidth * sizeof(half)), nBlockWidth * sizeof(half) / sizeof(copy_t), ldA>;
        static constexpr size_t mRange = Base::m / Piece::m;
        static constexpr size_t nRange = Base::n / Piece::n;
        template <size_t im>
        __device__ const copy_t *get() const { return &(Base::template slice<Piece>(im, 0)[0][0]); }
    };
    struct BlockCopyMatrixA_Global : CombineMatrix<KBlockCopyMatrixA_Global, 1, kBlocks>
    {
        using Base = CombineMatrix<KBlockCopyMatrixA_Global, 1, kBlocks>;
        using Base::operator[];
        __device__ const KBlockCopyMatrixA_Global &getKBlock(size_t ik) const { return Base::template slice<KBlockCopyMatrixA_Global>(0, ik); }
    };
    struct BlockCopyMatrixB_Global : CombineMatrix<KBlockCopyMatrixB_Global, kBlocks, 1>
    {
        using Base = CombineMatrix<KBlockCopyMatrixB_Global, kBlocks, 1>;
        using Base::operator[];
        __device__ const KBlockCopyMatrixB_Global &getKBlock(size_t ik) const { return Base::template slice<KBlockCopyMatrixB_Global>(ik, 0); }
    };
    struct WarpMatrixC_Global : public Matrix<float, mWarpWidth, nWarpWidth, ldC>
    {
        using Base = Matrix<float, mWarpWidth, nWarpWidth, ldC>;
        using Base::operator[];
        using Piece = Matrix<float, mCoreWidth, nCoreWidth, ldC>;
        __device__ copy_t *get(size_t im, size_t in) { return (copy_t *)&(Base::template slice<Piece>(im, in)[0][0]); }
    };
    struct MatrixA_Global : public Matrix<half, M, K, ldA>
    {
        using Base = Matrix<half, M, K, ldA>;
        using Base::operator[];
        __device__ const BlockCopyMatrixA_Global &getCopyBlock() const
        {
            return Base::template convert()
                .template slice<BlockCopyMatrixA_Global>(getY(), 0)
                .move(threadIdx.x / (kBlockWidth * sizeof(half) / sizeof(copy_t)), threadIdx.x % (kBlockWidth * sizeof(half) / sizeof(copy_t)))
                .template convert<BlockCopyMatrixA_Global>();
        }
    };
    struct MatrixB_Global : public Matrix<half, K, N, ldB>
    {
        using Base = Matrix<half, K, N, ldB>;
        using Base::operator[];
        __device__ const BlockCopyMatrixB_Global &getCopyBlock() const
        {
            return Base::template convert()
                .template slice<BlockCopyMatrixB_Global>(0, getX())
                .move(threadIdx.x / (nBlockWidth * sizeof(half) / sizeof(copy_t)), threadIdx.x % (nBlockWidth * sizeof(half) / sizeof(copy_t)))
                .template convert<BlockCopyMatrixB_Global>();
        }
    };
    struct MatrixC_Global : public Matrix<float, M, N, ldC>
    {
        using Base = Matrix<float, M, N, ldC>;
        using Base::operator[];
        __device__ WarpMatrixC_Global &getWarpBlock()
        {
            return Base::template slice<Matrix<float, mBlockWidth, nBlockWidth, ldC>>(getY(), getX())
                .template slice<Matrix<float, mWarpWidth, nWarpWidth, ldC>>((threadIdx.x / warpSize) / nBlockWarps, (threadIdx.x / warpSize) % nBlockWarps)
                .template convert<WarpMatrixC_Global>()
                .move((threadIdx.x % warpSize) / 16, (threadIdx.x % 16) * 4)
                .template convert<WarpMatrixC_Global>();
        }
    };

    struct SmemCopyMatrixA
        : public Matrix<copy_t, mBlockWidth, kBlockWidth * sizeof(half) / sizeof(copy_t), sldA>
    {
        using Base = Matrix<copy_t, mBlockWidth, kBlockWidth * sizeof(half) / sizeof(copy_t), sldA>;
        using Base::operator[];
        using Piece = Matrix<copy_t, BlockThreads * sizeof(copy_t) / (kBlockWidth * sizeof(half)), kBlockWidth * sizeof(half) / sizeof(copy_t), sldA>;
        static constexpr size_t mRange = Base::m / Piece::m;
        static constexpr size_t nRange = Base::n / Piece::n;
        template <size_t im>
        __device__ copy_t *get() { return &(Base::template slice<Piece>(im, 0)[0][0]); }
    };
    struct SmemCopyMatrixB
        : public Matrix<copy_t, kBlockWidth, nBlockWidth * sizeof(half) / sizeof(copy_t), sldB>
    {
        using Base = Matrix<copy_t, kBlockWidth, nBlockWidth * sizeof(half) / sizeof(copy_t), sldB>;
        using Base::operator[];
        using Piece = Matrix<copy_t, BlockThreads * sizeof(copy_t) / (nBlockWidth * sizeof(half)), nBlockWidth * sizeof(half) / sizeof(copy_t), sldB>;
        static constexpr size_t mRange = Base::m / Piece::m;
        static constexpr size_t nRange = Base::n / Piece::n;
        template <size_t im>
        __device__ copy_t *get() { return &(Base::template slice<Piece>(im, 0)[0][0]); }
    };

    struct CoreMatrixA : public Matrix<copy_t, mCoreWidth, kCoreWidth * sizeof(half) / sizeof(copy_t), sldA>
    {
        using Base = Matrix<copy_t, mCoreWidth, kCoreWidth * sizeof(half) / sizeof(copy_t), sldA>;
        using Base::operator[];
    };
    struct CoreMatrixB : public Matrix<copy_t, kCoreWidth, nCoreWidth * sizeof(half) / sizeof(copy_t), sldB>
    {
        using Base = Matrix<copy_t, kCoreWidth, nCoreWidth * sizeof(half) / sizeof(copy_t), sldB>;
        using Base::operator[];
    };
    struct CoreMatrixC : public Matrix<reg_t, mCoreWidth, nCoreWidth, sldC>
    {
        using Base = Matrix<reg_t, mCoreWidth, nCoreWidth, sldC>;
        using Base::operator[];
    };
    struct WarpMatrixA : public CombineMatrix<CoreMatrixA, mWarpCores, kWarpCores>
    {
        using Base = CombineMatrix<CoreMatrixA, mWarpCores, kWarpCores>;
        using Base::operator[];
        __device__ const copy_t *get(size_t i, size_t j)
        {
            return &((*this)[i * 16 + (threadIdx.x % 16)][(2 * j) ^ ((threadIdx.x % 32) / 16) ^ (threadIdx.x % 8)]);
        }
    };
    struct WarpMatrixB : public CombineMatrix<CoreMatrixB, kWarpCores, nWarpCores>
    {
        using Base = CombineMatrix<CoreMatrixB, kWarpCores, nWarpCores>;
        using Base::operator[];
        __device__ const copy_t *get(size_t i, size_t j)
        {
            return &((*this)[i * 16 + (threadIdx.x % 16)][(2 * j) ^ ((threadIdx.x % 32) / 16) ^ (threadIdx.x % 8)]);
        }
    };
    struct WarpMatrixC : public CombineMatrix<CoreMatrixC, 1, nWarpCores>
    {
        using Base = CombineMatrix<CoreMatrixC, 1, nWarpCores>;
        using Base::operator[];
    };

    struct SmemMatrixA : public Matrix<half, mBlockWidth, kBlockWidth, sldA>
    {
        using Base = Matrix<half, mBlockWidth, kBlockWidth, sldA>;
        using Base::operator[];
        __device__ SmemCopyMatrixA &getCopyBlock()
        {
            return Base::template convert<SmemCopyMatrixA>()
                .move(threadIdx.x / (kBlockWidth * sizeof(half) / sizeof(copy_t)), (threadIdx.x % (kBlockWidth * sizeof(half) / sizeof(copy_t))) ^ ((threadIdx.x / (kBlockWidth * sizeof(half) / sizeof(copy_t))) % 8))
                .template convert<SmemCopyMatrixA>();
        }
        __device__ WarpMatrixA &getWarpBlock()
        {
            return Base::template convert().template slice<WarpMatrixA>((threadIdx.x / warpSize) / nBlockWarps, 0);
        }
    };
    struct SmemMatrixB : public Matrix<half, kBlockWidth, nBlockWidth, sldB>
    {
        using Base = Matrix<half, kBlockWidth, nBlockWidth, sldB>;
        using Base::operator[];
        __device__ SmemCopyMatrixB &getCopyBlock()
        {
            return Base::template convert<SmemCopyMatrixB>()
                .move(threadIdx.x / (nBlockWidth * sizeof(half) / sizeof(copy_t)), (threadIdx.x % (nBlockWidth * sizeof(half) / sizeof(copy_t))) ^ ((threadIdx.x / (nBlockWidth * sizeof(half) / sizeof(copy_t))) % 8))
                .template convert<SmemCopyMatrixB>();
        }
        __device__ WarpMatrixB &getWarpBlock()
        {
            return Base::template convert().template slice<WarpMatrixB>(0, (threadIdx.x / warpSize) % nBlockWarps);
        }
    };
    struct SmemMatrixC : public CombineMatrix<WarpMatrixC, mBlockWarps, nBlockWarps>
    {
        using Base = CombineMatrix<WarpMatrixC, mBlockWarps, nBlockWarps>;
        using Base::operator[];
        __device__ WarpMatrixC &getWarpBlock() { return Base::template slice<WarpMatrixC>((threadIdx.x / warpSize) / nBlockWarps, (threadIdx.x / warpSize) % nBlockWarps); }
    };
    struct Smem
    {
        struct AB
        {
            SmemMatrixA a[3];
            SmemMatrixB b[3];
        };
        union
        {
            AB ab;
            SmemMatrixC c;
        };
    };
    __device__ static SmemCopyMatrixA &__selectBuffer(SmemCopyMatrixA &p, size_t i) { return *(SmemCopyMatrixA *)(i + (SmemMatrixA *)&p); }
    __device__ static SmemCopyMatrixB &__selectBuffer(SmemCopyMatrixB &p, size_t i) { return *(SmemCopyMatrixB *)(i + (SmemMatrixB *)&p); }
    __device__ static WarpMatrixA &__selectBuffer(WarpMatrixA &p, size_t i) { return *(WarpMatrixA *)(i + (SmemMatrixA *)&p); }
    __device__ static WarpMatrixB &__selectBuffer(WarpMatrixB &p, size_t i) { return *(WarpMatrixB *)(i + (SmemMatrixB *)&p); }
    __device__ static void gemm(Smem &smem, const MatrixA_Global &GlobalA, const MatrixB_Global &GlobalB, MatrixC_Global &GlobalC)
    {
        const BlockCopyMatrixA_Global &blockA = GlobalA.getCopyBlock();
        const BlockCopyMatrixB_Global &blockB = GlobalB.getCopyBlock();
        WarpMatrixC_Global &warpC_global = GlobalC.getWarpBlock();
        WarpMatrixC &warpC = smem.c.getWarpBlock();
        reg_t A[2][mWarpCores][4];
        reg_t B[2][nWarpCores][2];
        reg_t C[mWarpCores][nWarpCores][4];
#pragma unroll
        for (size_t i = 0; i < mWarpCores; ++i)
#pragma unroll
            for (size_t j = 0; j < nWarpCores; ++j)
#pragma unroll
                for (size_t k = 0; k < kWarpCores; ++k)
                    C[i][j][k] = 0;
        SmemCopyMatrixA &smemA3 = smem.ab.a[0].getCopyBlock();
        SmemCopyMatrixB &smemB3 = smem.ab.b[0].getCopyBlock();
        WarpMatrixA &warpA3 = smem.ab.a[0].getWarpBlock();
        WarpMatrixB &warpB3 = smem.ab.b[0].getWarpBlock();
#pragma unroll
        for (size_t k = 0; k < 2; ++k)
        {
            SmemCopyMatrixA &smemA = __selectBuffer(smemA3, k);
            SmemCopyMatrixB &smemB = __selectBuffer(smemB3, k);
            const KBlockCopyMatrixA_Global &kBlockA = blockA.getKBlock(k);
            const KBlockCopyMatrixB_Global &kBlockB = blockB.getKBlock(k);
            cp_async(smemA.template get<0>(), kBlockA.template get<0>());
            cp_async(smemA.template get<1>(), kBlockA.template get<1>());
            cp_async(smemA.template get<2>(), kBlockA.template get<2>());
            cp_async(smemA.template get<3>(), kBlockA.template get<3>());
            cp_async(smemB.template get<0>(), kBlockB.template get<0>());
            cp_async(smemB.template get<1>(), kBlockB.template get<1>());
            cp_async(smemB.template get<2>(), kBlockB.template get<2>());
            cp_async(smemB.template get<3>(), kBlockB.template get<3>());
            cp_async(smemB.template get<4>(), kBlockB.template get<4>());
            cp_async(smemB.template get<5>(), kBlockB.template get<5>());
            cp_async(smemB.template get<6>(), kBlockB.template get<6>());
            cp_async(smemB.template get<7>(), kBlockB.template get<7>());
            asm volatile("cp.async.commit_group;\n");
        }
        {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
            __syncthreads();
            WarpMatrixA &warpA = __selectBuffer(warpA3, 0);
            WarpMatrixB &warpB = __selectBuffer(warpB3, 0);
            SmemCopyMatrixA &smemA = __selectBuffer(smemA3, 2);
            SmemCopyMatrixB &smemB = __selectBuffer(smemB3, 2);
            const KBlockCopyMatrixA_Global &kBlockA = blockA.getKBlock(2);
            const KBlockCopyMatrixB_Global &kBlockB = blockB.getKBlock(2);
#pragma unroll
            for (size_t i = 0; i < 4; ++i)
            {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[0][i][0]), "=r"(A[0][i][1]), "=r"(A[0][i][2]), "=r"(A[0][i][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(i, 0))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[0][i * 2][0]), "=r"(B[0][i * 2][1]), "=r"(B[0][i * 2 + 1][0]), "=r"(B[0][i * 2 + 1][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(0, i))));
            }
            {
                size_t kCoreIndex = 1;
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                cp_async(smemA.template get<0>(), kBlockA.template get<0>());
                cp_async(smemA.template get<1>(), kBlockA.template get<1>());
                cp_async(smemA.template get<2>(), kBlockA.template get<2>());
                cp_async(smemA.template get<3>(), kBlockA.template get<3>());
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
            }
            {
                size_t kCoreIndex = 2;
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                cp_async(smemB.template get<0>(), kBlockB.template get<0>());
                cp_async(smemB.template get<1>(), kBlockB.template get<1>());
                cp_async(smemB.template get<2>(), kBlockB.template get<2>());
                cp_async(smemB.template get<3>(), kBlockB.template get<3>());
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
            }
            {
                size_t kCoreIndex = 3;
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                cp_async(smemB.template get<4>(), kBlockB.template get<4>());
                cp_async(smemB.template get<5>(), kBlockB.template get<5>());
                cp_async(smemB.template get<6>(), kBlockB.template get<6>());
                cp_async(smemB.template get<7>(), kBlockB.template get<7>());
                asm volatile("cp.async.commit_group;\n");
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
            }
        }
        for (size_t k = 1; k < kBlocks - 4; k += 3)
        {
            {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
                __syncthreads();
                WarpMatrixA &warpA = __selectBuffer(warpA3, 1);
                WarpMatrixB &warpB = __selectBuffer(warpB3, 1);
                SmemCopyMatrixA &smemA = __selectBuffer(smemA3, 0);
                SmemCopyMatrixB &smemB = __selectBuffer(smemB3, 0);
                const KBlockCopyMatrixA_Global &kBlockA = blockA.getKBlock(k + 2);
                const KBlockCopyMatrixB_Global &kBlockB = blockB.getKBlock(k + 2);
                {
                    size_t kCoreIndex = 0;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 1;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemA.template get<0>(), kBlockA.template get<0>());
                    cp_async(smemA.template get<1>(), kBlockA.template get<1>());
                    cp_async(smemA.template get<2>(), kBlockA.template get<2>());
                    cp_async(smemA.template get<3>(), kBlockA.template get<3>());
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 2;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemB.template get<0>(), kBlockB.template get<0>());
                    cp_async(smemB.template get<1>(), kBlockB.template get<1>());
                    cp_async(smemB.template get<2>(), kBlockB.template get<2>());
                    cp_async(smemB.template get<3>(), kBlockB.template get<3>());
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 3;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemB.template get<4>(), kBlockB.template get<4>());
                    cp_async(smemB.template get<5>(), kBlockB.template get<5>());
                    cp_async(smemB.template get<6>(), kBlockB.template get<6>());
                    cp_async(smemB.template get<7>(), kBlockB.template get<7>());
                    asm volatile("cp.async.commit_group;\n");
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
            }
            {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
                __syncthreads();
                WarpMatrixA &warpA = __selectBuffer(warpA3, 2);
                WarpMatrixB &warpB = __selectBuffer(warpB3, 2);
                SmemCopyMatrixA &smemA = __selectBuffer(smemA3, 1);
                SmemCopyMatrixB &smemB = __selectBuffer(smemB3, 1);
                const KBlockCopyMatrixA_Global &kBlockA = blockA.getKBlock(k + 3);
                const KBlockCopyMatrixB_Global &kBlockB = blockB.getKBlock(k + 3);
                {
                    size_t kCoreIndex = 0;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 1;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemA.template get<0>(), kBlockA.template get<0>());
                    cp_async(smemA.template get<1>(), kBlockA.template get<1>());
                    cp_async(smemA.template get<2>(), kBlockA.template get<2>());
                    cp_async(smemA.template get<3>(), kBlockA.template get<3>());
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 2;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemB.template get<0>(), kBlockB.template get<0>());
                    cp_async(smemB.template get<1>(), kBlockB.template get<1>());
                    cp_async(smemB.template get<2>(), kBlockB.template get<2>());
                    cp_async(smemB.template get<3>(), kBlockB.template get<3>());
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 3;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemB.template get<4>(), kBlockB.template get<4>());
                    cp_async(smemB.template get<5>(), kBlockB.template get<5>());
                    cp_async(smemB.template get<6>(), kBlockB.template get<6>());
                    cp_async(smemB.template get<7>(), kBlockB.template get<7>());
                    asm volatile("cp.async.commit_group;\n");
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
            }
            {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
                __syncthreads();
                WarpMatrixA &warpA = __selectBuffer(warpA3, 0);
                WarpMatrixB &warpB = __selectBuffer(warpB3, 0);
                SmemCopyMatrixA &smemA = __selectBuffer(smemA3, 2);
                SmemCopyMatrixB &smemB = __selectBuffer(smemB3, 2);
                const KBlockCopyMatrixA_Global &kBlockA = blockA.getKBlock(k + 4);
                const KBlockCopyMatrixB_Global &kBlockB = blockB.getKBlock(k + 4);
                {
                    size_t kCoreIndex = 0;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 1;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemA.template get<0>(), kBlockA.template get<0>());
                    cp_async(smemA.template get<1>(), kBlockA.template get<1>());
                    cp_async(smemA.template get<2>(), kBlockA.template get<2>());
                    cp_async(smemA.template get<3>(), kBlockA.template get<3>());
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 2;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemB.template get<0>(), kBlockB.template get<0>());
                    cp_async(smemB.template get<1>(), kBlockB.template get<1>());
                    cp_async(smemB.template get<2>(), kBlockB.template get<2>());
                    cp_async(smemB.template get<3>(), kBlockB.template get<3>());
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
                {
                    size_t kCoreIndex = 3;
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                    cp_async(smemB.template get<4>(), kBlockB.template get<4>());
                    cp_async(smemB.template get<5>(), kBlockB.template get<5>());
                    cp_async(smemB.template get<6>(), kBlockB.template get<6>());
                    cp_async(smemB.template get<7>(), kBlockB.template get<7>());
                    asm volatile("cp.async.commit_group;\n");
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
                }
            }
        }
#pragma unroll
        for (size_t k = kBlocks - 2 - kBlocks % 3; k < kBlocks - 2; ++k)
        {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
            __syncthreads();
            WarpMatrixA &warpA = __selectBuffer(warpA3, k % 3);
            WarpMatrixB &warpB = __selectBuffer(warpB3, k % 3);
            SmemCopyMatrixA &smemA = __selectBuffer(smemA3, (k + 2) % 3);
            SmemCopyMatrixB &smemB = __selectBuffer(smemB3, (k + 2) % 3);
            const KBlockCopyMatrixA_Global &kBlockA = blockA.getKBlock(k + 2);
            const KBlockCopyMatrixB_Global &kBlockB = blockB.getKBlock(k + 2);
            {
                size_t kCoreIndex = 0;
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
            }
            {
                size_t kCoreIndex = 1;
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                cp_async(smemA.template get<0>(), kBlockA.template get<0>());
                cp_async(smemA.template get<1>(), kBlockA.template get<1>());
                cp_async(smemA.template get<2>(), kBlockA.template get<2>());
                cp_async(smemA.template get<3>(), kBlockA.template get<3>());
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
            }
            {
                size_t kCoreIndex = 2;
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                cp_async(smemB.template get<0>(), kBlockB.template get<0>());
                cp_async(smemB.template get<1>(), kBlockB.template get<1>());
                cp_async(smemB.template get<2>(), kBlockB.template get<2>());
                cp_async(smemB.template get<3>(), kBlockB.template get<3>());
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
            }
            {
                size_t kCoreIndex = 3;
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[0][j][0]), "=r"(C[0][j][1]), "=r"(C[0][j][2]), "=r"(C[0][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][0][0]), "r"(A[(kCoreIndex + 1) % 2][0][1]), "r"(A[(kCoreIndex + 1) % 2][0][2]), "r"(A[(kCoreIndex + 1) % 2][0][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[0][j][0]), "r"(C[0][j][1]), "r"(C[0][j][2]), "r"(C[0][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][0][0]), "=r"(A[kCoreIndex % 2][0][1]), "=r"(A[kCoreIndex % 2][0][2]), "=r"(A[kCoreIndex % 2][0][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(0, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][0][0]), "=r"(B[kCoreIndex % 2][0][1]), "=r"(B[kCoreIndex % 2][1][0]), "=r"(B[kCoreIndex % 2][1][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 0))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[1][j][0]), "=r"(C[1][j][1]), "=r"(C[1][j][2]), "=r"(C[1][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][1][0]), "r"(A[(kCoreIndex + 1) % 2][1][1]), "r"(A[(kCoreIndex + 1) % 2][1][2]), "r"(A[(kCoreIndex + 1) % 2][1][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[1][j][0]), "r"(C[1][j][1]), "r"(C[1][j][2]), "r"(C[1][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][1][0]), "=r"(A[kCoreIndex % 2][1][1]), "=r"(A[kCoreIndex % 2][1][2]), "=r"(A[kCoreIndex % 2][1][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(1, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][2][0]), "=r"(B[kCoreIndex % 2][2][1]), "=r"(B[kCoreIndex % 2][3][0]), "=r"(B[kCoreIndex % 2][3][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 1))));
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[2][j][0]), "=r"(C[2][j][1]), "=r"(C[2][j][2]), "=r"(C[2][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][2][0]), "r"(A[(kCoreIndex + 1) % 2][2][1]), "r"(A[(kCoreIndex + 1) % 2][2][2]), "r"(A[(kCoreIndex + 1) % 2][2][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[2][j][0]), "r"(C[2][j][1]), "r"(C[2][j][2]), "r"(C[2][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][2][0]), "=r"(A[kCoreIndex % 2][2][1]), "=r"(A[kCoreIndex % 2][2][2]), "=r"(A[kCoreIndex % 2][2][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(2, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][4][0]), "=r"(B[kCoreIndex % 2][4][1]), "=r"(B[kCoreIndex % 2][5][0]), "=r"(B[kCoreIndex % 2][5][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 2))));
                cp_async(smemB.template get<4>(), kBlockB.template get<4>());
                cp_async(smemB.template get<5>(), kBlockB.template get<5>());
                cp_async(smemB.template get<6>(), kBlockB.template get<6>());
                cp_async(smemB.template get<7>(), kBlockB.template get<7>());
                asm volatile("cp.async.commit_group;\n");
#pragma unroll
                for (size_t j = 0; j < nWarpCores; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=r"(C[3][j][0]), "=r"(C[3][j][1]), "=r"(C[3][j][2]), "=r"(C[3][j][3])
                        : "r"(A[(kCoreIndex + 1) % 2][3][0]), "r"(A[(kCoreIndex + 1) % 2][3][1]), "r"(A[(kCoreIndex + 1) % 2][3][2]), "r"(A[(kCoreIndex + 1) % 2][3][3]),
                          "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                          "r"(C[3][j][0]), "r"(C[3][j][1]), "r"(C[3][j][2]), "r"(C[3][j][3]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(A[kCoreIndex % 2][3][0]), "=r"(A[kCoreIndex % 2][3][1]), "=r"(A[kCoreIndex % 2][3][2]), "=r"(A[kCoreIndex % 2][3][3])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(3, kCoreIndex))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(B[kCoreIndex % 2][6][0]), "=r"(B[kCoreIndex % 2][6][1]), "=r"(B[kCoreIndex % 2][7][0]), "=r"(B[kCoreIndex % 2][7][1])
                             : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, 3))));
            }
        }
#pragma unroll
        for (size_t k = kBlocks - 2; k < kBlocks; ++k)
        {
            WarpMatrixA &warpA = __selectBuffer(warpA3, k % 3);
            WarpMatrixB &warpB = __selectBuffer(warpB3, k % 3);
            asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
            __syncthreads();
#pragma unroll
            for (size_t kCoreIndex = 0; kCoreIndex < kWarpCores; ++kCoreIndex)
#pragma unroll
                for (size_t i = 0; i < 4; ++i)
                {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(A[kCoreIndex % 2][i][0]), "=r"(A[kCoreIndex % 2][i][1]), "=r"(A[kCoreIndex % 2][i][2]), "=r"(A[kCoreIndex % 2][i][3])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpA.get(i, kCoreIndex))));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(B[kCoreIndex % 2][i * 2][0]), "=r"(B[kCoreIndex % 2][i * 2][1]), "=r"(B[kCoreIndex % 2][i * 2 + 1][0]), "=r"(B[kCoreIndex % 2][i * 2 + 1][1])
                                 : "r"((__uint32_t)__cvta_generic_to_shared(warpB.get(kCoreIndex, i))));
#pragma unroll
                    for (size_t j = 0; j < nWarpCores; ++j)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=r"(C[i][j][0]), "=r"(C[i][j][1]), "=r"(C[i][j][2]), "=r"(C[i][j][3])
                            : "r"(A[(kCoreIndex + 1) % 2][i][0]), "r"(A[(kCoreIndex + 1) % 2][i][1]), "r"(A[(kCoreIndex + 1) % 2][i][2]), "r"(A[(kCoreIndex + 1) % 2][i][3]),
                              "r"(B[(kCoreIndex + 1) % 2][j][0]), "r"(B[(kCoreIndex + 1) % 2][j][1]),
                              "r"(C[i][j][0]), "r"(C[i][j][1]), "r"(C[i][j][2]), "r"(C[i][j][3]));
                }
            asm volatile("cp.async.commit_group;\n");
        }
#pragma unroll
        for (size_t i = 0; i < mWarpCores; ++i)
#pragma unroll
            for (size_t j = 0; j < nWarpCores; ++j)
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=r"(C[i][j][0]), "=r"(C[i][j][1]), "=r"(C[i][j][2]), "=r"(C[i][j][3])
                    : "r"(A[1][i][0]), "r"(A[1][i][1]), "r"(A[1][i][2]), "r"(A[1][i][3]),
                      "r"(B[1][j][0]), "r"(B[1][j][1]),
                      "r"(C[i][j][0]), "r"(C[i][j][1]), "r"(C[i][j][2]), "r"(C[i][j][3]));
#pragma unroll
        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
        {
#pragma unroll
            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                asm volatile(
                    "st.shared.v2.u32 [%0], {%1, %2};\n"
                    :
                    : "r"((__uint32_t)__cvta_generic_to_shared(&(warpC[(threadIdx.x % warpSize) / 4][nCoreIndex * 8 + (threadIdx.x % 4) * 2]))), "r"(C[mCoreIndex][nCoreIndex][0]), "r"(C[mCoreIndex][nCoreIndex][1]));
#pragma unroll
            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                asm volatile(
                    "st.shared.v2.u32 [%0], {%1, %2};\n"
                    :
                    : "r"((__uint32_t)__cvta_generic_to_shared(&(warpC[(threadIdx.x % warpSize) / 4 + 8][nCoreIndex * 8 + (threadIdx.x % 4) * 2]))), "r"(C[mCoreIndex][nCoreIndex][2]), "r"(C[mCoreIndex][nCoreIndex][3]));
#pragma unroll
            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                asm volatile(
                    "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(C[mCoreIndex][nCoreIndex][0]), "=r"(C[mCoreIndex][nCoreIndex][1]), "=r"(C[mCoreIndex][nCoreIndex][2]), "=r"(C[mCoreIndex][nCoreIndex][3])
                    : "r"((__uint32_t)__cvta_generic_to_shared(&(warpC[nCoreIndex * 2 + (threadIdx.x % warpSize) / 16][(threadIdx.x % 16) * 4]))));
        }
#pragma unroll
        for (size_t mCoreIndex = 0; mCoreIndex < mWarpCores; ++mCoreIndex)
#pragma unroll
            for (size_t nCoreIndex = 0; nCoreIndex < nWarpCores; ++nCoreIndex)
                asm volatile(
                    "st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
                    :
                    : "l"(&(warpC_global[mCoreIndex * 16 + nCoreIndex * 2][0])),
                      "r"(C[mCoreIndex][nCoreIndex][0]),
                      "r"(C[mCoreIndex][nCoreIndex][1]),
                      "r"(C[mCoreIndex][nCoreIndex][2]),
                      "r"(C[mCoreIndex][nCoreIndex][3]));
    }
#ifdef COMPILING
};
template <size_t M, size_t K, size_t N>
__global__
__launch_bounds__(PTX_GEMM<M, K, N>::mBlockWarps *PTX_GEMM<M, K, N>::nBlockWarps *PTX_GEMM<M, K, N>::WarpSize) void ptx_gemm(const half *A, const half *B, float *C)
{
    extern __shared__ typename PTX_GEMM<M, K, N>::Smem smem[];
    PTX_GEMM<M, K, N>::gemm(*smem, *(const typename PTX_GEMM<M, K, N>::MatrixA_Global *)A, *(const typename PTX_GEMM<M, K, N>::MatrixB_Global *)B, *(typename PTX_GEMM<M, K, N>::MatrixC_Global *)C);
}
#endif