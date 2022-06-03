#include <gtest/gtest.h>
#include "gemm.h"
bool f_eq(float a, float b)
{
    return (a - b) / (a + b + 1.0f) < 0.001f;
}
TEST(ptx_gemm, test_8192_8192_8192)
{
    MatMul<8192, 8192, 8192> matmul{};
    half *a = matmul.getA();
    for (int i = 0; i < 8192 * 8192; ++i)
        a[i] = half(((i / 8192) % 128) + ((i / 8192) / 64));
    a = matmul.getB();
    for (int i = 0; i < 8192 * 8192; ++i)
        a[i] = half(((i / 8192) % 128) + ((i / 8192) / 64));
    matmul.set();
    auto ans1 = new float[8192 * 8192];
    matmul.run(-1);
    memcpy(ans1, matmul.get(), 8192 * 8192 * sizeof(float));
    matmul.run(1);
    auto ans2 = matmul.get();
    int diff = 0;
    for (int i = 0; i < 8192 * 8192; ++i)
        if (!f_eq(ans1[i], ans2[i]))
            ++diff;
    EXPECT_EQ(diff, 0);
    delete[] ans1;
};
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}