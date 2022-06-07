#include <gtest/gtest.h>
#include "gemm.h"
bool f_eq(float a, float b)
{
    if (((a - b) / (a + b + 0.1f) < 0.1f) && (b - a) / (a + b + 0.1f) < 0.1f)
        return true;
    return a==b;
}
TEST(ptx_gemm, test_256_256_256)
{
    MatMul<256, 256, 256> matmul{};
    half *a = matmul.getA();
    for (int i = 0; i < 256 * 256; ++i)
        a[i] = half(i/2);
    a = matmul.getB();
    for (int i = 0; i < 256 * 256; ++i)
        a[i] = half(((i % 256)) - ((i / 256)));
    matmul.set();
    auto ans1 = new float[256 * 256];
    matmul.run(0);
    memcpy(ans1, matmul.get(), 256 * 256 * sizeof(float));
    matmul.run(1);
    auto ans2 = matmul.get();
    int diff = 0;
    for (int i = 0; i < 256 * 256; ++i)
        if (!f_eq(ans1[i], ans2[i]))
        {
            ++diff;
            printf("%u, %u, %g, %g\n", i / 256, i % 256, ans1[i], ans2[i]);
        }
    EXPECT_EQ(diff, 0);
    delete[] ans1;
};
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}