#include <gtest/gtest.h>
#include "gemm.h"
bool f_eq(float a, float b)
{
    if (((a - b) / (a + b + 0.1f) < 0.1f) && (b - a) / (a + b + 0.1f) < 0.1f)
        return true;
    return a == b;
}
TEST(my_gemm, test_8192_8192_8192)
{
    MatMul<8192, 8192, 8192> matmul{};
    half *a = matmul.getA();
    for (int i = 0; i < 8192 * 8192; ++i)
        a[i] = half((i % 8192) + (i / 8192));
    a = matmul.getB();
    for (int i = 0; i < 8192 * 8192; ++i)
        a[i] = half((i % 8192) - (i / 8192));
    matmul.set();
    auto ans1 = new float[8192 * 8192];
    matmul.run(0);
    memcpy(ans1, matmul.get(), 8192 * 8192 * sizeof(float));
    matmul.run(1);
    auto ans2 = matmul.get();
    int diff = 0;
    for (int i = 0; i < 8192 * 8192; ++i)
        if (!f_eq(ans1[i], ans2[i]))
        {
            ++diff;
            printf("%u, %u, %g, %g\n", i / 8192, i % 8192, ans1[i], ans2[i]);
        }
    EXPECT_EQ(diff, 0);
    delete[] ans1;
};
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}