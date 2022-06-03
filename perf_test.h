#pragma once
template <typename TestT, typename TimerT>
class PerfTest
{
    TestT tester;
    TimerT timer;

public:
    template <typename... Args>
    PerfTest(Args &&...args) : tester(((Args &&) args)...), timer() {}
    template <size_t warmUpCount, size_t testCount, typename... Args>
    PerfTest<TestT, TimerT> &run(const Args&...args)
    {
        for (size_t i = 0; i < warmUpCount; ++i)
            tester.run(args...);
        timer.start();
        for (size_t i = 0; i < testCount; ++i)
            tester.run(args...);
        tester.print(timer.stop() / testCount, args...);
        return *this;
    }
};