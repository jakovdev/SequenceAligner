#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "args.h"
#include "print.h"

static struct
{
    double write;
    double align;
} g_times = { 0 };

#define bench_io_add(expr)                                                                         \
    do                                                                                             \
    {                                                                                              \
        double _time_start = 0.0;                                                                  \
        if (args_mode_benchmark() && args_mode_write())                                            \
        {                                                                                          \
            _time_start = time_current();                                                          \
        }                                                                                          \
        expr;                                                                                      \
        if (args_mode_benchmark() && args_mode_write())                                            \
        {                                                                                          \
            g_times.write += (time_current() - _time_start);                                       \
        }                                                                                          \
    } while (false)

static inline void
bench_align_end(void)
{
    if (args_mode_benchmark())
    {
        print(TIMING, MSG_NONE, "Computation: %.3f sec", g_times.align);
    }
}

static inline void
bench_io_end(void)
{
    if (args_mode_benchmark() && args_mode_write())
    {
        print(TIMING, MSG_NONE, "I/O operations: %.3f sec", g_times.write);
    }
}

static inline void
bench_total(size_t alignments)
{
    if (args_mode_benchmark())
    {
        double time_total = g_times.align + g_times.write * args_mode_write();
        print(SECTION, MSG_NONE, "Performance Summary");
        print(TIMING, MSG_LOC(FIRST), "Timing breakdown:");

        print(TIMING,
              MSG_LOC(MIDDLE),
              "Compute: %.3f sec (%.1f%%)",
              g_times.align,
              (g_times.align / time_total) * 100);

        if (args_mode_write())
        {
            print(TIMING,
                  MSG_LOC(MIDDLE),
                  "I/O: %.3f sec (%.1f%%)",
                  g_times.write,
                  (g_times.write / time_total) * 100);
        }

        print(TIMING, MSG_LOC(LAST), "Total: %.3f sec", time_total, 100.0);

        double alignments_per_sec = alignments / g_times.align;
        print(TIMING, MSG_LOC(FIRST), "Alignments per second: %.2f", alignments_per_sec);

        if (args_thread_num() > 1)
        {
            double avg_alignment_time = g_times.align / args_thread_num();
            print(TIMING, MSG_LOC(MIDDLE), "Average time per thread: %.3f sec", avg_alignment_time);
            print(TIMING,
                  MSG_LOC(LAST),
                  "Alignments per second per thread: %.2f",
                  alignments_per_sec / args_thread_num());
        }
    }
}

#endif