#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "arch.h"
#include "args.h"
#include "print.h"

typedef struct
{
    double init_start;
    double init;
    double elapsed_init_time;
    double align_start;
    double align;
    double write_start;
    double write;
    double total;
    double accumulated_io_time;
    size_t total_alignments;
} BenchmarkTimes;

static BenchmarkTimes g_times = { 0 };

INLINE void
bench_init_start(void)
{
    if (args_mode_benchmark())
    {
        g_times.init_start = time_current();
        g_times.elapsed_init_time = 0.0;
    }
}

INLINE void
bench_init_end(void)
{
    if (args_mode_benchmark())
    {
        g_times.init = time_current() - g_times.init_start - g_times.elapsed_init_time;
        print(TIMING, MSG_NONE, "Initialization: %.3f sec", g_times.init);
    }
}

INLINE double
bench_pause_init(void)
{
    if (!args_mode_benchmark())
    {
        return 0.0;
    }

    double current_time = time_current();
    g_times.elapsed_init_time += (current_time - g_times.init_start);
    return current_time;
}

INLINE void
bench_resume_init(double saved_time)
{
    if (args_mode_benchmark())
    {
        g_times.init_start = time_current() - (saved_time - g_times.init_start) +
                             g_times.elapsed_init_time;
    }
}

INLINE void
bench_align_start(void)
{
    if (args_mode_benchmark())
    {
        g_times.align_start = time_current();
    }
}

INLINE void
bench_align_end(void)
{
    if (args_mode_benchmark())
    {
        g_times.align = time_current() - g_times.align_start;
        print(TIMING, MSG_NONE, "Computation: %.3f sec", g_times.align);
    }
}

INLINE void
add_io_time(double elapsed)
{
    if (args_mode_benchmark() && args_mode_write())
    {
        g_times.accumulated_io_time += elapsed;
    }
}

INLINE void
bench_write_start(void)
{
    if (args_mode_benchmark() && args_mode_write())
    {
        g_times.accumulated_io_time = 0.0;
        g_times.write_start = time_current();
    }
}

INLINE void
bench_write_end(void)
{
    if (args_mode_benchmark() && args_mode_write())
    {
        g_times.write = g_times.accumulated_io_time;
        print(TIMING, MSG_NONE, "I/O operations: %.3f sec", g_times.write);
    }
}

INLINE void
bench_set_alignments(size_t total_alignments)
{
    if (args_mode_benchmark())
    {
        g_times.total_alignments = total_alignments;
    }
}

INLINE void
bench_total(void)
{
    if (args_mode_benchmark())
    {
        g_times.total = g_times.init + g_times.align + g_times.write * args_mode_write();
        print(SECTION, MSG_NONE, "Performance Summary");
        print(TIMING, MSG_LOC(FIRST), "Timing breakdown:");

        print(TIMING,
              MSG_LOC(MIDDLE),
              "Init: %.3f",
              g_times.init,
              (g_times.init / g_times.total) * 100);

        print(TIMING,
              MSG_LOC(MIDDLE),
              "Compute: %.3f",
              g_times.align,
              (g_times.align / g_times.total) * 100);

        if (args_mode_write())
        {
            print(TIMING,
                  MSG_LOC(MIDDLE),
                  "I/O: %.3f",
                  g_times.write,
                  (g_times.write / g_times.total) * 100);
        }

        print(TIMING, MSG_LOC(LAST), "Total: %.3f", g_times.total, 100.0);

        double alignments_per_sec = g_times.total_alignments / g_times.align;
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