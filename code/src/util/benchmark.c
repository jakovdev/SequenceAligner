#include "util/benchmark.h"

#include "core/app/args.h"
#include "system/arch.h"
#include "util/print.h"

static struct
{
    double io_start;
    double io;
    double align_start;
    double align;
    double filter_start;
    double filter;
} g_times = { 0 };

void
bench_io_start(void)
{
    if (args_mode_benchmark())
    {
        g_times.io_start = time_current();
    }
}

void
bench_align_start(void)
{
    if (args_mode_benchmark())
    {
        g_times.align_start = time_current();
    }
}

void
bench_filter_start(void)
{
    if (args_mode_benchmark())
    {
        g_times.filter_start = time_current();
    }
}

void
bench_io_end(void)
{
    if (args_mode_benchmark())
    {
        g_times.io += time_current() - g_times.io_start;
    }
}

void
bench_align_end(void)
{
    if (args_mode_benchmark())
    {
        g_times.align += time_current() - g_times.align_start;
    }
}

void
bench_filter_end(void)
{
    if (args_mode_benchmark())
    {
        g_times.filter += time_current() - g_times.filter_start;
    }
}

void
bench_io_print(void)
{
    if (args_mode_benchmark())
    {
        print(TIMING, MSG_NONE, "I/O operations: %.3f sec", g_times.io);
    }
}

void
bench_align_print(void)
{
    if (args_mode_benchmark())
    {
        print(TIMING, MSG_NONE, "Computation: %.3f sec", g_times.align);
    }
}

void
bench_filter_print(size_t filtered)
{
    if (args_mode_benchmark())
    {
        print(TIMING, MSG_NONE, "Filtered %zu sequences in %.3f seconds", filtered, g_times.filter);
    }
}

void
bench_total_print(size_t alignments)
{
    if (args_mode_benchmark())
    {
        double time_total = g_times.align + g_times.io + g_times.filter;
        print(SECTION, MSG_NONE, "Performance Summary");
        print(TIMING, MSG_LOC(FIRST), "Timing breakdown:");

        double align_percent = (g_times.align / time_total) * 100;
        print(TIMING, MSG_LOC(MIDDLE), "Compute: %.3f sec (%.1f%%)", g_times.align, align_percent);

        double io_percent = (g_times.io / time_total) * 100;
        print(TIMING, MSG_LOC(MIDDLE), "I/O: %.3f sec (%.1f%%)", g_times.io, io_percent);

        if (g_times.filter > 0)
        {
            double filter_percent = (g_times.filter / time_total) * 100;
            const char* filter_msg = "Filtering: %.3f sec (%.1f%%)";
            print(TIMING, MSG_LOC(MIDDLE), filter_msg, g_times.filter, filter_percent);
        }

        print(TIMING, MSG_LOC(LAST), "Total: %.3f sec", time_total, 100.0);

        double alignments_per_sec = (double)alignments / g_times.align;
        print(TIMING, MSG_LOC(FIRST), "Alignments per second: %.2f", alignments_per_sec);

        if ((args_thread_num() > 1) && (!args_mode_cuda()))
        {
            double time_thread = g_times.align / (double)args_thread_num();
            double time_thread_sec = alignments_per_sec / (double)args_thread_num();
            print(TIMING, MSG_LOC(MIDDLE), "Average time per thread: %.3f sec", time_thread);
            print(TIMING, MSG_LOC(LAST), "Alignments per second per thread: %.2f", time_thread_sec);
        }
    }
}
