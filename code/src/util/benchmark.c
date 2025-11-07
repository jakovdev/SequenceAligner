#include "util/benchmark.h"

#include "core/app/args.h"
#include "system/os.h"
#include "util/print.h"

static struct {
	double io_start;
	double io;
	double align_start;
	double align;
	double filter_start;
	double filter;
} g_times = { 0 };

void bench_io_start(void)
{
	if (args_mode_benchmark()) {
		g_times.io_start = time_current();
	}
}

void bench_align_start(void)
{
	if (args_mode_benchmark()) {
		g_times.align_start = time_current();
	}
}

void bench_filter_start(void)
{
	if (args_mode_benchmark()) {
		g_times.filter_start = time_current();
	}
}

void bench_io_end(void)
{
	if (args_mode_benchmark()) {
		g_times.io += time_current() - g_times.io_start;
	}
}

void bench_align_end(void)
{
	if (args_mode_benchmark()) {
		g_times.align += time_current() - g_times.align_start;
	}
}

void bench_filter_end(void)
{
	if (args_mode_benchmark()) {
		g_times.filter += time_current() - g_times.filter_start;
	}
}

void bench_io_print(void)
{
	if (args_mode_benchmark()) {
		print(M_NONE, INFO "I/O operations: %.3f sec", g_times.io);
	}
}

void bench_align_print(void)
{
	if (args_mode_benchmark()) {
		print(M_NONE, INFO "Computation: %.3f sec", g_times.align);
	}
}

void bench_filter_print(u64 filtered)
{
	if (args_mode_benchmark()) {
		print(M_NONE,
		      INFO "Filtered " Pu64 " sequences in %.3f seconds",
		      filtered, g_times.filter);
	}
}

void bench_total_print(u64 alignments)
{
	if (args_mode_benchmark()) {
		double time_total = g_times.align + g_times.io + g_times.filter;
		print(M_NONE, SECTION "Performance Summary");
		print(M_LOC(FIRST), INFO "Timing breakdown:");

		double align_percent = (g_times.align / time_total) * 100;
		print(M_LOC(MIDDLE), INFO "Compute: %.3f sec (%.1f%%)",
		      g_times.align, align_percent);

		double io_percent = (g_times.io / time_total) * 100;
		print(M_LOC(MIDDLE), INFO "I/O: %.3f sec (%.1f%%)", g_times.io,
		      io_percent);

		if (g_times.filter > 0) {
			double f_percent = (g_times.filter / time_total) * 100;
			const char *f_msg = INFO "Filtering: %.3f sec (%.1f%%)";
			print(M_LOC(MIDDLE), f_msg, g_times.filter, f_percent);
		}

		print(M_LOC(LAST), INFO "Total: %.3f sec", time_total, 100.0);

		double aps = (double)alignments / g_times.align;
		print(M_LOC(FIRST), INFO "Alignments per second: %.2f", aps);

		if ((args_thread_num() > 1) && (!args_mode_cuda())) {
			double time_thread =
				g_times.align / (double)args_thread_num();
			double time_thread_sec =
				aps / (double)args_thread_num();
			print(M_LOC(MIDDLE),
			      INFO "Average time per thread: %.3f sec",
			      time_thread);
			print(M_LOC(LAST),
			      INFO "Alignments per second per thread: %.2f",
			      time_thread_sec);
		}
	}
}
