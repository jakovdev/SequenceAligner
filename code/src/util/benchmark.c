#include "util/benchmark.h"

#include "interface/seqalign_cuda.h"
#include "system/os.h"
#include "util/args.h"
#include "util/print.h"

static struct {
	double io_start;
	double io;
	double align_start;
	double align;
	double filter_start;
	double filter;
} g_times = { 0 };

static bool mode_benchmark;

void bench_io_start(void)
{
	if (mode_benchmark)
		g_times.io_start = time_current();
}

void bench_align_start(void)
{
	if (mode_benchmark)
		g_times.align_start = time_current();
}

void bench_filter_start(void)
{
	if (mode_benchmark)
		g_times.filter_start = time_current();
}

void bench_io_end(void)
{
	if (mode_benchmark)
		g_times.io += time_current() - g_times.io_start;
}

void bench_align_end(void)
{
	if (mode_benchmark)
		g_times.align += time_current() - g_times.align_start;
}

void bench_filter_end(void)
{
	if (mode_benchmark)
		g_times.filter += time_current() - g_times.filter_start;
}

void bench_io_print(void)
{
	if (mode_benchmark)
		pinfo("I/O operations: %.3f sec", g_times.io);
}

void bench_align_print(void)
{
	if (mode_benchmark)
		pinfo("Computation: %.3f sec", g_times.align);
}

void bench_filter_print(void)
{
	if (mode_benchmark)
		pinfo("Filtering: %.3f sec", g_times.filter);
}

void bench_total_print(s64 alignments)
{
	if (mode_benchmark) {
		double time_total = g_times.align + g_times.io + g_times.filter;
		psection("Performance Summary");
		pinfo("Timing breakdown:");

		pinfom("Compute: %.3f sec (%.1f%%)", g_times.align,
		       (g_times.align / time_total) * 100);

		pinfom("I/O: %.3f sec (%.1f%%)", g_times.io,
		       (g_times.io / time_total) * 100);

		if (g_times.filter > 0) {
			pinfom("Filtering: %.3f sec (%.1f%%)", g_times.filter,
			       (g_times.filter / time_total) * 100);
		}

		pinfol("Total: %.3f sec", time_total);

		double aps = (double)alignments / g_times.align;
		pinfo("Alignments per second: %.2f", aps);

		if ((arg_threads() > 1) && (!arg_mode_cuda())) {
			pinfom("Average time per thread: %.3f sec",
			       g_times.align / (double)arg_threads());
			pinfol("Alignments per second per thread: %.2f",
			       aps / (double)arg_threads());
		}
	}
}

static void print_benchmark(void)
{
	pinfo("Benchmarking mode: Enabled");
}

ARGUMENT(benchmark) = {
	.opt = 'B',
	.lopt = "benchmark",
	.help = "Enable timing of various steps",
	.set = &mode_benchmark,
	.action_callback = print_benchmark,
	.action_phase = ARG_CALLBACK_IF_SET,
	.help_weight = 410,
};
