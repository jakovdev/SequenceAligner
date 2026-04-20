#include "util/benchmark.h"

#include <args.h>
#include <print.h>

#include "system/os.h"

static struct {
	double total;
	double input_start;
	double input;
	double output_start;
	double output;
	double align_start;
	double align;
	double filter_start;
	double filter;
} g_times;

static bool mode_benchmark;
#define BENCH(type, name)                                                    \
	void bench_##type##_start(void)                                      \
	{                                                                    \
		if (mode_benchmark)                                          \
			g_times.type##_start = time_current();               \
	}                                                                    \
	void bench_##type##_end(void)                                        \
	{                                                                    \
		if (mode_benchmark) {                                        \
			double time = time_current() - g_times.type##_start; \
			g_times.type += time;                                \
			g_times.total += time;                               \
		}                                                            \
	}                                                                    \
	void bench_##type##_print(void)                                      \
	{                                                                    \
		if (mode_benchmark)                                          \
			pinfo(name ": %.3f sec", g_times.type);              \
	}

BENCH(input, "Input")
BENCH(filter, "Filtering")
BENCH(align, "Alignment")
BENCH(output, "Output")

#define BENCH_TOTAL(type, name)                          \
	pinfom(name ": %.3f sec (%.1f%%)", g_times.type, \
	       (g_times.type / g_times.total) * 100)

void bench_total_print(double alignments)
{
	if (!mode_benchmark)
		return;
	psection("Performance Summary");
	pinfo("Timing breakdown:");
	BENCH_TOTAL(input, "Input");
	if (g_times.filter > 0.0)
		BENCH_TOTAL(filter, "Filtering");
	BENCH_TOTAL(align, "Alignment");
	if (g_times.output > 0.0)
		BENCH_TOTAL(output, "Output");
	pinfol("Total: %.3f sec", g_times.total);
	pinfo("Alignments per second: %.2f", alignments / g_times.align);
}

static void print_benchmark(void)
{
	pinfo("Benchmarking mode: Enabled");
}

ARG_EXTERN(compression);

ARGUMENT(benchmark) = {
	.opt = 'B',
	.lopt = "benchmark",
	.help = "Enable timing of various steps",
	.set = &mode_benchmark,
	.action_callback = print_benchmark,
	.action_phase = ARG_CALLBACK_IF_SET,
	.help_order = ARG_ORDER_AFTER(ARG(compression)),
};
