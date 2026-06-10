#include <args.h>
#include <print.h>
#include <progress.h>
#include <string.h>

#include "bio/align.h"
#include "io/input.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

static float threshold;

bool filter(struct input *in)
{
	if (threshold <= 0.0f)
		return true;

	s32 num = in->num;
	bool *lost = calloc(num, sizeof(*lost));
	if (!lost) {
		perr("Out of memory during sequence filtering");
		return false;
	}

	if (!progress_start(num - 1, THREAD_NUM, "Filtering sequences")) {
		free(lost);
		return false;
	}

	bench_filter_start();
#pragma omp parallel
	{
#pragma omp for schedule(dynamic)
		for (s32 i = 1; i < num; i++) {
			auto m1 = in->meta[i];
			s32 l1 = m1.len;
			seq s1 = in->letters + m1.off;
			for (s32 j = 0; j < i; j++) {
				if (lost[j])
					continue;

				auto m2 = in->meta[j];
				s32 ml = min(l1, m2.len);
				seq s2 = in->letters + m2.off;
				if (LEN_BAD(ml) || SEQ_BAD(s1) || SEQ_BAD(s2))
					unreachable_release();

				s32 matches = 0;
				for (s32 k = 0; k < ml; k++)
					matches += s1[k] == s2[k];
				if ((float)matches / (float)ml >= threshold) {
					lost[i] = true;
					break;
				}
			}

			progress_add(1);
		}

		progress_flush();
	}
	progress_end();

	in->max = 0;
	in->num = 0;
	for (s32 read = 0, used = 0; read < num; read++) {
		if (lost[read])
			continue;

		auto meta = in->meta[read];
		s32 len = meta.len;
		s32 off = meta.off;
		if (used != off)
			memmove(in->letters + used, in->letters + off, len + 1);
		in->meta[in->num].off = used;
		in->meta[in->num++].len = len;
		in->max = max(in->max, len);
		used += len + 1;
	}
	free(lost);
	bench_filter_end();

	if (in->num < SEQ_N_MIN) {
		perr("Not enough filtered sequences: %d (min: %d)", in->num,
		     SEQ_N_MIN);
		return false;
	}

	bench_filter_print();
	return true;
}

ARG_PARSE_F(filter, float, , (val < 0.0f || val > 1.0f),
	    "Filter threshold must be between 0.0 and 1.0")

static void print_filter(void)
{
	if (threshold > 0.0f)
		pinfom("Filter threshold: %.1f%%", threshold * 100.0f);
	else
		pwarnm("Filter: Ignored");
}

ARG_EXTERN(gap_penalty);
ARG_EXTERN(list_matrices);

ARGUMENT(filter_threshold) = {
	.opt = 'f',
	.lopt = "filter",
	.help = "Filter sequences with similarity above threshold [0.0-1.0]",
	.param = "FLOAT",
	.param_req = ARG_PARAM_REQUIRED,
	.dest = &threshold,
	.parse_callback = parse_filter,
	.action_callback = print_filter,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(gap_penalty)),
	.help_order = ARG_ORDER_AFTER(ARG(list_matrices)),
};
