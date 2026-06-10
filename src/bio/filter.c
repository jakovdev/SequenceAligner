#include <args.h>
#include <print.h>
#include <progress.h>
#include <string.h>

#include "bio/sequence.h"
#include "io/input.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

static float threshold;

[[gnu::nonnull]]
static double similarity(const struct sequence *restrict seq1,
			 const struct sequence *restrict seq2)
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2))
		unreachable_release();

	s32 min_len = min(seq1->length, seq2->length);
	s32 matches = 0;
	for (s32 i = 0; i < min_len; i++)
		matches += (seq1->letters[i] == seq2->letters[i]);

	return (double)matches / (double)min_len;
}

bool filter(struct input *in)
{
	if (threshold <= 0.0f)
		return true;

	size_t seq_n = (size_t)in->seqs_n;
	bool *lost = calloc(seq_n, sizeof(*lost));
	if unlikely (!lost) {
		perr("Out of memory during sequence filtering");
		return false;
	}

	if (!progress_start(seq_n - 1, THREAD_NUM, "Filtering sequences")) {
		free(lost);
		return false;
	}

	s32 seqs_n = in->seqs_n;
	const struct sequence *restrict seqs = in->seqs;
	bench_filter_start();
#pragma omp parallel
	{
#pragma omp for schedule(dynamic)
		for (s32 i = 1; i < seqs_n; i++) {
			const struct sequence *restrict seq1 = &seqs[i];

			for (s32 j = 0; j < i; j++) {
				if (lost[j])
					continue;
				if (similarity(seq1, &seqs[j]) >= threshold) {
					lost[i] = true;
					break;
				}
			}

			progress_add(1);
		}

		progress_flush();
	}
	progress_end();

	in->lengths_max = 0;
	s32 write = 0;
	s64 used = 0;
	for (s32 read = 0; read < in->seqs_n; read++) {
		if (lost[read])
			continue;

		s32 len = in->lengths[read];
		s64 off = in->offsets[read];
		char *dst = in->letters + used;
		if (used != off)
			memmove(dst, in->letters + off, len + 1);
		in->lengths[write] = len;
		in->offsets[write] = used;
		in->seqs[write].length = len;
		in->seqs[write++].letters = dst;
		in->lengths_max = max(in->lengths_max, len);
		used += len + 1;
	}
	free(lost);
	bench_filter_end();

	in->seqs_n = write;
	if (write < SEQ_N_MIN) {
		perr("Not enough filtered sequences: %d (min: %d)", write,
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
