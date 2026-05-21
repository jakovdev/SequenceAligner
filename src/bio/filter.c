#include "bio/alignment.h"

#include <args.h>
#include <print.h>
#include <progress.h>

#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

static double threshold;

[[gnu::nonnull]]
static double similarity(seq_ptr seq1, seq_ptr seq2)
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2))
		unreachable_release();

	const s32 min_len = min(seq1->length, seq2->length);
	s32 matches = 0;
	for (s32 i = 0; i < min_len; i++)
		matches += (seq1->letters[i] == seq2->letters[i]);

	return (double)matches / (double)min_len;
}

bool filter(struct input *dataset)
{
	if (!threshold)
		return true;

	size_t seq_n = (size_t)dataset->seqs_n;
	bool *lost = calloc(seq_n, sizeof(*lost));
	if unlikely (!lost) {
		perr("Out of memory during sequence filtering");
		return false;
	}

	if (!progress_start(seq_n - 1, THREAD_NUM, "Filtering sequences")) {
		free(lost);
		return false;
	}

	s32 seqs_n = dataset->seqs_n;
	struct sequence *seqs = dataset->seqs;
	bench_filter_start();
#pragma omp parallel
	{
#pragma omp for schedule(dynamic)
		for (s32 i = 1; i < seqs_n; i++) {
			seq_ptr seq1 = &seqs[i];

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
	if (!input_lose(dataset, lost)) {
		free(lost);
		return false;
	}
	free(lost);
	bench_filter_end();
	bench_filter_print();
	return true;
}

ARG_PARSE_D(filter, double, , (val < 0.0 || val > 1.0),
	    "Filter threshold must be between 0.0 and 1.0")

static void print_filter(void)
{
	if (threshold > 0.0)
		pinfom("Filter threshold: %.1f%%", threshold * 100.0);
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
	.dest = &filter,
	.parse_callback = parse_filter,
	.action_callback = print_filter,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(gap_penalty)),
	.help_order = ARG_ORDER_AFTER(ARG(list_matrices)),
};
