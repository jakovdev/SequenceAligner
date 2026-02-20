#include "bio/sequence/filtering.h"

#include <stdatomic.h>
#include <string.h>

#include "system/compiler.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/args.h"
#include "util/benchmark.h"
#include "util/print.h"
#include "util/progress.h"

static double filter;

static double similarity(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2))
		unreachable_release();

	s32 min_len = seq1->length < seq2->length ? seq1->length : seq2->length;
	s32 matches = 0;

	for (s32 i = 0; i < min_len; i++)
		matches += (seq1->letters[i] == seq2->letters[i]);

	return (double)matches / (double)min_len;
}

bool filter_seqs(sequence_t *seqs, bool *kept, s32 seq_n, s32 *seq_n_filter)
{
	if (!seqs || !kept || !seq_n_filter || seq_n <= SEQ_N_MIN) {
		pdev("Invalid parameters in filter_seqs()");
		perr("Internal error during sequence filtering");
		pabort();
	}

	*seq_n_filter = 0;
	size_t seq_n_s = (size_t)seq_n;
	memset(kept, 1, bytesof(kept, seq_n_s));

	if (!progress_start(seq_n_s - 1, arg_threads(), "Filtering sequences"))
		return false;

	s32 filtered_total = 0;
#pragma omp parallel reduction(+ : filtered_total)
	{
		s32 filtered = 0;
		s32 i;
#pragma omp for schedule(dynamic)
		for (i = 1; i < seq_n; i++) {
			bool should_keep = true;

			for (s32 j = 0; j < i; j++) {
				if (!kept[j])
					continue;

				double pid = similarity(&seqs[i], &seqs[j]);
				if (pid >= filter) {
					should_keep = false;
					filtered++;
					break;
				}
			}

			kept[i] = should_keep;
			progress_add(1);
		}

		progress_flush();
		filtered_total += filtered;
	}

	bench_filter_end();
	progress_end();
	bench_filter_start();
	*seq_n_filter = filtered_total;
	return true;
}

bool arg_mode_filter(void)
{
	return filter > 0.0;
}

ARG_PARSE_D(filter, double, , (val < 0.0 || val > 1.0),
	    "Filter threshold must be between 0.0 and 1.0")

static void print_filter(void)
{
	if (arg_mode_filter())
		pinfom("Filter threshold: %.1f%%", filter * 100.0);
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
	.action_order = ARG_ORDER_AFTER(gap_penalty),
	.help_order = ARG_ORDER_AFTER(list_matrices),
};
