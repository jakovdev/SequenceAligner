#include "bio/sequence/filtering.h"

#include <string.h>

#include "bio/sequence/sequences.h"
#include "bio/types.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/args.h"
#include "util/benchmark.h"
#include "util/print.h"
#include "util/progress.h"

static double filter;

static double similarity(SEQUENCE_PTR_T(seq1), SEQUENCE_PTR_T(seq2))
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2))
		unreachable_release();

	const s32 min_len = min(seq1->length, seq2->length);
	s32 matches = 0;
	for (s32 i = 0; i < min_len; i++)
		matches += (seq1->letters[i] == seq2->letters[i]);

	return (double)matches / (double)min_len;
}

bool filter_seqs(void)
{
	if (!(filter > 0.0))
		return true;

	if (!LENGTHS || !OFFSETS || !LETTERS || !SEQS || SEQS_N < SEQ_N_MIN) {
		pdev("Invalid globals in filter_seqs()");
		perr("Internal error during sequence filtering");
		pabort();
	}

	size_t seq_n = (size_t)SEQS_N;
	bool *lost = calloc(seq_n, sizeof(*lost));
	if unlikely (!lost) {
		perr("Out of memory allocating filtering array");
		return false;
	}

	if (!progress_start(seq_n - 1, arg_threads(), "Filtering sequences")) {
		free(lost);
		return false;
	}

	bench_filter_start();
#pragma omp parallel
	{
		s32 i;
#pragma omp for schedule(dynamic)
		for (i = 1; i < SEQS_N; i++) {
			sequence_ptr_t seq1 = &SEQS[i];

			for (s32 j = 0; j < i; j++) {
				if (lost[j])
					continue;

				if (similarity(seq1, &SEQS[j]) >= filter) {
					lost[i] = true;
					break;
				}
			}

			progress_add(1);
		}

		progress_flush();
	}
	progress_end();

	size_t sum = (size_t)(OFFSETS[seq_n - 1] + LENGTHS[seq_n - 1] + 1);
	LENGTHS_MAX = 0;
	s32 write_index = 0;
	s64 used = 0;
	for (s32 read_index = 0; read_index < SEQS_N; read_index++) {
		if (lost[read_index])
			continue;

		s32 len = LENGTHS[read_index];
		s64 off = OFFSETS[read_index];
		char *new = LETTERS + used;
		if (used != off)
			memmove(new, LETTERS + off, (size_t)len + 1);
		LENGTHS[write_index] = len;
		OFFSETS[write_index] = used;
		SEQS[write_index].length = len;
		SEQS[write_index++].letters = new;
		used += len + 1;
		if ((size_t)len > LENGTHS_MAX)
			LENGTHS_MAX = (size_t)len;
	}
	bench_filter_end();

	free(lost);

	SEQS_N = write_index;
	ALIGNMENTS = ((s64)SEQS_N * (SEQS_N - 1)) / 2;

	if (seq_n > (size_t)SEQS_N && (sum - (size_t)used) >= PAGE_SIZE) {
		REALLOCA(LENGTHS, (size_t)SEQS_N)
			pverb("Could not shrink sequence lengths array");
		REALLOCA(OFFSETS, (size_t)SEQS_N)
			pverb("Could not shrink sequence offsets array");
		REALLOC_AL(LETTERS, PAGE_SIZE, sum, (size_t)used)
			pverb("Could not shrink sequence letters array");

		for (s32 i = 0; i < SEQS_N; i++)
			SEQS[i].letters = LETTERS + OFFSETS[i];

		pinfo("Filtered %zu sequences", seq_n - (size_t)SEQS_N);
	}

	if (SEQS_N < SEQ_N_MIN) {
		perr("Filtering removed too many sequences (" Ps32 " remain)",
		     SEQS_N);
		return false;
	}

	if (used - SEQS_N < SEQ_LEN_SUM_MIN) {
		perr("Not enough total sequence length after filtering: " Ps64,
		     used - SEQS_N);
		return false;
	}

	bench_filter_print();
	return true;
}

ARG_PARSE_D(filter, double, , (val < 0.0 || val > 1.0),
	    "Filter threshold must be between 0.0 and 1.0")

static void print_filter(void)
{
	if (filter > 0.0)
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
	.action_order = ARG_ORDER_AFTER(ARG(gap_penalty)),
	.help_order = ARG_ORDER_AFTER(ARG(list_matrices)),
};
