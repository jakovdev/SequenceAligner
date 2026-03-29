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

static double similarity(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2))
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

	if (!g_lengths || !g_offsets || !g_letters || !g_seqs ||
	    g_seq_n < SEQ_N_MIN) {
		pdev("Invalid globals in filter_seqs()");
		perr("Internal error during sequence filtering");
		pabort();
	}

	size_t seq_n = (size_t)g_seq_n;
	bool *MALLOCA_AL(kept, CACHE_LINE, seq_n);
	if unlikely (!kept) {
		perr("Out of memory allocating filtering array");
		return false;
	}
	memset(kept, 1, bytesof(kept, seq_n));

	if (!progress_start(seq_n - 1, arg_threads(), "Filtering sequences")) {
		free_aligned(kept);
		return false;
	}

	bench_filter_start();
#pragma omp parallel
	{
		s32 i;
#pragma omp for schedule(dynamic)
		for (i = 1; i < g_seq_n; i++) {
			bool should_keep = true;
			sequence_ptr_t seq1 = &g_seqs[i];

			for (s32 j = 0; j < i; j++) {
				if (!kept[j])
					continue;

				if (similarity(seq1, &g_seqs[j]) >= filter) {
					should_keep = false;
					break;
				}
			}

			kept[i] = should_keep;
			progress_add(1);
		}

		progress_flush();
	}
	progress_end();

	g_seq_len_max = 0;
	s32 write_index = 0;
	s64 used = 0;
	for (s32 read_index = 0; read_index < g_seq_n; read_index++) {
		if (!kept[read_index])
			continue;

		s32 len = g_lengths[read_index];
		s64 off = g_offsets[read_index];
		char *new = g_letters + used;
		if (used != off)
			memmove(new, g_letters + off, (size_t)len + 1);
		g_lengths[write_index] = len;
		g_offsets[write_index] = used;
		g_seqs[write_index].length = len;
		g_seqs[write_index++].letters = new;
		used += len + 1;
		if (len > g_seq_len_max)
			g_seq_len_max = len;
	}
	bench_filter_end();

	free_aligned(kept);

	g_seq_n = write_index;
	g_alignments = ((s64)g_seq_n * (g_seq_n - 1)) / 2;
	size_t sum = (size_t)(g_offsets[seq_n - 1] + g_lengths[seq_n - 1] + 1);

	if (seq_n > (size_t)g_seq_n && (sum - (size_t)used) >= PAGE_SIZE) {
		REALLOCA(g_lengths, (size_t)g_seq_n)
			pverb("Could not shrink sequence lengths array");
		REALLOCA(g_offsets, (size_t)g_seq_n)
			pverb("Could not shrink sequence offsets array");
		REALLOC_AL(g_letters, PAGE_SIZE, sum, (size_t)used)
			pverb("Could not shrink sequence letters array");

		for (s32 i = 0; i < g_seq_n; i++)
			g_seqs[i].letters = g_letters + g_offsets[i];

		pinfo("Filtered %zu sequences", seq_n - (size_t)g_seq_n);
	}

	if (g_seq_n < SEQ_N_MIN) {
		perr("Filtering removed too many sequences (" Ps32 " remain)",
		     g_seq_n);
		return false;
	}

	if (used - g_seq_n < SEQ_LEN_SUM_MIN) {
		perr("Not enough total sequence length after filtering: " Ps64,
		     used - g_seq_n);
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
