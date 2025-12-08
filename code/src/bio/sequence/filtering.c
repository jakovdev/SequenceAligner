#include "bio/sequence/filtering.h"

#include <stdatomic.h>
#include <string.h>

#include "system/compiler.h"
#include "system/memory.h"
#include "system/os.h"
#include "system/simd.h"
#include "util/args.h"
#include "util/benchmark.h"
#include "util/print.h"
#include "util/progress.h"

static double filter;

static double similarity(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2))
		unreachable();

	s32 min_len = seq1->length < seq2->length ? seq1->length : seq2->length;
	s32 matches = 0;

#if USE_SIMD == 1
	s32 vec_limit = (min_len / BYTES) * BYTES;

	for (s32 i = 0; i < vec_limit; i += BYTES * 2) {
		prefetch(seq1->letters + i + BYTES);
		prefetch(seq2->letters + i + BYTES);
	}

	for (s32 i = 0; i < vec_limit; i += BYTES) {
		veci_t v1 = loadu((const veci_t *)(seq1->letters + i));
		veci_t v2 = loadu((const veci_t *)(seq2->letters + i));

#if defined(__AVX512F__) && defined(__AVX512BW__)
		num_t mask = cmpeq_epi8(v1, v2);
		matches += __builtin_popcountll(mask);
#else
		num_t mask = movemask_epi8(cmpeq_epi8(v1, v2));
		matches += __builtin_popcount(mask);
#endif
	}

	for (s32 i = vec_limit; i < min_len; i++)
		matches += (seq1->letters[i] == seq2->letters[i]);
#else
	for (s32 i = 0; i < min_len; i++)
		matches += (seq1->letters[i] == seq2->letters[i]);
#endif

	return (double)matches / (double)min_len;
}

bool filter_seqs(sequence_t *seqs, bool *kept, s32 seq_n, s32 *seq_n_filter)
{
	if (!seqs || !kept || !seq_n_filter || seq_n <= SEQ_N_MIN) {
		pdev("Invalid parameters in filter_seqs()");
		perr("Internal error during sequence filtering");
		exit(EXIT_FAILURE);
	}

	*seq_n_filter = 0;
	memset(kept, 1, bytesof(kept, (size_t)seq_n));

	const s64 total = seq_n - 1;
	const s64 update_limit = total / ((s64)arg_thread_num() * 100);
	_Alignas(CACHE_LINE) _Atomic(s64) g_progress = 0;
	if unlikely (!progress_start(&g_progress, total, "Filtering sequences"))
		return false;

	s32 filtered_total = 0;
#pragma omp parallel reduction(+ : filtered_total)
	{
		s64 progress = 0;
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

			if (++progress >= update_limit) {
				atomic_add_relaxed(&g_progress, progress);
				progress = 0;
			}
		}

		if (progress > 0)
			atomic_add_relaxed(&g_progress, progress);

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
	.action_weight = 500,
	.help_weight = 950,
};
