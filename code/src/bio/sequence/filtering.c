#include "bio/sequence/filtering.h"

#include <stdatomic.h>

#include "util/args.h"
#include "bio/types.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "system/os.h"
#include "system/simd.h"
#include "util/benchmark.h"
#include "util/print.h"
#include "util/progress.h"

static double filter;

static double similarity_pairwise(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (UNLIKELY(!seq1->length || !seq2->length))
		return 0.0;

	u64 min_len = seq1->length < seq2->length ? seq1->length : seq2->length;
	u64 matches = 0;

#if USE_SIMD == 1
	u64 vec_limit = (min_len / BYTES) * BYTES;

	for (u64 i = 0; i < vec_limit; i += BYTES * 2) {
		prefetch(seq1->letters + i + BYTES);
		prefetch(seq2->letters + i + BYTES);
	}

	for (u64 i = 0; i < vec_limit; i += BYTES) {
		veci_t v1 = loadu((const veci_t *)(seq1->letters + i));
		veci_t v2 = loadu((const veci_t *)(seq2->letters + i));

#if defined(__AVX512F__) && defined(__AVX512BW__)
		num_t mask = cmpeq_epi8(v1, v2);
		matches += (u64)__builtin_popcountll(mask);
#else
		num_t mask = movemask_epi8(cmpeq_epi8(v1, v2));
		matches += (u64)__builtin_popcount(mask);
#endif
	}

	for (u64 i = vec_limit; i < min_len; i++)
		matches += (seq1->letters[i] == seq2->letters[i]);
#else
	for (u64 i = 0; i < min_len; i++)
		matches += (seq1->letters[i] == seq2->letters[i]);
#endif

	return (double)matches / (double)min_len;
}

bool filter_sequences(sequence_t *sequences, u32 sequence_count,
		      bool *keep_flags, u32 *filtered_count)
{
	if (!sequences || !keep_flags || !filtered_count) {
		perror("Invalid parameters to filter sequences");
		return false;
	}

	if ((sequence_count <= SEQUENCE_COUNT_MIN) || (filter <= 0.0) ||
	    (filter > 1.0)) {
		perror("Invalid sequence count or filter threshold");
		return false;
	}

	const u64 num_threads = (u64)arg_thread_num();
	const u64 progress_total = sequence_count - 1;
	_Alignas(CACHE_LINE) _Atomic(u64) g_progress = 0;
	*filtered_count = 0;
	u32 filtered_total = 0;
	const u64 update_limit = progress_total / (num_threads * 100);
	for (u32 i = 0; i < sequence_count; i++)
		keep_flags[i] = true;

	if (!progress_start(&g_progress, progress_total, "Filtering sequences"))
		return false;

	OMP_PARALLEL_REDUCTION(filtered_total, +)
	u64 progress = 0;
	u32 filtered = 0;

	OMP_FOR_DYNAMIC(i, 1, sequence_count) {
		OMP_START_DYNAMIC(i);
		bool should_keep = true;

		for (u32 j = 0; j < i; j++) {
			if (!keep_flags[j])
				continue;

			double similarity = similarity_pairwise(&sequences[i],
								&sequences[j]);
			if (similarity >= filter) {
				should_keep = false;
				filtered++;
				break;
			}
		}

		keep_flags[i] = should_keep;

		if (++progress >= update_limit) {
			atomic_add_relaxed(&g_progress, progress);
			progress = 0;
		}
	}

	if (progress > 0)
		atomic_add_relaxed(&g_progress, progress);

	filtered_total += filtered;
	OMP_PARALLEL_REDUCTION_END()

	bench_filter_end();
	progress_end();
	bench_filter_start();
	*filtered_count = filtered_total;
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
