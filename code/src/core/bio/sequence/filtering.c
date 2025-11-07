#include "core/bio/sequence/filtering.h"

#include <stdatomic.h>

#include "core/app/args.h"
#include "core/bio/types.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "system/os.h"
#include "system/simd.h"
#include "util/benchmark.h"
#include "util/print.h"
#include "util/progress.h"

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
		      double filter_threshold, bool *keep_flags,
		      u32 *filtered_count)
{
	if (!sequences || !keep_flags || !filtered_count) {
		print(M_NONE, ERR "Invalid parameters to filter sequences");
		return false;
	}

	if ((sequence_count <= SEQUENCE_COUNT_MIN) ||
	    (filter_threshold <= 0.0) || (filter_threshold > 1.0)) {
		print(M_NONE, ERR "Invalid sequence count or filter threshold");
		return false;
	}

	const u64 num_threads = (u64)args_thread_num();
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
			if (similarity >= filter_threshold) {
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
