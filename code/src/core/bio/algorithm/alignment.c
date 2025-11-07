#include "core/bio/algorithm/alignment.h"

#include <stdatomic.h>

#include "core/app/args.h"
#include "core/bio/sequence/sequences.h"
#include "core/bio/types.h"
#include "core/interface/seqalign_hdf5.h"
#include "system/types.h"
#include "system/os.h"
#include "system/memory.h"
#include "util/benchmark.h"
#include "util/progress.h"

bool align(void)
{
	const u32 sequence_count = sequences_count();
	const u64 alignment_count = sequences_alignment_count();
	const u64 num_threads = (u64)args_thread_num();
	const align_func_t align_func = align_function(args_align_method());
	s64 global_checksum = 0;
	_Alignas(CACHE_LINE) _Atomic(u64) global_progress = 0;
	const u64 update_limit = max(1, alignment_count / (num_threads * 100));

	bench_align_start();
	if (!progress_start(&global_progress, alignment_count,
			    "Aligning sequences"))
		return false;

#pragma omp parallel reduction(+ : global_checksum)
	{
		s64 local_checksum = 0;
		u64 local_progress = 0;

#ifdef _MSC_VER
		s64 si;
#pragma omp for schedule(dynamic)
		for (si = 0; si < (s64)sequence_count; si++) {
			u32 i = (u32)si;
#else
#pragma omp for schedule(dynamic)
		for (u32 i = 0; i < sequence_count; i++) {
#endif
			for (u32 j = i + 1; j < sequence_count; j++) {
				sequence_ptr_t seq1 = sequence_get(i);
				sequence_ptr_t seq2 = sequence_get(j);
				s32 score = align_func(seq1, seq2);
				local_checksum += score;
				h5_matrix_set(i, j, score);
				local_progress++;
			}

			if (local_progress >= update_limit) {
				atomic_add_relaxed(&global_progress,
						   local_progress);
				local_progress = 0;
			}
		}

		if (local_progress > 0)
			atomic_add_relaxed(&global_progress, local_progress);

		global_checksum += local_checksum;
	}

	bench_align_end();
	progress_end();
	h5_checksum_set(global_checksum * 2);
	return true;
}
