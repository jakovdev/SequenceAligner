#include "core/bio/algorithm/alignment.h"

#include <stdatomic.h>

#include "core/app/args.h"
#include "core/bio/sequence/sequences.h"
#include "core/bio/types.h"
#include "core/interface/seqalign_hdf5.h"
#include "system/compiler.h"
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
	s64 g_checksum = 0;
	_Alignas(CACHE_LINE) _Atomic(u64) g_progress = 0;
	const u64 update_limit = max(1, alignment_count / (num_threads * 100));

	bench_align_start();
	if (!progress_start(&g_progress, alignment_count, "Aligning sequences"))
		return false;

	OMP_PARALLEL_REDUCTION(g_checksum, +)
	s64 checksum = 0;
	u64 progress = 0;

	OMP_FOR_DYNAMIC(i, 0, sequence_count) {
		OMP_START_DYNAMIC(i);
		for (u32 j = i + 1; j < sequence_count; j++) {
			sequence_ptr_t seq1 = sequence_get(i);
			sequence_ptr_t seq2 = sequence_get(j);
			s32 score = align_func(seq1, seq2);
			checksum += score;
			h5_matrix_set(i, j, score);
			progress++;
		}

		if (progress >= update_limit) {
			atomic_add_relaxed(&g_progress, progress);
			progress = 0;
		}
	}

	if (progress > 0)
		atomic_add_relaxed(&g_progress, progress);

	g_checksum += checksum;
	OMP_PARALLEL_REDUCTION_END()

	bench_align_end();
	progress_end();
	h5_checksum_set(g_checksum * 2);
	return true;
}
