#include "bio/algorithm/alignment.h"

#include <stdatomic.h>

#include "bio/algorithm/matrix.h"
#include "bio/algorithm/indices.h"
#include "bio/sequence/sequences.h"
#include "bio/types.h"
#include "interface/seqalign_hdf5.h"
#include "system/compiler.h"
#include "system/os.h"
#include "system/memory.h"
#include "util/benchmark.h"
#include "util/progress.h"
#include "util/print.h"

bool align(void)
{
	const align_func_t align_func = align_function(arg_align_method());
	const u32 sequence_count = sequences_count();
	const u64 alignment_count = sequences_alignment_count();
	const u64 num_threads = (u64)arg_thread_num();
	const u64 update_limit = max(1, alignment_count / (num_threads * 100));
	_Alignas(CACHE_LINE) _Atomic(u64) g_progress = 0;
	s64 g_checksum = 0;

	pinfo("Will perform " Pu64 " pairwise alignments", alignment_count);
	perr_context("ALIGN");

	bench_align_start();
	if (!progress_start(&g_progress, alignment_count, "Aligning sequences"))
		return false;

	OMP_PARALLEL(reduction(+ : g_checksum))
	matrix_buffers_init(sequences_length_max());
	indices_buffers_init(sequences_length_max());
	s64 checksum = 0;
	u64 progress = 0;

	OMP_FOR_DYNAMIC(i, 0, sequence_count) {
		OMP_START_DYNAMIC(i);
		sequence_ptr_t seq1 = sequence(i);
		indices_precompute(seq1);
		for (u32 j = i + 1; j < sequence_count; j++) {
			sequence_ptr_t seq2 = sequence(j);
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
	indices_buffers_free();
	matrix_buffers_free();
	OMP_PARALLEL_END()

	bench_align_end();
	progress_end();
	h5_checksum_set(g_checksum * 2);
	bench_align_print();
	return true;
}
