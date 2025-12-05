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

	pinfo("Will perform " Pu64 " pairwise alignments", alignment_count);
	perr_context("ALIGN");

	_Alignas(CACHE_LINE) _Atomic(u64) g_progress = 0;
	if (!progress_start(&g_progress, alignment_count, "Aligning sequences"))
		return false;

	bench_align_start();
	s64 g_checksum = 0;
	OMP_PARALLEL(reduction(+ : g_checksum))
	matrix_buffers_init(sequences_length_max());
	indices_buffers_init(sequences_length_max());
	s32 *MALLOC_CL(column_buffer, sequence_count);
	if (!column_buffer) {
		perr("Failed to allocate column buffer");
		exit(EXIT_FAILURE);
	}
	s64 checksum = 0;
	u64 progress = 0;

	OMP_FOR_DYNAMIC(col, 1, sequence_count) {
		OMP_START_DYNAMIC(col);
		sequence_ptr_t seq = sequence(col);
		indices_precompute(seq);
		for (u32 row = 0; row < col; row++) {
			const s32 score = align_func(seq, sequence(row));
			column_buffer[row] = score;
			checksum += score;
		}

		h5_matrix_column_set(col, column_buffer);
		progress += col;
		if (progress >= update_limit) {
			atomic_add_relaxed(&g_progress, progress);
			progress = 0;
		}
	}

	if (progress > 0)
		atomic_add_relaxed(&g_progress, progress);

	g_checksum += checksum;
	free_aligned(column_buffer);
	indices_buffers_free();
	matrix_buffers_free();
	OMP_PARALLEL_END()

	bench_align_end();
	progress_end();
	h5_checksum_set(g_checksum * 2);
	bench_align_print();
	return true;
}
