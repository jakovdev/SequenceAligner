#include "bio/algorithm/alignment.h"

#include <stdatomic.h>

#include "bio/algorithm/matrix.h"
#include "bio/algorithm/indices.h"
#include "bio/sequence/sequences.h"
#include "bio/types.h"
#include "interface/seqalign_hdf5.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/progress.h"
#include "util/print.h"

bool align(void)
{
	const align_func_t method = align_method(arg_align_method());
	const s64 total = sequences_alignments();
	const s64 num_threads = (s64)arg_thread_num();
	const s64 update_limit = max(1, total / (num_threads * 100));
	const s32 seq_n = sequences_seq_n();
	const s32 seq_len_max = sequences_seq_len_max();

	pinfo("Will perform " Ps64 " pairwise alignments", total);

	_Alignas(CACHE_LINE) _Atomic(s64) g_progress = 0;
	if unlikely (!progress_start(&g_progress, total, "Aligning sequences"))
		return false;

	bench_align_start();
	s64 g_checksum = 0;
#pragma omp parallel reduction(+ : g_checksum)
	{
		matrix_buffers_init(seq_len_max);
		indices_buffers_init(seq_len_max);
		s32 *MALLOCA_CL(column_buffer, (size_t)seq_n);
		if unlikely (!column_buffer) {
			perr("Out of memory allocating similarity matrix columns");
			exit(EXIT_FAILURE);
		}
		s64 checksum = 0;
		s64 progress = 0;
		s32 col;
#pragma omp for schedule(dynamic)
		for (col = 1; col < seq_n; col++) {
			sequence_ptr_t seq = sequence(col);
			indices_precompute(seq);
			for (s32 row = 0; row < col; row++) {
				const s32 score = method(seq, sequence(row));
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
	}

	bench_align_end();
	progress_end();
	h5_checksum_set(g_checksum * 2);
	bench_align_print();
	return true;
}
